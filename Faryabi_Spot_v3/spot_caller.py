"""
spot_caller.py  —  Faryabi Spot Calling  v3  —  adaptive smFISH spot detector
════════════════════════════════════════════════════════════════════════════════

Classical detector: LoG bandpass → adaptive MAD threshold → NMS → Gaussian fit.
No trained model needed. Works on any smFISH data out of the box.

GPU ACCELERATION
  Install CuPy matching your CUDA version:
    pip install cupy-cuda12x   # CUDA 12
    pip install cupy-cuda11x   # CUDA 11
  CuPy is detected automatically. Falls back to NumPy/SciPy silently if absent.
  GPU accelerates: rolling-ball background, LoG filter, NMS (~10-50x speedup).
  Per-spot Gaussian fitting stays on CPU (inherently serial, small patches).

ALGORITHM
  1. Rolling-ball background subtraction  (morphological opening)
  2. LoG bandpass filter                  (matched to expected PSF)
  3. Adaptive MAD threshold map           (per-tile, bilinear interpolation)
  4. Non-maximum suppression              (local maxima above threshold)
  5. 2-D Gaussian sub-pixel fit           (sub-pixel xy, SNR, sigma, amplitude)
  6. Quality filtering                    (SNR, sigma bounds)

DROP-IN PISCIS REPLACEMENT
  from spot_caller import SpotCaller
  model  = SpotCaller(spot_yx=3, threshold=5.0)
  spots  = model.predict(image_2d)     # (N,2) [y,x] — same as Piscis
  df     = model.detect(image_2d)      # full DataFrame with quality metrics
  df_3d  = model.detect_3d(stack_zyx)  # 3-D detection
"""

import argparse
import csv
import logging
import warnings
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage
from scipy.ndimage import (
    gaussian_filter,
    maximum_filter,
    binary_erosion,
    generate_binary_structure,
    label as nd_label,
)
from scipy.optimize import least_squares

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── GPU via CuPy (optional, auto-detected) ────────────────────────────────────
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpnd
    _CUPY_AVAILABLE = cp.cuda.is_available()
except ImportError:
    cp = None
    cpnd = None
    _CUPY_AVAILABLE = False

log = logging.getLogger(__name__)


class SpotCaller:
    """
    Adaptive smFISH spot detector.

    Parameters
    ----------
    spot_yx : float
        Expected spot radius in XY (pixels). Typically 2–4 for smFISH.
    spot_z : float
        Expected spot radius in Z (planes). For 3-D detection only.
    threshold : float
        Detection threshold in MAD units above local background. 4–6 typical.
    tile_size : int
        Tile size for local noise estimation (pixels).
    min_snr : float
        Minimum SNR after Gaussian fitting.
    max_sigma_ratio : float
        Reject spots wider than N × spot_yx (aggregates, debris).
    subpixel : bool
        2-D Gaussian fitting for sub-pixel localisation.
    use_gpu : bool | None
        None = auto-detect CuPy. True/False = explicit override.
    """

    def __init__(
        self,
        spot_yx: float = 3.0,
        spot_z:  float = 2.0,
        threshold: float = 5.0,
        tile_size: int = 128,
        min_snr: float = 2.0,
        max_sigma_ratio: float = 2.5,
        subpixel: bool = True,
        use_gpu: "bool | None" = None,
    ):
        self.spot_yx         = float(spot_yx)
        self.spot_z          = float(spot_z)
        self.threshold       = float(threshold)
        self.tile_size       = int(tile_size)
        self.min_snr         = float(min_snr)
        self.max_sigma_ratio = float(max_sigma_ratio)
        self.subpixel        = subpixel

        # GPU
        if use_gpu is None:
            self.use_gpu = _CUPY_AVAILABLE
        else:
            self.use_gpu = bool(use_gpu) and _CUPY_AVAILABLE

        if self.use_gpu:
            log.info("SpotCaller: CuPy GPU — LoG/NMS on GPU, Gaussian fit on CPU")
        else:
            log.debug("SpotCaller: CPU mode (CuPy %s)",
                      "not installed" if not _CUPY_AVAILABLE else "disabled")

    # ── Public interface (Piscis-compatible) ──────────────────────────────────

    def predict(self, image: np.ndarray,
                threshold: "float | None" = None) -> np.ndarray:
        """Detect spots. Returns (N,2) [y,x]. Same signature as Piscis."""
        thr    = threshold if threshold is not None else self.threshold
        result = self.detect(image, threshold_mad=thr)
        if len(result) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return result[["y_sub", "x_sub"]].values.astype(np.float32)

    # ── 2-D detection ─────────────────────────────────────────────────────────

    def detect(self, image: np.ndarray,
               threshold_mad: "float | None" = None) -> "pd.DataFrame":
        """
        Full 2-D detection returning a DataFrame with all quality metrics.
        image: 2-D float array (already projected / best-plane selected).
        """
        import pandas as pd
        thr = threshold_mad if threshold_mad is not None else self.threshold
        img = np.asarray(image, dtype=np.float32)
        if img.ndim != 2:
            raise ValueError(f"Expected 2-D image, got shape {img.shape}")

        if self.use_gpu:
            # GPU path: heavy array ops on GPU
            img_gpu = cp.asarray(img)
            bg_gpu  = self._rolling_ball_gpu(img_gpu, self.spot_yx * 3)
            sub_gpu = img_gpu - bg_gpu
            log_gpu = self._log_filter_gpu(sub_gpu, self.spot_yx / np.sqrt(2))
            thr_map = self._mad_threshold_map(cp.asnumpy(log_gpu), thr)  # tiles on CPU
            peaks   = self._nms_peaks_gpu(log_gpu, thr_map, max(1, int(self.spot_yx)))
            sub     = cp.asnumpy(sub_gpu)
            # Free GPU memory
            del img_gpu, bg_gpu, sub_gpu, log_gpu
            cp.get_default_memory_pool().free_all_blocks()
        else:
            # CPU path
            bg      = self._rolling_ball(img, self.spot_yx * 3)
            sub     = img - bg
            log_img = self._log_filter(sub, self.spot_yx / np.sqrt(2))
            thr_map = self._mad_threshold_map(log_img, thr)
            peaks   = self._nms_peaks(log_img, thr_map, max(1, int(self.spot_yx)))

        if len(peaks) == 0:
            return pd.DataFrame(columns=[
                "y","x","y_sub","x_sub","z_plane",
                "snr","sigma_y","sigma_x","fit_residual","amplitude"])

        # Per-spot Gaussian fitting — always CPU (inherently serial)
        records = []
        half    = max(3, int(self.spot_yx * 2))
        H, W    = img.shape

        for (r, c) in peaks:
            r0,r1 = max(0,r-half), min(H,r+half+1)
            c0,c1 = max(0,c-half), min(W,c+half+1)
            patch = sub[r0:r1, c0:c1].copy()

            rec = {"y": float(r), "x": float(c),
                   "y_sub": float(r), "x_sub": float(c),
                   "z_plane": 0, "snr": 0.0,
                   "sigma_y": self.spot_yx, "sigma_x": self.spot_yx,
                   "fit_residual": 0.0, "amplitude": float(patch.max())}

            if self.subpixel and patch.size >= 9:
                fit = self._fit_gaussian_2d(patch)
                if fit is not None:
                    amp, y0, x0, sy, sx, bg_fit = fit
                    rec["y_sub"]        = r0 + y0
                    rec["x_sub"]        = c0 + x0
                    rec["sigma_y"]      = abs(sy)
                    rec["sigma_x"]      = abs(sx)
                    rec["amplitude"]    = amp
                    rec["fit_residual"] = bg_fit
                    edge  = self._patch_edge(patch)
                    noise = max(1e-9, 1.4826 * np.median(np.abs(edge - np.median(edge))))
                    rec["snr"] = amp / noise

            records.append(rec)

        df = pd.DataFrame(records)
        df = df[df["snr"] >= self.min_snr]
        if len(df):
            df = df[df["sigma_y"] <= self.spot_yx * self.max_sigma_ratio]
            df = df[df["sigma_x"] <= self.spot_yx * self.max_sigma_ratio]
            df = df[df["sigma_y"] >= self.spot_yx * 0.2]
            df = df[df["sigma_x"] >= self.spot_yx * 0.2]
        return df.reset_index(drop=True)

    # ── 3-D detection ─────────────────────────────────────────────────────────

    def detect_3d(self, stack: np.ndarray,
                  threshold_mad: "float | None" = None) -> "pd.DataFrame":
        """
        3-D LoG detection on (Z,Y,X) stack, refined with per-plane Gaussian fit.
        """
        import pandas as pd
        thr   = threshold_mad if threshold_mad is not None else self.threshold
        stack = np.asarray(stack, dtype=np.float32)
        if stack.ndim != 3:
            raise ValueError(f"Expected 3-D stack, got shape {stack.shape}")

        # 3-D background subtraction
        r   = max(1, int(round(self.spot_yx * 3)))
        bg  = ndimage.grey_opening(stack, size=(1, 2*r+1, 2*r+1))
        sub = stack - bg

        # 3-D LoG
        sigma   = [self.spot_z / np.sqrt(2), self.spot_yx / np.sqrt(2),
                   self.spot_yx / np.sqrt(2)]
        log3d   = np.clip(-ndimage.gaussian_laplace(sub, sigma=sigma), 0, None)

        # 3-D NMS
        footprint = generate_binary_structure(3, 1)
        for _ in range(int(max(self.spot_yx, self.spot_z))):
            footprint = ndimage.binary_dilation(footprint)
        local_max = (log3d == maximum_filter(log3d, footprint=footprint))

        # Threshold using max-projection of the threshold map
        mip_log = log3d.max(axis=0)
        thr_map = self._mad_threshold_map(mip_log, thr)
        above   = log3d >= thr_map[np.newaxis, :, :]
        peaks3d = np.argwhere(local_max & above)   # (N,3) [z,y,x]

        if len(peaks3d) == 0:
            return pd.DataFrame(columns=[
                "y","x","y_sub","x_sub","z_plane",
                "snr","sigma_y","sigma_x","fit_residual","amplitude"])

        records = []
        half    = max(3, int(self.spot_yx * 2))
        Z,H,W   = stack.shape

        for (z,r,c) in peaks3d:
            r0,r1 = max(0,r-half), min(H,r+half+1)
            c0,c1 = max(0,c-half), min(W,c+half+1)
            patch = sub[z, r0:r1, c0:c1].copy()

            rec = {"y": float(r), "x": float(c),
                   "y_sub": float(r), "x_sub": float(c),
                   "z_plane": int(z), "snr": 0.0,
                   "sigma_y": self.spot_yx, "sigma_x": self.spot_yx,
                   "fit_residual": 0.0, "amplitude": float(patch.max())}

            if self.subpixel and patch.size >= 9:
                fit = self._fit_gaussian_2d(patch)
                if fit is not None:
                    amp, y0, x0, sy, sx, bg_fit = fit
                    rec["y_sub"]        = r0 + y0
                    rec["x_sub"]        = c0 + x0
                    rec["sigma_y"]      = abs(sy)
                    rec["sigma_x"]      = abs(sx)
                    rec["amplitude"]    = amp
                    rec["fit_residual"] = bg_fit
                    edge  = self._patch_edge(patch)
                    noise = max(1e-9, 1.4826*np.median(np.abs(edge-np.median(edge))))
                    rec["snr"] = amp / noise

            records.append(rec)

        import pandas as pd
        df = pd.DataFrame(records)
        df = df[df["snr"] >= self.min_snr]
        if len(df):
            df = df[df["sigma_y"] <= self.spot_yx * self.max_sigma_ratio]
            df = df[df["sigma_x"] <= self.spot_yx * self.max_sigma_ratio]
            df = df[df["sigma_y"] >= self.spot_yx * 0.2]
            df = df[df["sigma_x"] >= self.spot_yx * 0.2]
        return df.reset_index(drop=True)

    # ── GPU methods (CuPy) ────────────────────────────────────────────────────

    @staticmethod
    def _rolling_ball_gpu(img_gpu, radius: float):
        """Rolling-ball background via morphological opening (CuPy)."""
        r    = max(1, int(round(radius)))
        y, x = cp.ogrid[-r:r+1, -r:r+1]
        disk = (x*x + y*y <= r*r)
        from cupyx.scipy.ndimage import minimum_filter as cp_min, maximum_filter as cp_max
        return cp_max(cp_min(img_gpu, footprint=disk), footprint=disk)

    @staticmethod
    def _log_filter_gpu(img_gpu, sigma: float):
        """Laplacian of Gaussian via CuPy (finite differences)."""
        blurred = cpnd.gaussian_filter(img_gpu, sigma=sigma)
        lap = (cp.roll(blurred,1,0) + cp.roll(blurred,-1,0) +
               cp.roll(blurred,1,1) + cp.roll(blurred,-1,1) - 4*blurred)
        return cp.clip(-lap, 0, None)

    @staticmethod
    def _nms_peaks_gpu(log_gpu, thr_map_cpu: np.ndarray, radius: int) -> np.ndarray:
        """GPU NMS. Returns peak coordinates as CPU numpy array."""
        size      = 2*radius + 1
        thr_gpu   = cp.asarray(thr_map_cpu)
        local_max = (log_gpu == cpnd.maximum_filter(log_gpu, size=size))
        peaks_gpu = cp.argwhere(local_max & (log_gpu >= thr_gpu))
        return cp.asnumpy(peaks_gpu)

    # ── CPU methods ───────────────────────────────────────────────────────────

    @staticmethod
    def _rolling_ball(img: np.ndarray, radius: float) -> np.ndarray:
        r    = max(1, int(round(radius)))
        y, x = np.ogrid[-r:r+1, -r:r+1]
        disk = (x*x + y*y <= r*r).astype(np.uint8)
        return ndimage.grey_opening(img, structure=disk)

    @staticmethod
    def _disk_struct(r: int) -> np.ndarray:
        y, x = np.ogrid[-r:r+1, -r:r+1]
        return (x*x + y*y <= r*r).astype(np.uint8)

    @staticmethod
    def _log_filter(img: np.ndarray, sigma: float) -> np.ndarray:
        return np.clip(-ndimage.gaussian_laplace(img, sigma=sigma), 0, None)

    def _mad_threshold_map(self, img: np.ndarray, k: float) -> np.ndarray:
        """Per-tile adaptive threshold map, bilinearly interpolated."""
        H, W = img.shape
        ts   = self.tile_size
        n_r  = max(1, H // ts)
        n_c  = max(1, W // ts)
        rows = np.linspace(0, H, n_r+1, dtype=int)
        cols = np.linspace(0, W, n_c+1, dtype=int)
        grid   = np.zeros((n_r, n_c), dtype=np.float32)
        grid_y = np.zeros(n_r)
        grid_x = np.zeros(n_c)
        for i in range(n_r):
            for j in range(n_c):
                tile = img[rows[i]:rows[i+1], cols[j]:cols[j+1]].ravel()
                tile = tile[tile > 0]
                if len(tile) < 10:
                    grid[i,j] = img.max(); continue
                med = np.median(tile)
                grid[i,j] = med + k * 1.4826 * np.median(np.abs(tile - med))
            grid_y[i] = (rows[i]+rows[i+1]) / 2
        for j in range(n_c):
            grid_x[j] = (cols[j]+cols[j+1]) / 2
        if n_r == 1 and n_c == 1:
            return np.full_like(img, grid[0,0])
        try:
            from scipy.interpolate import RegularGridInterpolator
            interp  = RegularGridInterpolator(
                (grid_y, grid_x), grid, method="linear",
                bounds_error=False, fill_value=None)
            yy, xx  = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            pts     = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float32)
            return interp(pts).reshape(H, W).astype(np.float32)
        except Exception:
            return np.full_like(img, float(np.median(grid)))

    @staticmethod
    def _nms_peaks(img: np.ndarray, thr_map: np.ndarray, radius: int) -> np.ndarray:
        size      = 2*radius + 1
        local_max = (img == maximum_filter(img, size=size))
        return np.argwhere(local_max & (img >= thr_map))

    @staticmethod
    def _patch_edge(patch: np.ndarray) -> np.ndarray:
        return np.concatenate([patch[0,:], patch[-1,:],
                               patch[1:-1,0], patch[1:-1,-1]])

    @staticmethod
    def _fit_gaussian_2d(patch: np.ndarray):
        """Fit 2-D Gaussian + background. Returns (amp,y0,x0,sy,sx,bg) or None."""
        try:
            from scipy.optimize import curve_fit
            H, W   = patch.shape
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            p      = patch - patch.min()
            total  = p.sum() + 1e-9
            y0     = (yy*p).sum() / total
            x0     = (xx*p).sum() / total
            amp0   = float(patch.max() - patch.min())
            sig0   = float(max(1.0, (H+W)/8))
            bg0    = float(patch.min())

            def gauss2d(xy, amp, y0, x0, sy, sx, bg):
                y, x = xy
                return amp * np.exp(-0.5*((y-y0)**2/sy**2 + (x-x0)**2/sx**2)) + bg

            popt, _ = curve_fit(
                gauss2d, (yy.ravel(), xx.ravel()), patch.ravel(),
                p0=[amp0,y0,x0,sig0,sig0,bg0],
                bounds=([0,0,0,0.3,0.3,-np.inf],[np.inf,H,W,H,W,np.inf]),
                maxfev=200)
            return tuple(popt)
        except Exception:
            return None


# ════════════════════════════════════════════════════════════════════════════════
# Standalone CLI (optional — normally called from run_piscis.py)
# ════════════════════════════════════════════════════════════════════════════════

def save_preview_png(plane, out_path, p_low=1.0, p_high=99.5):
    lo = np.percentile(plane, p_low)
    hi = np.percentile(plane, p_high)
    s  = np.clip((plane-lo)/max(hi-lo,1e-9), 0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((s*255).astype(np.uint8)).save(out_path)


def detect_channel(stem):
    u = stem.upper()
    if "DAPI" in u: return "DAPI"
    if "CY3"  in u: return "Cy3"
    if "CY5"  in u: return "Cy5"
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="SpotCaller standalone")
    p.add_argument("--input_dir",  required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--spot_yx",    type=float, default=3.0)
    p.add_argument("--threshold",  type=float, default=5.0)
    p.add_argument("--min_snr",    type=float, default=2.0)
    p.add_argument("--use_3d",     action="store_true")
    p.add_argument("--no_gpu",     action="store_true")
    p.add_argument("--save_previews", action="store_true")
    p.add_argument("--percentile_low",  type=float, default=1.0)
    p.add_argument("--percentile_high", type=float, default=99.5)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = SpotCaller(spot_yx=args.spot_yx, threshold=args.threshold,
                       min_snr=args.min_snr, use_gpu=not args.no_gpu)

    for tif in sorted(args.input_dir.glob("*.tif")):
        ch = detect_channel(tif.stem)
        if ch in (None, "DAPI"): continue
        with tifffile.TiffFile(tif) as f:
            data = f.asarray().astype(np.float32)
        stack = data if data.ndim == 3 else data[np.newaxis]
        plane = stack.max(axis=0)
        df    = (model.detect_3d(stack) if args.use_3d else model.detect(plane))
        out   = args.output_dir / f"{tif.stem}_spots.csv"
        df.insert(0, "channel", ch)
        df.to_csv(out, index=False)
        log.info("%s → %d spots → %s", tif.name, len(df), out.name)
        if args.save_previews:
            save_preview_png(plane,
                args.output_dir / "previews" / f"{tif.stem}.png",
                args.percentile_low, args.percentile_high)
