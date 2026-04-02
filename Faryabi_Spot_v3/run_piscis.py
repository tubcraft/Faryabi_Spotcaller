#!/usr/bin/env python3
"""
run_piscis.py  —  Faryabi Spot Calling  v3
════════════════════════════════════════════════════════════════════════════════

Spot detection (Piscis or SpotCaller) + Cellpose cell segmentation for smFISH.

ENVIRONMENT
  conda activate piscis_env_v2          # your existing environment
  pip install flask nd2                 # only if not already installed
  # GPU already configured via jax[cuda12_pip] + cupy-cuda12x

QUICK START
  # Piscis with GPU
  python run_piscis.py \\
      --input_dir  /data/C1_left/all_images \\
      --output_dir /results/C1_left \\
      --detector   piscis --model 20251212 --threshold 0.5 \\
      --cellpose_model Cellpose_models/20251103_CD4_ORCA_RNA \\
      --cellpose_diam 80 --expand_mask 3 \\
      --workers 4 --save_previews

  # SpotCaller with GPU (CuPy)
  python run_piscis.py \\
      --input_dir  /data/C1_left/all_images \\
      --output_dir /results/C1_left \\
      --detector   spot_caller --threshold 6 --spot_yx 3 --use_3d \\
      --workers 20 --save_previews

  # Disable GPU explicitly
  python run_piscis.py ... --no_gpu

GPU NOTES
  Piscis    : JAX auto-detects CUDA — no configuration needed.
  SpotCaller: CuPy accelerates LoG/NMS/thresholding; Gaussian fit stays CPU.
  Cellpose  : always CPU (avoids GPU memory contention with Piscis workers).
  Workers   : for GPU use --workers 2-4 (VRAM shared across processes).
              for SpotCaller CPU use --workers 10-20.

CHANNEL NAMING
  Files must contain: Cy3, Cy5, or DAPI  (case-insensitive, anywhere in name)
  FOV grouping: channel keyword is stripped → position key
    Location_Cy3_xy001.tif + Location_Cy5_xy001.tif + Location_DAPI_xy001.tif
    → FOV "location_xy001"

RUN SUBFOLDERS
  Each run creates: <output_dir>/<detector>_<model>_t<threshold>[_stack]_<timestamp>/
  Example: /results/C1_left/piscis_20251212_t0p50_20260319_142200/

OUTPUTS  (inside run subfolder)
  <fov>_spots.csv     fov,channel,y,x  (+ z in stack mode; + snr/sigma for SpotCaller)
  <fov>_cells.csv     per-cell spot counts
  all_cells.csv       all FOVs: fov,cell_id,Cy3_spots,Cy5_spots,total_spots
  cell_summary.csv    per-FOV stats: mean/median/std/max spots per cell
  spot_counts.csv     per-FOV spot count summary
  run_<ts>.log        full INFO-level run log
  previews/           8-bit PNG images + RGBA mask PNGs (alpha=cell_id for hover)

FILE FORMATS
  .tif / .tiff     Multi-page z-stack (16-bit grayscale, single or multi-channel)
  .nd2             Nikon ND2 (requires: pip install nd2)
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import csv
import logging
import multiprocessing
import os
import re
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import laplace


# ════════════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir: Path) -> Path:
    """Console (INFO) + file (INFO) logging. Suppresses noisy third-party loggers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"run_{ts}.log"

    for noisy in ("numba", "numba.core", "matplotlib", "PIL", "torch",
                  "cellpose", "httpx", "httpcore", "hf_xet"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S"))
    root.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  [%(threadName)s]  %(message)s",
        "%Y-%m-%d %H:%M:%S"))
    root.addHandler(fh)

    return log_path


log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# Image I/O
# ════════════════════════════════════════════════════════════════════════════════

def read_image(path: Path) -> np.ndarray:
    """Read .tif/.tiff or .nd2 into a NumPy array."""
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        with tifffile.TiffFile(path) as tif:
            data = tif.asarray()
    elif ext == ".nd2":
        try:
            import nd2 as _nd2
            data = _nd2.imread(str(path))
        except ImportError:
            raise ImportError("ND2 files require: pip install nd2")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    log.debug("Loaded %s  shape=%s  dtype=%s", path.name, data.shape, data.dtype)
    return data


def to_zyx(data: np.ndarray, n_channels: int = 0) -> np.ndarray:
    """
    Coerce any image array to (Z, Y, X) float32.

    Handles: (Y,X), (Z,Y,X), (C,Z,Y,X), (Z,C,Y,X), (T,Z,C,Y,X).
    Channel axis detected by matching size to n_channels, or smallest axis heuristic.
    Always takes channel index 0.
    """
    data = data.astype(np.float32)
    if data.ndim == 2:
        return data[np.newaxis]
    if data.ndim == 3:
        return data
    if data.ndim == 4:
        if n_channels > 1:
            cands  = [i for i, s in enumerate(data.shape) if s == n_channels]
            c_axis = cands[0] if cands else min(range(4), key=lambda i: data.shape[i])
        else:
            small  = [i for i, s in enumerate(data.shape) if s <= 8]
            c_axis = small[0] if small else 0
        log.debug("  4-D %s: axis %d = channels, using ch 0", data.shape, c_axis)
        return np.moveaxis(data, c_axis, 0)[0]
    if data.ndim == 5:
        return to_zyx(data[0], n_channels)   # drop time axis
    raise ValueError(f"Cannot interpret shape {data.shape} as a z-stack.")


def best_focus_plane(stack: np.ndarray) -> int:
    """Sharpest plane by Laplacian variance."""
    return int(np.argmax([float(np.var(laplace(p.astype(np.float32)))) for p in stack]))


def collapse_z(stack: np.ndarray, projection: str) -> np.ndarray:
    """Collapse (Z,Y,X) → (Y,X) with the chosen projection."""
    if projection == "max":        return stack.max(axis=0)
    if projection == "mean":       return stack.mean(axis=0)
    if projection == "best_focus": return stack[best_focus_plane(stack)]
    if projection == "middle":     return stack[len(stack) // 2]
    raise ValueError(f"Unknown projection '{projection}'.")


def load_and_project(path: Path, projection: str) -> np.ndarray:
    return collapse_z(to_zyx(read_image(path)), projection)


# ════════════════════════════════════════════════════════════════════════════════
# File filtering & channel detection
# ════════════════════════════════════════════════════════════════════════════════

SPOT_CHANNELS = {"CY3": "Cy3", "CY5": "Cy5"}


def detect_channel(stem: str) -> "str | None":
    upper = stem.upper()
    if "DAPI" in upper:
        return "DAPI"
    for k, v in SPOT_CHANNELS.items():
        if k in upper:
            return v
    return None


def is_valid_file(p: Path) -> bool:
    """Accept .tif/.tiff/.nd2, reject hidden/temp/non-ASCII names."""
    n = p.name
    if n.startswith(".") or n.startswith("~$") or n.startswith("__"):
        return False
    if p.suffix.lower() not in (".tif", ".tiff", ".nd2"):
        return False
    return bool(re.match(r"^[\x20-\x7E]+$", n))


# ════════════════════════════════════════════════════════════════════════════════
# Preview / mask PNG export
# ════════════════════════════════════════════════════════════════════════════════

def save_preview_png(plane: np.ndarray, out_path: Path,
                     p_low: float = 1.0, p_high: float = 99.5) -> None:
    lo = np.percentile(plane, p_low)
    hi = np.percentile(plane, p_high)
    s  = np.clip((plane - lo) / max(hi - lo, 1e-9), 0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((s * 255).astype(np.uint8)).save(out_path)


def save_mask_preview_png(mask: np.ndarray, out_path: Path) -> None:
    """
    RGBA PNG where RGB = cell colour, Alpha = cell_id (1-255).
    Alpha is used by the viewer for cell-hover hit-testing.

    Fully vectorised — O(1) regardless of cell count.
    Replaces the old per-cell loop which was O(n_cells × H × W).
    """
    import colorsys
    h, w    = mask.shape
    rgba    = np.zeros((h, w, 4), dtype=np.uint8)
    n_cells = int(mask.max())

    if n_cells > 0:
        # Build colour lookup table for all cells at once
        rng  = np.random.default_rng(42)
        hues = np.linspace(0, 1, n_cells + 1, endpoint=False)[1:]
        rng.shuffle(hues)
        # lut shape: (n_cells+1, 3) — index 0 = background (black)
        lut = np.zeros((n_cells + 1, 3), dtype=np.uint8)
        for i, h_ in enumerate(hues):
            r, g, b = colorsys.hsv_to_rgb(h_, 0.85, 0.95)
            lut[i + 1] = [int(r*255), int(g*255), int(b*255)]

        # Vectorised boundary detection on the whole mask at once
        try:
            from skimage.segmentation import find_boundaries as _fb
            boundaries = _fb(mask, mode="outer")
        except ImportError:
            from scipy.ndimage import binary_erosion
            boundaries = mask.astype(bool) & ~binary_erosion(mask.astype(bool))

        fg       = mask > 0
        interior = fg & ~boundaries
        boundary = fg &  boundaries

        # Apply colours via LUT indexing — one operation for entire image
        rgba[interior, :3] = (lut[mask[interior]].astype(np.float32) * 0.25).astype(np.uint8)
        rgba[boundary, :3] = lut[mask[boundary]]

        # Alpha = cell_id capped at 255 (for JS hit-testing)
        alpha          = np.minimum(mask, 255).astype(np.uint8)
        rgba[:, :, 3]  = alpha

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(out_path)
    log.debug("Mask saved: %s  (%d cells)", out_path.name, n_cells)


# ════════════════════════════════════════════════════════════════════════════════
# CSV writers
# ════════════════════════════════════════════════════════════════════════════════

def save_spots_csv_simple(spots: np.ndarray, out_path: Path,
                          channel: str, fov: str) -> None:
    """2-D: fov,channel,y,x  |  3-D stack mode: fov,channel,z,y,x
    Uses bulk numpy/pandas write — fast even for tens of thousands of spots.
    """
    import pandas as pd
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spots = np.asarray(spots)
    if len(spots) == 0:
        has_z = False
        df = pd.DataFrame(columns=["fov","channel","y","x"])
    elif spots.ndim == 2 and spots.shape[1] == 3:
        has_z = True
        df = pd.DataFrame({
            "fov":     fov,
            "channel": channel,
            "z":       spots[:, 0].astype(int),
            "y":       np.round(spots[:, 1], 2),
            "x":       np.round(spots[:, 2], 2),
        })
    else:
        has_z = False
        df = pd.DataFrame({
            "fov":     fov,
            "channel": channel,
            "y":       np.round(spots[:, 0], 2),
            "x":       np.round(spots[:, 1], 2),
        })
    df.to_csv(out_path, index=False)
    log.info("    → %d spots: %s", len(spots), out_path.name)


def save_spots_csv_rich(df, out_path: Path, channel: str, fov: str) -> None:
    """SpotCaller CSV with full quality metrics."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fov","channel","y","x","y_sub","x_sub","z_plane",
                    "snr","sigma_y","sigma_x","fit_residual","amplitude"])
        for _, row in df.iterrows():
            w.writerow([
                fov, channel,
                round(row["y"], 2),    round(row["x"], 2),
                round(row.get("y_sub", row["y"]), 4),
                round(row.get("x_sub", row["x"]), 4),
                int(row.get("z_plane", 0)),
                round(row.get("snr", 0), 3),
                round(row.get("sigma_y", 0), 3),
                round(row.get("sigma_x", 0), 3),
                round(row.get("fit_residual", 0), 3),
                round(row.get("amplitude", 0), 1),
            ])
    log.info("    → %d spots: %s", len(df), out_path.name)


def save_cells_csv(counts: dict, channels: list, fov: str, out_path: Path) -> None:
    """Per-cell spot counts, including cells with 0 spots."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fov", "cell_id"] + [f"{ch}_spots" for ch in channels])
        for cid in sorted(counts):
            w.writerow([fov, cid] + [counts[cid].get(ch, 0) for ch in channels])
    log.info("    → cells CSV: %s  (%d cells)", out_path.name, len(counts))


# ════════════════════════════════════════════════════════════════════════════════
# Spot → cell assignment
# ════════════════════════════════════════════════════════════════════════════════

def assign_spots_to_cells(spots_by_channel: dict, mask: np.ndarray,
                          fov: str, out_path: Path) -> dict:
    """
    Assign spots to cells by pixel lookup in the Cellpose mask.
    Handles 2-D (N,2) [y,x], 3-D (N,3) [z,y,x], and SpotCaller DataFrames.
    """
    n_cells  = int(mask.max())
    channels = list(spots_by_channel.keys())
    h, w     = mask.shape
    counts   = {cid: {ch: 0 for ch in channels} for cid in range(1, n_cells + 1)}

    for ch, spots in spots_by_channel.items():
        if spots is None or len(spots) == 0:
            continue
        if hasattr(spots, "iterrows"):
            coords = spots[["y", "x"]].values
        else:
            arr = np.asarray(spots)
            coords = arr[:, 1:3] if arr.ndim == 2 and arr.shape[1] == 3 else arr[:, :2]

        # Vectorised lookup — clip to bounds then index mask in one operation
        ys = np.clip(np.round(coords[:, 0]).astype(int), 0, h - 1)
        xs = np.clip(np.round(coords[:, 1]).astype(int), 0, w - 1)
        cell_ids = mask[ys, xs]          # (N,) array of cell labels

        # Count per cell using numpy bincount — O(n_spots) not O(n_spots × n_cells)
        valid    = cell_ids > 0
        for cid, cnt in zip(*np.unique(cell_ids[valid], return_counts=True)):
            if cid in counts:
                counts[int(cid)][ch] += int(cnt)

    save_cells_csv(counts, channels, fov, out_path)
    total = sum(sum(v.values()) for v in counts.values())
    log.info("    → %d cells, %d spots assigned", n_cells, total)
    return counts


# ════════════════════════════════════════════════════════════════════════════════
# Piscis wrappers
# ════════════════════════════════════════════════════════════════════════════════

def run_piscis_on_plane(plane: np.ndarray, model, threshold: float) -> np.ndarray:
    """2-D Piscis — returns (N,2) [y,x]."""
    mn, mx = plane.min(), plane.max()
    norm   = (plane - mn) / max(mx - mn, 1e-9)
    return np.array(model.predict(norm, threshold=threshold))


def run_piscis_on_stack(stack: np.ndarray, model, threshold: float) -> np.ndarray:
    """
    Full z-stack Piscis — returns (N,3) [z,y,x].

    Piscis >= 0.2.6 with stack=True can return either:
      - A flat (N,3) array [z,y,x]  ← newer API
      - A list of per-plane (K,2) arrays [y,x]  ← older API
    We handle both cases. Falls back to MIP on Piscis < 1.1.
    """
    norm = np.zeros_like(stack, dtype=np.float32)
    for i in range(stack.shape[0]):
        mn, mx  = stack[i].min(), stack[i].max()
        norm[i] = (stack[i] - mn) / max(mx - mn, 1e-9)
    try:
        pred = model.predict(norm, threshold=threshold, stack=True)
        pred = np.asarray(pred) if not isinstance(pred, list) else pred

        # Case 1: flat (N,3) array [z,y,x] — Piscis >= 0.2.6 direct output
        if isinstance(pred, np.ndarray):
            if pred.ndim == 2 and pred.shape[1] == 3:
                log.debug("    Piscis stack: flat (N,3) output, %d spots", len(pred))
                return pred.astype(np.float32)
            elif pred.ndim == 2 and pred.shape[1] == 2:
                # (N,2) [y,x] — no z info, prepend 0
                log.debug("    Piscis stack: flat (N,2) output, prepending z=0")
                z_col = np.zeros((len(pred), 1), dtype=np.float32)
                return np.hstack([z_col, pred]).astype(np.float32)
            elif len(pred) == 0:
                return np.zeros((0, 3), dtype=np.float32)

        # Case 2: list of per-plane (K,2) arrays — older Piscis API
        if isinstance(pred, list):
            zyx_list = []
            for z_idx, plane_spots in enumerate(pred):
                plane_spots = np.asarray(plane_spots)
                if plane_spots.ndim == 2 and len(plane_spots):
                    z_col = np.full((len(plane_spots), 1), z_idx, dtype=np.float32)
                    zyx_list.append(np.hstack([z_col, plane_spots]))
            result = np.vstack(zyx_list) if zyx_list else np.zeros((0, 3), dtype=np.float32)
            log.debug("    Piscis stack: list-of-planes output, %d spots", len(result))
            return result.astype(np.float32)

        log.warning("    Piscis stack: unexpected output type %s — falling back to MIP", type(pred))
        raise TypeError("unexpected")

    except (TypeError, AttributeError):
        log.warning("    Piscis stack=True not supported or failed — falling back to MIP.")
        spots = np.asarray(model.predict(norm.max(axis=0), threshold=threshold))
        if len(spots):
            z_col = np.zeros((len(spots), 1), dtype=np.float32)
            return np.hstack([z_col, spots]).astype(np.float32)
        return np.zeros((0, 3), dtype=np.float32)


# ════════════════════════════════════════════════════════════════════════════════
# Cellpose wrapper  (CPU only — avoids VRAM contention with Piscis)
# ════════════════════════════════════════════════════════════════════════════════

def run_cellpose(dapi_path: Path, cp_model, diameter: float,
                 expand_px: int = 0) -> np.ndarray:
    """Max-project DAPI → Cellpose segmentation (CPU). Optionally expand masks."""
    plane  = load_and_project(dapi_path, "max")
    lo, hi = np.percentile(plane, 1), np.percentile(plane, 99.5)
    img8   = (np.clip((plane - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    masks, _, _ = cp_model.eval(img8,
                                diameter=diameter if diameter > 0 else None,
                                channels=[0, 0], do_3D=False)
    masks = masks.astype(np.int32)
    if expand_px > 0:
        try:
            from skimage.segmentation import expand_labels
            masks = expand_labels(masks, expand_px).astype(np.int32)
        except ImportError:
            log.warning("expand_mask needs scikit-image; skipping")
    return masks


# ════════════════════════════════════════════════════════════════════════════════
# Worker process — initializer + per-FOV function
# ════════════════════════════════════════════════════════════════════════════════

_W: dict = {}   # worker-local model store (populated by _worker_init once per process)


def _worker_init(detector, model_name, sc_params, cellpose_model_path,
                 run_spot_tool, run_cellpose_tool, script_dir,
                 log_level, log_queue, no_gpu):
    """
    Called once per worker process at pool start.
    Loads models into _W; routes logging through log_queue back to main process.
    """
    import os, logging, sys
    from logging.handlers import QueueHandler

    # Force JAX to CPU if --no_gpu set — must happen before any jax import
    if no_gpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Pin each worker to 1 BLAS/OMP thread — prevents oversubscription
    for k, v in [("OMP_NUM_THREADS","1"), ("OPENBLAS_NUM_THREADS","1"),
                 ("MKL_NUM_THREADS","1"), ("BLIS_NUM_THREADS","1"),
                 ("VECLIB_MAXIMUM_THREADS","1"), ("NUMEXPR_NUM_THREADS","1")]:
        os.environ[k] = v

    # Route all logging back to main process via queue
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(QueueHandler(log_queue))
    root.setLevel(log_level)
    for noisy in ("numba","numba.core","matplotlib","PIL","torch",
                  "cellpose","httpx","httpcore","hf_xet"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    wlog = logging.getLogger(__name__)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    if detector == "piscis" and run_spot_tool:
        from piscis import Piscis
        p = Piscis(model_name=model_name)
        # Log which JAX device Piscis is using
        try:
            import jax as _jax
            devs = _jax.devices()
            wlog.info("Worker %d: Piscis on %s", os.getpid(), devs)
        except Exception:
            pass
        _W["piscis_model"] = p

    elif detector == "spot_caller" and run_spot_tool:
        from spot_caller import SpotCaller
        use_gpu = not no_gpu   # SpotCaller will check CuPy availability itself
        wlog.info("Worker %d: SpotCaller  use_gpu=%s", os.getpid(), use_gpu)
        _W["spot_caller_model"] = SpotCaller(**sc_params, use_gpu=use_gpu)

    if cellpose_model_path and run_cellpose_tool:
        from cellpose import models as cp_models
        wlog.info("Worker %d: Cellpose on CPU", os.getpid())
        _W["cp_model"] = cp_models.CellposeModel(
            pretrained_model=cellpose_model_path, gpu=False)


def process_fov(fov, ch_paths, output_dir, detector,
                piscis_threshold, piscis_stack,
                sc_threshold, sc_use_3d, projection,
                cellpose_diam, expand_mask,
                run_spot_tool, run_cellpose_tool,
                save_previews, p_low, p_high) -> dict:
    """
    Process one FOV in a worker process using models from _W.

    Cy3 and Cy5 are detected in parallel using threads — they are independent
    and JAX/CuPy release the GIL during inference, so true parallelism applies.
    Cellpose runs after both channels finish (needs spots for assignment).
    """
    summary = {"fov": fov, "channels": {}, "n_cells": 0, "error": None}
    spots_by_channel = {}

    def _detect_channel(ch: str):
        """Detect spots in one channel. Returns (ch, spots_or_df, plane, n_spots)."""
        tif_path = ch_paths.get(ch)
        if tif_path is None or not run_spot_tool:
            return ch, None, None, 0
        log.info("  [%s] %s", ch, tif_path.name)
        try:
            stack = to_zyx(read_image(tif_path))

            if detector == "piscis":
                if piscis_stack and stack.shape[0] > 1:
                    log.debug("    stack mode (%d planes)", stack.shape[0])
                    spots = run_piscis_on_stack(stack, _W["piscis_model"], piscis_threshold)
                    plane = stack.max(axis=0)
                else:
                    plane = collapse_z(stack, projection)
                    spots = run_piscis_on_plane(plane, _W["piscis_model"], piscis_threshold)
                save_spots_csv_simple(spots, output_dir / f"{tif_path.stem}_spots.csv", ch, fov)
                return ch, spots, plane, len(spots)

            else:   # spot_caller
                sc = _W["spot_caller_model"]
                if sc_use_3d and stack.shape[0] > 1:
                    df    = sc.detect_3d(stack, threshold_mad=sc_threshold)
                    plane = stack.max(axis=0)
                else:
                    plane = collapse_z(stack, projection)
                    df    = sc.detect(plane, threshold_mad=sc_threshold)
                save_spots_csv_rich(df, output_dir / f"{tif_path.stem}_spots.csv", ch, fov)
                return ch, df, plane, len(df)

        except Exception as exc:
            log.error("  FAILED [%s]: %s", ch, exc)
            log.debug(traceback.format_exc())
            summary["error"] = str(exc)
            return ch, None, None, 0

    # Cy3, Cy5 and DAPI/Cellpose run in parallel below

    # DAPI / Cellpose — run concurrently with spot calling channels
    # Cellpose on DAPI is independent of Cy3/Cy5; start it in a thread
    # alongside the channel futures so all three run in parallel.
    dapi_path = ch_paths.get("DAPI")
    mask      = None

    def _run_cellpose():
        if dapi_path is None: return None
        log.info("  [DAPI] %s", dapi_path.name)
        try:
            if save_previews:
                save_preview_png(load_and_project(dapi_path, "max"),
                    output_dir / "previews" / f"{dapi_path.stem}.png", p_low, p_high)
            if _W.get("cp_model") is not None and run_cellpose_tool:
                m = run_cellpose(dapi_path, _W["cp_model"], cellpose_diam, expand_mask)
                np.save(output_dir / f"{dapi_path.stem}_mask.npy", m)
                if save_previews:
                    save_mask_preview_png(m,
                        output_dir / "previews" / f"{dapi_path.stem}_mask.png")
                log.info("    [DAPI] → %d cells", int(m.max()))
                return m
        except Exception as exc:
            log.error("  FAILED [DAPI/Cellpose]: %s", exc)
            log.debug(traceback.format_exc())
            summary["error"] = str(exc)
        return None

    # Run Cy3, Cy5, and DAPI/Cellpose all in parallel
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="ch") as tex:
        cp_future  = tex.submit(_run_cellpose)
        ch_futures = {tex.submit(_detect_channel, ch): ch for ch in ("Cy3", "Cy5")}
        for fut in ch_futures:
            ch, result, plane, n_spots = fut.result()
            if result is not None:
                spots_by_channel[ch] = result
                summary["channels"][ch] = n_spots
                log.info("    [%s] → %d spots", ch, n_spots)
                if save_previews and plane is not None:
                    save_preview_png(plane,
                        output_dir / "previews" / f"{ch_paths[ch].stem}.png", p_low, p_high)
        mask = cp_future.result()

    # Cell assignment needs both spots and mask — done after all threads finish
    if mask is not None:
        n_cells = int(mask.max())
        summary["n_cells"] = n_cells
        if spots_by_channel:
            assign_spots_to_cells(spots_by_channel, mask, fov,
                                  output_dir / f"{fov}_cells.csv")

    return summary


# ════════════════════════════════════════════════════════════════════════════════
# Post-run summary + IQR outlier detection
# ════════════════════════════════════════════════════════════════════════════════

def print_run_summary(all_results: list, output_dir: Path) -> None:
    sep = "═" * 58
    log.info(sep); log.info("  RUN SUMMARY"); log.info(sep)

    ch_counts: dict = {}
    for r in all_results:
        if r.get("error"): continue
        for ch, n in r.get("channels", {}).items():
            ch_counts.setdefault(ch, []).append((r["fov"], n))

    n_ok  = sum(1 for r in all_results if not r.get("error"))
    n_err = sum(1 for r in all_results if r.get("error"))
    log.info("  FOVs processed : %d", n_ok)
    if n_err: log.warning("  FOVs failed    : %d", n_err)

    for ch, fov_counts in sorted(ch_counts.items()):
        counts = [n for _, n in fov_counts]
        arr    = np.array(counts, dtype=float)
        log.info("  --- Channel: %s ---", ch)
        log.info("    Total spots    : %d", int(arr.sum()))
        log.info("    Per-FOV mean   : %.1f  (median %.1f  std %.1f)",
                 arr.mean(), float(np.median(arr)), arr.std())
        log.info("    Range          : %d – %d", int(arr.min()), int(arr.max()))
        if len(arr) >= 4:
            q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
            iqr    = q3 - q1
            lo_t, hi_t = q1 - 1.5*iqr, q3 + 1.5*iqr
            for fov_name, n in fov_counts:
                if n < lo_t or n > hi_t:
                    log.warning("    ⚠ FOV outlier [%s]: %s → %d spots",
                                "HIGH" if n > hi_t else "LOW", fov_name, n)

    # Per-cell stats
    all_per_cell: dict = {"Cy3": [], "Cy5": []}
    cell_warnings = []
    for csv_path in sorted(p for p in output_dir.glob("*_cells.csv")
                           if "all_cells" not in p.name):
        try:
            with csv_path.open() as fh:
                rows = list(csv.DictReader(fh))
            if not rows: continue
            fov_name = rows[0].get("fov", csv_path.stem.replace("_cells",""))
            for ch in ["Cy3","Cy5"]:
                col = next((k for k in rows[0]
                            if ch.lower() in k.lower() and "spots" in k.lower()), None)
                if not col: continue
                try: vals = [int(float(r[col])) for r in rows]
                except (ValueError,KeyError): continue
                all_per_cell[ch].extend(vals)
                if len(vals) >= 4:
                    arr2 = np.array(vals, dtype=float)
                    q1,q3 = np.percentile(arr2,25), np.percentile(arr2,75)
                    hi_t  = q3 + 1.5*(q3-q1)
                    hot   = [v for v in vals if v > hi_t and v > 0]
                    if hot:
                        cell_warnings.append(
                            f"  ⚠ {fov_name} {ch}: {len(hot)} cell(s) "
                            f"unusually high (>{hi_t:.0f}), max={max(hot)}")
        except Exception: continue

    for ch, per_cell in all_per_cell.items():
        if not per_cell: continue
        arr = np.array(per_cell, dtype=float)
        pos = arr[arr > 0]
        log.info("  --- Per-cell: %s ---", ch)
        log.info("    Total cells    : %d", len(arr))
        if len(pos):
            log.info("    Cells w/ spots : %d  (%.1f%%)", len(pos), 100*len(pos)/len(arr))
            log.info("    Spots/cell     : mean=%.2f  median=%.2f  std=%.2f",
                     pos.mean(), float(np.median(pos)), pos.std())
    for w in cell_warnings:
        log.warning(w)
    log.info(sep)


# ════════════════════════════════════════════════════════════════════════════════
# Batch runner
# ════════════════════════════════════════════════════════════════════════════════

def batch_process(
    input_dir: Path, output_dir: Path, pattern: str,
    detector: str,
    model_name: str, piscis_threshold: float, piscis_stack: bool = False,
    spot_yx: float = 3.0, spot_z: float = 2.0,
    sc_threshold: float = 5.0, min_snr: float = 2.0,
    max_sigma_ratio: float = 2.5, tile_size: int = 128, sc_use_3d: bool = False,
    projection: str = "max",
    cellpose_model_path: "str | None" = None, cellpose_diam: float = 0,
    expand_mask: int = 0,
    run_only: str = "both",
    save_previews: bool = True,
    percentile_low: float = 1.0, percentile_high: float = 99.5,
    workers: int = 4, batch_size: int = 8, no_gpu: bool = False,
) -> None:
    """Orchestrate the full pipeline across all FOVs via a process pool."""
    run_spot_tool     = run_only in ("piscis", "both")
    run_cellpose_tool = run_only in ("cellpose", "both")
    workers           = max(1, min(workers, 20))

    # Set thread limits before spawning — children inherit os.environ
    for k, v in [("OMP_NUM_THREADS","1"),("OPENBLAS_NUM_THREADS","1"),
                 ("MKL_NUM_THREADS","1"),("BLIS_NUM_THREADS","1"),
                 ("VECLIB_MAXIMUM_THREADS","1"),("NUMEXPR_NUM_THREADS","1"),
                 ("XLA_FLAGS","--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")]:
        os.environ.setdefault(k, v)

    # JAX compilation cache — no admin rights needed, lives in user home dir.
    # First run compiles and caches; all subsequent runs load from cache.
    # Set externally via: export JAX_COMPILATION_CACHE_DIR=~/.jax_cache
    # or we auto-set a sensible default here.
    default_cache = Path.home() / ".jax_cache"
    if not os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        os.environ["JAX_COMPILATION_CACHE_DIR"] = str(default_cache)
        os.environ["JAX_CACHE_DIR"]              = str(default_cache)
        default_cache.mkdir(parents=True, exist_ok=True)
    cache_dir   = os.environ["JAX_COMPILATION_CACHE_DIR"]
    cache_files = len(list(Path(cache_dir).rglob("*"))) if Path(cache_dir).exists() else 0
    log.info("  JAX cache   : %s  (%d files%s)", cache_dir, cache_files,
             " — warm" if cache_files > 10 else " — COLD, first run will be slow"
             if detector == "piscis" else "")

    # GPU status
    try:
        import torch as _t
        gpu_info = (f"{_t.cuda.get_device_name(0)} (CUDA {_t.version.cuda})"
                    if _t.cuda.is_available() else "none")
    except Exception:
        gpu_info = "unknown"

    if detector == "spot_caller":
        try:
            import cupy as _cp
            cp_ok = _cp.cuda.is_available() and not no_gpu
        except ImportError:
            cp_ok = False

    log.info("════════════════════════════════════════════")
    log.info("  GPU        : %s%s", gpu_info,
             "  [DISABLED --no_gpu]" if no_gpu else "")
    if detector == "spot_caller":
        log.info("  SpotCaller : CuPy GPU = %s", cp_ok if detector=="spot_caller" else "n/a")
    log.info("  Detector   : %s", detector.upper())
    log.info("  Projection : %s  |  Stack: %s", projection, piscis_stack)
    log.info("  Workers    : %d", workers)
    log.info("  Input      : %s", input_dir)
    log.info("  Output     : %s", output_dir)
    log.info("════════════════════════════════════════════")

    # Pre-download Piscis model in main process to prevent worker race condition
    if detector == "piscis" and run_spot_tool:
        try:
            from piscis import Piscis as _P
            log.info("Pre-downloading/verifying Piscis model '%s' …", model_name)
            _ = _P(model_name=model_name)
            log.info("Model cached — workers load from local cache.")
        except ImportError as exc:
            raise ImportError("Piscis not installed. Run: pip install piscis") from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load model '{model_name}': {exc}") from exc

    # Discover files (TIF + ND2)
    nd2_pattern = re.sub(r"\.tif+$", ".nd2", pattern, flags=re.IGNORECASE)
    _seen: set = set()
    tif_files: list = []
    for pat in [pattern, nd2_pattern]:
        for p in sorted(input_dir.glob(pat)):
            if is_valid_file(p) and p not in _seen:
                tif_files.append(p); _seen.add(p)

    if not tif_files:
        log.warning("No files matched '%s' in %s", pattern, input_dir)
        return

    def fov_key(stem: str) -> str:
        return re.sub(r"(?i)[_-]?(cy3|cy5|dapi)[_-]?", "_", stem).strip("_").lower()

    groups: dict = defaultdict(dict)
    for p in tif_files:
        ch = detect_channel(p.stem)
        if ch is None:
            log.warning("SKIP (unknown channel): %s", p.name)
            continue
        groups[fov_key(p.stem)][ch] = p

    n_fovs = len(groups)
    log.info("Found %d FOVs across %d files", n_fovs, len(tif_files))
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_previews:
        (output_dir / "previews").mkdir(exist_ok=True)

    counts_path = output_dir / "spot_counts.csv"
    with counts_path.open("w", newline="") as fh:
        csv.writer(fh).writerow(["fov","channel","n_spots","n_cells","tif_file"])

    # Log queue: worker → main process
    ctx          = multiprocessing.get_context("spawn")
    log_queue    = ctx.Queue(maxsize=-1)
    q_listener   = QueueListener(log_queue, *logging.getLogger().handlers,
                                 respect_handler_level=True)
    q_listener.start()

    sc_params = dict(spot_yx=spot_yx, spot_z=spot_z, threshold=sc_threshold,
                     tile_size=tile_size, min_snr=min_snr,
                     max_sigma_ratio=max_sigma_ratio, subpixel=True)
    init_args = (detector, model_name, sc_params, cellpose_model_path,
                 run_spot_tool, run_cellpose_tool,
                 str(Path(__file__).parent), logging.getLogger().level,
                 log_queue, no_gpu)

    fov_kwargs = dict(
        output_dir=output_dir, detector=detector,
        piscis_threshold=piscis_threshold, piscis_stack=piscis_stack,
        sc_threshold=sc_threshold, sc_use_3d=sc_use_3d,
        projection=projection, cellpose_diam=cellpose_diam, expand_mask=expand_mask,
        run_spot_tool=run_spot_tool, run_cellpose_tool=run_cellpose_tool,
        save_previews=save_previews, p_low=percentile_low, p_high=percentile_high,
    )

    eff_workers = min(workers, n_fovs)
    log.info("Dispatching %d FOVs to %d worker processes …", n_fovs, eff_workers)

    all_results: list = []
    completed = 0

    with ProcessPoolExecutor(max_workers=eff_workers, mp_context=ctx,
                             initializer=_worker_init, initargs=init_args) as pool:
        futures = {pool.submit(process_fov, fov, ch_paths, **fov_kwargs): fov
                   for fov, ch_paths in sorted(groups.items())}
        for future in as_completed(futures):
            fov = futures[future]
            completed += 1
            try:
                result = future.result()
                result["fov"] = fov
                all_results.append(result)
                log.info("✓ [%d/%d] FOV %s — %s  cells=%d",
                         completed, n_fovs, fov,
                         " ".join(f"{ch}={n}" for ch,n in result["channels"].items()),
                         result["n_cells"])
                with counts_path.open("a", newline="") as fh:
                    w = csv.writer(fh)
                    for ch, n in result["channels"].items():
                        w.writerow([fov, ch, n, result["n_cells"],
                                    groups[fov].get(ch, Path("")).name])
            except Exception as exc:
                log.error("✗ [%d/%d] FOV %s FAILED: %s", completed, n_fovs, fov, exc)
                log.debug(traceback.format_exc())
                all_results.append({"fov":fov,"channels":{},"n_cells":0,"error":str(exc)})

    q_listener.stop()

    # ── Consolidate all_cells.csv ────────────────────────────────────────────
    fov_csvs = sorted(p for p in output_dir.glob("*_cells.csv")
                      if "all_cells" not in p.name)
    if fov_csvs:
        all_rows, col_order = [], None
        for csv_path in fov_csvs:
            try:
                with csv_path.open() as fh:
                    rows = list(csv.DictReader(fh))
                if not rows: continue
                ch_cols = [k for k in rows[0] if k.endswith("_spots")]
                if col_order is None: col_order = ch_cols
                for row in rows:
                    row["total_spots"] = sum(int(float(row.get(c,0))) for c in ch_cols)
                    all_rows.append(row)
            except Exception: continue
        if all_rows and col_order is not None:
            with (output_dir / "all_cells.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh,
                    fieldnames=["fov","cell_id"]+col_order+["total_spots"],
                    extrasaction="ignore")
                w.writeheader(); w.writerows(all_rows)
            log.info("all_cells.csv written  (%d total cells)", len(all_rows))

    # ── cell_summary.csv ─────────────────────────────────────────────────────
    summary_rows = []
    for csv_path in fov_csvs:
        try:
            with csv_path.open() as fh:
                rows = list(csv.DictReader(fh))
            if not rows: continue
            fov_name = rows[0].get("fov", csv_path.stem.replace("_cells",""))
            for col in [k for k in rows[0] if k.endswith("_spots") and k != "total_spots"]:
                try: vals = [int(float(r[col])) for r in rows]
                except (ValueError,KeyError): continue
                arr = np.array(vals, dtype=float)
                pos = arr[arr > 0]
                summary_rows.append({
                    "fov": fov_name, "channel": col.replace("_spots",""),
                    "n_cells": len(arr),
                    "n_cells_with_spots": len(pos),
                    "pct_cells_with_spots": round(100*len(pos)/len(arr),1) if len(arr) else 0,
                    "total_spots": int(arr.sum()),
                    "mean_spots_per_cell":   round(float(arr.mean()),3),
                    "median_spots_per_cell": round(float(np.median(arr)),3),
                    "std_spots_per_cell":    round(float(arr.std()),3),
                    "max_spots_in_cell":     int(arr.max()),
                })
        except Exception: continue
    if summary_rows:
        fields = ["fov","channel","n_cells","n_cells_with_spots","pct_cells_with_spots",
                  "total_spots","mean_spots_per_cell","median_spots_per_cell",
                  "std_spots_per_cell","max_spots_in_cell"]
        with (output_dir / "cell_summary.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader(); w.writerows(summary_rows)
        log.info("cell_summary.csv written  (%d rows)", len(summary_rows))

    try:
        print_run_summary(all_results, output_dir)
    except Exception as exc:
        log.debug("Summary error: %s", exc)

    log.info("════ Done. %d/%d FOVs. Output: %s ════", completed, n_fovs, output_dir)


# ════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Faryabi Spot Calling v3",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)

    io = p.add_argument_group("Input / Output")
    io.add_argument("--input_dir",  required=True, type=Path)
    io.add_argument("--output_dir", required=True, type=Path)
    io.add_argument("--pattern",    default="*.tif")

    det = p.add_argument_group("Detector")
    det.add_argument("--detector", default="piscis", choices=["piscis","spot_caller"])

    pi = p.add_argument_group("Piscis")
    pi.add_argument("--model",     default="20230905", dest="model_name")
    pi.add_argument("--threshold", type=float, default=0.5)
    pi.add_argument("--pvalue",    type=float, default=None, help="Alias for --threshold")
    pi.add_argument("--stack",     action="store_true",
                    help="Full z-stack to Piscis (Piscis>=1.1, slower but 3-D aware)")

    sc = p.add_argument_group("SpotCaller")
    sc.add_argument("--spot_yx",         type=float, default=3.0)
    sc.add_argument("--spot_z",          type=float, default=2.0)
    sc.add_argument("--min_snr",         type=float, default=2.0)
    sc.add_argument("--max_sigma_ratio", type=float, default=2.5)
    sc.add_argument("--tile_size",       type=int,   default=128)
    sc.add_argument("--use_3d",          action="store_true")

    zp = p.add_argument_group("Z-projection")
    zp.add_argument("--projection", default="max",
                    choices=["max","mean","best_focus","middle"])

    cp = p.add_argument_group("Cellpose  (CPU only)")
    cp.add_argument("--cellpose_model", default=None)
    cp.add_argument("--cellpose_diam",  type=float, default=0)
    cp.add_argument("--expand_mask",    type=int,   default=0,
                    help="Grow cell masks outward N px (catches border spots)")

    ctl = p.add_argument_group("Run control")
    ctl.add_argument("--run_only",   default="both", choices=["piscis","cellpose","both"])
    ctl.add_argument("--no_gpu",     action="store_true",
                     help="Disable GPU for Piscis and SpotCaller (force CPU)")
    ctl.add_argument("--workers",    type=int, default=4,
                     help="Parallel processes (max 20). GPU: use 2-4. CPU: up to 20.")
    ctl.add_argument("--batch_size", type=int, default=8)

    out = p.add_argument_group("Previews")
    out.add_argument("--save_previews",   action="store_true")
    out.add_argument("--percentile_low",  type=float, default=1.0)
    out.add_argument("--percentile_high", type=float, default=99.5)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    base_dir    = Path(args.output_dir)
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = Path(args.model_name).stem[:16] if args.model_name else "nomodel"
    thr         = args.pvalue if args.pvalue is not None else args.threshold
    thr_str     = f"t{thr:.2f}".replace(".", "p")
    extra       = "_stack" if getattr(args, "stack", False) else ""
    run_dir     = base_dir / f"{args.detector}_{model_short}_{thr_str}{extra}_{ts}"

    log_path = setup_logging(run_dir)
    log.info("Log file: %s", log_path)
    log.info("Command:  %s", " ".join(sys.argv))

    threshold = args.pvalue if args.pvalue is not None else args.threshold
    piscis_threshold = threshold if args.detector == "piscis" else 0.5
    sc_threshold     = threshold if args.detector == "spot_caller" else 5.0

    if args.workers > 20:
        log.warning("--workers capped at 20.")

    batch_process(
        input_dir=Path(args.input_dir), output_dir=run_dir, pattern=args.pattern,
        detector=args.detector,
        model_name=args.model_name, piscis_threshold=piscis_threshold,
        piscis_stack=args.stack,
        spot_yx=args.spot_yx, spot_z=args.spot_z, sc_threshold=sc_threshold,
        min_snr=args.min_snr, max_sigma_ratio=args.max_sigma_ratio,
        tile_size=args.tile_size, sc_use_3d=args.use_3d,
        projection=args.projection,
        cellpose_model_path=args.cellpose_model, cellpose_diam=args.cellpose_diam,
        expand_mask=args.expand_mask,
        run_only=args.run_only,
        save_previews=args.save_previews,
        percentile_low=args.percentile_low, percentile_high=args.percentile_high,
        workers=args.workers, batch_size=args.batch_size, no_gpu=args.no_gpu,
    )
