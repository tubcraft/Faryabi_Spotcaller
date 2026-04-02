# Faryabi Spot Calling

> End-to-end smFISH spot detection, cell segmentation, and interactive visualization.

📖 **Full documentation:** Download and open [`docs.html`](Faryabi_Spot_v3/docs.html) in your browser for the complete reference — installation, all flags, tuning guide, API, FAQ, and more.

A complete pipeline for **single-molecule fluorescence in-situ hybridisation (smFISH)** data analysis. It combines two spot detectors, Cellpose cell segmentation, an interactive browser-based viewer, and a Flask web GUI — processing raw TIF stacks through to per-cell spot counts.

```
TIF stacks → z-projection → spot detection → cell segmentation → CSVs + previews
```

---

## Files

| File | Purpose |
|------|---------|
| `run_piscis.py` | Main pipeline — spot detection + cell segmentation |
| `spot_caller.py` | Classical adaptive spot detector (SpotCaller) |
| `app.py` | Flask web GUI backend |
| `templates/index.html` | GUI frontend |
| `spot_viewer.html` | Interactive spot viewer (open directly in browser) |
| `docs.html` | Full documentation |
| `environment_piscis_v3.yml` | Conda environment specification |
| `warm_jax_cache.sh` | JAX cache warm-up script |

---

## Installation

### Conda (recommended)

```bash
conda env create -f environment_piscis_v3.yml
conda activate piscis_env

# Verify
python -c "import piscis, cellpose; print('OK')"
```

### Manual

```bash
pip install piscis
pip install "cellpose<4"    # pin to v3 for CP3-trained models
pip install flask tifffile scipy pillow numpy pandas scikit-image
```

> **⚠ Cellpose version:** If your model was trained with Cellpose 3 (most custom models are), you must install `cellpose<4`. Cellpose 4 changed the model format and is incompatible with CP3 weights.

### GPU support (Piscis / JAX)

```bash
pip install "jax[cuda12]"   # CUDA 12
pip install "jax[cuda11]"   # CUDA 11
```

SpotCaller uses only NumPy/SciPy and runs on CPU only.

---

## Quick Start

### Command line

```bash
# Piscis (deep learning) + Cellpose segmentation
python run_piscis.py \
  --input_dir      /data/experiment \
  --output_dir     /results/run1 \
  --detector       piscis \
  --model          /models/piscis_smfish \
  --cellpose_model /models/cellpose_cd4 \
  --workers        8 \
  --save_previews

# SpotCaller (classical, no model needed)
python run_piscis.py \
  --input_dir  /data/experiment \
  --output_dir /results/run2 \
  --detector   spot_caller \
  --spot_yx    3.0 \
  --threshold  5.0 \
  --workers    16 \
  --save_previews
```

### Web GUI

```bash
python app.py --host 0.0.0.0 --port 5000
# Open http://<server>:5000 in your browser
```

### Spot viewer

Open `spot_viewer.html` directly in any modern browser. Drop your entire output folder onto the **📂 folder drop zone** — images, CSVs, and masks load automatically.

---

## Input Files

### Naming convention

Files must contain a channel keyword anywhere in their filename (case-insensitive):

| Keyword | Channel | Processing |
|---------|---------|------------|
| `Cy3` | RNA channel 1 | Spot detection |
| `Cy5` | RNA channel 2 | Spot detection |
| `DAPI` | Nuclei | Cell segmentation only |

Files are automatically grouped by FOV — the channel keyword is stripped and the remainder becomes the FOV identifier:

```
Location_Cy3_xy001.tif   →  FOV: location_xy001  channel: Cy3
Location_Cy5_xy001.tif   →  FOV: location_xy001  channel: Cy5
Location_DAPI_xy001.tif  →  FOV: location_xy001  channel: DAPI
```

Supported format: multi-page TIF stacks (16-bit grayscale). Use `--pattern "*.tiff"` for four-letter extensions.

---

## All Flags

### Input / Output

| Flag | Default | Description |
|------|---------|-------------|
| `--input_dir` | required | Folder containing TIF stacks |
| `--output_dir` | required | Output destination (created if absent) |
| `--pattern` | `*.tif` | Glob pattern for TIF files |

### Detector

| Flag | Default | Description |
|------|---------|-------------|
| `--detector` | `piscis` | `piscis` or `spot_caller` |

### Piscis

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `20230905` | Model name or local path |
| `--threshold` | `0.5` | Detection probability 0–1. Lower = more spots. |

### SpotCaller

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold` | `5.0` | MAD multiplier above local noise |
| `--spot_yx` | `3.0` | Expected spot radius in XY (pixels) |
| `--spot_z` | `2.0` | Expected spot radius in Z (planes) |
| `--min_snr` | `2.0` | Minimum SNR after Gaussian fit |
| `--max_sigma_ratio` | `2.5` | Reject spots with sigma > N × spot_yx |
| `--tile_size` | `128` | Tile size for adaptive noise estimation |
| `--use_3d` | off | Full 3-D LoG detection (slower) |

### Cellpose

| Flag | Default | Description |
|------|---------|-------------|
| `--cellpose_model` | — | Path to Cellpose model. Omit to skip segmentation. |
| `--cellpose_diam` | `0` | Expected cell diameter in px. 0 = auto. |

### Run control

| Flag | Default | Description |
|------|---------|-------------|
| `--run_only` | `both` | `piscis` \| `cellpose` \| `both` |
| `--workers` | `4` | Parallel threads (max 20) |
| `--projection` | `max` | `max` \| `mean` \| `best_focus` \| `middle` |
| `--save_previews` | off | Export 8-bit PNGs for the viewer |

---

## Outputs

| File | Description |
|------|-------------|
| `<stem>_spots.csv` | One row per detected spot (y, x + SpotCaller quality metrics) |
| `<stem>_cells.csv` | Per-cell spot counts (fov, cell_id, Cy3_spots, Cy5_spots) |
| `spot_counts.csv` | Per-FOV summary |
| `cell_summary.csv` | Aggregate per-FOV and per-channel statistics |
| `run_YYYYMMDD_HHMMSS.log` | Full timestamped log with tracebacks |
| `previews/<stem>.png` | 8-bit contrast-stretched preview |
| `previews/<stem>_mask.png` | Coloured cell boundary overlay |
| `<stem>_mask.npy` | Raw Cellpose integer label mask |

---

## Python API

```python
from spot_caller import SpotCaller

# Drop-in replacement for Piscis.predict()
model = SpotCaller(spot_yx=3.0, threshold=5.0, min_snr=2.0)
spots = model.predict(image_2d)     # returns (N, 2) float32 [y, x]

# Full output with quality metrics
df = model.detect(image_2d)
# columns: y, x, y_sub, x_sub, z_plane, snr, sigma_y, sigma_x, amplitude

# 3-D detection on a z-stack
df_3d = model.detect_3d(stack_zyx)
```

```python
from run_piscis import batch_process
from pathlib import Path

batch_process(
    input_dir=Path("/data/experiment"),
    output_dir=Path("/results/run1"),
    pattern="*.tif",
    detector="spot_caller",
    spot_yx=3.0, threshold=5.0, min_snr=2.0,
    workers=8,
    save_previews=True,
)
```

---

## Web GUI Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Main GUI |
| `GET /viewer` | Serve spot_viewer.html |
| `POST /api/run` | Submit a new job (JSON body) |
| `GET /api/jobs` | List all jobs |
| `GET /api/jobs/<id>` | Job details + last 500 log lines |
| `GET /api/stream/<id>` | SSE live log stream |
| `POST /api/cancel/<id>` | Cancel a running job |

---

## Detector Comparison

| Criterion | Piscis | SpotCaller |
|-----------|--------|------------|
| Speed (GPU) | Very fast | N/A |
| Model required | Yes | No |
| Sub-pixel coords | No | Yes |
| Per-spot metrics | No | SNR, sigma, amplitude |
| Uneven illumination | Partial | Good (adaptive noise) |
| Interpretability | Black box | Fully interpretable |

For new datasets, run both and compare in the viewer's **comparison mode**.

---

## License

Private repository — all rights reserved.

# SpotCaller (classical, no model needed)
python run_piscis.py \
  --input_dir  /data/experiment \
  --output_dir /results/run2 \
  --detector   spot_caller \
  --spot_yx    3.0 \
  --threshold  5.0 \
  --workers    16 \
  --save_previews
```

### Web GUI

```bash
python app.py --host 0.0.0.0 --port 5000
# Open http://<server>:5000 in your browser
```

### Spot viewer

Open `spot_viewer.html` directly in any modern browser. Drop your entire output folder onto the **📂 folder drop zone** — images, CSVs, and masks load automatically.

---

## Input Files

### Naming convention

Files must contain a channel keyword anywhere in their filename (case-insensitive):

| Keyword | Channel | Processing |
|---------|---------|------------|
| `Cy3` | RNA channel 1 | Spot detection |
| `Cy5` | RNA channel 2 | Spot detection |
| `DAPI` | Nuclei | Cell segmentation only |

Files are automatically grouped by FOV — the channel keyword is stripped and the remainder becomes the FOV identifier:

```
Location_Cy3_xy001.tif   →  FOV: location_xy001  channel: Cy3
Location_Cy5_xy001.tif   →  FOV: location_xy001  channel: Cy5
Location_DAPI_xy001.tif  →  FOV: location_xy001  channel: DAPI
```

Supported format: multi-page TIF stacks (16-bit grayscale). Use `--pattern "*.tiff"` for four-letter extensions.

---

## All Flags

### Input / Output

| Flag | Default | Description |
|------|---------|-------------|
| `--input_dir` | required | Folder containing TIF stacks |
| `--output_dir` | required | Output destination (created if absent) |
| `--pattern` | `*.tif` | Glob pattern for TIF files |

### Detector

| Flag | Default | Description |
|------|---------|-------------|
| `--detector` | `piscis` | `piscis` or `spot_caller` |

### Piscis

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `20230905` | Model name or local path |
| `--threshold` | `0.5` | Detection probability 0–1. Lower = more spots. |

### SpotCaller

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold` | `5.0` | MAD multiplier above local noise |
| `--spot_yx` | `3.0` | Expected spot radius in XY (pixels) |
| `--spot_z` | `2.0` | Expected spot radius in Z (planes) |
| `--min_snr` | `2.0` | Minimum SNR after Gaussian fit |
| `--max_sigma_ratio` | `2.5` | Reject spots with sigma > N × spot_yx |
| `--tile_size` | `128` | Tile size for adaptive noise estimation |
| `--use_3d` | off | Full 3-D LoG detection (slower) |

### Cellpose

| Flag | Default | Description |
|------|---------|-------------|
| `--cellpose_model` | — | Path to Cellpose model. Omit to skip segmentation. |
| `--cellpose_diam` | `0` | Expected cell diameter in px. 0 = auto. |

### Run control

| Flag | Default | Description |
|------|---------|-------------|
| `--run_only` | `both` | `piscis` \| `cellpose` \| `both` |
| `--workers` | `4` | Parallel threads (max 20) |
| `--projection` | `max` | `max` \| `mean` \| `best_focus` \| `middle` |
| `--save_previews` | off | Export 8-bit PNGs for the viewer |

---

## Outputs

| File | Description |
|------|-------------|
| `<stem>_spots.csv` | One row per detected spot (y, x + SpotCaller quality metrics) |
| `<stem>_cells.csv` | Per-cell spot counts (fov, cell_id, Cy3_spots, Cy5_spots) |
| `spot_counts.csv` | Per-FOV summary |
| `cell_summary.csv` | Aggregate per-FOV and per-channel statistics |
| `run_YYYYMMDD_HHMMSS.log` | Full timestamped log with tracebacks |
| `previews/<stem>.png` | 8-bit contrast-stretched preview |
| `previews/<stem>_mask.png` | Coloured cell boundary overlay |
| `<stem>_mask.npy` | Raw Cellpose integer label mask |

---

## Python API

```python
from spot_caller import SpotCaller

# Drop-in replacement for Piscis.predict()
model = SpotCaller(spot_yx=3.0, threshold=5.0, min_snr=2.0)
spots = model.predict(image_2d)     # returns (N, 2) float32 [y, x]

# Full output with quality metrics
df = model.detect(image_2d)
# columns: y, x, y_sub, x_sub, z_plane, snr, sigma_y, sigma_x, amplitude

# 3-D detection on a z-stack
df_3d = model.detect_3d(stack_zyx)
```

```python
from run_piscis import batch_process
from pathlib import Path

batch_process(
    input_dir=Path("/data/experiment"),
    output_dir=Path("/results/run1"),
    pattern="*.tif",
    detector="spot_caller",
    spot_yx=3.0, threshold=5.0, min_snr=2.0,
    workers=8,
    save_previews=True,
)
```

---

## Web GUI Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Main GUI |
| `GET /viewer` | Serve spot_viewer.html |
| `POST /api/run` | Submit a new job (JSON body) |
| `GET /api/jobs` | List all jobs |
| `GET /api/jobs/<id>` | Job details + last 500 log lines |
| `GET /api/stream/<id>` | SSE live log stream |
| `POST /api/cancel/<id>` | Cancel a running job |

---

## Detector Comparison

| Criterion | Piscis | SpotCaller |
|-----------|--------|------------|
| Speed (GPU) | Very fast | N/A |
| Model required | Yes | No |
| Sub-pixel coords | No | Yes |
| Per-spot metrics | No | SNR, sigma, amplitude |
| Uneven illumination | Partial | Good (adaptive noise) |
| Interpretability | Black box | Fully interpretable |

For new datasets, run both and compare in the viewer's **comparison mode**.

---

## License

Private repository — all rights reserved.
