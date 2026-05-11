# Weed Real-Data Pipeline

End-to-end training and inference pipeline for **real open-source weed datasets**
in YOLO format. Designed to work with `weed_archive_detection` (5 classes, 1370 images)
and any similarly-structured YOLO dataset.

Built to be future-ready for **5-band multispectral** imagery — simply swap the
simulated `.npy` files for real sensor data when available.

## Architecture

```
Local YOLO dataset  (images/ + labels/ + data.yaml)
        │
        ▼
┌──────────────────────────────────────┐
│  Stage 1-4  — Data Preparation      │
│  • Auto-detect classes + structure   │
│  • Fix data.yaml absolute paths      │
│  • Simulate 5-band MS arrays (RGB→5) │
│    or load real .npy sensor files    │
│  • Extract per-class crop patches    │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Stage 5  — YOLOv8 Detection        │
│  CSPDarknet → PANet → Head          │
│  Fine-tuned on your weed classes    │
└──────────────────┬───────────────────┘
                   │  crop each box
                   ▼
┌──────────────────────────────────────┐
│  Stage 6  — ResNeXt-50              │
│  Classification (2-phase training)   │
│  Phase 1: frozen backbone            │
│  Phase 2: unfreeze last 2 groups    │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Stage 7  — Distribution Maps       │
│  • Density heatmap (KDE)            │
│  • Per-species dot map              │
│  • Per-species KDE heatmaps         │
│  • Summary statistics               │
└──────────────────────────────────────┘
```

## Quick Start

```bash
# From the repo root — uses weed_archive_detection (auto-detected from config):
python weed_real_pipeline/run_pipeline.py

# Point at a specific dataset:
python weed_real_pipeline/run_pipeline.py \
    --dataset_dir /path/to/weed_archive_detection

# GPU training:
python weed_real_pipeline/run_pipeline.py --device cuda

# Skip training (test inference only):
python weed_real_pipeline/run_pipeline.py --skip_train \
    --yolo_weights outputs/real_run/yolo/weed_detection/weights/best.pt \
    --cls_weights  outputs/real_run/classifier/best_classifier.pt
```

## Multispectral Fusion Modes

| `--fusion_mode` | Channels fed to model | Use case |
|-----------------|----------------------|----------|
| `rgb`           | R, G, B (3ch)        | Baseline |
| `rgb_ndvi`      | R, G, NDVI (3ch)     | **Default** — NDVI highlights vegetation stress |
| `all5`          | R, G, B, NIR, RE (5ch) | Maximum info; set `in_channels: 5` in config |
| `vi_stack`      | NDVI, GNDVI, RE-NDVI (3ch) | Pure vegetation-index analysis |

```bash
# Use all 5 bands with real 5-band sensor data:
python weed_real_pipeline/run_pipeline.py \
    --fusion_mode all5 --in_channels 5 --no-simulate_ms
```

## Using Real 5-Band Imagery

When real multispectral data is available (e.g. MicaSense RedEdge, Parrot Sequoia):

1. Convert your GeoTIFF or raw sensor files to `.npy` arrays of shape `(H, W, 5)`
   with band order `[R, G, B, NIR, RedEdge]` and float32 values in `[0, 1]`.
2. Place them alongside the RGB images as `<image_stem>_ms.npy`.
3. Run with `--no-simulate_ms`.

The rest of the pipeline handles everything automatically.

## Configuration

All defaults are in `configs/config.yaml`. CLI flags override config values.

Key settings:
```yaml
dataset:
  root: "../weed_archive_detection"   # ← point at your dataset

multispectral:
  fusion_mode: "rgb_ndvi"             # ← or "all5" for real 5-band

detection:
  model_size: "n"                     # ← "s"/"m"/"l"/"x" for more accuracy
  epochs: 50

classification:
  in_channels: 3                      # ← set to 5 for all5 fusion with real MS
```

## Output Files

```
outputs/real_run/
  data.yaml                          dataset config with absolute paths
  multispectral/{train,val}/         5-band .npy arrays (simulated or real)
  classification/{train,val}/<cls>/  per-class crop patches for classifier
  yolo/weed_detection/
    weights/best.pt                  best YOLOv8 checkpoint
    results.png                      training curves
  classifier/
    best_classifier.pt               best ResNeXt-50 checkpoint
  inference_results/
    *_detected.png                   annotated images
    density_heatmap.png              overall weed pressure map
    category_map.png                 per-species distribution map
    per_species_heatmaps/            per-species KDE maps
    summary_stats.txt                counts + spatial statistics
```

## Dataset Format

Expects standard YOLO format (Roboflow / Label Studio / CVAT export):

```
dataset_root/
  data.yaml            ← nc, names, train/val paths
  images/
    train/  *.jpg / *.png
    valid/  *.jpg / *.png
    test/   *.jpg / *.png   (optional)
  labels/
    train/  *.txt   (one row per box: class_id cx cy w h, normalised)
    valid/  *.txt
    test/   *.txt
```

The loader auto-handles `val` vs `valid` naming, and rewrites `data.yaml`
with correct absolute paths (fixing paths exported for Kaggle / Colab).
