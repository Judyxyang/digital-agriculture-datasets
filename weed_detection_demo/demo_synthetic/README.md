# Synthetic Demo Archive

This folder preserves the **synthetic data demo** — a self-contained pipeline
that generates artificial weed imagery so the full pipeline can be tested
without any real labelled dataset.

## Files

| File | Purpose |
|------|---------|
| `sample_generator.py` | Generates 640×640 synthetic RGB images (soil background + weed blobs), YOLO `.txt` annotations, 5-band `.npy` multispectral arrays, and per-class crop patches. 9 DeepWeeds classes, 120 train + 30 val + 20 test. |
| `run_demo.py` | One-command end-to-end runner: generate → preprocess → train YOLOv8 → train ResNeXt-50 → inference + maps. |

## Usage

From the repo root:

```bash
# Quick smoke test (3 YOLO epochs, 3 ResNeXt epochs)
python weed_detection_demo/demo_synthetic/run_demo.py

# Full training run
python weed_detection_demo/demo_synthetic/run_demo.py --full

# Skip training — just test preprocessing and map generation
python weed_detection_demo/demo_synthetic/run_demo.py --skip_train
```

> **Note**: This is purely synthetic data for pipeline verification.
> For real-data training see `../../weed_real_pipeline/`.
