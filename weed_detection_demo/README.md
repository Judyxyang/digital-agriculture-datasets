# Weed Detection & Classification Demo

End-to-end pipeline for detecting and mapping weed species from RGB and
multispectral UAV/field imagery.

## Architecture

```
RGB image + Multispectral (.npy)
        │
        ▼
┌──────────────────────────────────┐
│  Stage 1 – Preprocessing        │
│  • Letterbox resize (640×640)    │
│  • Percentile band normalisation │
│  • Vegetation index fusion       │
│    (NDVI, GNDVI, RE-NDVI)        │
│  • Albumentations augmentation   │
└───────────────┬──────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│  Stage 2 – YOLOv8 Detection     │
│  CSPDarknet → PANet → Head      │
│  Output: bounding boxes per     │
│  weed instance                   │
└───────────────┬──────────────────┘
                │  crop each box
                ▼
┌──────────────────────────────────┐
│  Stage 3 – ResNeXt-50           │
│  Classification                  │
│  Output: species label +        │
│  confidence per box             │
└───────────────┬──────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│  Stage 4 – Distribution Maps    │
│  • Overall density heatmap       │
│  • Per-species category map      │
│  • Per-species KDE heatmaps      │
│  • Summary statistics            │
└──────────────────────────────────┘
```

## Weed Classes (DeepWeeds taxonomy)

| ID | Species | Description |
|----|---------|-------------|
| 0 | Chinee Apple | Ziziphus mauritiana |
| 1 | Lantana | Lantana camara |
| 2 | Parkinsonia | Parkinsonia aculeata |
| 3 | Parthenium | Parthenium hysterophorus |
| 4 | Prickly Acacia | Vachellia nilotica |
| 5 | Rubber Vine | Cryptostegia grandiflora |
| 6 | Siam Weed | Chromolaena odorata |
| 7 | Snake Weed | Stachytarpheta spp. |
| 8 | Negative | Background / no weed |

## Do You Need Annotations?

**Yes.** Training requires:

| Model | Annotation type | Format |
|-------|----------------|--------|
| YOLOv8 | Bounding box per weed instance | `class_id cx cy w h` (YOLO normalised) in `.txt` |
| ResNeXt-50 | Class label per image crop | Folder-per-class directory structure |

For **real data**, use [Label Studio](https://labelstud.io/),
[CVAT](https://cvat.ai/), or [Roboflow](https://roboflow.com/) to annotate.
Export in YOLO format. This demo generates synthetic annotated data automatically.

## Quick Start

```bash
# Install dependencies
pip install -r weed_detection_demo/requirements.txt

# Run full demo (generates data, trains 3 epochs, runs inference, makes maps)
python weed_detection_demo/run_demo.py

# Full training run (50 YOLO epochs + 30 ResNeXt epochs)
python weed_detection_demo/run_demo.py --full

# Skip training – just test preprocessing and map generation
python weed_detection_demo/run_demo.py --skip_train

# GPU training
python weed_detection_demo/run_demo.py --device cuda
```

## Individual Stage Scripts

```bash
# 1. Generate synthetic dataset only
python -m weed_detection_demo.data.sample_generator

# 2. Train YOLO detector
python -m weed_detection_demo.training.train_detector \
  --dataset_root outputs/sample_dataset --epochs 50 --model_size n

# 3. Train ResNeXt classifier
python -m weed_detection_demo.training.train_classifier \
  --dataset_root outputs/sample_dataset --epochs 30

# 4. Inference + distribution maps
python -m weed_detection_demo.inference.inference_pipeline \
  --yolo_weights  outputs/yolo_runs/weed_detection/weights/best.pt \
  --cls_weights   outputs/classifier_runs/best_classifier.pt \
  --input_dir     outputs/sample_dataset/test/rgb \
  --ms_dir        outputs/sample_dataset/test/ms

# 5. Distribution map standalone demo (no trained models needed)
python -m weed_detection_demo.inference.distribution_map
```

## Output Files

```
outputs/
  sample_dataset/
    detection/images/{train,val}/   RGB images (PNG)
    detection/labels/{train,val}/   YOLO annotations (TXT)
    classification/{train,val}/<class>/  Weed crop patches (PNG)
    multispectral/{train,val}/      5-band arrays (NPY)
    test/{rgb,ms}/                  Unlabelled test images
  yolo_runs/weed_detection/
    weights/best.pt                 Best YOLOv8 checkpoint
    results.png                     Training curves
  classifier_runs/
    best_classifier.pt              Best ResNeXt-50 checkpoint
  inference_results/
    *_detected.png                  Annotated images
    density_heatmap.png             Overall weed pressure map
    category_map.png                Per-species distribution map
    per_species_heatmaps/           Per-species KDE heatmaps
    summary_stats.txt               Detection counts & spatial stats
```

## Multispectral Fusion Modes

| Mode | Channels | Use case |
|------|----------|----------|
| `rgb` | R, G, B | Baseline – identical to RGB model |
| `rgb_ndvi` | R, G, NDVI | Best balance; NDVI highlights vegetation |
| `all5` | R, G, B, NIR, RE | Maximum information; needs custom first conv |
| `vi_stack` | NDVI, GNDVI, RE-NDVI | Pure vegetation index analysis |

## Model Notes

- **YOLOv8** (requested as "YOLOv26" — does not exist): Current state-of-the-art
  single-stage detector from Ultralytics. Uses CSPDarknet backbone + PANet neck +
  decoupled detection head. Available in 5 sizes (n/s/m/l/x).

- **ResNeXt-50-32x4d** (requested as "ResNeXt52" — does not exist): Standard
  torchvision model with 50 layers and 32 groups of width 4. Trained with
  two-phase strategy: frozen backbone warmup → full fine-tuning.
