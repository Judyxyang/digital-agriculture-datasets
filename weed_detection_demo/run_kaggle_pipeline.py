"""
Full pipeline using the Kaggle weed detection dataset.

  https://www.kaggle.com/datasets/thaslimvs/weeds-detection-dataset

Steps
─────
  1. Download dataset via kaggle CLI  (skipped if already present)
  2. Auto-detect classes + YOLO structure
  3. Simulate multispectral bands (NIR, RedEdge) from RGB
  4. Extract per-class crops for ResNeXt-50 classifier training
  5. Train YOLOv8 detector
  6. Train ResNeXt-50 classifier
  7. Inference on val/test images → annotated images + distribution maps

Prerequisites
─────────────
  pip install -r weed_detection_demo/requirements.txt
  # Kaggle API token in ~/.kaggle/kaggle.json
  #   {"username": "YOUR_USERNAME", "key": "YOUR_API_KEY"}

Usage
─────
  # Full pipeline (downloads data + trains):
  python weed_detection_demo/run_kaggle_pipeline.py

  # If dataset already downloaded:
  python weed_detection_demo/run_kaggle_pipeline.py \
      --dataset_dir outputs/kaggle_dataset --skip_download

  # Skip training, just run inference + maps on existing weights:
  python weed_detection_demo/run_kaggle_pipeline.py \
      --skip_download --skip_train \
      --yolo_weights  outputs/yolo_runs/weed_detection/weights/best.pt \
      --cls_weights   outputs/classifier_runs/best_classifier.pt

  # GPU:
  python weed_detection_demo/run_kaggle_pipeline.py --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from data.kaggle_loader import KaggleWeedDataset, download_kaggle
from preprocessing.rgb_preprocessing import WeedClassificationDataset
from preprocessing.multispectral_preprocessing import (
    MultispectralPreprocessor, fuse_channels, compute_ndvi, FUSION_MODES
)
from models.yolov8_detector import WeedDetector
from models.resnext50_classifier import (
    WeedClassifier, LabelSmoothingCrossEntropy,
    build_optimizer, build_scheduler, save_checkpoint
)
from inference.inference_pipeline import WeedInferencePipeline
from inference.distribution_map import WeedDistributionMapper


def step(title: str):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Weed detection pipeline using Kaggle dataset"
    )
    p.add_argument("--dataset_dir",   default="outputs/kaggle_dataset",
                   help="Where to download / look for the Kaggle dataset")
    p.add_argument("--output_dir",    default="outputs")
    p.add_argument("--skip_download", action="store_true",
                   help="Dataset already downloaded; skip kaggle download")
    p.add_argument("--skip_train",    action="store_true",
                   help="Skip training; requires --yolo_weights and --cls_weights")
    p.add_argument("--yolo_weights",  default=None)
    p.add_argument("--cls_weights",   default=None)
    p.add_argument("--det_epochs",    type=int, default=50)
    p.add_argument("--cls_epochs",    type=int, default=30)
    p.add_argument("--batch",         type=int, default=16)
    p.add_argument("--device",        default="cpu")
    p.add_argument("--model_size",    default="n",
                   choices=["n", "s", "m", "l", "x"])
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.output_dir)
    ds_dir = Path(args.dataset_dir)

    # ── Stage 1: Download ─────────────────────────────────────────────────────
    step("Stage 1 — Kaggle Dataset Download")
    if not args.skip_download:
        download_kaggle(str(ds_dir))
    else:
        print(f"  Skipping download. Using: {ds_dir}")

    # ── Stage 2: Detect dataset structure ────────────────────────────────────
    step("Stage 2 — Dataset Structure Detection")
    ds = KaggleWeedDataset(str(ds_dir))
    stats = ds.stats()
    print(f"\n  Classes ({stats['num_classes']}): {stats['class_names']}")
    print(f"  Train images : {stats['train_images']}")
    print(f"  Val images   : {stats['val_images']}")

    # Ensure valid data.yaml
    yaml_path = ds.write_yaml()
    CLASS_NAMES = ds.class_names

    # ── Stage 3: Multispectral simulation ────────────────────────────────────
    step("Stage 3 — Simulate Multispectral Bands (NIR, RedEdge)")
    ms_train = str(ds_dir / "multispectral" / "train")
    ms_val   = str(ds_dir / "multispectral" / "val")
    ds.generate_multispectral(ms_train, split="train")
    ds.generate_multispectral(ms_val,   split="val")

    # Verify one MS array
    import numpy as np
    ms_files = list(Path(ms_train).glob("*.npy"))
    if ms_files:
        ms = np.load(str(ms_files[0]))
        for mode in FUSION_MODES:
            fused = fuse_channels(ms, mode)
            print(f"  fusion='{mode}' → {fused.shape}  "
                  f"range [{fused.min():.2f}, {fused.max():.2f}]")
        ndvi = compute_ndvi(ms)
        print(f"  NDVI range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")

    # ── Stage 4: Extract crops for classifier ─────────────────────────────────
    step("Stage 4 — Extract Weed Crops for ResNeXt-50 Classifier")
    crops_train = str(ds_dir / "classification" / "train")
    crops_val   = str(ds_dir / "classification" / "val")
    ds.extract_crops(crops_train, split="train")
    ds.extract_crops(crops_val,   split="val")

    if args.skip_train:
        print("\n  --skip_train set. Using provided weights.")
        yolo_weights = args.yolo_weights
        cls_weights  = args.cls_weights
    else:
        # ── Stage 5: Train YOLOv8 ────────────────────────────────────────────
        step(f"Stage 5 — YOLOv8-{args.model_size} Training "
             f"({args.det_epochs} epochs)")
        detector = WeedDetector(
            model_size=args.model_size,
            num_classes=len(CLASS_NAMES),
            device=args.device,
        )
        yolo_out = str(out / "yolo_runs")
        t0 = time.time()
        yolo_weights = detector.train(
            dataset_yaml=yaml_path,
            epochs=args.det_epochs,
            imgsz=640,
            batch=args.batch,
            lr0=0.01,
            output_dir=yolo_out,
            patience=20,
            cos_lr=True,
            mosaic=1.0,
            mixup=0.1,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            fliplr=0.5,
            flipud=0.2,
        )
        print(f"  YOLOv8 training done in {time.time()-t0:.0f}s")
        print(f"  Best weights: {yolo_weights}")

        # Validate
        metrics = detector.validate(yaml_path)
        print("\n  Validation metrics:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

        # ── Stage 6: Train ResNeXt-50 ─────────────────────────────────────────
        step(f"Stage 6 — ResNeXt-50 Training ({args.cls_epochs} epochs)")
        import torch
        from torch.utils.data import DataLoader

        train_cls_ds = WeedClassificationDataset(
            crops_train, CLASS_NAMES, split="train"
        )
        val_cls_ds = WeedClassificationDataset(
            crops_val, CLASS_NAMES, split="val"
        )
        if len(train_cls_ds) == 0:
            print("  WARNING: No classification crops found. "
                  "Check labels directory.")
        else:
            print(f"  Train crops: {len(train_cls_ds)}, "
                  f"Val crops: {len(val_cls_ds)}")

        train_loader = DataLoader(train_cls_ds, batch_size=min(32, args.batch),
                                  shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_cls_ds,   batch_size=min(32, args.batch),
                                  shuffle=False, num_workers=0)

        device = torch.device(args.device)
        model  = WeedClassifier(
            num_classes=len(CLASS_NAMES),
            pretrained=True,
            in_channels=3,
            dropout_rate=0.3,
            freeze_backbone=True,
        ).to(device)

        total = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  ResNeXt-50-32x4d | total params: {total:,} | "
              f"trainable (phase 1): {train_p:,}")

        criterion = LabelSmoothingCrossEntropy(0.1)
        optimizer = build_optimizer(model, lr=1e-3)
        scheduler = build_scheduler(optimizer, args.cls_epochs, warmup_epochs=5)

        best_acc  = 0.0
        cls_ckpt  = str(out / "classifier_runs" / "best_classifier.pt")
        Path(cls_ckpt).parent.mkdir(parents=True, exist_ok=True)
        WARMUP    = 5

        t0 = time.time()
        for epoch in range(1, args.cls_epochs + 1):
            if epoch == WARMUP + 1:
                model.unfreeze_backbone(layers_from_end=2)
                train_p = sum(p.numel() for p in model.parameters()
                              if p.requires_grad)
                print(f"\n  Phase 2: backbone unfrozen. "
                      f"Trainable: {train_p:,}")
                optimizer = build_optimizer(model, lr=1e-4)
                scheduler = build_scheduler(optimizer,
                                            args.cls_epochs - WARMUP,
                                            warmup_epochs=0)

            model.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            correct = total_n = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    preds = model(imgs.to(device)).argmax(1).cpu()
                    correct += (preds == labels).sum().item()
                    total_n += labels.size(0)
            val_acc = correct / total_n if total_n else 0.0
            scheduler.step()
            print(f"  Epoch {epoch:3d}/{args.cls_epochs}  "
                  f"val_acc={val_acc:.3f}")

            if val_acc >= best_acc:
                best_acc = val_acc
                save_checkpoint(model, optimizer, epoch, val_acc, cls_ckpt)

        print(f"  ResNeXt-50 training done in {time.time()-t0:.0f}s")
        print(f"  Best val acc: {best_acc:.4f}")
        cls_weights = cls_ckpt

    # ── Stage 7: Inference + Distribution Maps ────────────────────────────────
    step("Stage 7 — Inference + Weed Distribution Map")

    # Collect val images for inference
    val_imgs_dir = ds.val_imgs or ds.train_imgs
    img_paths = (list(val_imgs_dir.rglob("*.jpg")) +
                 list(val_imgs_dir.rglob("*.png")))
    img_paths = [str(p) for p in sorted(img_paths)[:50]]   # cap at 50 for demo

    # Match MS arrays
    ms_dir_p = Path(ms_val)
    ms_paths = []
    for ip in img_paths:
        ms_p = ms_dir_p / f"{Path(ip).stem}_ms.npy"
        ms_paths.append(str(ms_p) if ms_p.exists() else None)

    print(f"  Running on {len(img_paths)} images…")
    pipeline = WeedInferencePipeline(
        yolo_weights=yolo_weights,
        cls_weights=cls_weights,
        class_names=CLASS_NAMES,
        device=args.device,
        det_conf=0.25,
        det_iou=0.45,
    )

    inf_out = str(out / "inference_results")
    results = pipeline.run_batch(img_paths, ms_paths, output_dir=inf_out)

    total_inst = sum(len(r.instances) for r in results)
    print(f"  {len(results)} images, {total_inst} weed instances detected")

    mapper = WeedDistributionMapper(class_names=CLASS_NAMES)
    mapper.generate_all(results, output_dir=inf_out)

    step("COMPLETE")
    print(f"  Outputs → {out.resolve()}")
    print(f"    {inf_out}/density_heatmap.png   — weed pressure map")
    print(f"    {inf_out}/category_map.png       — per-species dot map")
    print(f"    {inf_out}/per_species_heatmaps/  — per-species KDE maps")
    print(f"    {inf_out}/summary_stats.txt      — detection statistics")
    print(f"    {inf_out}/*_detected.png         — annotated images")


if __name__ == "__main__":
    main()
