"""
One-command demo runner.

Runs all five pipeline stages with synthetic data:
  1. Generate synthetic RGB + multispectral sample dataset
  2. Preprocess (verify transforms run correctly)
  3. Build YOLO detector + ResNeXt classifier
  4. Train both models (short demo: 3 epochs each)
  5. Inference on test images → annotated images + distribution maps

Run:
  python weed_detection_demo/run_demo.py

  # Full training run (longer):
  python weed_detection_demo/run_demo.py --full

  # Skip training, just generate maps from untrained model:
  python weed_detection_demo/run_demo.py --skip_train
"""

import argparse
import sys
import time
from pathlib import Path

# Make imports work from the repo root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from data.sample_generator import generate_dataset, WEED_CLASSES
from preprocessing.rgb_preprocessing import (
    WeedClassificationDataset, WeedDetectionDataset, DetectionPreprocessor
)
from preprocessing.multispectral_preprocessing import (
    MultispectralPreprocessor, fuse_channels, compute_ndvi, FUSION_MODES
)
from models.yolov8_detector import WeedDetector
from models.resnext50_classifier import WeedClassifier, CropClassifier
from inference.inference_pipeline import WeedInferencePipeline
from inference.distribution_map import WeedDistributionMapper


def step(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--full",       action="store_true", help="Full training (50/30 epochs)")
    p.add_argument("--skip_train", action="store_true", help="Skip training; demo map only")
    p.add_argument("--device",     default="cpu")
    p.add_argument("--output_dir", default=str(ROOT / ".." / "outputs"))
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device
    out    = Path(args.output_dir)

    det_epochs = (50 if args.full else 3)
    cls_epochs = (30 if args.full else 3)

    # ── Stage 1: Data Generation ──────────────────────────────────────────────
    step("Stage 1 — Synthetic Dataset Generation")
    dataset_root = out / "sample_dataset"
    t0 = time.time()
    generate_dataset(str(dataset_root))
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Stage 2: Preprocessing verification ──────────────────────────────────
    step("Stage 2 — Preprocessing Verification")
    import numpy as np

    # RGB classification
    cls_ds = WeedClassificationDataset(
        str(dataset_root / "classification" / "train"),
        WEED_CLASSES, split="train"
    )
    img_t, lbl = cls_ds[0]
    print(f"  [RGB Classifier] tensor: {tuple(img_t.shape)}, "
          f"label: {WEED_CLASSES[lbl]}")

    # Detection
    det_pre = DetectionPreprocessor()
    sample_img = next((dataset_root / "detection" / "images" / "train").glob("*.png"))
    t, hw, sc, pd = det_pre(str(sample_img))
    print(f"  [RGB Detector]   tensor: {tuple(t.shape)}, scale: {sc:.3f}")

    # Multispectral
    ms_file = next((dataset_root / "multispectral" / "train").glob("*.npy"))
    ms = np.load(str(ms_file))
    for mode in FUSION_MODES:
        fused = fuse_channels(ms, mode)
        print(f"  [MS fusion='{mode}'] shape: {fused.shape}, "
              f"range [{fused.min():.2f}, {fused.max():.2f}]")

    ndvi = compute_ndvi(ms)
    print(f"  [NDVI] range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")

    if args.skip_train:
        print("\n  --skip_train set: jumping to inference with untrained models")
        yolo_weights = None
        cls_weights  = None
    else:
        # ── Stage 3+4: Model build + Training ────────────────────────────────
        step(f"Stage 3+4 — YOLOv8 Training ({det_epochs} epochs)")
        detector = WeedDetector(
            model_size="n", num_classes=len(WEED_CLASSES), device=device
        )
        yaml_path = str(dataset_root / "detection" / "dataset.yaml")
        yolo_out  = str(out / "yolo_runs")
        yolo_weights = detector.train(
            dataset_yaml=yaml_path,
            epochs=det_epochs,
            batch=4,
            imgsz=640,
            output_dir=yolo_out,
        )
        print(f"  YOLOv8 best weights: {yolo_weights}")

        step(f"Stage 3+4 — ResNeXt-50 Training ({cls_epochs} epochs)")
        import torch
        from torch.utils.data import DataLoader
        from models.resnext50_classifier import (
            LabelSmoothingCrossEntropy, build_optimizer, build_scheduler,
            save_checkpoint
        )

        cls_root = dataset_root / "classification"
        train_ds = WeedClassificationDataset(
            str(cls_root / "train"), WEED_CLASSES, split="train"
        )
        val_ds = WeedClassificationDataset(
            str(cls_root / "val"), WEED_CLASSES, split="val"
        )
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)

        model     = WeedClassifier(num_classes=len(WEED_CLASSES),
                                   pretrained=False, freeze_backbone=True).to(device)
        criterion = LabelSmoothingCrossEntropy(0.1)
        optimizer = build_optimizer(model, lr=1e-3)
        scheduler = build_scheduler(optimizer, cls_epochs)

        best_acc  = 0.0
        cls_ckpt  = str(out / "classifier_runs" / "best_classifier.pt")
        Path(cls_ckpt).parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, cls_epochs + 1):
            model.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs), labels)
                loss.backward()
                optimizer.step()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    preds = model(imgs.to(device)).argmax(1).cpu()
                    correct += (preds == labels).sum().item()
                    total   += labels.size(0)
            val_acc = correct / total if total else 0.0
            scheduler.step()
            print(f"  Epoch {epoch}/{cls_epochs}  val_acc={val_acc:.3f}")

            if val_acc >= best_acc:
                best_acc = val_acc
                save_checkpoint(model, optimizer, epoch, val_acc, cls_ckpt)

        cls_weights = cls_ckpt
        print(f"  ResNeXt-50 best weights: {cls_weights}")

    # ── Stage 5: Inference ────────────────────────────────────────────────────
    step("Stage 5 — Inference + Weed Distribution Map")

    test_imgs = sorted((dataset_root / "test" / "rgb").glob("*.png"))
    test_ms   = [
        str(dataset_root / "test" / "ms" / f"{p.stem}_ms.npy")
        for p in test_imgs
    ]
    img_paths = [str(p) for p in test_imgs]
    ms_paths  = [p if Path(p).exists() else None for p in test_ms]

    pipeline = WeedInferencePipeline(
        yolo_weights=yolo_weights,
        cls_weights=cls_weights,
        device=device,
    )

    inf_out  = str(out / "inference_results")
    print(f"  Running on {len(img_paths)} test images…")
    results = pipeline.run_batch(img_paths, ms_paths, output_dir=inf_out)

    total_inst = sum(len(r.instances) for r in results)
    print(f"  {len(results)} images processed, {total_inst} weed instances")

    # Distribution maps
    mapper = WeedDistributionMapper(class_names=WEED_CLASSES)
    mapper.generate_all(results, output_dir=inf_out)

    step("COMPLETE")
    print(f"  All outputs in: {out.resolve()}")
    print(f"    {inf_out}/density_heatmap.png     — overall weed density")
    print(f"    {inf_out}/category_map.png         — per-species dot map")
    print(f"    {inf_out}/per_species_heatmaps/    — per-species heatmaps")
    print(f"    {inf_out}/summary_stats.txt        — detection statistics")
    print(f"    {inf_out}/*_detected.png           — annotated images")


if __name__ == "__main__":
    main()
