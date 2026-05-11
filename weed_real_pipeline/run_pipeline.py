"""
Real-data weed detection + classification pipeline.

Designed for any locally-available YOLO-format weed dataset, including
weed_archive_detection (5 classes, 958/274/138 images).

Stages
──────
  1. Load & validate dataset structure
  2. Rewrite data.yaml with absolute paths (fixes Kaggle/Roboflow paths)
  3. Simulate 5-band multispectral arrays from RGB  (or load real .npy if available)
  4. Extract per-class crop patches for ResNeXt-50 classifier
  5. Train YOLOv8 detector
  6. Train ResNeXt-50 classifier (2-phase: frozen backbone → partial unfreeze)
  7. Run inference on val/test images → annotated images + distribution maps

Usage
─────
  # Full run with default config:
  python weed_real_pipeline/run_pipeline.py

  # Point at a different dataset:
  python weed_real_pipeline/run_pipeline.py --dataset_dir /path/to/my_dataset

  # Skip training (use existing weights):
  python weed_real_pipeline/run_pipeline.py --skip_train \\
      --yolo_weights outputs/real_run/yolo/weed_detection/weights/best.pt \\
      --cls_weights  outputs/real_run/classifier/best_classifier.pt

  # GPU:
  python weed_real_pipeline/run_pipeline.py --device cuda

  # Override config fusion mode:
  python weed_real_pipeline/run_pipeline.py --fusion_mode all5

Configuration
─────────────
  All defaults live in weed_real_pipeline/configs/config.yaml.
  CLI flags override config values.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_DIR))   # data/, preprocessing/, models/, inference/

import yaml

from data.dataset_loader import LocalWeedDataset
from preprocessing.preprocessing import RGBClassificationDataset, FUSION_MODES
from models.yolov8_detector import WeedDetector
from models.resnext50_classifier import (
    WeedClassifier, LabelSmoothingCrossEntropy,
    build_optimizer, build_scheduler, save_checkpoint, load_checkpoint,
)
from inference.inference_pipeline import WeedInferencePipeline
from inference.distribution_map import WeedDistributionMapper
from evaluation.model_assessment import ClassifierAssessment


# ── Helpers ───────────────────────────────────────────────────────────────────

def step(title: str):
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")


def load_config() -> dict:
    cfg_path = PIPELINE_DIR / "configs" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def parse_args(cfg: dict):
    p = argparse.ArgumentParser(
        description="Real-data weed detection + classification pipeline"
    )
    ds_cfg  = cfg.get("dataset", {})
    det_cfg = cfg.get("detection", {})
    cls_cfg = cfg.get("classification", {})
    inf_cfg = cfg.get("inference", {})
    ms_cfg  = cfg.get("multispectral", {})

    p.add_argument("--dataset_dir",  default=ds_cfg.get("root", "../weed_archive_detection"))
    p.add_argument("--output_dir",   default=ds_cfg.get("output_dir", "outputs/real_run"))
    p.add_argument("--skip_train",   action="store_true")
    p.add_argument("--skip_yolo",    action="store_true",
                   help="Skip YOLO detection stage (use for classification-only datasets like DeepWeeds)")
    p.add_argument("--yolo_weights", default=None)
    p.add_argument("--cls_weights",  default=None)
    p.add_argument("--det_epochs",   type=int,   default=det_cfg.get("epochs", 50))
    p.add_argument("--cls_epochs",   type=int,   default=cls_cfg.get("epochs", 30))
    p.add_argument("--batch",        type=int,   default=det_cfg.get("batch", 16))
    p.add_argument("--device",       default=cfg.get("device", "cpu"))
    p.add_argument("--model_size",   default=det_cfg.get("model_size", "n"),
                   choices=["n", "s", "m", "l", "x"])
    p.add_argument("--fusion_mode",  default=ms_cfg.get("fusion_mode", "rgb_ndvi"),
                   choices=list(FUSION_MODES))
    p.add_argument("--in_channels",  type=int,
                   default=cls_cfg.get("in_channels", 3),
                   help="3 for rgb/rgb_ndvi/vi_stack; 5 for all5")
    p.add_argument("--simulate_ms",  default=ms_cfg.get("simulate", True),
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--max_infer",    type=int,
                   default=inf_cfg.get("max_images", 50),
                   help="Max images for inference demo (0 = unlimited)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg  = load_config()
    args = parse_args(cfg)

    cls_cfg = cfg.get("classification", {})
    det_cfg = cfg.get("detection", {})
    inf_cfg = cfg.get("inference", {})

    out    = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds_dir = Path(args.dataset_dir).resolve()

    # ── Stage 1: Load dataset ─────────────────────────────────────────────────
    step("Stage 1 — Load & Validate Dataset")
    ds    = LocalWeedDataset(str(ds_dir))
    stats = ds.stats()
    print(f"\n  Classes ({stats['num_classes']}): {stats['class_names']}")
    print(f"  Train : {stats['train_images']} images")
    print(f"  Val   : {stats['val_images']} images")
    print(f"  Test  : {stats['test_images']} images")
    CLASS_NAMES = ds.class_names

    # ── Stage 2: Fix data.yaml ────────────────────────────────────────────────
    step("Stage 2 — Write data.yaml with Absolute Paths")
    yaml_path = ds.write_yaml()

    # ── Stage 3: Multispectral ────────────────────────────────────────────────
    step("Stage 3 — Multispectral Arrays")
    ms_train = str(out / "multispectral" / "train")
    ms_val   = str(out / "multispectral" / "val")

    if args.simulate_ms:
        print(f"  Simulating 5-band MS (fusion mode: {args.fusion_mode})")
        ds.generate_multispectral(ms_train, split="train")
        ds.generate_multispectral(ms_val,   split="val")
    else:
        print("  --no-simulate_ms: expecting pre-existing .npy files in:")
        print(f"    {ms_train}")
        print(f"    {ms_val}")

    # Quick sanity check
    import numpy as np
    from preprocessing.multispectral_preprocessing import (
        fuse_channels, compute_ndvi, FUSION_MODES as _FM,
    )
    ms_files = list(Path(ms_train).glob("*.npy"))
    if ms_files:
        ms = np.load(str(ms_files[0]))
        print(f"\n  Sample MS array: {ms.shape}  dtype={ms.dtype}")
        fused = fuse_channels(ms, args.fusion_mode)
        print(f"  Fused ({args.fusion_mode}): {fused.shape}  "
              f"range [{fused.min():.2f}, {fused.max():.2f}]")
        ndvi = compute_ndvi(ms)
        print(f"  NDVI range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")

    # ── Stage 4: Extract crops ────────────────────────────────────────────────
    step("Stage 4 — Extract Crop Patches for ResNeXt-50")
    crops_train = str(out / "classification" / "train")
    crops_val   = str(out / "classification" / "val")
    ds.extract_crops(crops_train, split="train",
                     crop_size=cls_cfg.get("crop_size", 96))
    ds.extract_crops(crops_val,   split="val",
                     crop_size=cls_cfg.get("crop_size", 96))

    if args.skip_train:
        print("\n  --skip_train: using provided weights.")
        yolo_weights = args.yolo_weights
        cls_weights  = args.cls_weights
    else:
        # ── Stage 5: Train YOLOv8 ────────────────────────────────────────────
        if args.skip_yolo:
            step("Stage 5 — YOLOv8 Training [SKIPPED — classification-only dataset]")
            yolo_weights = None
        else:
            step(f"Stage 5 — YOLOv8-{args.model_size} Training ({args.det_epochs} epochs)")
            detector = WeedDetector(
                model_size=args.model_size,
                num_classes=len(CLASS_NAMES),
                device=args.device,
            )
            yolo_out = str(out / "yolo")
            t0 = time.time()
            yolo_weights = detector.train(
                dataset_yaml=yaml_path,
                epochs=args.det_epochs,
                imgsz=640,
                batch=args.batch,
                lr0=det_cfg.get("lr0", 0.01),
                output_dir=yolo_out,
                patience=det_cfg.get("patience", 20),
                cos_lr=det_cfg.get("cos_lr", True),
                mosaic=det_cfg.get("mosaic", 1.0),
                mixup=det_cfg.get("mixup", 0.1),
                hsv_h=det_cfg.get("hsv_h", 0.015),
                hsv_s=det_cfg.get("hsv_s", 0.7),
                hsv_v=det_cfg.get("hsv_v", 0.4),
                fliplr=det_cfg.get("fliplr", 0.5),
                flipud=det_cfg.get("flipud", 0.2),
            )
            print(f"  YOLOv8 done in {time.time()-t0:.0f}s  →  {yolo_weights}")

            metrics = detector.validate(yaml_path)
            print("\n  Validation:")
            for k, v in metrics.items():
                print(f"    {k}: {v:.4f}")

        # ── Stage 6: Train ResNeXt-50 ─────────────────────────────────────────
        step(f"Stage 6 — ResNeXt-50 Classifier ({args.cls_epochs} epochs)")
        import torch
        from torch.utils.data import DataLoader

        img_size  = cls_cfg.get("img_size", 224)
        train_ds  = RGBClassificationDataset(crops_train, CLASS_NAMES, "train", img_size)
        val_ds    = RGBClassificationDataset(crops_val,   CLASS_NAMES, "val",   img_size)

        if len(train_ds) == 0:
            print("  WARNING: No crops found — check label files and split paths.")
        else:
            print(f"  Train crops: {len(train_ds)}  |  Val crops: {len(val_ds)}")

        bs = min(cls_cfg.get("batch", 32), args.batch)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0)

        # Resolve device — "0,1" means multi-GPU; use cuda:0 as primary for torch
        _dev_str = args.device
        if "," in _dev_str:
            primary_device = torch.device("cuda:0")
        elif _dev_str.isdigit():
            primary_device = torch.device(f"cuda:{_dev_str}")
        else:
            primary_device = torch.device(_dev_str)
        device = primary_device

        model  = WeedClassifier(
            num_classes=len(CLASS_NAMES),
            pretrained=True,
            in_channels=args.in_channels,
            dropout_rate=cls_cfg.get("dropout", 0.3),
            freeze_backbone=True,
        ).to(device)

        # Wrap in DataParallel if multiple GPUs requested
        use_multi_gpu = (
            cfg.get("multi_gpu_classifier", False) and
            torch.cuda.device_count() > 1 and
            "," in _dev_str
        )
        if use_multi_gpu:
            gpu_ids = [int(x) for x in _dev_str.split(",")]
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
            print(f"  Using DataParallel on GPUs: {gpu_ids}")

        total   = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  ResNeXt-50-32x4d | total: {total:,} | trainable (phase 1): {train_p:,}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

        # build_optimizer / unfreeze_backbone need the inner model, not DataParallel wrapper
        inner_model = model.module if isinstance(model, torch.nn.DataParallel) else model

        criterion = LabelSmoothingCrossEntropy(cls_cfg.get("label_smoothing", 0.1))
        optimizer = build_optimizer(inner_model, lr=cls_cfg.get("lr_head", 1e-3))
        scheduler = build_scheduler(optimizer, args.cls_epochs,
                                    warmup_epochs=cls_cfg.get("warmup_epochs", 5))

        WARMUP    = cls_cfg.get("warmup_epochs", 5)
        best_acc  = 0.0
        cls_ckpt  = str(out / "classifier" / "best_classifier.pt")
        Path(cls_ckpt).parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        for epoch in range(1, args.cls_epochs + 1):
            if epoch == WARMUP + 1:
                inner_model.unfreeze_backbone(layers_from_end=2)
                train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"\n  Phase 2: backbone partially unfrozen. Trainable: {train_p:,}")
                optimizer = build_optimizer(inner_model, lr=cls_cfg.get("lr_backbone", 1e-4))
                scheduler = build_scheduler(optimizer,
                                            args.cls_epochs - WARMUP, warmup_epochs=0)

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
                    preds    = model(imgs.to(device)).argmax(1).cpu()
                    correct += (preds == labels).sum().item()
                    total_n += labels.size(0)
            val_acc = correct / total_n if total_n else 0.0
            scheduler.step()
            print(f"  Epoch {epoch:3d}/{args.cls_epochs}  val_acc={val_acc:.3f}")

            if val_acc >= best_acc:
                best_acc = val_acc
                save_checkpoint(inner_model, optimizer, epoch, val_acc, cls_ckpt)

        print(f"\n  ResNeXt-50 done in {time.time()-t0:.0f}s")
        print(f"  Best val acc: {best_acc:.4f}  →  {cls_ckpt}")
        cls_weights = cls_ckpt

    # ── Stage 7: Inference + Distribution Maps ────────────────────────────────
    step("Stage 7 — Inference + Weed Distribution Maps")

    inf_out = str(out / "inference_results")
    if args.skip_yolo:
        print("  --skip_yolo: no YOLO detector — skipping inference & distribution maps.")
        results = []
    else:
        # Prefer test split → val → train (in that order) so the distribution map
        # is built from held-out data rather than training images.
        if ds.test_imgs is not None:
            infer_src   = ds.test_imgs
            infer_split = "test"
        elif ds.val_imgs is not None:
            infer_src   = ds.val_imgs
            infer_split = "val"
        else:
            infer_src   = ds.train_imgs
            infer_split = "train"

        print(f"  Inference split: '{infer_split}'  ({infer_src})")

        img_paths = sorted(
            list(infer_src.rglob("*.jpg")) + list(infer_src.rglob("*.png"))
        )
        total_avail = len(img_paths)
        if args.max_infer > 0:
            img_paths = img_paths[:args.max_infer]
        if len(img_paths) < total_avail:
            print(f"  NOTE: capped at {args.max_infer}/{total_avail} images "
                  f"(set --max_infer 0 for all)")
        img_paths = [str(p) for p in img_paths]

        # Locate multispectral files for the chosen split
        if infer_split == "test":
            ms_infer_dir = str(out / "multispectral" / "test")
            if args.simulate_ms:
                ds.generate_multispectral(ms_infer_dir, split="test")
        elif infer_split == "val":
            ms_infer_dir = ms_val
        else:
            ms_infer_dir = ms_train

        ms_dir_p = Path(ms_infer_dir)
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
            det_conf=inf_cfg.get("det_conf", 0.25),
            det_iou=inf_cfg.get("det_iou", 0.45),
        )

        results = pipeline.run_batch(img_paths, ms_paths, output_dir=inf_out)

        total_inst = sum(len(r.instances) for r in results)
        print(f"  {len(results)} images | {total_inst} weed instances detected")

    mapper = WeedDistributionMapper(class_names=CLASS_NAMES)
    mapper.generate_all(results, output_dir=inf_out)

    # ── Stage 8: Model Assessment (DeepWeeds-style metrics) ──────────────────
    step("Stage 8 — Model Assessment (DeepWeeds-style metrics)")

    if cls_weights and Path(cls_weights).exists():
        import torch as _torch
        _dev_str = args.device
        if "," in _dev_str:
            _assess_device = _torch.device("cuda:0")
        elif _dev_str.isdigit():
            _assess_device = _torch.device(f"cuda:{_dev_str}")
        else:
            _assess_device = _torch.device(_dev_str)

        # Use val crops for assessment (always available; test crops extracted if present)
        assess_crops = crops_val
        assess_split = "val"
        if ds.test_imgs is not None:
            test_crops = str(out / "classification" / "test")
            ds.extract_crops(test_crops, split="test",
                             crop_size=cls_cfg.get("crop_size", 96))
            assess_crops = test_crops
            assess_split = "test"

        img_size = cls_cfg.get("img_size", 224)
        assess_ds = RGBClassificationDataset(assess_crops, CLASS_NAMES,
                                             "val", img_size)

        if len(assess_ds) == 0:
            print("  WARNING: No crop patches found for assessment — skipping.")
        else:
            print(f"  Assessment split : '{assess_split}' ({len(assess_ds)} crops)")
            clf_model = load_checkpoint(cls_weights, CLASS_NAMES,
                                        device=str(_assess_device))
            assessor = ClassifierAssessment(
                class_names=CLASS_NAMES,
                output_dir=str(out / "assessment"),
            )
            metrics = assessor.evaluate(clf_model, assess_ds, _assess_device,
                                        batch_size=cls_cfg.get("batch", 32))

            # Collect YOLO detection metrics if we trained
            _yolo_metrics = None
            if not args.skip_train and yolo_weights and Path(yolo_weights).exists():
                try:
                    _det = WeedDetector(
                        model_size=args.model_size,
                        num_classes=len(CLASS_NAMES),
                        weights_path=yolo_weights,
                        device=args.device,
                    )
                    _yolo_metrics = _det.validate(yaml_path)
                    print(f"  YOLO mAP@0.5: {_yolo_metrics['mAP50']:.4f}  "
                          f"mAP@0.5-0.95: {_yolo_metrics['mAP50-95']:.4f}")
                except Exception as e:
                    print(f"  YOLO validation skipped: {e}")

            assessor.save_report(metrics, _yolo_metrics, split_name=assess_split)
    else:
        print("  No classifier weights available — skipping assessment.")

    step("COMPLETE")
    print(f"  All outputs → {out.resolve()}")
    if not args.skip_train:
        yolo_run_dir = out / "yolo" / "weed_detection"
        print(f"\n  YOLOv8 training plots:")
        print(f"    {yolo_run_dir}/results.png")
        print(f"    {yolo_run_dir}/confusion_matrix.png")
        print(f"    {yolo_run_dir}/PR_curve.png")
        print(f"    {yolo_run_dir}/weights/best.pt")
        print(f"\n  ResNeXt-50 checkpoint:")
        print(f"    {out / 'classifier' / 'best_classifier.pt'}")
    print(f"\n  Inference results:")
    print(f"    {inf_out}/density_heatmap.png")
    print(f"    {inf_out}/category_map.png")
    print(f"    {inf_out}/per_species_heatmaps/")
    print(f"    {inf_out}/summary_stats.txt")
    print(f"    {inf_out}/*_detected.png")
    print(f"\n  Model assessment (DeepWeeds-style metrics):")
    print(f"    {out / 'assessment'}/assessment_report.txt")
    print(f"    {out / 'assessment'}/confusion_matrix.png")
    print(f"    {out / 'assessment'}/per_class_metrics.png")
    print(f"    {out / 'assessment'}/metrics.json")


if __name__ == "__main__":
    main()
