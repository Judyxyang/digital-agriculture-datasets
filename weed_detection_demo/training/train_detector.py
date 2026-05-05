"""
YOLOv8 weed detector training script.

Usage
─────
  python -m training.train_detector \
    --dataset_root outputs/sample_dataset \
    --epochs 50 \
    --model_size n \
    --batch 16 \
    --device cpu

Training stages
───────────────
  Stage 1 (epochs 1-N): Fine-tune all YOLOv8 layers on the weed dataset.
  YOLOv8 internally handles its own two-phase training (warm-up + cosine LR).
  The ultralytics trainer also handles:
    - Mosaic augmentation (disabled in last 10 epochs)
    - Mixed precision (AMP) on CUDA
    - EMA model weights
    - Auto anchor generation
    - Multi-scale training
"""

import argparse
import sys
from pathlib import Path

# Allow running as a module from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.sample_generator import generate_dataset, WEED_CLASSES
from models.yolov8_detector import WeedDetector, build_dataset_yaml


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 weed detector")
    p.add_argument("--dataset_root", default="outputs/sample_dataset",
                   help="Root directory of the dataset (will be generated if missing)")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch",       type=int,   default=16)
    p.add_argument("--imgsz",       type=int,   default=640)
    p.add_argument("--model_size",  default="n",
                   choices=["n", "s", "m", "l", "x"],
                   help="YOLOv8 variant (n=nano … x=xlarge)")
    p.add_argument("--lr",          type=float, default=0.01)
    p.add_argument("--device",      default="cpu",
                   help="cuda / cpu / mps")
    p.add_argument("--output_dir",  default="outputs/yolo_runs")
    p.add_argument("--weights",     default=None,
                   help="Resume from checkpoint .pt")
    p.add_argument("--generate_data", action="store_true",
                   help="Force re-generate synthetic dataset")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Data ──────────────────────────────────────────────────────────────
    dataset_root = Path(args.dataset_root)
    yaml_path    = dataset_root / "detection" / "dataset.yaml"

    if args.generate_data or not yaml_path.exists():
        print("[Train] Generating synthetic dataset…")
        generate_dataset(str(dataset_root))

    print(f"[Train] Using dataset yaml: {yaml_path}")
    print(f"[Train] Classes: {WEED_CLASSES}")

    # ── 2. Model ─────────────────────────────────────────────────────────────
    detector = WeedDetector(
        model_size=args.model_size,
        num_classes=len(WEED_CLASSES),
        weights_path=args.weights,
        device=args.device,
    )
    print(f"[Train] YOLOv8-{args.model_size} loaded on {args.device}")

    # ── 3. Training ───────────────────────────────────────────────────────────
    print(f"[Train] Starting training for {args.epochs} epochs…")
    best_pt = detector.train(
        dataset_yaml=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr,
        output_dir=args.output_dir,
        # Additional ultralytics hyper-parameters
        patience=20,          # early stopping patience
        save=True,
        plots=True,
        rect=False,           # rectangular training (False for square crops)
        cos_lr=True,
        label_smoothing=0.0,
        hsv_h=0.015,          # colour augmentation
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.3,
        mosaic=1.0,
        mixup=0.1,
    )

    print(f"\n[Train] Best checkpoint saved: {best_pt}")

    # ── 4. Validation ─────────────────────────────────────────────────────────
    print("[Train] Running validation on best checkpoint…")
    detector.load_weights(best_pt)
    metrics = detector.validate(str(yaml_path), imgsz=args.imgsz)
    print("[Train] Validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n[Train] Done. To run inference:")
    print(f"  python -m inference.inference_pipeline "
          f"--yolo_weights {best_pt} --input_dir outputs/sample_dataset/test/rgb")


if __name__ == "__main__":
    main()
