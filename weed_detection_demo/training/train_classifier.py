"""
ResNeXt-50 weed classifier training script.

Usage
─────
  python -m training.train_classifier \
    --dataset_root outputs/sample_dataset \
    --epochs 30 \
    --batch 32 \
    --device cpu

Two-phase training strategy
────────────────────────────
  Phase 1 (freeze backbone): Train only the classification head for
    `warmup_epochs` epochs at a high LR. This avoids destroying pre-trained
    features before the head learns meaningful representations.

  Phase 2 (unfreeze last 2 layer groups): Fine-tune the full network at a
    lower LR with cosine annealing. Produces the best accuracy.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.sample_generator import generate_dataset, WEED_CLASSES
from preprocessing.rgb_preprocessing import WeedClassificationDataset
from models.resnext50_classifier import (
    WeedClassifier,
    LabelSmoothingCrossEntropy,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
)


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train ResNeXt-50 weed classifier")
    p.add_argument("--dataset_root",  default="outputs/sample_dataset")
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch",         type=int,   default=32)
    p.add_argument("--img_size",      type=int,   default=224)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int,   default=5,
                   help="Epochs with frozen backbone (head-only training)")
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--label_smooth",  type=float, default=0.1)
    p.add_argument("--workers",       type=int,   default=2)
    p.add_argument("--device",        default="cpu")
    p.add_argument("--output_dir",    default="outputs/classifier_runs")
    p.add_argument("--generate_data", action="store_true")
    p.add_argument("--no_pretrain",   action="store_true",
                   help="Train from scratch (no ImageNet weights)")
    return p.parse_args()


# ── Training / validation loops ──────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total   = 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds  = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"  [E{epoch}] step {batch_idx+1}/{len(loader)} "
                  f"loss={loss.item():.4f} "
                  f"acc={correct/total:.3f}")

    return total_loss / total, correct / total


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total   = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

    return total_loss / total, correct / total


# ── Per-class accuracy ────────────────────────────────────────────────────────

@torch.no_grad()
def per_class_accuracy(model, loader, class_names, device):
    model.eval()
    n = len(class_names)
    correct_per = torch.zeros(n)
    total_per   = torch.zeros(n)

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels
        preds = model(imgs.to(device)).argmax(dim=1).cpu()
        for cls_id in range(n):
            mask = labels == cls_id
            correct_per[cls_id] += (preds[mask] == cls_id).sum()
            total_per[cls_id]   += mask.sum()

    print("\n[Classifier] Per-class accuracy:")
    for i, name in enumerate(class_names):
        denom = total_per[i].item()
        acc   = (correct_per[i] / denom).item() if denom > 0 else 0.0
        print(f"  {name:<25} {acc*100:5.1f}%  (n={int(denom)})")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    dataset_root = Path(args.dataset_root)
    cls_root     = dataset_root / "classification"

    if args.generate_data or not cls_root.exists():
        print("[Train] Generating synthetic dataset…")
        generate_dataset(str(dataset_root))

    train_ds = WeedClassificationDataset(
        str(cls_root / "train"), WEED_CLASSES, split="train", img_size=args.img_size
    )
    val_ds = WeedClassificationDataset(
        str(cls_root / "val"), WEED_CLASSES, split="val", img_size=args.img_size
    )
    print(f"[Train] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=device.type == "cuda")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    model = WeedClassifier(
        num_classes=len(WEED_CLASSES),
        pretrained=not args.no_pretrain,
        in_channels=3,
        dropout_rate=args.dropout,
        freeze_backbone=True,           # Phase 1: train head only
    ).to(device)

    total_params  = sum(p.numel() for p in model.parameters())
    trainable     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model: ResNeXt-50-32x4d  |  Total params: {total_params:,}  "
          f"|  Trainable (phase 1): {trainable:,}")

    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smooth)
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)

    best_val_acc = 0.0
    best_ckpt    = str(out_dir / "best_classifier.pt")

    # ── 3. Training loop ──────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Phase 2: unfreeze backbone after warm-up
        if epoch == args.warmup_epochs + 1:
            model.unfreeze_backbone(layers_from_end=2)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n[Train] Phase 2: backbone unfrozen. Trainable params: {trainable:,}")
            # Rebuild optimizer & scheduler for phase 2
            optimizer = build_optimizer(model, lr=args.lr * 0.1,
                                        weight_decay=args.weight_decay)
            scheduler = build_scheduler(optimizer,
                                        args.epochs - args.warmup_epochs,
                                        warmup_epochs=0)

        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device, epoch)
        val_loss, val_acc     = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        dt = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.3f} | "
              f"{dt:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, best_ckpt)
            print(f"  ✓ New best val acc: {val_acc:.4f} → saved to {best_ckpt}")

    # ── 4. Per-class accuracy on validation set ───────────────────────────────
    print(f"\n[Train] Best val accuracy: {best_val_acc:.4f}")
    model_best = WeedClassifier(num_classes=len(WEED_CLASSES), pretrained=False)
    ckpt = torch.load(best_ckpt, map_location=device)
    model_best.load_state_dict(ckpt["state_dict"])
    model_best.to(device)
    per_class_accuracy(model_best, val_loader, WEED_CLASSES, device)

    print(f"\n[Train] Done. Best checkpoint: {best_ckpt}")
    print("[Train] To run inference:")
    print(f"  python -m inference.inference_pipeline "
          f"--classifier_weights {best_ckpt}")


if __name__ == "__main__":
    main()
