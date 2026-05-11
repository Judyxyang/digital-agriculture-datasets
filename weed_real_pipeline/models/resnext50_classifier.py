"""
ResNeXt-50 weed classification model.

Architecture note
─────────────────
The user requested "ResNeXt52" — no such variant exists.
ResNeXt-50-32x4d (50 layers, 32 groups of width 4) is the standard
published model from "Aggregated Residual Transformations" (Xie et al., 2017)
and is the correct closest match.

Variants available in torchvision:
  resnext50_32x4d   ← used here (default)
  resnext101_32x8d
  resnext101_64x4d

The classifier is used as the second stage of the pipeline:
  Image → YOLOv8 (detect bounding boxes) → crop each box →
  ResNeXt-50 (classify each crop) → fine-grained weed species label
"""

from typing import List, Optional, Dict, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ── Model definition ──────────────────────────────────────────────────────────

class WeedClassifier(nn.Module):
    """
    ResNeXt-50-32x4d fine-tuned for weed species classification.

    Parameters
    ----------
    num_classes    : number of weed species (including background)
    pretrained     : load ImageNet weights for the backbone
    in_channels    : 3 for RGB/rgb_ndvi/vi_stack, 5 for all5 multispectral
    dropout_rate   : dropout before the classification head
    freeze_backbone: freeze all backbone layers (train head only); useful for
                     limited data / fast demo
    """

    def __init__(
        self,
        num_classes: int = 9,
        pretrained: bool = True,
        in_channels: int = 3,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        weights = tvm.ResNeXt50_32X4D_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tvm.resnext50_32x4d(weights=weights)

        # Adapt first conv for non-RGB input
        if in_channels != 3:
            orig_conv = backbone.conv1
            new_conv  = nn.Conv2d(
                in_channels, orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False,
            )
            # Initialise new channels by averaging the pre-trained RGB weights
            with torch.no_grad():
                new_conv.weight[:, :3] = orig_conv.weight
                if in_channels > 3:
                    extra = orig_conv.weight.mean(dim=1, keepdim=True)
                    for c in range(3, in_channels):
                        new_conv.weight[:, c] = extra[:, 0]
            backbone.conv1 = new_conv

        # Strip the final FC and replace with our head
        in_features = backbone.fc.in_features
        backbone.fc  = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes),
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, layers_from_end: int = 2):
        """Unfreeze the last N ResNet layer groups (for fine-tuning phase 2)."""
        trainable_groups = list(self.backbone.children())[-layers_from_end:]
        for layer in trainable_groups:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)    # (B, 2048)
        logits   = self.head(features) # (B, num_classes)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        return F.softmax(self.forward(x), dim=-1)


# ── Loss ─────────────────────────────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing to reduce overconfident predictions."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_cls  = logits.size(-1)
        log_p  = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(log_p, self.smoothing / (n_cls - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = (-smooth_targets * log_p).sum(dim=-1).mean()
        return loss


# ── Training utilities ────────────────────────────────────────────────────────

def build_optimizer(model: WeedClassifier, lr: float = 1e-4, weight_decay: float = 1e-4):
    """
    Different learning rates for backbone vs head (standard transfer learning).
    """
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = list(model.head.parameters())
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params,     "lr": lr},
    ], weight_decay=weight_decay)


def build_scheduler(optimizer, num_epochs: int, warmup_epochs: int = 5):
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model: WeedClassifier, optimizer, epoch: int,
                    val_acc: float, path: str):
    torch.save({
        "epoch":      epoch,
        "state_dict": model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "val_acc":    val_acc,
        "num_classes":model.num_classes,
        "in_channels":model.in_channels,
    }, path)


def load_checkpoint(path: str, class_names: List[str],
                    in_channels: int = 3, device: str = "cpu") -> WeedClassifier:
    # Resolve multi-GPU device string to a single device for loading
    if isinstance(device, str) and "," in device:
        map_dev = "cuda:0"
    else:
        map_dev = device
    ckpt  = torch.load(path, map_location=map_dev, weights_only=False)
    model = WeedClassifier(
        num_classes=len(class_names),
        pretrained=False,
        in_channels=ckpt.get("in_channels", in_channels),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(map_dev)
    model.eval()
    return model


# ── Per-crop inference helper ─────────────────────────────────────────────────

class CropClassifier:
    """
    Classifies a single image crop (numpy BGR) using a trained WeedClassifier.
    Used inside the inference pipeline after YOLOv8 detection.
    """

    def __init__(self, model: WeedClassifier, class_names: List[str],
                 img_size: int = 224, device: str = "cpu"):
        self.model       = model.eval()
        self.class_names = class_names
        self.img_size    = img_size
        # Resolve multi-GPU string to single device for inference tensors
        self.device      = "cuda:0" if (isinstance(device, str) and "," in device) else device

        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    @torch.no_grad()
    def classify(self, crop_bgr: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Returns (class_name, confidence, probability_array).
        """
        import cv2
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        t = self.transform(image=crop_rgb)["image"].unsqueeze(0).to(self.device)
        probs = self.model.predict_proba(t).squeeze(0).cpu().numpy()
        cls_id = int(probs.argmax())
        return self.class_names[cls_id], float(probs[cls_id]), probs

    @torch.no_grad()
    def classify_batch(self, crops_bgr: List[np.ndarray]) -> List[Tuple[str, float, np.ndarray]]:
        import cv2
        tensors = []
        for crop in crops_bgr:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            t = self.transform(image=crop_rgb)["image"]
            tensors.append(t)
        batch = torch.stack(tensors).to(self.device)
        probs = self.model.predict_proba(batch).cpu().numpy()
        results = []
        for p in probs:
            cls_id = int(p.argmax())
            results.append((self.class_names[cls_id], float(p[cls_id]), p))
        return results


if __name__ == "__main__":
    from data.sample_generator import WEED_CLASSES

    model = WeedClassifier(num_classes=len(WEED_CLASSES), pretrained=False)
    total = sum(p.numel() for p in model.parameters())
    print(f"[ResNeXt-50] Parameters: {total:,}")

    dummy = torch.randn(2, 3, 224, 224)
    out   = model(dummy)
    print(f"[ResNeXt-50] Output shape: {out.shape}")  # (2, 9)
