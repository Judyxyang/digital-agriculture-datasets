"""
Unified preprocessing for the real-data pipeline.

Supports both:
  • 3-band RGB  — standard ImageNet-normalised pipeline
  • 5-band multispectral (.npy) — percentile normalisation + vegetation-index fusion

RGB pipeline
────────────
  Train: RandomResizedCrop → HFlip → VFlip → ColorJitter → GaussNoise → Normalize
  Val  : Resize → Normalize

Multispectral pipeline (5-band)
────────────────────────────────
  Band order in .npy files: [R, G, B, NIR, RedEdge]
  Fusion modes (set in config):
    "rgb"       → (H,W,3) R, G, B
    "rgb_ndvi"  → (H,W,3) R, G, NDVI          ← recommended default
    "all5"      → (H,W,5) all bands            (needs in_channels=5 in model)
    "vi_stack"  → (H,W,3) NDVI, GNDVI, RE-NDVI

To switch to real 5-band imagery:
  Replace LocalWeedDataset.generate_multispectral() with a loader that reads
  GeoTIFF / raw sensor .npy files directly. The rest of this pipeline is
  already band-count agnostic.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Re-export multispectral utilities from the shared module so callers only
# need to import from this file.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "weed_detection_demo"))
from preprocessing.multispectral_preprocessing import (   # noqa: E402
    fuse_channels, normalise_ms_image, compute_ndvi, compute_gndvi,
    compute_re_ndvi, standardise, FUSION_MODES, FUSION_STATS,
)

__all__ = [
    "IMAGENET_MEAN", "IMAGENET_STD",
    "get_rgb_transforms",
    "RGBClassificationDataset",
    "MultispectralClassificationDataset",
    "DetectionPreprocessor",
    "fuse_channels", "compute_ndvi", "FUSION_MODES",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ── RGB augmentation ──────────────────────────────────────────────────────────

def get_rgb_transforms(split: str, img_size: int = 224) -> A.Compose:
    """Albumentations pipeline for RGB classification (train or val)."""
    if split == "train":
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.65, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.3, hue=0.1, p=0.6),
            A.GaussNoise(p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


# ── RGB dataset ───────────────────────────────────────────────────────────────

class RGBClassificationDataset(Dataset):
    """
    Folder-per-class crop dataset for ResNeXt-50 classifier training.

    root/
      <class_name_0>/  *.png / *.jpg
      <class_name_1>/  *.png / *.jpg
      ...
    """

    def __init__(self, root: str, class_names: List[str],
                 split: str = "train", img_size: int = 224):
        self.class_names  = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform    = get_rgb_transforms(split, img_size)
        self.samples: List[Tuple[str, int]] = []

        root_p = Path(root)
        for cls_name in class_names:
            cls_dir = root_p / cls_name
            if not cls_dir.exists():
                continue
            for img_file in list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg")):
                self.samples.append((str(img_file), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(image=img_rgb)["image"], label


# ── Multispectral classification dataset ──────────────────────────────────────

class MultispectralClassificationDataset(Dataset):
    """
    Paired (MS .npy crop, label) dataset for 5-band classifier training.

    Expects the same folder-per-class layout as RGBClassificationDataset,
    but each file is a .npy array of shape (H, W, 5).
    """

    def __init__(self, root: str, class_names: List[str],
                 split: str = "train", img_size: int = 224,
                 fusion_mode: str = "rgb_ndvi"):
        self.class_names  = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.fusion_mode  = fusion_mode
        self.img_size     = img_size
        self.is_train     = split == "train"
        self.samples: List[Tuple[str, int]] = []

        root_p = Path(root)
        for cls_name in class_names:
            cls_dir = root_p / cls_name
            if not cls_dir.exists():
                continue
            for npy_file in cls_dir.glob("*.npy"):
                self.samples.append((str(npy_file), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        npy_path, label = self.samples[idx]
        ms    = np.load(npy_path)                          # (H, W, 5) float32
        fused = fuse_channels(ms, self.fusion_mode)        # (H, W, C)
        fused = cv2.resize(fused, (self.img_size, self.img_size))

        if self.is_train:
            if np.random.rand() < 0.5:
                fused = np.fliplr(fused)
            if np.random.rand() < 0.3:
                fused = np.flipud(fused)
            k = np.random.choice([0, 1, 2, 3])
            fused = np.rot90(fused, k).copy()

        fused = standardise(fused, self.fusion_mode)
        tensor = torch.from_numpy(fused.astype(np.float32)).permute(2, 0, 1)
        return tensor, label


# ── Detection preprocessing ───────────────────────────────────────────────────

class DetectionPreprocessor:
    """
    Letterbox-resize a single RGB image for YOLOv8 inference.
    Returns (tensor CHW float32, orig_hw, scale, pad).
    """

    def __init__(self, target_size: int = 640):
        self.target_size = target_size

    def letterbox(self, img: np.ndarray):
        h, w  = img.shape[:2]
        scale = min(self.target_size / h, self.target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h, pad_w = self.target_size - new_h, self.target_size - new_w
        top,  left   = pad_h // 2, pad_w // 2
        padded = cv2.copyMakeBorder(
            resized, top, pad_h - top, left, pad_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        return padded, scale, (top, left)

    def __call__(self, img_path: str):
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_hw = img_rgb.shape[:2]
        lb, scale, pad = self.letterbox(img_rgb)
        tensor = torch.from_numpy(lb.astype(np.float32) / 255.0).permute(2, 0, 1)
        return tensor, orig_hw, scale, pad
