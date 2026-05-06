"""
RGB image preprocessing pipeline for weed detection and classification.

Two modes:
  - DetectionPreprocessor  : returns (tensor, original_size) ready for YOLOv8
  - ClassificationPreprocessor : returns tensor ready for ResNeXt-50
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Normalisation stats (ImageNet defaults work well as starting point) ──────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ── Detection preprocessing ──────────────────────────────────────────────────

class DetectionPreprocessor:
    """
    Prepares a single RGB image for YOLOv8 inference.

    Steps:
      1. Read image (BGR → RGB)
      2. Letterbox resize to target_size × target_size
      3. Normalize to [0, 1]
      4. HWC → CHW, float32 tensor
    """

    def __init__(self, target_size: int = 640):
        self.target_size = target_size

    def letterbox(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w = img.shape[:2]
        scale = min(self.target_size / h, self.target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        top, left = pad_h // 2, pad_w // 2

        img_padded = cv2.copyMakeBorder(
            img_resized, top, pad_h - top, left, pad_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return img_padded, scale, (top, left)

    def __call__(self, img_path: str) -> Tuple[torch.Tensor, Tuple[int, int], float, Tuple[int, int]]:
        img_bgr  = cv2.imread(img_path)
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_hw  = img_rgb.shape[:2]

        img_lb, scale, pad = self.letterbox(img_rgb)
        img_f    = img_lb.astype(np.float32) / 255.0
        tensor   = torch.from_numpy(img_f).permute(2, 0, 1)   # CHW
        return tensor, orig_hw, scale, pad


# ── Classification preprocessing ────────────────────────────────────────────

def get_classification_transforms(split: str, img_size: int = 224):
    """
    Returns albumentations pipeline for train / val.
    Output: normalised CHW float32 tensor.
    """
    if split == "train":
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
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


class WeedClassificationDataset(Dataset):
    """
    Reads weed crop images from a folder-per-class directory tree.

    Expected layout:
      root/
        <class_name_0>/  *.png
        <class_name_1>/  *.png
        ...
    """

    def __init__(self, root: str, class_names: List[str], split: str = "train",
                 img_size: int = 224):
        self.class_names  = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform    = get_classification_transforms(split, img_size)
        self.samples: List[Tuple[str, int]] = []

        root_p = Path(root)
        for cls_name in class_names:
            cls_dir = root_p / cls_name
            if not cls_dir.exists():
                continue
            for img_file in cls_dir.glob("*.png"):
                self.samples.append((str(img_file), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=img_rgb)
        return transformed["image"], label


# ── Detection dataset (for YOLOv8 custom training via ultralytics API) ───────

class WeedDetectionDataset(Dataset):
    """
    Used for inspection / visualisation; actual YOLOv8 training
    uses dataset.yaml + ultralytics Trainer directly.
    """

    def __init__(self, images_dir: str, labels_dir: str,
                 class_names: List[str], img_size: int = 640):
        self.img_size     = img_size
        self.class_names  = class_names
        self.preprocessor = DetectionPreprocessor(img_size)

        images_dir_p = Path(images_dir)
        labels_dir_p = Path(labels_dir)
        self.pairs: List[Tuple[str, str]] = []

        for img_path in sorted(images_dir_p.glob("*.png")):
            lbl_path = labels_dir_p / (img_path.stem + ".txt")
            if lbl_path.exists():
                self.pairs.append((str(img_path), str(lbl_path)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.pairs[idx]
        tensor, orig_hw, scale, pad = self.preprocessor(img_path)

        boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    boxes.append([float(p) for p in parts])

        return tensor, torch.tensor(boxes, dtype=torch.float32), orig_hw


if __name__ == "__main__":
    import sys
    root = Path(__file__).parent.parent / "outputs" / "sample_dataset"
    if not root.exists():
        print("Run data/sample_generator.py first.")
        sys.exit(1)

    from data.sample_generator import WEED_CLASSES

    ds = WeedClassificationDataset(
        str(root / "classification" / "train"), WEED_CLASSES, split="train"
    )
    print(f"[RGB] Classification dataset: {len(ds)} samples")
    img_t, lbl = ds[0]
    print(f"  tensor shape: {img_t.shape}, label: {WEED_CLASSES[lbl]}")

    pre = DetectionPreprocessor()
    sample_img = list((root / "detection" / "images" / "train").glob("*.png"))[0]
    t, hw, sc, pd = pre(str(sample_img))
    print(f"[RGB] Detection tensor: {t.shape}, orig: {hw}, scale: {sc:.3f}, pad: {pd}")
