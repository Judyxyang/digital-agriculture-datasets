"""
Multispectral image preprocessing pipeline.

Handles 5-band images: [R, G, B, NIR, RedEdge] stored as float32 .npy arrays.

Key operations
──────────────
  1. Per-band normalisation (percentile clip then min-max)
  2. Vegetation index computation (NDVI, GNDVI, RedEdge NDVI)
  3. Optional channel fusion: RGB + selected indices → 3-channel tensor
     for feeding into RGB-pretrained backbones
  4. Augmentation-aware dataset class for training

Fusion modes
────────────
  "rgb"       – discard NIR/RE; use R, G, B only
  "rgb_ndvi"  – R, G, NDVI as 3 channels (NDVI replaces blue)
  "all5"      – all 5 bands (requires custom first conv layer)
  "vi_stack"  – NDVI, GNDVI, RedEdge-NDVI (all vegetation indices)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional
import albumentations as A

FUSION_MODES = ("rgb", "rgb_ndvi", "all5", "vi_stack")

# ── Band indices inside the 5-channel array ──────────────────────────────────
R_IDX, G_IDX, B_IDX, NIR_IDX, RE_IDX = 0, 1, 2, 3, 4


# ── Normalisation ────────────────────────────────────────────────────────────

def percentile_normalise(band: np.ndarray,
                         lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Clip to [lo, hi] percentile then rescale to [0, 1]."""
    lo_val = np.percentile(band, lo)
    hi_val = np.percentile(band, hi)
    clipped = np.clip(band, lo_val, hi_val)
    denom = hi_val - lo_val
    if denom < 1e-8:
        return np.zeros_like(clipped)
    return (clipped - lo_val) / denom


def normalise_ms_image(ms: np.ndarray) -> np.ndarray:
    """Normalise every band independently. Returns float32 (H,W,5) in [0,1]."""
    out = np.empty_like(ms, dtype=np.float32)
    for i in range(ms.shape[-1]):
        out[..., i] = percentile_normalise(ms[..., i])
    return out


# ── Vegetation indices ───────────────────────────────────────────────────────

def compute_ndvi(ms: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - R) / (NIR + R + ε)"""
    nir = ms[..., NIR_IDX].astype(np.float32)
    r   = ms[..., R_IDX].astype(np.float32)
    return (nir - r) / (nir + r + 1e-8)


def compute_gndvi(ms: np.ndarray) -> np.ndarray:
    """GNDVI = (NIR - G) / (NIR + G + ε)"""
    nir = ms[..., NIR_IDX].astype(np.float32)
    g   = ms[..., G_IDX].astype(np.float32)
    return (nir - g) / (nir + g + 1e-8)


def compute_re_ndvi(ms: np.ndarray) -> np.ndarray:
    """RedEdge NDVI = (NIR - RE) / (NIR + RE + ε)"""
    nir = ms[..., NIR_IDX].astype(np.float32)
    re  = ms[..., RE_IDX].astype(np.float32)
    return (nir - re) / (nir + re + 1e-8)


# ── Channel fusion ───────────────────────────────────────────────────────────

def fuse_channels(ms: np.ndarray, mode: str = "rgb_ndvi") -> np.ndarray:
    """
    Convert normalised (H,W,5) float32 array into a (H,W,C) tensor-ready array.
    All output channels are in [-1, 1] after normalisation where appropriate.

    mode = "rgb"       → (H,W,3)  R, G, B
    mode = "rgb_ndvi"  → (H,W,3)  R, G, NDVI
    mode = "all5"      → (H,W,5)  R, G, B, NIR, RE
    mode = "vi_stack"  → (H,W,3)  NDVI, GNDVI, RE-NDVI
    """
    assert mode in FUSION_MODES, f"mode must be one of {FUSION_MODES}"
    ms_norm = normalise_ms_image(ms)

    if mode == "rgb":
        return ms_norm[..., :3]

    elif mode == "rgb_ndvi":
        r    = ms_norm[..., R_IDX]
        g    = ms_norm[..., G_IDX]
        ndvi = np.clip((compute_ndvi(ms) + 1) / 2, 0, 1)   # [-1,1] → [0,1]
        return np.stack([r, g, ndvi], axis=-1)

    elif mode == "all5":
        return ms_norm

    elif mode == "vi_stack":
        ndvi   = np.clip((compute_ndvi(ms)    + 1) / 2, 0, 1)
        gndvi  = np.clip((compute_gndvi(ms)   + 1) / 2, 0, 1)
        re_ndvi= np.clip((compute_re_ndvi(ms) + 1) / 2, 0, 1)
        return np.stack([ndvi, gndvi, re_ndvi], axis=-1)


# ── Mean / std for each fusion mode (approximate) ───────────────────────────
# Used to standardise tensors to zero mean, unit variance.

FUSION_STATS = {
    "rgb":      {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "rgb_ndvi": {"mean": (0.485, 0.456, 0.5),   "std": (0.229, 0.224, 0.2)},
    "all5":     {"mean": (0.485, 0.456, 0.406, 0.4, 0.45),
                 "std":  (0.229, 0.224, 0.225, 0.22, 0.22)},
    "vi_stack": {"mean": (0.5, 0.5, 0.5),       "std": (0.2, 0.2, 0.2)},
}


def standardise(arr: np.ndarray, mode: str) -> np.ndarray:
    stats = FUSION_STATS[mode]
    mean  = np.array(stats["mean"], dtype=np.float32)
    std   = np.array(stats["std"],  dtype=np.float32)
    return (arr - mean) / (std + 1e-8)


# ── Augmentation ─────────────────────────────────────────────────────────────

def _ms_augment(arr: np.ndarray, is_train: bool) -> np.ndarray:
    """Spatial augmentations that apply identically to all channels."""
    if not is_train:
        return arr
    if np.random.rand() < 0.5:
        arr = np.fliplr(arr)
    if np.random.rand() < 0.3:
        arr = np.flipud(arr)
    k = np.random.choice([0, 1, 2, 3])
    arr = np.rot90(arr, k)
    return arr.copy()


# ── Dataset ──────────────────────────────────────────────────────────────────

class MultispectralDataset(Dataset):
    """
    Loads paired (RGB detection image, multispectral array) samples.

    For each detection split image, the corresponding *_ms.npy file is loaded,
    fused, and returned alongside the YOLO label boxes.

    Parameters
    ----------
    rgb_images_dir   : path to detection/images/<split>/
    ms_dir           : path to multispectral/<split>/
    labels_dir       : path to detection/labels/<split>/
    class_names      : ordered list of class names
    fusion_mode      : one of FUSION_MODES
    split            : "train" or "val"
    img_size         : spatial resize target
    """

    def __init__(self, rgb_images_dir: str, ms_dir: str, labels_dir: str,
                 class_names: List[str], fusion_mode: str = "rgb_ndvi",
                 split: str = "train", img_size: int = 640):
        self.fusion_mode  = fusion_mode
        self.class_names  = class_names
        self.is_train     = split == "train"
        self.img_size     = img_size

        rgb_dir_p = Path(rgb_images_dir)
        ms_dir_p  = Path(ms_dir)
        lbl_dir_p = Path(labels_dir)

        self.triplets: List[Tuple[str, str, str]] = []
        for img_path in sorted(rgb_dir_p.glob("*.png")):
            stem   = img_path.stem
            ms_p   = ms_dir_p  / f"{stem}_ms.npy"
            lbl_p  = lbl_dir_p / f"{stem}.txt"
            if ms_p.exists() and lbl_p.exists():
                self.triplets.append((str(img_path), str(ms_p), str(lbl_p)))

    def __len__(self):
        return len(self.triplets)

    def _load_ms(self, ms_path: str) -> np.ndarray:
        ms = np.load(ms_path)                     # (H, W, 5)
        fused = fuse_channels(ms, self.fusion_mode)
        fused = _ms_augment(fused, self.is_train)
        import cv2
        fused = cv2.resize(fused, (self.img_size, self.img_size))
        fused = standardise(fused, self.fusion_mode)
        return fused.astype(np.float32)

    def __getitem__(self, idx: int):
        _, ms_path, lbl_path = self.triplets[idx]
        arr = self._load_ms(ms_path)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)   # CHW

        boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    boxes.append([float(p) for p in parts])
        return tensor, torch.tensor(boxes, dtype=torch.float32)


# ── Single-image preprocessing helper (for inference) ────────────────────────

class MultispectralPreprocessor:
    """Preprocess a single MS .npy file for inference."""

    def __init__(self, fusion_mode: str = "rgb_ndvi", img_size: int = 640):
        self.fusion_mode = fusion_mode
        self.img_size    = img_size

    def __call__(self, ms_path: str) -> torch.Tensor:
        import cv2
        ms    = np.load(ms_path).astype(np.float32)
        fused = fuse_channels(ms, self.fusion_mode)
        fused = cv2.resize(fused, (self.img_size, self.img_size))
        fused = standardise(fused, self.fusion_mode)
        tensor = torch.from_numpy(fused).permute(2, 0, 1)
        return tensor


if __name__ == "__main__":
    from pathlib import Path
    root = Path(__file__).parent.parent / "outputs" / "sample_dataset"
    ms_file = list((root / "multispectral" / "train").glob("*.npy"))[0]

    ms = np.load(str(ms_file))
    print(f"[MS] Raw array shape: {ms.shape}, dtype: {ms.dtype}")
    for mode in FUSION_MODES:
        fused = fuse_channels(ms, mode)
        print(f"  fusion='{mode}' → shape {fused.shape}, "
              f"range [{fused.min():.2f}, {fused.max():.2f}]")

    ndvi = compute_ndvi(ms)
    print(f"[MS] NDVI range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")
