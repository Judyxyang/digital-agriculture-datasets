"""
Synthetic weed dataset generator.

Produces:
  - RGB images (3-channel, uint8 PNG)
  - Multispectral images (5-channel: R,G,B,NIR,RedEdge, float32 NPY)
  - YOLO-format bounding-box annotation .txt files (for detection training)
  - Class-labelled crop directories (for ResNeXt classification training)

Weed classes mirror the DeepWeeds taxonomy (9 species + background).
"""

import os
import random
import shutil
import numpy as np
import cv2
from pathlib import Path

# ── Dataset configuration ────────────────────────────────────────────────────

WEED_CLASSES = [
    "Chinee_Apple",
    "Lantana",
    "Parkinsonia",
    "Parthenium",
    "Prickly_Acacia",
    "Rubber_Vine",
    "Siam_Weed",
    "Snake_Weed",
    "Negative",           # background / no weed
]

# BGR palette per class (used to paint synthetic weed blobs)
CLASS_COLORS_BGR = {
    "Chinee_Apple":    (34,  139, 34),
    "Lantana":         (147, 20,  255),
    "Parkinsonia":     (0,   205, 102),
    "Parthenium":      (255, 255, 0),
    "Prickly_Acacia":  (0,   128, 128),
    "Rubber_Vine":     (205, 92,  92),
    "Siam_Weed":       (255, 140, 0),
    "Snake_Weed":      (64,  224, 208),
    "Negative":        (34,  139, 34),
}

IMG_SIZE   = 640          # pixels (square)
N_TRAIN    = 120          # images for training
N_VAL      = 30           # images for validation
N_TEST     = 20           # images for inference demo
MAX_WEEDS  = 6            # max weed instances per image
CROP_SIZE  = 96           # px – classifier crop size


# ── Helpers ──────────────────────────────────────────────────────────────────

def _soil_background(h: int, w: int) -> np.ndarray:
    """Return a noisy brown soil-like BGR image."""
    base = np.full((h, w, 3), (45, 80, 120), dtype=np.uint8)
    noise = np.random.randint(-30, 30, (h, w, 3), dtype=np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _paint_weed(img: np.ndarray, cls_name: str):
    """
    Paint an elliptical weed blob on *img* in-place.
    Returns the bounding box (x1, y1, x2, y2) or None if placement failed.
    """
    h, w = img.shape[:2]
    color = CLASS_COLORS_BGR[cls_name]

    rx = random.randint(20, 80)
    ry = random.randint(20, 80)
    cx = random.randint(rx, w - rx)
    cy = random.randint(ry, h - ry)

    # Main blob
    cv2.ellipse(img, (cx, cy), (rx, ry), random.randint(0, 180), 0, 360, color, -1)

    # Texture – random small circles to simulate leaves
    n_leaves = random.randint(5, 15)
    for _ in range(n_leaves):
        lx = cx + random.randint(-rx, rx)
        ly = cy + random.randint(-ry, ry)
        lr = random.randint(4, 14)
        shade = tuple(max(0, min(255, c + random.randint(-40, 40))) for c in color)
        cv2.circle(img, (lx, ly), lr, shade, -1)

    x1, y1 = max(0, cx - rx), max(0, cy - ry)
    x2, y2 = min(w, cx + rx), min(h, cy + ry)
    return x1, y1, x2, y2


def _make_multispectral(rgb: np.ndarray) -> np.ndarray:
    """
    Simulate a 5-band multispectral image (R, G, B, NIR, RedEdge).
    NIR is approximated from the green channel + noise (vegetation reflects NIR).
    RedEdge is approximated as (NIR + Red) / 2 with noise.
    Returns float32 array normalised to [0, 1], shape (H, W, 5).
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 2], rgb_f[..., 1], rgb_f[..., 0]

    nir = np.clip(g * 1.4 + np.random.normal(0, 0.05, g.shape), 0, 1).astype(np.float32)
    re  = np.clip((nir + r) / 2 + np.random.normal(0, 0.03, r.shape), 0, 1).astype(np.float32)

    ms = np.stack([r, g, b, nir, re], axis=-1).astype(np.float32)
    return ms


def _yolo_label(cx_n, cy_n, w_n, h_n, cls_id: int) -> str:
    return f"{cls_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}"


# ── Core generation ──────────────────────────────────────────────────────────

def generate_dataset(root: str, seed: int = 42):
    """
    Generate the full synthetic dataset under *root*.

    Directory layout produced
    ─────────────────────────
    root/
      detection/
        images/train/  *.png
        images/val/    *.png
        labels/train/  *.txt   (YOLO format)
        labels/val/    *.txt
        dataset.yaml
      classification/
        train/<class>/  *.png  (cropped weed patches)
        val/<class>/    *.png
      multispectral/
        train/  *_ms.npy
        val/    *_ms.npy
      test/
        rgb/    *.png
        ms/     *_ms.npy
    """
    random.seed(seed)
    np.random.seed(seed)

    root = Path(root)
    if root.exists():
        shutil.rmtree(root)

    # Detection dirs
    for split in ("train", "val"):
        (root / "detection" / "images" / split).mkdir(parents=True)
        (root / "detection" / "labels" / split).mkdir(parents=True)

    # Classification dirs
    for split in ("train", "val"):
        for cls in WEED_CLASSES:
            (root / "classification" / split / cls).mkdir(parents=True)

    # Multispectral dirs
    for split in ("train", "val"):
        (root / "multispectral" / split).mkdir(parents=True)

    # Test dirs
    (root / "test" / "rgb").mkdir(parents=True)
    (root / "test" / "ms").mkdir(parents=True)

    splits = [("train", N_TRAIN), ("val", N_VAL)]

    for split, n_imgs in splits:
        for idx in range(n_imgs):
            img_name = f"{split}_{idx:04d}"
            img = _soil_background(IMG_SIZE, IMG_SIZE)
            labels = []
            n_weeds = random.randint(1, MAX_WEEDS)

            for _ in range(n_weeds):
                cls_name = random.choice(WEED_CLASSES[:-1])   # skip Negative
                cls_id   = WEED_CLASSES.index(cls_name)
                bbox = _paint_weed(img, cls_name)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                bw, bh = x2 - x1, y2 - y1
                if bw < 10 or bh < 10:
                    continue

                # YOLO normalised label
                cx_n = (x1 + bw / 2) / IMG_SIZE
                cy_n = (y1 + bh / 2) / IMG_SIZE
                w_n  = bw / IMG_SIZE
                h_n  = bh / IMG_SIZE
                labels.append(_yolo_label(cx_n, cy_n, w_n, h_n, cls_id))

                # Save crop for classifier
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
                    crop_path = root / "classification" / split / cls_name / f"{img_name}_{cls_id}.png"
                    cv2.imwrite(str(crop_path), crop_resized)

            # Save RGB image
            img_path = root / "detection" / "images" / split / f"{img_name}.png"
            cv2.imwrite(str(img_path), img)

            # Save YOLO label
            lbl_path = root / "detection" / "labels" / split / f"{img_name}.txt"
            lbl_path.write_text("\n".join(labels))

            # Save multispectral
            ms = _make_multispectral(img)
            ms_path = root / "multispectral" / split / f"{img_name}_ms.npy"
            np.save(str(ms_path), ms)

    # Generate test images (no labels needed)
    for idx in range(N_TEST):
        img = _soil_background(IMG_SIZE, IMG_SIZE)
        n_weeds = random.randint(2, MAX_WEEDS)
        for _ in range(n_weeds):
            cls_name = random.choice(WEED_CLASSES[:-1])
            _paint_weed(img, cls_name)

        cv2.imwrite(str(root / "test" / "rgb" / f"test_{idx:04d}.png"), img)
        ms = _make_multispectral(img)
        np.save(str(root / "test" / "ms" / f"test_{idx:04d}_ms.npy"), ms)

    # Write YOLO dataset.yaml
    yaml_content = f"""# Auto-generated weed detection dataset config
path: {root / 'detection'}
train: images/train
val:   images/val

nc: {len(WEED_CLASSES)}
names: {WEED_CLASSES}
"""
    (root / "detection" / "dataset.yaml").write_text(yaml_content)

    print(f"[DataGen] Dataset written to: {root}")
    print(f"  Detection  – train: {N_TRAIN} images, val: {N_VAL} images")
    print(f"  Classifier – crops saved per class under classification/")
    print(f"  Test       – {N_TEST} unlabelled images for inference demo")
    return root


if __name__ == "__main__":
    out = Path(__file__).parent.parent / "outputs" / "sample_dataset"
    generate_dataset(str(out))
