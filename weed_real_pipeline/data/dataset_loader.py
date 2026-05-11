"""
Local YOLO-format weed dataset loader.

Designed for any locally-available YOLO-format dataset, including:
  - weed_archive_detection (5-class, 958/274/138 split)
  - Any dataset exported from Roboflow, CVAT, or Label Studio in YOLO format

Expected layout
───────────────
  <dataset_root>/
    data.yaml          ← class names + split paths
    images/
      train/   *.jpg / *.png
      valid/   *.jpg / *.png   (or val/)
      test/    *.jpg / *.png   (optional)
    labels/
      train/   *.txt
      valid/   *.txt
      test/    *.txt

Usage
─────
  from data.dataset_loader import LocalWeedDataset
  ds = LocalWeedDataset("/path/to/weed_archive_detection")
  print(ds.stats())
  yaml_path = ds.write_yaml()            # fixes absolute paths for YOLO trainer
  ds.generate_multispectral("ms/train")  # simulate 5-band arrays if needed
  ds.extract_crops("crops/train")        # per-class patches for classifier
"""

import shutil
import yaml
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np


# ── Structure detection helpers ───────────────────────────────────────────────

def _find_yaml(root: Path) -> Optional[Path]:
    for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
        p = root / name
        if p.exists():
            return p
    for p in root.rglob("data.yaml"):
        return p
    return None


def _find_split_dir(root: Path, split: str) -> Optional[Path]:
    """Locate images directory; handles val/valid naming variants."""
    alt = "valid" if split == "val" else ("val" if split == "valid" else split)
    candidates = [
        root / "images" / split,
        root / "images" / alt,
        root / split,
        root / alt,
    ]
    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return c
    return None


def _infer_classes(labels_dir: Path) -> List[str]:
    max_id = 0
    for f in labels_dir.rglob("*.txt"):
        for line in f.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                try:
                    max_id = max(max_id, int(parts[0]))
                except ValueError:
                    pass
    return [f"weed_class_{i}" for i in range(max_id + 1)]


# ── Main dataset class ────────────────────────────────────────────────────────

class LocalWeedDataset:
    """
    Adapter for any locally-stored YOLO-format weed dataset.

    Attributes
    ----------
    root         : dataset root Path
    class_names  : ordered list of class name strings
    yaml_path    : path to data.yaml (absolute-path version for YOLO trainer)
    train_imgs   : Path to training images
    val_imgs     : Path to validation images
    test_imgs    : Path to test images (may be None)
    train_labels : Path to training label txts
    val_labels   : Path to validation label txts
    """

    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self._detect_structure()

    def _detect_structure(self):
        yaml_p = _find_yaml(self.root)

        if yaml_p:
            with open(yaml_p) as f:
                cfg = yaml.safe_load(f)
            names = cfg.get("names")
            if isinstance(names, dict):
                # ultralytics ≥8.1 stores names as {0: "class", 1: ...}
                names = [names[i] for i in sorted(names)]
            self.class_names: List[str] = names or [f"class_{i}" for i in range(cfg.get("nc", 1))]
            self.yaml_path = str(yaml_p)
        else:
            self.class_names = None
            self.yaml_path   = None

        self.train_imgs = _find_split_dir(self.root, "train")
        self.val_imgs   = (
            _find_split_dir(self.root, "val") or
            _find_split_dir(self.root, "valid")
        )
        self.test_imgs  = _find_split_dir(self.root, "test")

        if self.train_imgs is None:
            raise FileNotFoundError(
                f"Cannot locate training images under {self.root}. "
                "Ensure images/train/ (or train/) exists and is non-empty."
            )

        def _labels_dir(imgs: Optional[Path]) -> Optional[Path]:
            if imgs is None:
                return None
            lbl = Path(str(imgs).replace("images", "labels"))
            return lbl if lbl.exists() else None

        self.train_labels = _labels_dir(self.train_imgs)
        self.val_labels   = _labels_dir(self.val_imgs)
        self.test_labels  = _labels_dir(self.test_imgs)

        if self.class_names is None and self.train_labels:
            self.class_names = _infer_classes(self.train_labels)
        if self.class_names is None:
            self.class_names = ["weed"]

        print(f"[Dataset] Root        : {self.root}")
        print(f"[Dataset] Classes ({len(self.class_names)}): {self.class_names}")
        print(f"[Dataset] Train imgs  : {self.train_imgs}")
        print(f"[Dataset] Val imgs    : {self.val_imgs}")
        print(f"[Dataset] Test imgs   : {self.test_imgs}")
        print(f"[Dataset] Train labels: {self.train_labels}")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        def _count(d: Optional[Path]) -> int:
            if d is None:
                return 0
            return len(list(d.rglob("*.jpg")) + list(d.rglob("*.png")))

        return {
            "num_classes":   len(self.class_names),
            "class_names":   self.class_names,
            "train_images":  _count(self.train_imgs),
            "val_images":    _count(self.val_imgs),
            "test_images":   _count(self.test_imgs),
        }

    # ── Write / fix data.yaml with absolute paths ─────────────────────────────

    def write_yaml(self, output_path: Optional[str] = None) -> str:
        """
        Write data.yaml with absolute paths so ultralytics can find
        the data regardless of working directory.
        """
        if output_path is None:
            output_path = str(self.root / "data.yaml")

        def _rel(p: Optional[Path]) -> str:
            if p is None:
                return ""
            try:
                return str(p.relative_to(self.root))
            except ValueError:
                return str(p)

        content = {
            "path":  str(self.root),
            "train": _rel(self.train_imgs),
            "val":   _rel(self.val_imgs) if self.val_imgs else _rel(self.train_imgs),
            "nc":    len(self.class_names),
            "names": self.class_names,
        }
        if self.test_imgs:
            content["test"] = _rel(self.test_imgs)

        with open(output_path, "w") as f:
            yaml.dump(content, f, default_flow_style=False, sort_keys=False)

        self.yaml_path = output_path
        print(f"[Dataset] data.yaml written → {output_path}")
        return output_path

    # ── Multispectral simulation (for RGB-only datasets) ─────────────────────

    def generate_multispectral(self, output_dir: str, split: str = "train",
                                num_bands: int = 5):
        """
        Simulate a 5-band multispectral array from RGB for each image.

        Bands: [R, G, B, NIR, RedEdge]
          NIR      = G × 1.4 + N(0, 0.05)   (green channel correlates with NIR)
          RedEdge  = (NIR + R) / 2 + N(0, 0.03)

        When real multispectral data is available, replace this with a loader
        that reads .tif / .npy files directly (see note in README).
        """
        import cv2

        src = self.train_imgs if split == "train" else (
              self.val_imgs   if split == "val"   else self.test_imgs)
        if src is None:
            print(f"[MS] No images for split '{split}'")
            return

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        img_files = list(src.rglob("*.jpg")) + list(src.rglob("*.png"))
        print(f"[MS] Simulating {num_bands}-band MS for {len(img_files)} '{split}' images…")

        for img_path in img_files:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            rgb_f   = img_rgb.astype(np.float32) / 255.0
            r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]

            nir = np.clip(g * 1.4 + np.random.normal(0, 0.05, g.shape), 0, 1).astype(np.float32)
            re  = np.clip((nir + r) / 2 + np.random.normal(0, 0.03, r.shape), 0, 1).astype(np.float32)

            if num_bands == 5:
                ms = np.stack([r, g, b, nir, re], axis=-1)
            elif num_bands == 3:
                ms = np.stack([r, g, b], axis=-1)
            else:
                raise ValueError(f"num_bands must be 3 or 5, got {num_bands}")

            np.save(str(out / f"{img_path.stem}_ms.npy"), ms.astype(np.float32))

        print(f"[MS] Saved {len(img_files)} arrays → {out}")

    # ── Classification crop extraction ────────────────────────────────────────

    def extract_crops(self, output_dir: str, split: str = "train",
                      crop_size: int = 96):
        """
        Extract per-class weed crop patches from YOLO bounding box labels.
        Creates a folder-per-class layout:
          <output_dir>/<class_name>/*.png
        """
        import cv2

        src_imgs = (self.train_imgs if split == "train" else
                    self.val_imgs   if split == "val"   else self.test_imgs)
        src_lbls = (self.train_labels if split == "train" else
                    self.val_labels   if split == "val"   else self.test_labels)

        if src_imgs is None or src_lbls is None:
            print(f"[Crops] Missing images or labels for split '{split}'")
            return

        out = Path(output_dir)
        for cls_name in self.class_names:
            (out / cls_name).mkdir(parents=True, exist_ok=True)

        img_files = list(src_imgs.rglob("*.jpg")) + list(src_imgs.rglob("*.png"))
        n_saved = 0

        for img_path in img_files:
            lbl_path = src_lbls / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]

            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx_n, cy_n, bw_n, bh_n = map(float, parts[1:5])

                x1 = max(0, int((cx_n - bw_n / 2) * w))
                y1 = max(0, int((cy_n - bh_n / 2) * h))
                x2 = min(w, int((cx_n + bw_n / 2) * w))
                y2 = min(h, int((cy_n + bh_n / 2) * h))

                crop = img_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                cls_name = (self.class_names[cls_id]
                            if cls_id < len(self.class_names)
                            else f"class_{cls_id}")
                crop_resized = cv2.resize(crop, (crop_size, crop_size))
                save_path = out / cls_name / f"{img_path.stem}_{n_saved}.png"
                cv2.imwrite(str(save_path), crop_resized)
                n_saved += 1

        print(f"[Crops] Extracted {n_saved} crops → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys

    p = argparse.ArgumentParser(description="Inspect and prepare a local YOLO weed dataset")
    p.add_argument("--dataset_dir",    required=True, help="Path to dataset root")
    p.add_argument("--gen_ms",         action="store_true", help="Simulate 5-band MS arrays")
    p.add_argument("--extract_crops",  action="store_true", help="Extract per-class crop patches")
    p.add_argument("--output_dir",     default="outputs/real_data")
    args = p.parse_args()

    ds = LocalWeedDataset(args.dataset_dir)
    stats = ds.stats()
    print("\n[Stats]")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    yaml_path = ds.write_yaml()
    print(f"\n  data.yaml → {yaml_path}")

    out = Path(args.output_dir)
    if args.gen_ms:
        ds.generate_multispectral(str(out / "multispectral" / "train"), "train")
        ds.generate_multispectral(str(out / "multispectral" / "val"),   "val")

    if args.extract_crops:
        ds.extract_crops(str(out / "classification" / "train"), "train")
        ds.extract_crops(str(out / "classification" / "val"),   "val")
