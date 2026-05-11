"""
Weed dataset loaders.

  LocalWeedDataset    — YOLO-format datasets (images/ + labels/ folders)
  DeepWeedsDataset    — DeepWeeds CSV-format (flat images/ + labels.csv)

Expected layout — YOLO format
───────────────────────────────
  <dataset_root>/
    data.yaml
    images/  train/  valid/  test/
    labels/  train/  valid/  test/

Expected layout — DeepWeeds CSV format
────────────────────────────────────────
  <dataset_root>/
    labels.csv          ← columns: Filename, Label, Species
    images/  *.jpg      ← all images in one flat directory

Usage
─────
  from data.dataset_loader import LocalWeedDataset, DeepWeedsDataset
  ds = DeepWeedsDataset("/path/to/deepweeds")
  ds.extract_crops("crops/train", split="train")   # copies full images per class
"""

import csv
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
            if "images" in str(imgs):
                lbl = Path(str(imgs).replace("images", "labels"))
                if lbl.exists():
                    return lbl
            lbl = imgs.parent / "labels" / imgs.name
            if lbl.exists():
                return lbl
            lbl = imgs / "labels"
            if lbl.exists():
                return lbl
            if any(imgs.glob("*.txt")):
                return imgs
            return None

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

    def write_yaml(self, output_path: Optional[str] = None) -> str:
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

    def generate_multispectral(self, output_dir: str, split: str = "train",
                                num_bands: int = 5):
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

    def extract_crops(self, output_dir: str, split: str = "train",
                      crop_size: int = 96):
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


# ── DeepWeeds CSV dataset ─────────────────────────────────────────────────────

DEEPWEEDS_CLASSES = [
    "Chinee apple",
    "Lantana",
    "Parkinsonia",
    "Parthenium",
    "Prickly acacia",
    "Rubber vine",
    "Siam weed",
    "Snake weed",
    "Negative",
]


class DeepWeedsDataset:
    """
    Adapter for the DeepWeeds dataset (Olsen et al., 2019).

    Layout expected
    ───────────────
      <root>/
        labels.csv      ← Filename, Label, Species
        images/         ← flat directory of all 17,509 .jpg images

    Stratified 60/20/20 train/val/test split, seeded for reproducibility.
    Split cache written to <root>/splits.csv on first run.
    """

    SPLIT_SEED = 42

    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self._load()

    def _load(self):
        csv_candidates = list(self.root.glob("*.csv"))
        if not csv_candidates:
            raise FileNotFoundError(f"No CSV file found in {self.root}")
        csv_path = next((p for p in csv_candidates if p.stem.lower() == "labels"),
                        csv_candidates[0])

        img_dir = self.root / "images"
        if not img_dir.exists():
            img_dir = self.root

        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            headers = [h.strip() for h in (reader.fieldnames or [])]
            filename_col = next((h for h in headers if h.lower() == "filename"), None)
            label_col    = next((h for h in headers if h.lower() == "label"),    None)
            species_col  = next((h for h in headers if h.lower() == "species"),  None)
            if filename_col is None or label_col is None:
                raise ValueError(
                    f"CSV must have 'Filename' and 'Label' columns. Found: {headers}"
                )
            for row in reader:
                fname   = row[filename_col].strip()
                label   = int(row[label_col])
                species = row[species_col].strip() if species_col else None
                img_p   = img_dir / fname
                if img_p.exists():
                    rows.append((fname, label, species, img_p))

        if not rows:
            raise FileNotFoundError(
                f"No matching images found. CSV: {csv_path}, images dir: {img_dir}"
            )

        if rows[0][2] is not None:
            id_to_name: Dict[int, str] = {}
            for _, lbl, species, _ in rows:
                id_to_name.setdefault(lbl, species)
            max_id = max(id_to_name)
            self.class_names: List[str] = [
                id_to_name.get(i, f"class_{i}") for i in range(max_id + 1)
            ]
        else:
            self.class_names = DEEPWEEDS_CLASSES

        split_cache = self.root / "splits.csv"
        if split_cache.exists():
            splits: Dict[str, str] = {}
            with open(split_cache, newline="") as f:
                for row in csv.DictReader(f):
                    splits[row["Filename"]] = row["Split"]
        else:
            import random
            rng = random.Random(self.SPLIT_SEED)
            by_label: Dict[int, list] = {}
            for fname, lbl, _, _ in rows:
                by_label.setdefault(lbl, []).append(fname)
            splits = {}
            for lbl, fnames in by_label.items():
                shuffled = fnames[:]
                rng.shuffle(shuffled)
                n = len(shuffled)
                n_val  = max(1, int(n * 0.20))
                n_test = max(1, int(n * 0.20))
                for i, fn in enumerate(shuffled):
                    if i < n_val:
                        splits[fn] = "val"
                    elif i < n_val + n_test:
                        splits[fn] = "test"
                    else:
                        splits[fn] = "train"
            with open(split_cache, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Filename", "Split"])
                for fn, sp in splits.items():
                    w.writerow([fn, sp])
            print(f"[DeepWeeds] Split cache written → {split_cache}")

        self._train_rows = [(fn, lbl, ip) for fn, lbl, _, ip in rows if splits.get(fn) == "train"]
        self._val_rows   = [(fn, lbl, ip) for fn, lbl, _, ip in rows if splits.get(fn) == "val"]
        self._test_rows  = [(fn, lbl, ip) for fn, lbl, _, ip in rows if splits.get(fn) == "test"]

        self.train_imgs   = None
        self.val_imgs     = None
        self.test_imgs    = None
        self.train_labels = None
        self.val_labels   = None
        self.test_labels  = None
        self.yaml_path    = None

        print(f"[DeepWeeds] Root       : {self.root}")
        print(f"[DeepWeeds] Classes    : {len(self.class_names)} — {self.class_names}")
        print(f"[DeepWeeds] Train      : {len(self._train_rows)} images")
        print(f"[DeepWeeds] Val        : {len(self._val_rows)} images")
        print(f"[DeepWeeds] Test       : {len(self._test_rows)} images")

    def stats(self) -> Dict:
        return {
            "num_classes":  len(self.class_names),
            "class_names":  self.class_names,
            "train_images": len(self._train_rows),
            "val_images":   len(self._val_rows),
            "test_images":  len(self._test_rows),
        }

    def write_yaml(self, output_path: Optional[str] = None) -> str:
        if output_path is None:
            output_path = str(self.root / "data.yaml")
        content = {
            "path":  str(self.root),
            "train": "",
            "val":   "",
            "nc":    len(self.class_names),
            "names": self.class_names,
        }
        with open(output_path, "w") as f:
            yaml.dump(content, f, default_flow_style=False, sort_keys=False)
        self.yaml_path = output_path
        print(f"[DeepWeeds] data.yaml stub → {output_path}")
        return output_path

    def generate_multispectral(self, output_dir: str, split: str = "train",
                                num_bands: int = 5):
        import cv2
        rows = (self._train_rows if split == "train" else
                self._val_rows   if split == "val"   else self._test_rows)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        print(f"[MS] Simulating {num_bands}-band MS for {len(rows)} '{split}' images…")
        for fn, _, img_path in rows:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            rgb_f   = img_rgb.astype(np.float32) / 255.0
            r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
            nir = np.clip(g * 1.4 + np.random.normal(0, 0.05, g.shape), 0, 1).astype(np.float32)
            re  = np.clip((nir + r) / 2 + np.random.normal(0, 0.03, r.shape), 0, 1).astype(np.float32)
            ms  = np.stack([r, g, b, nir, re], axis=-1) if num_bands == 5 else np.stack([r, g, b], axis=-1)
            np.save(str(out / f"{Path(fn).stem}_ms.npy"), ms)
        print(f"[MS] Saved → {out}")

    def extract_crops(self, output_dir: str, split: str = "train",
                      crop_size: int = 96):
        """
        For DeepWeeds the full image IS the crop (image-level classification).
        Copies/resizes images into per-class folders.
        """
        import cv2
        rows = (self._train_rows if split == "train" else
                self._val_rows   if split == "val"   else self._test_rows)
        out = Path(output_dir)
        for cls_name in self.class_names:
            (out / cls_name).mkdir(parents=True, exist_ok=True)
        n_saved = 0
        for fn, lbl, img_path in rows:
            cls_name = (self.class_names[lbl]
                        if lbl < len(self.class_names) else f"class_{lbl}")
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_resized = cv2.resize(img_bgr, (crop_size, crop_size))
            save_path = out / cls_name / Path(fn).name
            cv2.imwrite(str(save_path), img_resized)
            n_saved += 1
        print(f"[DeepWeeds] Organised {n_saved} '{split}' images → {out}")


def load_dataset(root: str) -> "LocalWeedDataset | DeepWeedsDataset":
    """
    Auto-detect dataset type:
      - If a *.csv file is present → DeepWeedsDataset
      - Otherwise → LocalWeedDataset (YOLO format)
    """
    root_p = Path(root).resolve()
    if list(root_p.glob("*.csv")):
        return DeepWeedsDataset(root)
    return LocalWeedDataset(root)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys

    p = argparse.ArgumentParser(description="Inspect and prepare a local YOLO weed dataset")
    p.add_argument("--dataset_dir",    required=True, help="Path to dataset root")
    p.add_argument("--gen_ms",         action="store_true", help="Simulate 5-band MS arrays")
    p.add_argument("--extract_crops",  action="store_true", help="Extract per-class crop patches")
    p.add_argument("--output_dir",     default="outputs/real_data")
    args = p.parse_args()

    ds = load_dataset(args.dataset_dir)
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
