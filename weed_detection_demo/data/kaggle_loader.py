"""
Kaggle weed detection dataset downloader and adapter.

Target dataset
──────────────
  https://www.kaggle.com/datasets/thaslimvs/weeds-detection-dataset

Dataset characteristics (as publicly documented)
─────────────────────────────────────────────────
  • YOLO-format bounding-box annotations (.txt label files)
  • Classes: broadleaf, grass, sedge, etc. (auto-detected from data.yaml)
  • Directory layout after download:
      weeds-detection-dataset/
        data.yaml          ← class names + split paths
        images/
          train/   *.jpg / *.png
          valid/   *.jpg / *.png
          test/    *.jpg / *.png   (may be absent)
        labels/
          train/   *.txt
          valid/   *.txt
          test/    *.txt

Usage
─────
  # Option A – kaggle CLI (recommended):
  #   1. Install:  pip install kaggle
  #   2. Place API token in ~/.kaggle/kaggle.json
  #              {"username":"YOUR_USER","key":"YOUR_KEY"}
  #   3. Run this script:
  python weed_detection_demo/data/kaggle_loader.py \
      --output_dir outputs/kaggle_dataset

  # Option B – manual download:
  #   Download from the Kaggle page and unzip into outputs/kaggle_dataset/
  #   then run with --skip_download

Auto-detection
──────────────
  The loader reads data.yaml to discover class names and split paths.
  If data.yaml is absent it scans label files to infer class IDs and
  builds a synthetic class list.
"""

import argparse
import shutil
import subprocess
import sys
import yaml
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np


KAGGLE_DATASET = "thaslimvs/weeds-detection-dataset"
KAGGLE_ZIP     = "weeds-detection-dataset.zip"


# ── Download ─────────────────────────────────────────────────────────────────

def download_kaggle(output_dir: str) -> Path:
    """Download and unzip the dataset via the kaggle CLI."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(["kaggle", "--version"], check=True,
                       capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            "kaggle CLI not found. Install with: pip install kaggle\n"
            "Then put your API token in ~/.kaggle/kaggle.json\n"
            "Or download manually from:\n"
            f"  https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
            "and unzip into: " + str(out)
        )

    print(f"[Kaggle] Downloading {KAGGLE_DATASET} → {out} …")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "-p", str(out), "--unzip"],
        check=True,
    )
    print("[Kaggle] Download complete.")
    return out


# ── Dataset structure detection ───────────────────────────────────────────────

def _find_yaml(root: Path) -> Optional[Path]:
    """Locate data.yaml / dataset.yaml in the downloaded folder."""
    for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
        p = root / name
        if p.exists():
            return p
    # Search one level deeper
    for p in root.rglob("data.yaml"):
        return p
    return None


def _find_split_dir(root: Path, split: str) -> Optional[Path]:
    """
    Locate the images directory for a given split.
    Handles: images/train, images/valid, images/val, train, valid …
    """
    candidates = [
        root / "images" / split,
        root / "images" / ("valid" if split == "val" else split),
        root / split,
        root / ("valid" if split == "val" else split),
    ]
    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return c
    return None


def _infer_classes_from_labels(labels_dir: Path) -> List[str]:
    """Read all label files and find the max class ID to build a class list."""
    max_id = 0
    for lbl_file in labels_dir.rglob("*.txt"):
        for line in lbl_file.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                try:
                    max_id = max(max_id, int(parts[0]))
                except ValueError:
                    pass
    return [f"weed_class_{i}" for i in range(max_id + 1)]


class KaggleWeedDataset:
    """
    Adapter that reads the downloaded Kaggle weed dataset and exposes it
    in the same interface expected by the rest of this pipeline.

    Attributes
    ----------
    root         : Path to the dataset root
    class_names  : list of weed class name strings
    yaml_path    : path to data.yaml (for ultralytics YOLO trainer)
    train_imgs   : Path to training images directory
    val_imgs     : Path to validation images directory
    train_labels : Path to training labels directory
    val_labels   : Path to validation labels directory
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self._detect_structure()

    def _detect_structure(self):
        # ── 1. Find data.yaml ─────────────────────────────────────────────────
        yaml_p = _find_yaml(self.root)

        if yaml_p:
            with open(yaml_p) as f:
                cfg = yaml.safe_load(f)
            self.class_names: List[str] = (
                cfg.get("names") or
                [cfg[f"name{i}"] for i in range(cfg.get("nc", 1))]
            )
            self.yaml_path = str(yaml_p)
        else:
            self.class_names = None   # will infer below
            self.yaml_path   = None

        # ── 2. Find split directories ─────────────────────────────────────────
        self.train_imgs   = _find_split_dir(self.root, "train")
        self.val_imgs     = (
            _find_split_dir(self.root, "val") or
            _find_split_dir(self.root, "valid")
        )

        if self.train_imgs is None:
            raise FileNotFoundError(
                f"Could not locate train images under {self.root}. "
                "Make sure the dataset was downloaded and unzipped correctly."
            )

        # ── 3. Derive label directories ───────────────────────────────────────
        def _labels_dir(imgs_dir: Path) -> Optional[Path]:
            lbl = Path(str(imgs_dir).replace("images", "labels"))
            return lbl if lbl.exists() else None

        self.train_labels = _labels_dir(self.train_imgs)
        self.val_labels   = (
            _labels_dir(self.val_imgs) if self.val_imgs else None
        )

        # ── 4. Infer class names if yaml absent ───────────────────────────────
        if self.class_names is None and self.train_labels:
            self.class_names = _infer_classes_from_labels(self.train_labels)

        if self.class_names is None:
            self.class_names = ["weed"]   # fallback

        print(f"[KaggleDataset] Root        : {self.root}")
        print(f"[KaggleDataset] Classes     : {self.class_names}")
        print(f"[KaggleDataset] Train imgs  : {self.train_imgs}")
        print(f"[KaggleDataset] Val imgs    : {self.val_imgs}")
        print(f"[KaggleDataset] Train labels: {self.train_labels}")
        print(f"[KaggleDataset] yaml        : {self.yaml_path}")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        n_train = len(list(self.train_imgs.rglob("*.jpg")) +
                      list(self.train_imgs.rglob("*.png"))) if self.train_imgs else 0
        n_val   = len(list(self.val_imgs.rglob("*.jpg")) +
                      list(self.val_imgs.rglob("*.png"))) if self.val_imgs else 0
        return {
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "train_images": n_train,
            "val_images":   n_val,
        }

    # ── Write / fix data.yaml ─────────────────────────────────────────────────

    def write_yaml(self, output_path: Optional[str] = None) -> str:
        """
        Write (or overwrite) data.yaml with absolute paths so ultralytics
        YOLO trainer can find the data regardless of working directory.
        """
        if output_path is None:
            output_path = str(self.root / "data.yaml")

        # Determine relative paths from yaml location to image dirs
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
        with open(output_path, "w") as f:
            yaml.dump(content, f, default_flow_style=False, sort_keys=False)

        self.yaml_path = output_path
        print(f"[KaggleDataset] data.yaml written → {output_path}")
        return output_path

    # ── Multispectral simulation ──────────────────────────────────────────────

    def generate_multispectral(self, output_dir: str, split: str = "train"):
        """
        For each RGB image in the split, simulate a 5-band multispectral
        array and save as <stem>_ms.npy.  Useful for the MS pipeline stage
        when real multispectral imagery is unavailable.
        """
        import cv2
        src = self.train_imgs if split == "train" else self.val_imgs
        if src is None:
            print(f"[MS] No images found for split '{split}'")
            return

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        img_files = list(src.rglob("*.jpg")) + list(src.rglob("*.png"))
        print(f"[MS] Simulating multispectral for {len(img_files)} {split} images…")

        for img_path in img_files:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            rgb_f   = img_rgb.astype(np.float32) / 255.0
            r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]

            nir = np.clip(g * 1.4 + np.random.normal(0, 0.05, g.shape), 0, 1).astype(np.float32)
            re  = np.clip((nir + r) / 2 + np.random.normal(0, 0.03, r.shape), 0, 1).astype(np.float32)
            ms  = np.stack([r, g, b, nir, re], axis=-1).astype(np.float32)

            ms_path = out / f"{img_path.stem}_ms.npy"
            np.save(str(ms_path), ms)

        print(f"[MS] Saved {len(img_files)} multispectral arrays → {out}")

    # ── Classification crop extraction ────────────────────────────────────────

    def extract_crops(self, output_dir: str, split: str = "train",
                      crop_size: int = 96):
        """
        Read YOLO label files and save cropped weed patches per class.
        Creates the folder-per-class layout expected by WeedClassificationDataset.
        """
        import cv2
        src_imgs = self.train_imgs if split == "train" else self.val_imgs
        src_lbls = self.train_labels if split == "train" else self.val_labels
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

                x1 = int((cx_n - bw_n / 2) * w)
                y1 = int((cy_n - bh_n / 2) * h)
                x2 = int((cx_n + bw_n / 2) * w)
                y2 = int((cy_n + bh_n / 2) * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crop = img_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                cls_name = (self.class_names[cls_id]
                            if cls_id < len(self.class_names)
                            else f"class_{cls_id}")
                crop_resized = cv2.resize(crop, (crop_size, crop_size))
                save_path = out / cls_name / f"{img_path.stem}_{cls_id}_{n_saved}.png"
                cv2.imwrite(str(save_path), crop_resized)
                n_saved += 1

        print(f"[Crops] Extracted {n_saved} crops → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Download and prepare the Kaggle weed detection dataset"
    )
    p.add_argument("--output_dir",    default="outputs/kaggle_dataset",
                   help="Where to download / look for the dataset")
    p.add_argument("--skip_download", action="store_true",
                   help="Skip download; dataset already in --output_dir")
    p.add_argument("--gen_ms",        action="store_true",
                   help="Simulate multispectral arrays for each image")
    p.add_argument("--extract_crops", action="store_true",
                   help="Extract per-class crops for classifier training")
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.output_dir)

    if not args.skip_download:
        download_kaggle(str(out))

    ds = KaggleWeedDataset(str(out))
    stats = ds.stats()
    print("\n[Dataset Stats]")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Ensure a valid data.yaml exists
    ds.write_yaml()

    if args.gen_ms:
        ds.generate_multispectral(str(out / "multispectral" / "train"), "train")
        ds.generate_multispectral(str(out / "multispectral" / "val"),   "val")

    if args.extract_crops:
        ds.extract_crops(str(out / "classification" / "train"), "train")
        ds.extract_crops(str(out / "classification" / "val"),   "val")

    print("\n[Done] Dataset ready. Next step:")
    print(f"  python -m weed_detection_demo.training.train_detector "
          f"--dataset_root {out} --epochs 50")


if __name__ == "__main__":
    main()
