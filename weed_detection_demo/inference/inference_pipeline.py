"""
Two-stage weed detection + classification inference pipeline.

Stage 1 – Detection   : YOLOv8  → bounding boxes over weed instances
Stage 2 – Classification: ResNeXt-50 → fine-grained species label per box

Each image produces:
  - Annotated RGB image with boxes, species labels, and confidence scores
  - A WeedDetection dataclass for downstream map generation

Usage
─────
  python -m inference.inference_pipeline \
    --yolo_weights  outputs/yolo_runs/weed_detection/weights/best.pt \
    --cls_weights   outputs/classifier_runs/best_classifier.pt \
    --input_dir     outputs/sample_dataset/test/rgb \
    --ms_dir        outputs/sample_dataset/test/ms \
    --output_dir    outputs/inference_results \
    --device        cpu

  # Or point at a single image:
  --input_dir path/to/single_image.png

Modes without trained weights (demo)
──────────────────────────────────────
  If no weights are provided the pipeline runs with untrained models so
  you can verify the full flow without training first:
    python -m inference.inference_pipeline --demo
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.sample_generator import WEED_CLASSES, CLASS_COLORS_BGR
from preprocessing.rgb_preprocessing import DetectionPreprocessor
from preprocessing.multispectral_preprocessing import MultispectralPreprocessor
from models.yolov8_detector import WeedDetector, draw_detections
from models.resnext50_classifier import WeedClassifier, CropClassifier, load_checkpoint


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class WeedInstance:
    """Single detected weed instance in one image."""
    image_id:    str
    class_name:  str
    class_id:    int
    confidence:  float
    box_xyxy:    Tuple[int, int, int, int]   # pixel coordinates
    center_xy:   Tuple[float, float]          # normalised [0,1] image centre
    proba:       np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


@dataclass
class ImageResult:
    """All detections for one image."""
    image_id:   str
    image_hw:   Tuple[int, int]
    instances:  List[WeedInstance]
    annotated:  Optional[np.ndarray] = field(repr=False, default=None)


# ── Pipeline ─────────────────────────────────────────────────────────────────

class WeedInferencePipeline:
    """
    End-to-end inference: image(s) → annotated output + WeedInstance list.

    Parameters
    ----------
    yolo_weights     : path to YOLOv8 .pt checkpoint (None → untrained demo)
    cls_weights      : path to ResNeXt-50 .pt checkpoint (None → untrained demo)
    class_names      : ordered list of weed class names
    device           : "cpu" / "cuda" / "mps"
    det_conf         : YOLO detection confidence threshold
    det_iou          : YOLO NMS IoU threshold
    fusion_mode      : multispectral fusion mode (only used when MS data provided)
    """

    def __init__(
        self,
        yolo_weights:  Optional[str],
        cls_weights:   Optional[str],
        class_names:   List[str]     = WEED_CLASSES,
        device:        str           = "cpu",
        det_conf:      float         = 0.25,
        det_iou:       float         = 0.45,
        fusion_mode:   str           = "rgb_ndvi",
    ):
        self.class_names = class_names
        self.device      = device
        self.det_conf    = det_conf
        self.det_iou     = det_iou

        # ── Detector ─────────────────────────────────────────────────────────
        self.detector = WeedDetector(
            model_size="n",
            num_classes=len(class_names),
            weights_path=yolo_weights,
            device=device,
        )

        # ── Classifier ───────────────────────────────────────────────────────
        if cls_weights and Path(cls_weights).exists():
            clf_model = load_checkpoint(cls_weights, class_names, device=device)
        else:
            clf_model = WeedClassifier(
                num_classes=len(class_names), pretrained=False
            ).to(device).eval()

        self.crop_clf = CropClassifier(clf_model, class_names, device=device)

        # ── Preprocessors ────────────────────────────────────────────────────
        self.det_preproc = DetectionPreprocessor(target_size=640)
        self.ms_preproc  = MultispectralPreprocessor(
            fusion_mode=fusion_mode, img_size=640
        )

        # BGR color per class for drawing
        self.color_map: Dict[int, Tuple[int,int,int]] = {
            i: CLASS_COLORS_BGR.get(name, (0, 255, 0))
            for i, name in enumerate(class_names)
        }

    # ── Single image inference ────────────────────────────────────────────────

    def run_single(
        self,
        img_path: str,
        ms_path:  Optional[str] = None,
    ) -> ImageResult:
        """
        Run detection + classification on one image.

        If ms_path is provided the multispectral tensor is computed (for
        logging / future fusion), but the current detection stage uses RGB.
        """
        img_bgr = cv2.imread(img_path)
        h, w    = img_bgr.shape[:2]
        image_id = Path(img_path).stem

        # ── Stage 1: Detection ───────────────────────────────────────────────
        det_results = self.detector.predict(
            img_bgr,
            conf=self.det_conf,
            iou=self.det_iou,
        )
        det = det_results[0]   # single image

        # ── Stage 2: Classify each crop ──────────────────────────────────────
        instances: List[WeedInstance] = []
        if len(det["boxes"]) > 0:
            crops = []
            for box in det["boxes"].astype(int):
                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = img_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    crop = np.zeros((32, 32, 3), dtype=np.uint8)
                crops.append(crop)

            cls_results = self.crop_clf.classify_batch(crops)

            for i, (box, det_score) in enumerate(zip(det["boxes"].astype(int), det["scores"])):
                cls_name, cls_conf, proba = cls_results[i]
                cls_id = self.class_names.index(cls_name)
                x1, y1, x2, y2 = box
                cx_norm = ((x1 + x2) / 2) / w
                cy_norm = ((y1 + y2) / 2) / h

                # Final confidence = geometric mean of detection & classification
                combined_conf = float(np.sqrt(det_score * cls_conf))

                instances.append(WeedInstance(
                    image_id   = image_id,
                    class_name = cls_name,
                    class_id   = cls_id,
                    confidence = combined_conf,
                    box_xyxy   = (x1, y1, x2, y2),
                    center_xy  = (cx_norm, cy_norm),
                    proba      = proba,
                ))

        # ── Annotate image ────────────────────────────────────────────────────
        annotated = img_bgr.copy()
        for inst in instances:
            x1, y1, x2, y2 = inst.box_xyxy
            color = self.color_map.get(inst.class_id, (0, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{inst.class_name} {inst.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return ImageResult(
            image_id  = image_id,
            image_hw  = (h, w),
            instances = instances,
            annotated = annotated,
        )

    # ── Batch inference ───────────────────────────────────────────────────────

    def run_batch(
        self,
        img_paths: List[str],
        ms_paths:  Optional[List[str]] = None,
        output_dir: Optional[str]      = None,
    ) -> List[ImageResult]:
        results = []
        ms_paths = ms_paths or [None] * len(img_paths)

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for img_path, ms_path in zip(img_paths, ms_paths):
            print(f"  Processing: {Path(img_path).name}")
            result = self.run_single(img_path, ms_path)
            results.append(result)

            if output_dir and result.annotated is not None:
                out_path = Path(output_dir) / f"{result.image_id}_detected.png"
                cv2.imwrite(str(out_path), result.annotated)

        return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Weed detection + classification inference")
    p.add_argument("--yolo_weights",  default=None)
    p.add_argument("--cls_weights",   default=None)
    p.add_argument("--input_dir",     default=None,
                   help="Directory of .png images or single image path")
    p.add_argument("--ms_dir",        default=None,
                   help="Directory of *_ms.npy multispectral files (optional)")
    p.add_argument("--output_dir",    default="outputs/inference_results")
    p.add_argument("--det_conf",      type=float, default=0.25)
    p.add_argument("--det_iou",       type=float, default=0.45)
    p.add_argument("--device",        default="cpu")
    p.add_argument("--demo",          action="store_true",
                   help="Generate sample data and run with untrained models")
    return p.parse_args()


def main():
    args = parse_args()

    if args.demo:
        from data.sample_generator import generate_dataset
        dataset_root = Path("outputs/sample_dataset")
        generate_dataset(str(dataset_root))
        args.input_dir = str(dataset_root / "test" / "rgb")
        args.ms_dir    = str(dataset_root / "test" / "ms")

    if not args.input_dir:
        print("Provide --input_dir or use --demo")
        sys.exit(1)

    # Collect images
    input_p = Path(args.input_dir)
    if input_p.is_dir():
        img_paths = sorted(input_p.glob("*.png")) + sorted(input_p.glob("*.jpg"))
        img_paths = [str(p) for p in img_paths]
    else:
        img_paths = [str(input_p)]

    # Match MS files
    ms_paths = None
    if args.ms_dir:
        ms_dir_p = Path(args.ms_dir)
        ms_paths = []
        for img_p in img_paths:
            stem   = Path(img_p).stem
            ms_p   = ms_dir_p / f"{stem}_ms.npy"
            ms_paths.append(str(ms_p) if ms_p.exists() else None)

    print(f"[Inference] Found {len(img_paths)} images")
    pipeline = WeedInferencePipeline(
        yolo_weights=args.yolo_weights,
        cls_weights=args.cls_weights,
        device=args.device,
        det_conf=args.det_conf,
        det_iou=args.det_iou,
    )

    print(f"[Inference] Running inference…")
    results = pipeline.run_batch(img_paths, ms_paths, output_dir=args.output_dir)

    total_instances = sum(len(r.instances) for r in results)
    print(f"\n[Inference] Done. {len(results)} images, {total_instances} weed instances detected.")
    print(f"[Inference] Annotated images saved to: {args.output_dir}")

    # Generate distribution map
    from inference.distribution_map import WeedDistributionMapper
    mapper = WeedDistributionMapper(class_names=WEED_CLASSES)
    mapper.generate_all(results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
