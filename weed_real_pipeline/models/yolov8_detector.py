"""
YOLOv8 weed detection model wrapper.

Uses the `ultralytics` library which provides:
  - Pre-trained YOLOv8n/s/m/l/x backbones
  - YAML-driven custom training
  - Built-in augmentation, mixed-precision, and export

Architecture note
─────────────────
YOLOv8 = CSPDarknet backbone + PANet neck + decoupled detection head.
The user requested "YOLOv26" — no such version exists; YOLOv8 is the
current state-of-the-art from Ultralytics (2023-2025) and is the
appropriate choice for single-stage object detection.

Model sizes (nano → xlarge):
  yolov8n – fastest, least accurate  (good for demo)
  yolov8s – small (good balance)
  yolov8m – medium
  yolov8l – large
  yolov8x – most accurate, slowest
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import torch
import numpy as np
import cv2


# ── Model builder ────────────────────────────────────────────────────────────

class WeedDetector:
    """
    Thin wrapper around ultralytics YOLO for weed detection.

    Parameters
    ----------
    model_size   : "n", "s", "m", "l", or "x"
    num_classes  : number of weed classes
    weights_path : path to a .pt checkpoint; None → load pretrained COCO weights
    device       : "cpu", "cuda", "mps"
    """

    MODEL_VARIANTS = {
        "n": "yolov8n.pt",
        "s": "yolov8s.pt",
        "m": "yolov8m.pt",
        "l": "yolov8l.pt",
        "x": "yolov8x.pt",
    }

    def __init__(
        self,
        model_size: str = "n",
        num_classes: int = 9,
        weights_path: Optional[str] = None,
        device: str = "cpu",
    ):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("pip install ultralytics")

        self.num_classes = num_classes
        self.device      = device

        if weights_path and Path(weights_path).exists():
            self.model = YOLO(weights_path)
        else:
            base_weights = self.MODEL_VARIANTS.get(model_size, "yolov8n.pt")
            self.model   = YOLO(base_weights)

        # "0,1" is a multi-GPU string for ultralytics train(); .to() doesn't accept it
        if "," not in str(device):
            self.model.to(device)

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        dataset_yaml: str,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        lr0: float = 0.01,
        output_dir: str = "outputs/yolo_runs",
        **kwargs,
    ) -> str:
        """
        Fine-tune on a custom weed dataset.

        Returns path to the best checkpoint.
        Training plots (results.png, confusion_matrix.png, PR_curve.png, etc.)
        are saved to <output_dir>/weed_detection/.
        """
        results = self.model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            project=output_dir,
            name="weed_detection",
            exist_ok=True,
            plots=True,
            device=self.device,
            **kwargs,
        )
        run_dir  = Path(output_dir) / "weed_detection"
        best_pt  = run_dir / "weights" / "best.pt"
        print(f"\n[YOLO] Run directory : {run_dir.resolve()}")
        print(f"[YOLO] Best weights  : {best_pt}")
        results_png = run_dir / "results.png"
        if results_png.exists():
            print(f"[YOLO] Training plots: {results_png}")
        else:
            # ultralytics ≥8.2 saves individual curve files instead
            curves = list(run_dir.glob("*.png"))
            if curves:
                print(f"[YOLO] Training plots: {[p.name for p in curves]}")
        return str(best_pt)

    # ── Validation ───────────────────────────────────────────────────────────

    def validate(self, dataset_yaml: str, imgsz: int = 640) -> Dict[str, float]:
        metrics = self.model.val(data=dataset_yaml, imgsz=imgsz, device=self.device)
        return {
            "mAP50":   float(metrics.box.map50),
            "mAP50-95":float(metrics.box.map),
            "precision":float(metrics.box.mp),
            "recall":   float(metrics.box.mr),
        }

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(
        self,
        image_input,           # str path, np.ndarray (BGR), or list of either
        conf: float = 0.25,
        iou:  float = 0.45,
        imgsz: int  = 640,
    ) -> List[Dict[str, Any]]:
        """
        Run detection on one or more images.

        Returns a list of result dicts (one per image):
          {
            "boxes":   np.ndarray (N, 4) – xyxy pixel coordinates
            "scores":  np.ndarray (N,)   – confidence scores
            "classes": np.ndarray (N,)   – integer class IDs
          }
        """
        results = self.model.predict(
            source=image_input,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )

        outputs = []
        for r in results:
            boxes_t = r.boxes
            if boxes_t is None or len(boxes_t) == 0:
                outputs.append({"boxes": np.empty((0, 4)), "scores": np.empty(0), "classes": np.empty(0)})
                continue
            outputs.append({
                "boxes":   boxes_t.xyxy.cpu().numpy(),
                "scores":  boxes_t.conf.cpu().numpy(),
                "classes": boxes_t.cls.cpu().numpy().astype(int),
            })
        return outputs

    # ── Export ───────────────────────────────────────────────────────────────

    def export(self, format: str = "onnx", output_path: Optional[str] = None):
        """Export to ONNX / TorchScript / CoreML etc."""
        path = self.model.export(format=format)
        if output_path:
            import shutil
            shutil.copy(path, output_path)
            return output_path
        return path

    def load_weights(self, weights_path: str):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.model.to(self.device)


# ── YOLO dataset.yaml builder ─────────────────────────────────────────────────

def build_dataset_yaml(
    dataset_root: str,
    class_names: List[str],
    output_yaml: Optional[str] = None,
) -> str:
    """
    Write a dataset.yaml file compatible with ultralytics YOLO training.
    """
    root = Path(dataset_root)
    content = f"""# Weed detection dataset
path: {root}
train: images/train
val:   images/val

nc: {len(class_names)}
names: {class_names}
"""
    if output_yaml is None:
        output_yaml = str(root / "dataset.yaml")
    Path(output_yaml).write_text(content)
    return output_yaml


# ── Detection result visualisation ───────────────────────────────────────────

def draw_detections(
    img_bgr: np.ndarray,
    detections: Dict[str, Any],
    class_names: List[str],
    color_map: Optional[Dict[int, tuple]] = None,
) -> np.ndarray:
    """Overlay bounding boxes + labels on a BGR image."""
    img = img_bgr.copy()
    boxes   = detections["boxes"]
    scores  = detections["scores"]
    classes = detections["classes"]

    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, classes)):
        x1, y1, x2, y2 = box.astype(int)
        color = color_map.get(cls_id, (0, 255, 0)) if color_map else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cls_id]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


if __name__ == "__main__":
    from data.sample_generator import WEED_CLASSES
    root = Path(__file__).parent.parent / "outputs" / "sample_dataset"
    yaml_path = str(root / "detection" / "dataset.yaml")

    detector = WeedDetector(model_size="n", num_classes=len(WEED_CLASSES))
    print(f"[YOLO] Model loaded: {type(detector.model)}")
    print(f"[YOLO] Dataset yaml: {yaml_path}")
    print("[YOLO] Ready. Call detector.train(yaml_path) to fine-tune.")
