from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from banana_ripeness.ripeness_classifier import RipenessClassifier, RIPENESS_STAGES

# Confidence below this threshold is surfaced as low-confidence to the caller.
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.6


@dataclass(frozen=True)
class Prediction:
    """Result of a single inference call.

    Attributes:
        stage:          One of RIPENESS_STAGES.
        confidence:     Softmax probability of the top class, in [0, 1].
        low_confidence: True when confidence < threshold — caller should prompt
                        the user to retake the photo.
    """
    stage: str
    confidence: float
    low_confidence: bool

    def __str__(self) -> str:
        flag = " ⚠ low confidence — retake photo" if self.low_confidence else ""
        return f"{self.stage} ({self.confidence:.1%}){flag}"


class InferencePipeline:
    """Lightweight end-to-end inference pipeline for edge/mobile deployment.

    Loads a trained checkpoint once at construction time, then serves
    predictions with a single predict() call. Uses ImagePreprocessor
    internally — no duplicated preprocessing logic.

    Usage:
        pipeline = InferencePipeline("checkpoints/best.pth")
        result = pipeline.predict(pil_image)
        print(result.stage, result.confidence, result.low_confidence)
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        device: str | None = None,
        pretrained: bool = False,
    ):
        """
        Args:
            checkpoint_path:        Path to a saved state dict (.pth).
                                    When None, uses random weights (for testing).
            low_confidence_threshold: Confidence below this marks the prediction
                                    as low_confidence=True.
            device:                 Force a specific device ('cpu', 'cuda').
                                    Auto-detected when None.
            pretrained:             Load ImageNet pretrained weights when no
                                    checkpoint is given. Requires network access.
        """
        self._threshold = low_confidence_threshold
        self._classifier = RipenessClassifier(
            checkpoint_path=checkpoint_path,
            device=device,
            pretrained=pretrained,
        )

    def predict(
        self,
        image: Image.Image,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> Prediction:
        """Run inference on a single banana image.

        Args:
            image:   A PIL Image. Any mode accepted — converted to RGB internally.
            roi_box: Optional (left, upper, right, lower) bounding box to crop
                     the banana ROI before classification.

        Returns:
            A Prediction with stage, confidence, and low_confidence flag.
        """
        stage, confidence = self._classifier.predict(image, roi_box=roi_box)
        return Prediction(
            stage=stage,
            confidence=confidence,
            low_confidence=confidence < self._threshold,
        )
