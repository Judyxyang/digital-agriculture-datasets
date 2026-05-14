from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models

from banana_ripeness.image_preprocessor import ImagePreprocessor

RIPENESS_STAGES = ("unripe", "nearly-ripe", "ripe", "overripe")


class RipenessClassifier:
    """MobileNetV2-based banana ripeness classifier.

    Wraps a fine-tuned MobileNetV2 backbone and exposes a single
    predict() call. The backbone is swappable via the `backbone` argument
    for experimentation without touching the rest of the pipeline.

    Interface: classifier.predict(image) → (ripeness_stage, confidence_score)
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        backbone: nn.Module | None = None,
        device: str | None = None,
        pretrained: bool = True,
    ):
        self._device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self._preprocessor = ImagePreprocessor()
        self._model = self._build_model(backbone)

        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location=self._device)
            self._model.load_state_dict(state)

        self._model.to(self._device)
        self._model.eval()

    def _build_model(self, backbone: nn.Module | None) -> nn.Module:
        if backbone is not None:
            return backbone
        model = models.mobilenet_v2(weights=self._weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(RIPENESS_STAGES))
        return model

    def predict(
        self, image, roi_box=None
    ) -> Tuple[str, float]:
        """Classify the ripeness stage of a banana image.

        Args:
            image: A PIL Image of a banana.
            roi_box: Optional (left, upper, right, lower) ROI bounding box.

        Returns:
            A tuple of (ripeness_stage, confidence_score) where ripeness_stage
            is one of RIPENESS_STAGES and confidence_score is in [0, 1].
        """
        tensor = self._preprocessor.preprocess(image, roi_box=roi_box)
        tensor = tensor.unsqueeze(0).to(self._device)   # (1, 3, H, W)

        with torch.no_grad():
            logits = self._model(tensor)                  # (1, 4)
            probs = torch.softmax(logits, dim=1)          # (1, 4)

        top_idx = probs.argmax(dim=1).item()
        confidence = probs[0, top_idx].item()
        return RIPENESS_STAGES[top_idx], confidence

    def state_dict(self) -> dict:
        return self._model.state_dict()
