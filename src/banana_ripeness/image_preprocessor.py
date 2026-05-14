from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

# ImageNet statistics — consistent between training and inference
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

TARGET_SIZE = (224, 224)  # (height, width)


@dataclass
class ImagePreprocessor:
    """Shared preprocessing for both training and inference.

    Accepts a PIL Image or a (image, roi_box) tuple. When an ROI bounding box
    is provided the image is cropped to the box before resizing — this is the
    primary mechanism for eliminating background noise in field images.

    Interface: preprocessor.preprocess(image[, roi_box]) → torch.Tensor
    """

    target_size: Tuple[int, int] = field(default=TARGET_SIZE)
    mean: list = field(default_factory=lambda: list(_IMAGENET_MEAN))
    std: list = field(default_factory=lambda: list(_IMAGENET_STD))

    def __post_init__(self):
        self._normalize = transforms.Normalize(mean=self.mean, std=self.std)

    def preprocess(
        self,
        image: Image.Image,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> torch.Tensor:
        """Preprocess a raw PIL image into a normalised tensor.

        Args:
            image: A PIL Image in any mode. Converted to RGB internally.
            roi_box: Optional (left, upper, right, lower) bounding box in
                pixel coordinates. When provided the image is cropped to
                this region before resizing, isolating the banana ROI.

        Returns:
            Float32 tensor of shape (3, H, W) normalised with ImageNet stats.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image, got {type(image).__name__}")

        image = image.convert("RGB")

        if roi_box is not None:
            left, upper, right, lower = roi_box
            if left >= right or upper >= lower:
                raise ValueError(
                    f"Invalid roi_box {roi_box}: left must be < right and upper < lower"
                )
            image = image.crop(roi_box)

        image = image.resize(self.target_size[::-1], Image.BILINEAR)  # resize wants (W, H)
        tensor = F.to_tensor(image)          # (3, H, W), float32 in [0, 1]
        tensor = self._normalize(tensor)     # ImageNet normalisation
        return tensor
