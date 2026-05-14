import pytest
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from banana_ripeness.ripeness_classifier import RipenessClassifier, RIPENESS_STAGES


def _random_image(size=(224, 224)) -> Image.Image:
    arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _tiny_backbone() -> nn.Module:
    """4-class linear classifier over a flattened 3×224×224 input — fast, no pretrained weights."""
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(1280, len(RIPENESS_STAGES)),
    )


@pytest.fixture
def classifier():
    """RipenessClassifier with random-weight MobileNetV2 (no pretrained download)."""
    return RipenessClassifier(pretrained=False)


class TestPredictOutput:
    def test_returns_valid_ripeness_stage(self, classifier):
        stage, _ = classifier.predict(_random_image())
        assert stage in RIPENESS_STAGES

    def test_confidence_in_unit_interval(self, classifier):
        _, confidence = classifier.predict(_random_image())
        assert 0.0 <= confidence <= 1.0

    def test_returns_tuple_of_two(self, classifier):
        result = classifier.predict(_random_image())
        assert len(result) == 2

    def test_stage_is_string(self, classifier):
        stage, _ = classifier.predict(_random_image())
        assert isinstance(stage, str)

    def test_confidence_is_float(self, classifier):
        _, confidence = classifier.predict(_random_image())
        assert isinstance(confidence, float)


class TestBatchBehaviour:
    def test_consistent_predictions_on_same_image(self, classifier):
        img = _random_image()
        stage1, conf1 = classifier.predict(img)
        stage2, conf2 = classifier.predict(img)
        assert stage1 == stage2
        assert abs(conf1 - conf2) < 1e-5

    def test_multiple_images_all_return_valid_stages(self, classifier):
        for _ in range(5):
            stage, conf = classifier.predict(_random_image())
            assert stage in RIPENESS_STAGES
            assert 0.0 <= conf <= 1.0


class TestROIPassthrough:
    def test_predict_with_roi_returns_valid_stage(self, classifier):
        img = _random_image(size=(320, 240))
        stage, conf = classifier.predict(img, roi_box=(20, 20, 280, 200))
        assert stage in RIPENESS_STAGES
        assert 0.0 <= conf <= 1.0


class TestBackboneSwap:
    def test_custom_backbone_is_used(self):
        """A tiny backbone with a compatible head should work end-to-end."""
        backbone = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3, len(RIPENESS_STAGES)),
        )
        # Wrap in a minimal module that accepts (B, 3, H, W)
        class _Wrapper(nn.Module):
            def forward(self, x):
                x = x.mean(dim=[2, 3])          # (B, 3)
                return nn.Linear(3, 4)(x)        # (B, 4)

        clf = RipenessClassifier(backbone=_Wrapper())
        stage, conf = clf.predict(_random_image())
        assert stage in RIPENESS_STAGES
        assert 0.0 <= conf <= 1.0

    def test_state_dict_returns_dict(self, classifier):
        sd = classifier.state_dict()
        assert isinstance(sd, dict)
        assert len(sd) > 0
