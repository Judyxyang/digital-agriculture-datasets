import time
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from banana_ripeness.inference_pipeline import InferencePipeline, Prediction, DEFAULT_LOW_CONFIDENCE_THRESHOLD
from banana_ripeness.ripeness_classifier import RIPENESS_STAGES


def _random_image(size=(224, 224)) -> Image.Image:
    arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _solid_image(color=(128, 200, 50), size=(224, 224)) -> Image.Image:
    return Image.new("RGB", size, color)


@pytest.fixture
def pipeline():
    """InferencePipeline with random weights — no checkpoint or network needed."""
    return InferencePipeline(pretrained=False)


@pytest.fixture
def checkpoint_path(tmp_path):
    """A valid checkpoint file saved from a random-weight model."""
    from banana_ripeness.ripeness_classifier import RipenessClassifier
    clf = RipenessClassifier(pretrained=False)
    path = tmp_path / "test.pth"
    torch.save(clf.state_dict(), path)
    return path


class TestPredictionOutput:
    def test_returns_prediction_dataclass(self, pipeline):
        result = pipeline.predict(_random_image())
        assert isinstance(result, Prediction)

    def test_stage_is_valid_ripeness_stage(self, pipeline):
        result = pipeline.predict(_random_image())
        assert result.stage in RIPENESS_STAGES

    def test_confidence_in_unit_interval(self, pipeline):
        result = pipeline.predict(_random_image())
        assert 0.0 <= result.confidence <= 1.0

    def test_low_confidence_is_bool(self, pipeline):
        result = pipeline.predict(_random_image())
        assert isinstance(result.low_confidence, bool)

    def test_str_contains_stage_and_confidence(self, pipeline):
        result = pipeline.predict(_solid_image())
        text = str(result)
        assert result.stage in text
        assert "%" in text


class TestLowConfidenceFlag:
    def test_high_confidence_not_flagged(self):
        pipeline = InferencePipeline(pretrained=False, low_confidence_threshold=0.0)
        result = pipeline.predict(_random_image())
        assert result.low_confidence is False

    def test_low_confidence_flagged_when_threshold_is_one(self):
        pipeline = InferencePipeline(pretrained=False, low_confidence_threshold=1.0)
        result = pipeline.predict(_random_image())
        assert result.low_confidence is True

    def test_custom_threshold_respected(self):
        pipeline = InferencePipeline(pretrained=False, low_confidence_threshold=0.99)
        result = pipeline.predict(_random_image())
        expected = result.confidence < 0.99
        assert result.low_confidence == expected

    def test_low_confidence_str_contains_warning(self):
        pipeline = InferencePipeline(pretrained=False, low_confidence_threshold=1.0)
        result = pipeline.predict(_random_image())
        assert "low confidence" in str(result)

    def test_default_threshold_value(self):
        assert DEFAULT_LOW_CONFIDENCE_THRESHOLD == 0.6


class TestCheckpointLoading:
    def test_loads_checkpoint_and_predicts(self, checkpoint_path):
        pipeline = InferencePipeline(checkpoint_path=checkpoint_path)
        result = pipeline.predict(_random_image())
        assert result.stage in RIPENESS_STAGES

    def test_checkpoint_predictions_are_consistent(self, checkpoint_path):
        pipeline = InferencePipeline(checkpoint_path=checkpoint_path)
        img = _random_image()
        r1 = pipeline.predict(img)
        r2 = pipeline.predict(img)
        assert r1.stage == r2.stage
        assert abs(r1.confidence - r2.confidence) < 1e-5


class TestROIPassthrough:
    def test_predict_with_roi(self, pipeline):
        img = _random_image(size=(320, 240))
        result = pipeline.predict(img, roi_box=(20, 20, 280, 200))
        assert result.stage in RIPENESS_STAGES
        assert 0.0 <= result.confidence <= 1.0


class TestInputFormats:
    def test_rgba_image_accepted(self, pipeline):
        rgba = Image.new("RGBA", (224, 224), (100, 150, 200, 255))
        result = pipeline.predict(rgba)
        assert result.stage in RIPENESS_STAGES

    def test_grayscale_image_accepted(self, pipeline):
        gray = Image.new("L", (224, 224), 128)
        result = pipeline.predict(gray)
        assert result.stage in RIPENESS_STAGES

    def test_non_square_image_accepted(self, pipeline):
        img = _random_image(size=(480, 320))
        result = pipeline.predict(img)
        assert result.stage in RIPENESS_STAGES


class TestLatency:
    def test_inference_under_500ms(self, pipeline):
        img = _random_image()
        pipeline.predict(img)  # warm up
        start = time.perf_counter()
        pipeline.predict(img)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 500, f"Inference took {elapsed_ms:.1f}ms — exceeds 500ms budget"
