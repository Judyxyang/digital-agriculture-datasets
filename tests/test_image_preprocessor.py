import pytest
import torch
import numpy as np
from PIL import Image

from banana_ripeness.image_preprocessor import ImagePreprocessor, TARGET_SIZE, _IMAGENET_MEAN, _IMAGENET_STD


def _solid_image(color=(128, 200, 50), size=(320, 240)) -> Image.Image:
    """Return a solid-colour RGB PIL image."""
    return Image.new("RGB", size, color)


def _random_image(size=(320, 240)) -> Image.Image:
    arr = np.random.randint(0, 256, (*size[::-1], 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def preprocessor():
    return ImagePreprocessor()


class TestOutputShape:
    def test_output_shape_is_3_h_w(self, preprocessor):
        tensor = preprocessor.preprocess(_solid_image())
        assert tensor.shape == (3, *TARGET_SIZE)

    def test_output_dtype_is_float32(self, preprocessor):
        tensor = preprocessor.preprocess(_solid_image())
        assert tensor.dtype == torch.float32

    def test_output_shape_with_roi(self, preprocessor):
        img = _solid_image(size=(320, 240))
        tensor = preprocessor.preprocess(img, roi_box=(10, 10, 200, 180))
        assert tensor.shape == (3, *TARGET_SIZE)

    def test_custom_target_size(self):
        p = ImagePreprocessor(target_size=(128, 128))
        tensor = p.preprocess(_solid_image())
        assert tensor.shape == (3, 128, 128)


class TestNormalisation:
    def test_pixel_values_in_normalised_range(self, preprocessor):
        tensor = preprocessor.preprocess(_random_image())
        # After ImageNet normalisation values are typically in [-3, 3]
        assert tensor.min().item() >= -4.0
        assert tensor.max().item() <= 4.0

    def test_solid_white_normalised_correctly(self, preprocessor):
        white = Image.new("RGB", (64, 64), (255, 255, 255))
        tensor = preprocessor.preprocess(white)
        # White pixel (1.0) normalised: (1.0 - mean) / std
        expected_ch0 = (1.0 - _IMAGENET_MEAN[0]) / _IMAGENET_STD[0]
        assert abs(tensor[0, 0, 0].item() - expected_ch0) < 1e-4

    def test_solid_black_normalised_correctly(self, preprocessor):
        black = Image.new("RGB", (64, 64), (0, 0, 0))
        tensor = preprocessor.preprocess(black)
        # Black pixel (0.0) normalised: (0.0 - mean) / std
        expected_ch0 = (0.0 - _IMAGENET_MEAN[0]) / _IMAGENET_STD[0]
        assert abs(tensor[0, 0, 0].item() - expected_ch0) < 1e-4


class TestROIExtraction:
    def test_roi_crop_changes_content(self, preprocessor):
        # Left half red, right half blue
        img = Image.new("RGB", (200, 100), (255, 0, 0))
        blue_patch = Image.new("RGB", (100, 100), (0, 0, 255))
        img.paste(blue_patch, (100, 0))

        tensor_full = preprocessor.preprocess(img)
        tensor_roi = preprocessor.preprocess(img, roi_box=(100, 0, 200, 100))

        # ROI crops the blue half — tensors must differ
        assert not torch.allclose(tensor_full, tensor_roi)

    def test_roi_output_shape_unchanged(self, preprocessor):
        img = _solid_image(size=(400, 300))
        tensor = preprocessor.preprocess(img, roi_box=(50, 30, 350, 270))
        assert tensor.shape == (3, *TARGET_SIZE)

    def test_invalid_roi_raises(self, preprocessor):
        img = _solid_image()
        with pytest.raises(ValueError, match="Invalid roi_box"):
            preprocessor.preprocess(img, roi_box=(100, 50, 50, 200))  # left > right

    def test_invalid_roi_upper_lower_raises(self, preprocessor):
        img = _solid_image()
        with pytest.raises(ValueError, match="Invalid roi_box"):
            preprocessor.preprocess(img, roi_box=(10, 200, 100, 50))  # upper > lower


class TestInputHandling:
    def test_rgba_image_converted_to_rgb(self, preprocessor):
        rgba = Image.new("RGBA", (64, 64), (100, 150, 200, 128))
        tensor = preprocessor.preprocess(rgba)
        assert tensor.shape == (3, *TARGET_SIZE)

    def test_grayscale_image_converted_to_rgb(self, preprocessor):
        gray = Image.new("L", (64, 64), 128)
        tensor = preprocessor.preprocess(gray)
        assert tensor.shape == (3, *TARGET_SIZE)

    def test_non_image_raises_type_error(self, preprocessor):
        with pytest.raises(TypeError, match="Expected PIL.Image"):
            preprocessor.preprocess(np.zeros((224, 224, 3), dtype=np.uint8))

    def test_no_roi_returns_tensor(self, preprocessor):
        tensor = preprocessor.preprocess(_solid_image())
        assert isinstance(tensor, torch.Tensor)
