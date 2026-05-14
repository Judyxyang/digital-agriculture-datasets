import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from banana_ripeness.dataset_loader import (
    RIPENESS_STAGES,
    Split,
    _BananaDataset,
    _check_balance,
    _EVAL_TRANSFORM,
    _TRAIN_TRANSFORM,
    load,
)


def _make_hf_split(n_per_class: int = 10, n_classes: int = 4) -> MagicMock:
    """Return a mock HuggingFace dataset split with synthetic banana images."""
    rows = []
    for label in range(n_classes):
        for _ in range(n_per_class):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            rows.append({"image": img, "label": label})

    columns = {key: [r[key] for r in rows] for key in rows[0]}

    def getitem(self, idx):
        if isinstance(idx, str):
            return columns[idx]
        return rows[idx]

    mock = MagicMock()
    mock.__len__ = lambda self: len(rows)
    mock.__getitem__ = getitem
    mock.__iter__ = lambda self: iter(rows)
    return mock


class TestBananaDataset:
    def test_returns_tensor_and_label(self):
        hf_split = _make_hf_split(n_per_class=2)
        ds = _BananaDataset(hf_split, _EVAL_TRANSFORM)
        tensor, label = ds[0]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        assert isinstance(label, int)

    def test_pixel_values_normalised(self):
        hf_split = _make_hf_split(n_per_class=2)
        ds = _BananaDataset(hf_split, _EVAL_TRANSFORM)
        tensor, _ = ds[0]
        # After ImageNet normalisation values are typically in [-3, 3]
        assert tensor.min() >= -4.0
        assert tensor.max() <= 4.0

    def test_length_matches_input(self):
        hf_split = _make_hf_split(n_per_class=5)
        ds = _BananaDataset(hf_split, _EVAL_TRANSFORM)
        assert len(ds) == 20  # 5 per class × 4 classes


class TestCheckBalance:
    def test_balanced_dataset_raises_no_warning(self):
        hf_split = _make_hf_split(n_per_class=10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _check_balance(hf_split, "train")
        assert len(caught) == 0

    def test_imbalanced_dataset_warns(self):
        # Build a heavily skewed split: 37 unripe, 1 each of the rest
        hf_split = _make_hf_split(n_per_class=1)  # 4 total, then override
        labels = [0] * 37 + [1] * 1 + [2] * 1 + [3] * 1

        def getitem(self, idx):
            if isinstance(idx, str):
                return labels
            return {"image": Image.new("RGB", (64, 64)), "label": labels[idx]}

        hf_split.__len__ = lambda self: len(labels)
        hf_split.__getitem__ = getitem

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _check_balance(hf_split, "train")
        assert len(caught) == 1
        assert "imbalanced" in str(caught[0].message).lower()


class TestLoad:
    @patch("banana_ripeness.dataset_loader.load_dataset")
    def test_train_split_uses_augmentation(self, mock_load_dataset):
        mock_load_dataset.return_value = _make_hf_split(n_per_class=4)
        loader = load("train", batch_size=8, num_workers=0)
        assert loader.dataset._transform is _TRAIN_TRANSFORM

    @patch("banana_ripeness.dataset_loader.load_dataset")
    def test_validation_split_uses_eval_transform(self, mock_load_dataset):
        mock_load_dataset.return_value = _make_hf_split(n_per_class=4)
        loader = load("validation", batch_size=8, num_workers=0)
        assert loader.dataset._transform is _EVAL_TRANSFORM

    @patch("banana_ripeness.dataset_loader.load_dataset")
    def test_test_split_uses_eval_transform(self, mock_load_dataset):
        mock_load_dataset.return_value = _make_hf_split(n_per_class=4)
        loader = load("test", batch_size=8, num_workers=0)
        assert loader.dataset._transform is _EVAL_TRANSFORM

    @patch("banana_ripeness.dataset_loader.load_dataset")
    def test_returns_correct_batch_shape(self, mock_load_dataset):
        mock_load_dataset.return_value = _make_hf_split(n_per_class=4)
        loader = load("train", batch_size=8, num_workers=0)
        images, labels = next(iter(loader))
        assert images.shape == (8, 3, 224, 224)
        assert labels.shape == (8,)

    @patch("banana_ripeness.dataset_loader.load_dataset")
    def test_labels_are_valid_class_indices(self, mock_load_dataset):
        mock_load_dataset.return_value = _make_hf_split(n_per_class=4)
        loader = load("train", batch_size=16, num_workers=0)
        _, labels = next(iter(loader))
        assert labels.min() >= 0
        assert labels.max() < len(RIPENESS_STAGES)

    @patch("banana_ripeness.dataset_loader.load_dataset")
    def test_train_split_shuffles(self, mock_load_dataset):
        mock_load_dataset.return_value = _make_hf_split(n_per_class=10)
        loader = load("train", batch_size=40, num_workers=0)
        assert loader.sampler is not None  # shuffle=True uses RandomSampler

    @patch("banana_ripeness.dataset_loader.load_dataset")
    def test_dataset_is_balanced_across_classes(self, mock_load_dataset):
        mock_load_dataset.return_value = _make_hf_split(n_per_class=10)
        loader = load("train", batch_size=40, num_workers=0)
        _, labels = next(iter(loader))
        counts = torch.bincount(labels, minlength=len(RIPENESS_STAGES))
        # Each class should appear exactly 10 times in 40-sample batch
        assert (counts == 10).all()
