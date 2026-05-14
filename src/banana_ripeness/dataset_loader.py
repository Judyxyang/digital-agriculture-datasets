import warnings
from collections import Counter
from typing import Literal

from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

RIPENESS_STAGES = ("unripe", "nearly-ripe", "ripe", "overripe")

Split = Literal["train", "validation", "test"]

_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class _BananaDataset(Dataset):
    def __init__(self, hf_split, transform):
        self._data = hf_split
        self._transform = transform

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        row = self._data[idx]
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        return self._transform(image), row["label"]


def _check_balance(hf_split, split_name: str) -> None:
    counts = Counter(hf_split["label"])
    total = sum(counts.values())
    n_classes = len(RIPENESS_STAGES)
    expected = total / n_classes
    for label_id, count in counts.items():
        ratio = count / expected
        if ratio < 0.7 or ratio > 1.43:
            warnings.warn(
                f"Dataset split '{split_name}' is imbalanced: "
                f"class {label_id} has {count} samples "
                f"(expected ~{expected:.0f}). "
                "Consider resampling before training.",
                UserWarning,
                stacklevel=3,
            )
            return


def load(split: Split, batch_size: int = 32, num_workers: int = 2) -> DataLoader:
    """Load the luischuquimarca/Banana_Ripeness dataset for the given split.

    Args:
        split: One of 'train', 'validation', or 'test'.
        batch_size: Number of samples per batch.
        num_workers: DataLoader worker processes.

    Returns:
        A DataLoader yielding (image_tensor, label) batches.
        Labels are integer indices into RIPENESS_STAGES.
    """
    hf_ds = load_dataset("luischuquimarca/Banana_Ripeness", split=split)

    if split == "train":
        _check_balance(hf_ds, split)
        transform = _TRAIN_TRANSFORM
    else:
        transform = _EVAL_TRANSFORM

    dataset = _BananaDataset(hf_ds, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )
