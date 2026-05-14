from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from banana_ripeness.dataset_loader import load as load_split
from banana_ripeness.evaluator import Evaluator
from banana_ripeness.ripeness_classifier import RipenessClassifier


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 42
    num_workers: int = 2
    checkpoint_dir: str = "checkpoints"
    pretrained: bool = True


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_macro_f1: float
    saved_checkpoint: Optional[str] = None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrainingPipeline:
    """Orchestrates the full offline training workflow.

    Loads the banana ripeness dataset, fine-tunes a RipenessClassifier,
    evaluates after each epoch with the Evaluator, and saves the best
    checkpoint by validation macro F1.

    Usage:
        pipeline = TrainingPipeline(config)
        history = pipeline.run()
    """

    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self._evaluator = Evaluator()

    def run(
        self,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
    ) -> list[EpochResult]:
        """Run the full training loop.

        Args:
            train_loader: Optional pre-built DataLoader for the train split.
                          Loaded from HuggingFace if not provided.
            val_loader:   Optional pre-built DataLoader for the validation split.
                          Loaded from HuggingFace if not provided.

        Returns:
            List of EpochResult, one per epoch.
        """
        cfg = self.config
        _seed_everything(cfg.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_dir = Path(cfg.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if train_loader is None:
            train_loader = load_split("train", batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        if val_loader is None:
            val_loader = load_split("validation", batch_size=cfg.batch_size, num_workers=cfg.num_workers)

        classifier = RipenessClassifier(pretrained=cfg.pretrained, device=str(device))
        model = classifier._model
        model.to(device)

        optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        criterion = nn.CrossEntropyLoss()

        best_f1 = -1.0
        best_path: Optional[str] = None
        history: list[EpochResult] = []

        for epoch in range(1, cfg.epochs + 1):
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            val_f1 = self._validate(model, val_loader, device)

            saved = None
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_path = str(checkpoint_dir / "best.pth")
                torch.save(model.state_dict(), best_path)
                saved = best_path

            result = EpochResult(epoch, train_loss, val_f1, saved)
            history.append(result)
            print(
                f"Epoch {epoch:>3}/{cfg.epochs}  "
                f"loss={train_loss:.4f}  val_f1={val_f1:.4f}"
                + (f"  ✓ saved {saved}" if saved else "")
            )

        return history

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ) -> float:
        model.train()
        total_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        return total_loss / len(loader.dataset)

    def _validate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> float:
        model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())
        report = self._evaluator.evaluate(all_preds, all_labels)
        return report.macro_f1
