import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from banana_ripeness.training_pipeline import TrainingConfig, TrainingPipeline


def _synthetic_loader(n_samples: int = 40, batch_size: int = 8) -> DataLoader:
    """Return a DataLoader with random (3,224,224) images and 4-class labels."""
    images = torch.randn(n_samples, 3, 224, 224)
    labels = torch.randint(0, 4, (n_samples,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False)


@pytest.fixture
def cfg(tmp_path):
    return TrainingConfig(
        epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        seed=0,
        num_workers=0,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        pretrained=False,
    )


@pytest.fixture
def loaders():
    return _synthetic_loader(), _synthetic_loader(n_samples=16)


class TestRunReturnsHistory:
    def test_history_length_matches_epochs(self, cfg, loaders):
        train, val = loaders
        pipeline = TrainingPipeline(cfg)
        history = pipeline.run(train_loader=train, val_loader=val)
        assert len(history) == cfg.epochs

    def test_epoch_numbers_are_sequential(self, cfg, loaders):
        train, val = loaders
        history = TrainingPipeline(cfg).run(train_loader=train, val_loader=val)
        assert [r.epoch for r in history] == list(range(1, cfg.epochs + 1))

    def test_train_loss_is_positive(self, cfg, loaders):
        train, val = loaders
        history = TrainingPipeline(cfg).run(train_loader=train, val_loader=val)
        for r in history:
            assert r.train_loss >= 0.0

    def test_val_f1_in_unit_interval(self, cfg, loaders):
        train, val = loaders
        history = TrainingPipeline(cfg).run(train_loader=train, val_loader=val)
        for r in history:
            assert 0.0 <= r.val_macro_f1 <= 1.0


class TestCheckpointing:
    def test_best_checkpoint_file_created(self, cfg, loaders):
        train, val = loaders
        TrainingPipeline(cfg).run(train_loader=train, val_loader=val)
        assert Path(cfg.checkpoint_dir, "best.pth").exists()

    def test_checkpoint_is_loadable_state_dict(self, cfg, loaders):
        train, val = loaders
        TrainingPipeline(cfg).run(train_loader=train, val_loader=val)
        state = torch.load(
            Path(cfg.checkpoint_dir, "best.pth"), map_location="cpu"
        )
        assert isinstance(state, dict)
        assert len(state) > 0

    def test_at_least_one_epoch_marks_saved(self, cfg, loaders):
        train, val = loaders
        history = TrainingPipeline(cfg).run(train_loader=train, val_loader=val)
        saved = [r for r in history if r.saved_checkpoint is not None]
        assert len(saved) >= 1

    def test_checkpoint_dir_created_if_missing(self, tmp_path, loaders):
        train, val = loaders
        deep_dir = str(tmp_path / "a" / "b" / "checkpoints")
        cfg = TrainingConfig(epochs=1, num_workers=0, pretrained=False, checkpoint_dir=deep_dir)
        TrainingPipeline(cfg).run(train_loader=train, val_loader=val)
        assert Path(deep_dir, "best.pth").exists()


class TestReproducibility:
    def test_same_seed_same_first_loss(self, tmp_path, loaders):
        train, val = loaders
        cfg1 = TrainingConfig(epochs=1, seed=7, num_workers=0, pretrained=False,
                              checkpoint_dir=str(tmp_path / "run1"))
        cfg2 = TrainingConfig(epochs=1, seed=7, num_workers=0, pretrained=False,
                              checkpoint_dir=str(tmp_path / "run2"))
        h1 = TrainingPipeline(cfg1).run(train_loader=train, val_loader=val)
        h2 = TrainingPipeline(cfg2).run(train_loader=train, val_loader=val)
        assert abs(h1[0].train_loss - h2[0].train_loss) < 1e-4
