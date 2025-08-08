import torch
from src.utils import load_cifar10, validate_input_shape, save_model, load_model
from pathlib import Path
import tempfile
import os

def test_load_cifar10():
    dataloader = load_cifar10(batch_size=4)
    batch = next(iter(dataloader))
    assert batch.shape[1:] == (3, 32, 32)

def test_validate_input_shape_pass():
    dataloader = load_cifar10(batch_size=2)
    validate_input_shape(dataloader, (3, 32, 32))

def test_validate_input_shape_fail():
    dataloader = load_cifar10(batch_size=2)
    import pytest
    with pytest.raises(ValueError):
        validate_input_shape(dataloader, (1, 28, 28))

def test_save_and_load_model():
    from torch import nn
    model = nn.Linear(10, 5)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.pth"
        settings_path = Path(tmpdir) / "settings.json"
        save_model(model, {"lr": 0.01}, model_path, settings_path)
        assert model_path.exists()
        assert settings_path.exists()

        # Load state dict to new model
        new_model = torch.nn.Linear(10, 5)
        loaded_model = load_model(new_model, model_path)
        assert isinstance(loaded_model, torch.nn.Module)
