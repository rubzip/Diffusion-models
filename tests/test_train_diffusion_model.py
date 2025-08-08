import torch
from torch.utils.data import DataLoader, TensorDataset
from src.train_diffusion_model import train_diffusion_model
from src.models.diffusion_model import DiffusionModel, DenoisingModel, TimeEmbedding
import pytest


class DummyDenoisingModel(DenoisingModel):
    def forward(self, x, t_emb):
        return torch.zeros_like(x)

class DummyTimeEmbedding(TimeEmbedding):
    def forward(self, t):
        return torch.zeros(t.shape[0], 10)

@pytest.fixture
def dummy_model():
    denoising_model = DummyDenoisingModel()
    embedding_model = DummyTimeEmbedding()
    model = DiffusionModel(
        denoising_model=denoising_model,
        embedding_model=embedding_model,
        input_shape=(3, 32, 32),
        num_timesteps=10,
    )
    return model

@pytest.fixture
def dummy_dataloader():
    data = torch.randn(8, 3, 32, 32)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=4)

def test_train_runs(dummy_model, dummy_dataloader):
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=1e-3)
    model, history = train_diffusion_model(dummy_model, dummy_dataloader, optimizer, num_epochs=1)
    assert "loss" in history
    assert len(history["loss"]) == 1
    assert history["loss"][0] >= 0
