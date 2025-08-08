import torch
from src.models.sinusoidal_time_embedding import SinusoidalTimeEmbedding

def test_forward_shape():
    embedding = SinusoidalTimeEmbedding(embedding_dim=16, max_length=100)
    t = torch.tensor([0, 1, 50, 99])
    output = embedding(t)
    assert output.shape[1] == len(t)
    assert output.shape[2] == 16
