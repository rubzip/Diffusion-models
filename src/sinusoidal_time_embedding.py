import torch
import torch.nn as nn

from .diffusion_model import TimeEmbedding, DenoisingModel

class SinusoidalTimeEmbedding(TimeEmbedding):
    def __init__(self, embedding_dim: int, max_length: int = 1_000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_length = max_length

        precomputed_encodings = torch.zeros(max_length, embedding_dim)
        positions = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(max_length)) / embedding_dim))

        precomputed_encodings[:, 0::2] = torch.sin(positions * div_term)
        precomputed_encodings[:, 1::2] = torch.cos(positions * div_term)
        self.precomputed_encodings = nn.Parameter(precomputed_encodings, requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        assert t.dim() == 1, "Input tensor must be 1-dimensional"
        return self.precomputed_encodings[t.long() % self.max_length].unsqueeze(0)
