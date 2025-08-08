import torch
from src.models.diffusion_model import DiffusionModel

def test_get_coefficents():
    model = DiffusionModel(
        denoising_model=torch.nn.Identity(),
        embedding_model=torch.nn.Identity(),
        input_shape=(3, 32, 32),
        num_timesteps=10
    )
    t = torch.tensor([0, 5, 9])
    alpha_bar_t, beta_t, sigma_t = model._get_coefficents(t)
    assert alpha_bar_t.shape == t.shape
    assert beta_t.shape == t.shape
    assert sigma_t.shape == t.shape

def test_forward_noising_step():
    data = torch.ones(2, 3, 32, 32)
    noise = torch.zeros_like(data)
    alpha_bar_t = torch.tensor([1.0, 0.5])
    noised = DiffusionModel.forward_noising_step(data, noise, alpha_bar_t)
    assert noised.shape == data.shape
    # For alpha=1, noised should be data
    assert torch.allclose(noised[0], data[0])
