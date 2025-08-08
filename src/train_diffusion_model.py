from .models.diffusion_model import DiffusionModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer


def train_diffusion_model(model: DiffusionModel, dataloader: DataLoader, optimizer: Optimizer, num_epochs: int = 10, device: str = 'cpu'):
    model.train()
    model.to(device)

    history = {"loss": []}
    for epoch in range(num_epochs):
        total_loss = 0.0
        for data, _ in dataloader:
            data = data.to(device)
            optimizer.zero_grad()

            # Generating random time steps
            t = torch.randint(0, model.num_timesteps, (data.size(0),), device=device)
            alpha_bar_t, _, _ = model._get_coefficents(t)

            # Forward pass through the diffusion model
            noise = torch.randn_like(data)
            noised_data = model.forward_noising_step(data, noise, alpha_bar_t).to(device)
            predicted_noise = model(noised_data, t)

            # Simplified Loss Function makes the MSE between the predicted noise and the actual noise
            loss = F.mse_loss(noise, predicted_noise)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f}')
        history['loss'].append(avg_loss)
    return model, history
