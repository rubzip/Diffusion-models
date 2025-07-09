from .diffusion_model import DiffusionModel
import torch

def forward_noising_step(data: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, min_beta: float = 0.01, max_beta: float = 0.2, t_steps: int = 1000) -> torch.Tensor:
    """
    Forward noising steps.

    Args:
        data (torch.Tensor): Input data.
        noise (torch.Tensor): Input error.
        t (torch.Tensor): Time step tensor.

    Returns:
        torch.Tensor: Output tensor after applying the diffusion process.
    """
    betas = torch.linspace(min_beta, max_beta, t_steps)  # shape: [t_steps]

    # 2. Compute alpha and alpha_bar (cumulative product)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)  # shape: [t_steps]

    # 3. Gather alpha_bar_t for each sample in batch
    alpha_bar_t = alphas_bar[t].view(-1, 1, 1, 1)  # reshape for broadcasting

    # 4. Apply the forward noising formula
    noised_data = torch.sqrt(alpha_bar_t) * data + torch.sqrt(1.0 - alpha_bar_t) * noise
    return noised_data


def train_diffusion_model(model: DiffusionModel, dataloader, optimizer, num_epochs: int = 10, device: str = 'cpu'):
    model.train()
    model.to(device)

    history = {"loss": []}
    for epoch in range(num_epochs):
        total_loss = 0.0
        for data, _ in dataloader:
            data = data.to(device)
            optimizer.zero_grad()

            # Sample random time steps
            t = torch.randint(0, model.num_timesteps, (data.size(0),), device=device)

            # Forward pass through the diffusion model
            noise = torch.randn_like(data)
            noised_data = forward_noising_step(data, noise, t, t_steps=model.num_timesteps).to(device)
            predicted_noise = model(data, t)

            # Compute loss (e.g., MSE loss)
            loss = torch.nn.functional.mse_loss(noised_data, predicted_noise)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        history['loss'].append(avg_loss)
    return model