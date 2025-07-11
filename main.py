import torch

from src.diffusion_model import DiffusionModel
from src.train import train_diffusion_model
from src.sinusoidal_time_embedding import SinusoidalTimeEmbedding
from src.utils import load_pretrained_denoising_unet
from config import NUM_TIMESTEPS, EPOCHS, BETA_MIN, BETA_MAX, EMBEDDING_DIM, INPUT_SHAPE

if __name__ == "__main__":
    denoising_model = load_pretrained_denoising_unet(t_emb_dim=EMBEDDING_DIM)
    embedding_model = SinusoidalTimeEmbedding(embedding_dim=EMBEDDING_DIM, max_length=NUM_TIMESTEPS)
    model = DiffusionModel(
        denoising_model=denoising_model,
        embedding_model=embedding_model,
        input_shape=INPUT_SHAPE,
        num_timesteps=NUM_TIMESTEPS,
        beta_min=BETA_MIN,
        beta_max=BETA_MAX
    )

    model, history = train_diffusion_model(
        model=model,
        dataloader=dataloader,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        num_epochs=EPOCHS,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
