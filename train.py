from pathlib import Path
import torch

from src.models.diffusion_model import DiffusionModel
from src.train_diffusion_model import train_diffusion_model
from src.models.sinusoidal_time_embedding import SinusoidalTimeEmbedding
from src.utils import load_pretrained_denoising_unet, load_emoji, save_model, load_pretrained_diffusion_unet
from config import NUM_TIMESTEPS, EPOCHS, BETA_MIN, BETA_MAX, EMBEDDING_DIM, INPUT_SHAPE, BATCH_SIZE, MODEL_PATH, SETTINGS_PATH

if __name__ == "__main__":
    print("Loading emoji dataset...")
    dataloader = load_emoji(batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"Dataset already loaded, number of samples: {len(dataloader.dataset)}")

    print("Loading pretrained denoising UNet...")
    denoising_model = load_pretrained_denoising_unet(t_emb_dim=EMBEDDING_DIM, in_channels=INPUT_SHAPE[0], out_channels=INPUT_SHAPE[0])
    embedding_model = SinusoidalTimeEmbedding(embedding_dim=EMBEDDING_DIM, max_length=NUM_TIMESTEPS)
    model = DiffusionModel(
        denoising_model=denoising_model,
        embedding_model=embedding_model,
        input_shape=INPUT_SHAPE,
        num_timesteps=NUM_TIMESTEPS,
        beta_min=BETA_MIN,
        beta_max=BETA_MAX
    )
    # Next line loads the pretrained model as a checkpoint if it exists.
    # TODO: Create a cleaner way to handle with doing this (probably argparse)
    model = load_pretrained_diffusion_unet(MODEL_PATH, SETTINGS_PATH)
    print("Diffusion model initialized.")

    print("Starting training...")
    model, history = train_diffusion_model(
        model=model,
        dataloader=dataloader,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        num_epochs=EPOCHS,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print("Training completed.")

    print("Saving model and settings...")
    settings = {
        "num_timesteps": NUM_TIMESTEPS,
        "beta_min": BETA_MIN,
        "beta_max": BETA_MAX,
        "embedding_dim": EMBEDDING_DIM,
        "input_shape": INPUT_SHAPE,
    }
    save_model(model, settings, MODEL_PATH, SETTINGS_PATH)
    print("Model and settings saved.")    
