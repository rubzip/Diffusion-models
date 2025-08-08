from argparse import ArgumentParser

import torch
from torchvision.utils import save_image

from src.utils import load_pretrained_diffusion_unet
from config import SETTINGS_PATH, MODEL_PATH

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate images using a pretrained diffusion model.")
    parser.add_argument("--image_path", type=str, default="imgs/generated_image.png", help="Path to save the generated image.")
    args = parser.parse_args()

    model = load_pretrained_diffusion_unet(MODEL_PATH, SETTINGS_PATH)

    img = model.generate(num_samples=1)
    print(f"Generated image shape: {img.shape}")
    # Saving the generated image as PNG
    save_image(img, args.image_path, normalize=True)
