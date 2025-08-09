import random
from pathlib import Path
import json

import torch
from torchvision import transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, concatenate_datasets

from .models.u_net import DenoisingUNet
from .models.diffusion_model import DiffusionModel
from .models.sinusoidal_time_embedding import SinusoidalTimeEmbedding


def load_pretrained_denoising_unet(t_emb_dim: int = 256, in_channels: int = 3, out_channels: int = 3) -> DenoisingUNet:
    """Load a pretrained DenoisingUNet model.
    The pretrained model is based on the U-Net architecture and is trained on the Brain Segmentation dataset.
    The output channels of the pretrained model are set to 1, while the new model is set to 3.
    Also the pretrained model does not use time embeddings, while the new model does.
    """
    pretrained_model = torch.hub.load(
        "mateuszbuda/brain-segmentation-pytorch",
        "unet",
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=True,
    )
    model = DenoisingUNet(
        in_channels=in_channels, out_channels=out_channels, init_features=32, t_emb_dim=t_emb_dim
    )

    pretrained_state_dict = pretrained_model.state_dict()
    state_dict_to_load = {
        k: v
        for k, v in pretrained_state_dict.items()
        if k in model.state_dict() and v.shape == model.state_dict()[k].shape
    }

    model.load_state_dict(state_dict_to_load, strict=False)
    return model

class EmojiDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
#        if image.mode == "P":
#            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def load_emoji(batch_size=64, shuffle=True, num_workers=4):
    ds = load_dataset("rubzip/OpenMoji-Emoji-Dataset", "openmoji")
    ds_black = ds["black"]
    ds_color = ds["color"]
    ds_combined = concatenate_datasets([ds_black, ds_color])

    transform = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])

    emoji_dataset = EmojiDataset(ds_combined, transform=transform)
    loader = DataLoader(
        emoji_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return loader


def save_model(
    model: torch.nn.Module, settings: dict, model_path: Path, settings_path: Path
):
    """Save the model state dictionary to a file."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    settings_path = Path(settings_path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=4)


def load_model(model: torch.nn.Module, model_path: Path):
    """Load the model state dictionary from a file."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} does not exist.")

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    return model


def load_json(path: Path) -> dict:
    """Load a JSON file and return its content as a dictionary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file {path} does not exist.")

    with open(path, "r") as f:
        return json.load(f)


def load_pretrained_diffusion_unet(model_path, settings_path) -> DiffusionModel:
    """Load a pretrained DiffusionModel with a DenoisingUNet."""
    settings = load_json(settings_path)
    print(settings)
    num_timesteps, input_shape, embedding_dim, beta_min, beta_max = (
        settings[k]
        for k in (
            "num_timesteps",
            "input_shape",
            "embedding_dim",
            "beta_min",
            "beta_max",
        )
    )

    pretrained_model = torch.load(model_path)
    denoising_model = load_pretrained_denoising_unet(t_emb_dim=embedding_dim, in_channels=input_shape[0], out_channels=input_shape[0])
    embedding_model = SinusoidalTimeEmbedding(
        embedding_dim=embedding_dim, max_length=num_timesteps
    )
    model = DiffusionModel(
        denoising_model=denoising_model,
        embedding_model=embedding_model,
        input_shape=input_shape,
        num_timesteps=num_timesteps,
        beta_min=beta_min,
        beta_max=beta_max,
    )
    model.load_state_dict(pretrained_model, strict=False)
    return model
