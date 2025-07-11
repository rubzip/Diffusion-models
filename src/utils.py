import torch
from .u_net import DenoisingUNet


def load_pretrained_denoising_unet(t_emb_dim: int = 256) -> DenoisingUNet:
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
        in_channels=3, out_channels=3, init_features=32, t_emb_dim=t_emb_dim
    )

    pretrained_state_dict = pretrained_model.state_dict()
    state_dict_to_load = {
        k: v
        for k, v in pretrained_state_dict.items()
        if k in model.state_dict() and v.shape == model.state_dict()[k].shape
    }

    model.load_state_dict(state_dict_to_load, strict=False)
    return model
