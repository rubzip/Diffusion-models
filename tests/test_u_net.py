import torch
from src.models.u_net import DenoisingUNet

def test_unet_forward():
    model = DenoisingUNet(in_channels=3, out_channels=3, init_features=16, t_emb_dim=10)
    x = torch.randn(2, 3, 32, 32)
    t_emb = torch.randn(2, 10)
    output = model(x, t_emb)
    assert output.shape == x.shape
