import torch
from torch import nn

__all__ = [
    "Flatten"
]

class Flatten(nn.Module):
    def __init__(self, n_pixels, n_channels) -> None:
        super().__init__()
        self.embed_dim = n_pixels * n_channels

    def reset_parameters(self, generator=None):
        pass

    def forward(self, x):
        return torch.flatten(x, start_dim=1, end_dim=-1)