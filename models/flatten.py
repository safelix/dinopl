import torch
from torch import nn

__all__ = [
    #"Flatten",
    "flatten",
]

class Flatten(nn.Module):
    def __init__(self, n_pixels, n_channels, num_classes=None) -> None:
        super().__init__()
        self.embed_dim = n_pixels * n_channels

        self.fc = nn.Identity()
        if num_classes is not None:
            self.fc = nn.Linear(self.embed_dim, num_classes)

    def reset_parameters(self, generator=None):
        pass

    def forward(self, x):
        return torch.flatten(x, start_dim=1, end_dim=-1)

def flatten(n_pixels, n_channels, **kwargs) -> Flatten:
    return Flatten(n_pixels, n_channels, **kwargs)