import os
from torchvision.datasets import VisionDataset
from typing import Optional, Callable

__all__ = ['BaseDataset']

class BaseDataset(VisionDataset):
    img_size = (-1, -1)
    ds_pixels = -1
    ds_channels = -1
    ds_classes = -1
    mean = (float('nan'), float('nan'), float('nan'))   # = (torch.tensor(self.data) / 255).mean(dim=[0,1,2])
    std = (float('nan'), float('nan'), float('nan'))    # = (torch.tensor(self.data) / 255).std(dim=[0,1,2]) 

    def __init__(self, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, **kwargs) -> None:
        defaults = dict(root=os.environ['DINO_DATA'], download=True)
        defaults.update(kwargs)
        super().__init__(train=train, transform=transform, target_transform=target_transform, **defaults)
