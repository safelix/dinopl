import os
from typing import Callable, Optional

import torchvision.datasets as datasets

__all__ = [
    'CIFAR10',
    'CIFAR100'
]

class CIFAR10(datasets.CIFAR10):
    size = (32, 32)
    ds_pixels = 1024
    ds_channels = 3
    ds_classes = 10
    mean = (0.4914, 0.4822, 0.4465) # = (torch.tensor(self.data) / 255).mean(dim=[0,1,2])
    std = (0.247, 0.243, 0.261)     # = (torch.tensor(self.data) / 255).std(dim=[0,1,2]) 

    def __init__(self, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, **kwargs) -> None:
        defaults = dict(root=os.environ['DINO_DATA'], download=True)
        defaults.update(kwargs)
        super().__init__(train=train, transform=transform, target_transform=target_transform, **defaults)


class CIFAR100(datasets.CIFAR100):
    img_size = (32, 32)
    ds_pixels = 1024
    ds_channels = 3
    ds_classes = 10
    mean = (0.4914, 0.4822, 0.4465) # = (torch.tensor(self.data) / 255).mean(dim=[0,1,2])
    std = (0.247, 0.243, 0.261)     # = (torch.tensor(self.data) / 255).std(dim=[0,1,2]) 

    def __init__(self, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, **kwargs) -> None:
        defaults = dict(root=os.environ['DINO_DATA'], download=True)
        defaults.update(kwargs)
        super().__init__(train=train, transform=transform, target_transform=target_transform, **defaults)


if __name__ == '__main__':
    path_to_data = os.environ['DINO_DATA']
    os.makedirs(path_to_data, exist_ok=True) 
    CIFAR10(root=path_to_data, download=True)
    pass