import os
from typing import Callable, Optional

import torchvision.datasets as datasets

__all__ = [
    'MNIST',
]

class MNIST(datasets.MNIST):
    img_size = (28,28)
    ds_pixels = 1024
    ds_channels = 3
    ds_classes = 10
    mean = (0.1307, 0.1307, 0.1307)  # = (self.data / 255).mean()
    std = (0.3081, 0.3081, 0.3081)  # = (self.data / 255).std()

    def __init__(self, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, **kwargs) -> None:
        defaults = dict(root=os.environ['DINO_DATA'], download=True)
        defaults.update(kwargs)
        super().__init__(train=train, transform=transform, target_transform=target_transform, **defaults)

if __name__ == '__main__':
    path_to_data = os.environ['DINO_DATA']
    os.makedirs(path_to_data, exist_ok=True) 
    MNIST(root=path_to_data, download=True)


