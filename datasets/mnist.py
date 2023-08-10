import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image

import torchvision.datasets as datasets
from .base import BaseDataset

__all__ = [
    'MNIST',
]

class MNIST(BaseDataset, datasets.MNIST):
    img_size = (28,28)
    ds_pixels = 1024
    ds_channels = 3
    ds_classes = 10
    mean = (0.1307, 0.1307, 0.1307)  # = (self.data / 255).mean()
    std = (0.3081, 0.3081, 0.3081)  # = (self.data / 255).std()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")
        img = img.convert('RGB') # convert to RGB before transforms

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

# run with 'python -m datasets.mnist'
if __name__ == '__main__':
    path_to_data = os.environ['DINO_DATA']
    os.makedirs(path_to_data, exist_ok=True) 
    print(MNIST(root=path_to_data, download=True))


