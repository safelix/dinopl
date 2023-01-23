import torch
import torchvision
from typing import Type
from torchvision import transforms
from .base import BaseDataset

__all__ = [
    'hflip',
    'padcrop',
]

# Do not use underscore to specify augmentation! Underscore is used to concat names for parsing.

def hflip(DSet: Type[BaseDataset]):
    return transforms.RandomHorizontalFlip()

def padcrop(DSet: Type[BaseDataset]):
    return transforms.RandomCrop(DSet.img_size, 4)
