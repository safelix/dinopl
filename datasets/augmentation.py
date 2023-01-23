import torch
import torchvision
from typing import Type
from torchvision import transforms
from .base import BaseDataset

__all__ = [
    'flip',
    'crop',
]

def flip(DSet: Type[BaseDataset]):
    return transforms.RandomHorizontalFlip()

def crop(DSet: Type[BaseDataset]):
    return transforms.RandomCrop(DSet.img_size, 4)

