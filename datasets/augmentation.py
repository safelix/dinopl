import torch
import torchvision
from typing import Type
from torchvision import transforms
from .base import BaseDataset

__all__ = [
    'hflip',
    'padcrop',
    'dino'
]

# Do not use underscore to specify augmentation! Underscore is used to concat names for parsing.

def hflip(DSet: Type[BaseDataset]):
    return transforms.RandomHorizontalFlip()

def padcrop(DSet: Type[BaseDataset]):
    return transforms.RandomCrop(DSet.img_size, 4)

def dino(DSet: Type[BaseDataset]):
    '''From DINO_v2 but it's the same as DINO_v1: 
    https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/augmentations.py
    '''
    class GaussianBlur(transforms.RandomApply):
        '''From DINO_v2: but I don't agree that p needs to be inverted'''
        def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
            transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
            super().__init__(transforms=[transform], p=p)


    flip_and_color_jitter = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
    ])
    
    # Don't apply gaussian blur to images of size < 64 (exclude MNIST, CIFAR)
    apply_blur = DSet.img_size[0] >= 64

    global_transfo1 = transforms.Compose([
        flip_and_color_jitter, 
        GaussianBlur(p=1.0 if apply_blur else 0)
    ])
    global_transfo2 = transforms.Compose([
        flip_and_color_jitter,
        GaussianBlur(p=0.1 if apply_blur else 0),
        transforms.RandomSolarize(threshold=128, p=0.2)
    ])

    local_transfo = transforms.Compose([
        flip_and_color_jitter,
        GaussianBlur(p=0.5 if apply_blur else 0)
    ])
    
    # WARNING: the transforms are chosen randomly and applied
    # to global and local crops indepenently.
    return transforms.RandomChoice(
        [global_transfo1, global_transfo2, local_transfo],
        p=[0.4, 0.4, 0.2] 
    )