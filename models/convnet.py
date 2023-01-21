from typing import Optional, Type
from math import floor
import torch.nn as nn
import torch
from dinopl.modules import init

__all__ = [
    "ConvNet",
    "convnet_16_1",
    "convnet_16_1e",
    "convnet_16_2",
    "convnet_16_2e",
    "convnet_32_1",
    "convnet_32_1e",
    "convnet_32_2",
    "convnet_32_2e",
]


class ConvNet(nn.Module):
    def __init__(self, 
            width:int = 16, 
            depth:int = 2, 
            num_classes:Optional[int] = 10, 
            norm_layer:Type[nn.Module] = nn.BatchNorm2d,
    ) -> None :
        super().__init__()
        tail = str(depth).endswith('.5')
        depth = floor(depth)
        
        # make stem
        self.conv1 = nn.Conv2d(3, out_channels=width, kernel_size=3, padding='same')
        self.bn1 = nn.Identity()
        self.relu1 = nn.ReLU()
    
        module_list = []
        if depth > 1 or tail:
            module_list.append(nn.MaxPool2d(2, 2))

        for _ in range(depth - 1):
            module_list.append(nn.Conv2d(width, out_channels=width, kernel_size=3, padding='same'))
            module_list.append(norm_layer(width))
            module_list.append(nn.ReLU())

        if tail:
            module_list.append(nn.Conv2d(width, out_channels=width, kernel_size=3, padding='same'))
        
        self.block = nn.Sequential(*module_list)

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.embed_dim:int = width * 4
        
        self.fc = nn.Identity()
        if num_classes is not None:
            self.fc = nn.Linear(self.embed_dim, num_classes)

        self.reset_parameters()
    
    def reset_parameters(self, mode='fan_out', nonlinearity='relu', generator:torch.Generator=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode=mode, nonlinearity=nonlinearity, generator=generator)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc(x)
        return x


###################################################################################################
# WIDTH 16
###################################################################################################

def convnet_16_1(**kwargs):
    r"""Simple ConvNet with constant width 16 and depth 1.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=16, depth=1, **kwargs)


def convnet_16_1e(**kwargs):
    r"""Simple ConvNet with constant width 16 and depth 1.5.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=16, depth=1.5, **kwargs)


def convnet_16_2(**kwargs):
    r"""Simple ConvNet with constant width 16 and depth 2.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=16, depth=2, **kwargs)


def convnet_16_2e(**kwargs):
    r"""Simple ConvNet with constant width 16 and depth 2.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=16, depth=2.5, **kwargs)


###################################################################################################
# WIDTH 32
###################################################################################################

def convnet_32_1(**kwargs):
    r"""Simple ConvNet with constant width 16 and depth 1.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=32, depth=1, **kwargs)


def convnet_32_1e(**kwargs):
    r"""Simple ConvNet with constant width 16 and depth 1.5.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=32, depth=1.5, **kwargs)


def convnet_32_2(**kwargs):
    r"""Simple ConvNet with constant width 16 and depth 2.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=32, depth=2, **kwargs)


def convnet_32_2e(**kwargs):
    r"""Simple ConvNet with constant width 16 and depth 2.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=32, depth=2.5, **kwargs)