from typing import Optional, Type
import torch.nn as nn
import torch.nn.functional as F
import torch
from math import floor
from . import init


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
        self.fc = nn.Identity()
        
        if num_classes is not None:
            self.fc = nn.Linear(self.embed_dim, num_classes)
        self.embed_dim:int = width * 4

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


def convnet(**kwargs):
    r"""Simple ConvNet with constant width and specified depth.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(**kwargs)