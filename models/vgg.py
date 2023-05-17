# Copy from torchvision.models.vgg (src: https://github.com/pytorch/vision/blob/v0.14.0/torchvision/models/vgg.py). 
# - Include reset_parameters() functions to expose torch.Generator objects from underlying init implementations.
# - Remove functionality for pretrained weights and api logging.
from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn
from dinopl.modules import init

__all__ = [
    #"VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
]

class VGG(nn.Module):
    def __init__(
        self, 
        features: nn.Module, 
        num_classes: Optional[int] = None, 
        #dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = features
        self.embed_dim = 512 # cfgs[:][-1]
        
        self.fc = nn.Identity()
        if num_classes is not None:
            self.fc = nn.Linear(self.embed_dim, num_classes)
            #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            #self.classifier = nn.Sequential(
            #    nn.Linear(512 * 7 * 7, 4096),
            #    nn.ReLU(True),
            #    nn.Dropout(p=dropout),
            #    nn.Linear(4096, 4096),
            #    nn.ReLU(True),
            #    nn.Dropout(p=dropout),
            #    nn.Linear(4096, num_classes))

        self.reset_parameters()       
    
    def reset_parameters(self, mode='fan_out', nonlinearity='relu', generator:torch.Generator=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode=mode, nonlinearity=nonlinearity, generator=generator)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    init.constant_(m.weight, 1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01, generator=generator)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        #x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #x = self.classifier(x)
        x = self.fc(x)
        return x


def make_layers(cfg: List[Union[str, int]], norm_layer: Optional[nn.Module] = None) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if norm_layer is not None:
                layers += [conv2d, norm_layer(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def vgg11(norm_layer: Optional[nn.Module] = None, **kwargs: Any) -> VGG:
    """VGG-11 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        norm_layer (:class:`~torch.nn.Module`, optional): The normalization 
            layer to use. By default, no normalization layers are used.
        **kwargs: parameters passed to the ``models.vgg.VGG`` base class.
    """

    return VGG(make_layers(cfgs['A'], norm_layer=norm_layer), **kwargs)


def vgg11_bn(**kwargs: Any) -> VGG:
    """VGG-11-BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        **kwargs: parameters passed to the ``models.vgg.VGG`` base class.
    """

    return VGG(make_layers(cfgs['A'], norm_layer=nn.BatchNorm2d), **kwargs)


def vgg13(norm_layer: Optional[nn.Module] = None, **kwargs: Any) -> VGG:
    """VGG-13 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        norm_layer (:class:`~torch.nn.Module`, optional): The normalization 
            layer to use. By default, no normalization layers are used.
        **kwargs: parameters passed to the ``models.vgg.VGG`` base class.
    """

    return VGG(make_layers(cfgs['B'], norm_layer=norm_layer), **kwargs)


def vgg13_bn(**kwargs: Any) -> VGG:
    """VGG-13-BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        **kwargs: parameters passed to the ``models.vgg.VGG`` base class.
    """

    return VGG(make_layers(cfgs['B'], norm_layer=nn.BatchNorm2d), **kwargs)


def vgg16(norm_layer: Optional[nn.Module] = None, **kwargs: Any) -> VGG:
    """VGG-16 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        norm_layer (:class:`~torch.nn.Module`, optional): The normalization 
            layer to use. By default, no normalization layers are used.
        **kwargs: parameters passed to the ``models.vgg.VGG`` base class.
    """

    return VGG(make_layers(cfgs['D'], norm_layer=norm_layer), **kwargs)

def vgg16_bn(**kwargs: Any) -> VGG:
    """VGG-16-BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        **kwargs: parameters passed to the ``models.vgg.VGG`` base class.
    """

    return VGG(make_layers(cfgs['D'], norm_layer=nn.BatchNorm2d), **kwargs)


def vgg19(norm_layer: Optional[nn.Module] = None, **kwargs: Any) -> VGG:
    """VGG-19 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        norm_layer (:class:`~torch.nn.Module`, optional): The normalization 
            layer to use. By default, no normalization layers are used.
        **kwargs: parameters passed to the ``models.vgg.VGG`` base class.
    """

    return VGG(make_layers(cfgs['E'], norm_layer=norm_layer), **kwargs)


def vgg19_bn(**kwargs: Any) -> VGG:
    """VGG-19_BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        **kwargs: parameters passed to the ``models.vgg.VGG`` base class.
    """

    return VGG(make_layers(cfgs['E'], norm_layer=nn.BatchNorm2d), **kwargs)
    