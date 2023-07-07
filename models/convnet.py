from typing import Optional, Type, Tuple
from math import floor
import torch.nn as nn
import torch
from dinopl.modules import init

__all__ = [
    #"ConvNet",
    "convnet_16_1",
    "convnet_16_2",
    "convnet_16_3",
    "convnet_16_4",
    "convnet_16_5",
    "convnet_16_6",
    "convnet_16_7",
    "convnet_16_8",
    "convnet_32_1",
    "convnet_32_2",
    "convnet_32_3",
    "convnet_32_4",
    "convnet_32_5",
    "convnet_32_6",
    "convnet_32_7",
    "convnet_32_8",
]


class ConvNet(nn.Module):
    def __init__(self,
            widths:int = [16, 16], 
            num_classes:Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer:Type[nn.Module] = nn.Identity,
            pool_layer:Type[nn.Module] = nn.AvgPool2d,
            adaptive_final_pool: bool = True,
            img_size: Tuple[int, int] = None,
    ) -> None :
        super().__init__()
        img_size = img_size if isinstance(img_size, Tuple) or img_size is None else [img_size, img_size]

        # make adaptive final layer
        final_pool_layer = pool_layer
        if adaptive_final_pool and issubclass(pool_layer, nn.AvgPool2d):
            final_pool_layer = nn.AdaptiveAvgPool2d
        if adaptive_final_pool and issubclass(pool_layer, nn.MaxPool2d):
            final_pool_layer = nn.AdaptiveMaxPool2d

        # make sequential 
        module_list = []
        widths = [3] + widths
        for idx, (in_width, out_width) in enumerate(zip(widths[:-1], widths[1:])):
            block_list = [
                    nn.Conv2d(in_channels=in_width, out_channels=out_width, kernel_size=3, padding='same'),
                    act_layer(inplace=True),
                    norm_layer(num_features=out_width),
                    pool_layer((2,2)) if idx < len(widths) else final_pool_layer(2,2)
                ]
            module_list.append(nn.Sequential(*block_list))

            if img_size is not None:
                img_size = img_size[0] // 2, img_size[2] // 2
                
        self.sequential = nn.Sequential(*module_list)
        
        self.embed_dim:int = widths[out_width]
        adaptive_output_size = getattr(self.sequential[-1][-1], 'output_size', None)
        if adaptive_output_size is not None:
            self.embed_dim *= adaptive_output_size[0] * adaptive_output_size[0]
        elif img_size is not None:
            self.embed_dim *= img_size[0] * img_size[1]
        else:
            ValueError('Either adaptive_final_pool or img_size must be specified.')
    
            
        self.fc = nn.Identity()
        if num_classes is not None:
            self.fc = nn.Linear(self.embed_dim, num_classes)

        self.reset_parameters()
    
    def reset_parameters(self, mode='fan_out', nonlinearity='relu', generator:torch.Generator=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode=mode, nonlinearity=nonlinearity, generator=generator)

    def forward(self, x):
        x = self.sequential(x)
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
    return ConvNet(widths=1*[16], **kwargs)


def convnet_16_2(**kwargs):
    r"""Simple ConvNet with constant width 16 and depth 2.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=2*[16], **kwargs)

def convnet_16_3(**kwargs):
    r"""Simple ConvNet with constant width 16 and depth 3.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=3*[16], **kwargs)

def convnet_16_4(**kwargs):
    r"""Simple ConvNet with constant width 16 and depth 4.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=4*[16], **kwargs)

def convnet_16_5(**kwargs):
    r"""Simple ConvNet with constant width 16 and depth 5.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=5*[16], **kwargs)


def convnet_16_6(**kwargs):
    r"""Simple ConvNet with constant width 16 and depth 6.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=6*[16], **kwargs)


def convnet_16_7(**kwargs):
    r"""Simple ConvNet with constant width 16 and depth 7.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=7*[16], **kwargs)


def convnet_16_8(**kwargs):
    r"""Simple ConvNet with constant width 16 and depth 8.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=8*[16], **kwargs)



###################################################################################################
# WIDTH 32
###################################################################################################

def convnet_32_1(**kwargs):
    r"""Simple ConvNet with constant width 32 and depth 1.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=1*[32], **kwargs)


def convnet_32_2(**kwargs):
    r"""Simple ConvNet with constant width 32 and depth 2.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=2*[32], **kwargs)


def convnet_32_3(**kwargs):
    r"""Simple ConvNet with constant width 32 and depth 3.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=3*[32], **kwargs)


def convnet_32_4(**kwargs):
    r"""Simple ConvNet with constant width 32 and depth 4.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=4*[32], **kwargs)


def convnet_32_5(**kwargs):
    r"""Simple ConvNet with constant width 32 and depth 5.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=5*[32], **kwargs)


def convnet_32_6(**kwargs):
    r"""Simple ConvNet with constant width 32 and depth 6.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=6*[32], **kwargs)


def convnet_32_7(**kwargs):
    r"""Simple ConvNet with constant width 32 and depth 7.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=7*[32], **kwargs)


def convnet_32_8(**kwargs):
    r"""Simple ConvNet with constant width 32 and depth 8.
    Args:
        **kwargs: parameters passed to the ``models.convnet.ConvNet`` base class.
    """
    return ConvNet(width=8*[32], **kwargs)