from collections import OrderedDict
from math import sqrt
from typing import List, Optional, Type

import torch
import torch.nn as nn

from dinopl.modules import init

__all__ = [
    #"MLPEnc",
    "mlp_512_1",
    "mlp_512_2",
    "mlp_512_3",
    "mlp_512_4",
    "mlp_512_5",
    "mlp_1024_1",
    "mlp_1024_2",
    "mlp_1024_3",
    "mlp_1024_4",
    "mlp_1024_5"
]

# adjust MLP definition from dinopl.modules
def mlp_layer(in_dim:int, out_dim:int, act_fn:str = 'GELU', norm_layer:nn.Module = None):
    sublayers = OrderedDict()
    sublayers['lin'] = nn.Linear(in_dim, out_dim)

    if norm_layer is not None:
        sublayers['norm'] = norm_layer(out_dim)
    
    if act_fn.lower() == 'gelu':
        sublayers['act'] = nn.GELU()
    elif act_fn.lower() == 'relu':
        sublayers['act'] = nn.ReLU()
    else:
        raise RuntimeError('Unkown activation function.')
    return nn.Sequential(sublayers)


class MLPEnc(nn.Sequential):
    def __init__(self, 
            dims:List[int], 
            num_classes:int=None, 
            act_fn:str='GELU', 
            norm_layer:nn.Module = None
        ) -> None:
        super().__init__()

        layers = OrderedDict() # prepare layers
        layers['flatten'] = nn.Flatten(start_dim=1, end_dim=-1)
        for idx, (i, o) in enumerate(zip(dims[:-1], dims[1:])):
            layers[f'layer{idx}'] = mlp_layer(i, o, act_fn, norm_layer)

        self.embed_dim:int = dims[-1]

        layers['fc'] = nn.Identity()
        if num_classes is not None:
            layers['fc'] = nn.Linear(self.embed_dim, num_classes)

        super().__init__(layers)
        self.reset_parameters()
    
    def reset_parameters(self, method='default', generator:torch.Generator=None):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == 'default':
                    #m.reset_parameters() is equal to:
                    bound = 1 / sqrt(m.in_features)
                    init.uniform_(m.weight, -bound, bound, generator=generator)
                    if m.bias is not None:
                        init.uniform_(m.bias, -bound, bound, generator=generator)

                if method == 'kaiming_in':
                    init.kaiming_normal_(m.weight, mode='in', nonlinearity='relu', generator=generator)

                if method == 'kaiming_out':
                    init.kaiming_normal_(m.weight, mode='out', nonlinearity='relu', generator=generator)

                if method == 'trunc_normal':
                    init.trunc_normal_(m.weight, std=.02, generator=generator)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)

###################################################################################################
# WIDTH 512
###################################################################################################

def mlp_512_1(in_numel:int, **kwargs):
    r"""Simple MLP encoder with constant width 512 and depth 1.
    Args:
        in_numel: Number of elements in input image to encode.
        **kwargs: parameters passed to the ``models.mlp.MLPEnc`` base class.
    """
    return MLPEnc([in_numel, 512], **kwargs)


def mlp_512_2(in_numel:int, **kwargs):
    r"""Simple MLP encoder with constant width 512 and depth 2.
    Args:
        in_numel: Number of elements in input image to encode.
        **kwargs: parameters passed to the ``models.mlp.MLPEnc`` base class.
    """
    return MLPEnc([in_numel, 512, 512], **kwargs)


def mlp_512_3(in_numel:int, **kwargs):
    r"""Simple MLP encoder with constant width 512 and depth 3.
    Args:
        in_numel: Number of elements in input image to encode.
        **kwargs: parameters passed to the ``models.mlp.MLPEnc`` base class.
    """
    return MLPEnc([in_numel, 512, 512, 512], **kwargs)

def mlp_512_4(in_numel:int, **kwargs):
    r"""Simple MLP encoder with constant width 512 and depth 3.
    Args:
        in_numel: Number of elements in input image to encode.
        **kwargs: parameters passed to the ``models.mlp.MLPEnc`` base class.
    """
    return MLPEnc([in_numel, 512, 512, 512, 512], **kwargs)

def mlp_512_5(in_numel:int, **kwargs):
    r"""Simple MLP encoder with constant width 512 and depth 3.
    Args:
        in_numel: Number of elements in input image to encode.
        **kwargs: parameters passed to the ``models.mlp.MLPEnc`` base class.
    """
    return MLPEnc([in_numel, 512, 512, 512, 512, 512], **kwargs)


###################################################################################################
# WIDTH 1024
###################################################################################################

def mlp_1024_1(in_numel:int, **kwargs):
    r"""Simple MLP encoder with constant width 1024 and depth 1.
    Args:
        in_numel: Number of elements in input image to encode.
        **kwargs: parameters passed to the ``models.mlp.MLPEnc`` base class.
    """
    return MLPEnc([in_numel, 1024], **kwargs)


def mlp_1024_2(in_numel:int, **kwargs):
    r"""Simple MLP encoder with constant width 1024 and depth 2.
    Args:
        numel: Number of elements in input image to encode.
        **kwargs: parameters passed to the ``models.mlp.MLPEnc`` base class.
    """
    return MLPEnc([in_numel, 1024, 1024], **kwargs)


def mlp_1024_3(in_numel:int, **kwargs):
    r"""Simple MLP encoder with constant width 1024 and depth 3.
    Args:
        numel: Number of elements in input image to encode.
        **kwargs: parameters passed to the ``models.mlp.MLPEnc`` base class.
    """
    return MLPEnc([in_numel, 1024, 1024, 1024], **kwargs)

def mlp_1024_4(in_numel:int, **kwargs):
    r"""Simple MLP encoder with constant width 1024 and depth 3.
    Args:
        numel: Number of elements in input image to encode.
        **kwargs: parameters passed to the ``models.mlp.MLPEnc`` base class.
    """
    return MLPEnc([in_numel, 1024, 1024, 1024, 1024], **kwargs)

def mlp_1024_5(in_numel:int, **kwargs):
    r"""Simple MLP encoder with constant width 1024 and depth 3.
    Args:
        numel: Number of elements in input image to encode.
        **kwargs: parameters passed to the ``models.mlp.MLPEnc`` base class.
    """
    return MLPEnc([in_numel, 1024, 1024, 1024, 1024, 1024], **kwargs)