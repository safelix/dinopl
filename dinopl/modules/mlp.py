from collections import OrderedDict
from math import sqrt
from typing import List, Optional
import torch
from torch import nn
from .linear import Linear
from . import init

__all__ = [
    'MLP'
]

def mlp_layer(in_dim:int, out_dim:int, act_fn:str = 'GELU', use_bn:bool = False):
    sublayers = OrderedDict()
    sublayers['lin'] = Linear(in_dim, out_dim)

    if use_bn:
        sublayers['bn'] = nn.BatchNorm1d(out_dim)
    
    if act_fn.lower() == 'gelu':
        sublayers['act'] = nn.GELU()
    elif act_fn.lower() == 'relu':
        sublayers['act'] = nn.ReLU()
    else:
        raise RuntimeError('Unkown activation function.')
    
    return nn.Sequential(sublayers)



class MLP(nn.Sequential):
    def __init__(self, dims:List[int], act_fn:str = 'GELU', use_bn:bool = False):

        layers = OrderedDict() # prepare layers
        for idx, (i, o) in enumerate(zip(dims[:-1], dims[1:])):
            layers[f'layer{idx}'] = mlp_layer(i, o, act_fn, use_bn)

        # make sequential
        super().__init__(layers)
        self.reset_parameters()
    
    def reset_parameters(self, method='trunc_normal', generator:torch.Generator=None):
        for m in self.modules():
            if isinstance(m, Linear):
                m.reset_parameters(method=method, generator=generator)
            
            if isinstance(m, (nn.modules.batchnorm._NormBase, nn.GroupNorm, nn.LayerNorm)):
                if m.weight is not None:
                    init.constant_(m.weight, 1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        
