from collections import OrderedDict
from math import sqrt
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from . import init

__all__ = [
    'L2Bottleneck',
    'LpNormalize'
]


class L2Bottleneck(nn.Module):
    'L2-Bottleneck configuration string: \'{wn,-}/{l,lb,-}/{fn,-}/{wn,-}/{l,lb,-}/{wn,-}\'.'
    
    def __init__(self, in_dim:int, mid_dim:int, out_dim:int, cfg:str='-/lb/fn/wn/l/-'):
        super().__init__() 

        self.cfg = cfg.split('/')
        if len(self.cfg) != 6:
            raise ValueError('L2Bottleneck is specified by 6 strings separated with \'/\'.')
        for i, mode in enumerate(['input', 'output']):
            if self.cfg[3*i+0] not in ['wn', '-']:
                raise ValueError(f'cfg[{3*i+0}] describes {mode} weight normalization: can be eiter \'wn\' or \'-\'.')
            if self.cfg[3*i+1] not in ['l', 'lb', '-']:
                raise ValueError(f'cfg[{3*i+1}] describes {mode} linear layer: can be eiter \'l\', \'lb\' or \'-\'.')
            if self.cfg[3*i+2] not in ['fn', '-']:
                raise ValueError(f'cfg[{3*i+2}] describes {mode} feature normalization: can be eiter \'fn\' or \'-\'.')

        if self.cfg[1] == '-': 
            mid_dim = in_dim # overwrite mid_dim if lin1 doesn't exist
        elif self.cfg[4] == '-':
            mid_dim = out_dim # overwrite mid_dim if lin2 doesn't exist
        # TODO: Warn if both are not defined? store dimensions for parent modules?

        if 'l' in self.cfg[1]: # if lin1 exists
            self.wn1 = LpNormalize(p=2, dim=-1) if self.cfg[0] == 'wn' else nn.Identity()
            self.lin1 = nn.Linear(in_dim, mid_dim, bias='b' in self.cfg[1])
        self.fn1 = LpNormalize(p=2, dim=-1) if self.cfg[2] == 'fn' else nn.Identity()

        if 'l' in self.cfg[4]: # if lin2 exists
            self.wn2 = LpNormalize(p=2, dim=-1) if self.cfg[3] == 'wn' else nn.Identity()
            self.lin2 = nn.Linear(mid_dim, out_dim, bias='b' in self.cfg[4]) if 'l' in self.cfg[4] else None
        self.fn2 = LpNormalize(p=2, dim=-1) if self.cfg[5] == 'fn' else nn.Identity()

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

                if method == 'trunc_normal':
                    init.trunc_normal_(m.weight, std=.02, generator=generator)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)


    def forward(self, x):
        # transform to bottleneck
        if self.lin1 is not None:
            w1 = self.wn1(self.lin1.weight) # prepare weight normalization
            x = F.linear(input=x, weight=w1, bias=self.lin1.bias) # compute linear map
        x = self.fn1(x)     # compute feature normalization

        # transform to output
        if self.lin2 is not None:
            w2 = self.wn2(self.lin2.weight)    # prepare weight normalization
            x = F.linear(input=x, weight=w2, bias=self.lin2.bias) # compute linear map
        x = self.fn2(x)     # compute feature normalization

        return x


class LpNormalize(nn.Module):
    def __init__(self, p:float = 2, dim:int = -1):
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=-1)

    def extra_repr(self):
        return f'p={self.p}, dim={self.dim}'



class L2Bottleneck_OLD(nn.Sequential):
    def __init__(self, in_dim:int, mid_dim:int, out_dim:int):
        sublayers = OrderedDict()
        sublayers['lin'] = nn.Linear(in_dim, mid_dim)
        sublayers['featurenorm'] = LpNormalize(p=2, dim=-1)
        sublayers['weightnorm'] = WeightNormalizedLinear(mid_dim, out_dim, bias=False, norm=1)
        super().__init__(sublayers) 
 

class WeightNormalizedLinear(nn.Linear):
    def __init__(self, in_features:int, out_features:int, bias:bool = True, norm:Optional[float] = None):
        super().__init__(in_features, out_features, bias)

        # attach weight norm like in vision_transformer.py
        nn.utils.weight_norm(self)
        if norm:
            self.weight_g.data.fill_(norm)
            self.weight_g.requires_grad = False
            self.weight_g.requires_grad_ = (lambda x: x)

        # This doesn't work with deepcopy, see:
        # https://github.com/pytorch/pytorch/issues/28594 and
        # https://discuss.pytorch.org/t/when-can-you-not-deepcopy-a-model/153226/2
        
        # Workaround 1: detach the initiallized weight.. forward_pre_hook will reattach it again
        self.weight = self.weight_v.detach()

        # Workaround 2:
        #self.weight_g = nn.Parameter(torch.full((self.weight.shape[0],1), 1.0))
        #self.weight_g.requires_grad = False
        #self.weight.data = torch._weight_norm(self.weight, self.weight_g, 0)
        # _weight_norm() must be called manually before forward