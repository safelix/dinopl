from collections import OrderedDict
from math import sqrt
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .linear import Linear

__all__ = [
    'L2Bottleneck',
    'LpNormalize'
]


class L2Bottleneck(nn.Module):
    'L2-Bottleneck configuration string: \'{wn,-}/{l,lb,-}/{fn,fnd-}/{wn,-}/{l,lb,-}/{fn,fnd-}\'.'
    
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
            if self.cfg[3*i+2] not in ['fn', 'fnd', '-']:
                raise ValueError(f'cfg[{3*i+2}] describes {mode} feature normalization: can be eiter \'fn\' or \'-\'.')

        if self.cfg[1] == '-': 
            mid_dim = in_dim # overwrite mid_dim if lin1 doesn't exist
        elif self.cfg[4] == '-':
            mid_dim = out_dim # overwrite mid_dim if lin2 doesn't exist
        # TODO: Warn if both are not defined? store dimensions for parent modules?


        self.lin1, self.wn1 = None, None
        if 'l' in self.cfg[1]: # if lin1 exists
            self.wn1 = LpNormalize(p=2, dim=-1) if self.cfg[0] == 'wn' else nn.Identity()
            self.lin1 = Linear(in_dim, mid_dim, bias='b' in self.cfg[1])
        self.fn1 = LpNormalize(p=2, dim=-1, detach=(self.cfg[2]=='fnd')) if 'fn' in self.cfg[2] else nn.Identity()

        self.lin2, self.wn2 = None, None
        if 'l' in self.cfg[4]: # if lin2 exists
            self.wn2 = LpNormalize(p=2, dim=-1) if self.cfg[3] == 'wn' else nn.Identity()
            self.lin2 = Linear(mid_dim, out_dim, bias='b' in self.cfg[4])
        self.fn2 = LpNormalize(p=2, dim=-1, detach=(self.cfg[5] == 'fnd')) if 'fn' in self.cfg[5] else nn.Identity()

        self.reset_parameters()
    
    def reset_parameters(self, method='trunc_normal', generator:torch.Generator=None):
        if self.lin1 is not None:
            self.lin1.reset_parameters(method, generator=generator)

        if self.lin2 is not None:
            self.lin2.reset_parameters(method=method, generator=generator)


    def forward(self, x:torch.Tensor):
        # transform to bottleneck
        if self.lin1 is not None:
            w1 = self.wn1(self.lin1.weight) # prepare weight normalization
            x = self.lin1(input=x, weight=w1, bias=self.lin1.bias) # compute linear map
        x = self.fn1(x)     # compute feature normalization

        # transform to output
        if self.lin2 is not None:
            w2 = self.wn2(self.lin2.weight)    # prepare weight normalization
            x = self.lin2(input=x, weight=w2, bias=self.lin2.bias) # compute linear map
        x = self.fn2(x)     # compute feature normalization

        return x


class LpNormalize(nn.Module):
    def __init__(self, p:float = 2, dim:int = -1, eps:float = 1e-12, detach:bool = False):
        super().__init__()
        self.p = p
        self.dim = dim
        self.eps = eps
        if detach:
            self.forward = self.forward_detach

    def forward_detach(self, x:torch.Tensor):
        denom:torch.Tensor = x.norm(self.p, self.dim, keepdim=True).clamp(min=self.eps).expand_as(x)
        return x / denom.detach()

    def forward(self, x:torch.Tensor):
        #return F.normalize(x, p=self.p, dim=self.dim)
        denom:torch.Tensor = x.norm(self.p, self.dim, keepdim=True).clamp(min=self.eps).expand_as(x)
        return x / denom

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