import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def recprint(curr, recprefix='', prefix=''):
    t = type(curr)

    out = '\n'
    if t == list or t==tuple:
        out += (f'{recprefix}{prefix}{t.__name__} of length {len(curr)}')
        
        for idx, child in enumerate(curr, 1):
            out += recprint(child, recprefix=f'{recprefix}  ', prefix=f'{idx}. ') 
    
    elif t == dict:
        out += (f'{recprefix}{prefix}{t.__name__} of length {len(curr)}')
        
        for name, child in curr.items():
            out += recprint(child, recprefix=f'{recprefix}  ', prefix=f'\'{name}\': ') 
    
    elif t == torch.Tensor or t==np.ndarray:
        if len(curr.shape) == 1:
            out += (f'{recprefix}{prefix}{t.__name__} of shape {curr.shape}: {curr}')
        else:
            out += (f'{recprefix}{prefix}{t.__name__} of shape {curr.shape}')
    
    else:
        out += (f'{recprefix}{prefix}object of type {t.__name__}')
    
    return out


## Losses and Metrics
def entropy(prob:torch.Tensor, log_prob:torch.Tensor):
    return torch.mean(prob * -log_prob, dim=-1)

def cross_entropy(log_pred:torch.Tensor, targ:torch.Tensor):
    return torch.mean(targ * -log_pred, dim=-1)

def kl_divergence(log_pred:torch.Tensor, targ:torch.Tensor, log_targ:torch.Tensor,):
    #return cross_entropy(log_pred, targ) - entropy(targ, log_targ)
    return torch.mean(targ * -(log_pred - log_targ), dim=-1)

## torch.nn utilities
def is_bias(n:str, p:nn.Parameter): 
    return n.endswith('bias') or len(p.shape)==1

def module_to_vector(module:nn.Module):
    return nn.utils.parameters_to_vector(module.parameters())

## MLP Utilities
def mlp_layer(in_dim:int, out_dim:int, act_fn:str = 'GELU', use_bn:bool = False):
    sublayers = [nn.Linear(in_dim, out_dim)]
    if use_bn:
        sublayers.append(nn.BatchNorm1d(out_dim))
    
    if act_fn.lower() == 'gelu':
        sublayers.append(nn.GELU())
    elif act_fn.lower() == 'relu':
        sublayers.append(nn.ReLU())
    else:
        raise RuntimeError('Unkown activation function.')
    
    return nn.Sequential(*sublayers)

class L2Bottleneck(nn.Sequential):
    def __init__(self, in_dim:int, mid_dim:int, out_dim:int):
        sublayers = [nn.Linear(in_dim, mid_dim)]
        sublayers.append(LpNormalizeFeatures(p=1, dim=-1))
        sublayers.append(WeightNormalizedLinear(mid_dim, out_dim, bias=False))
        super().__init__(*sublayers) 

class LpNormalizeFeatures(nn.Module):
    def __init__(self, p:float = 2, dim:int = -1):
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=-1)

    def extra_repr(self):
        return f'p={self.p}, dim={self.dim}' 

class WeightNormalizedLinear(nn.Linear):
    def __init__(self, in_features:int, out_features:int, bias:bool = True):
        super().__init__(in_features, out_features, bias)

        # attach weight norm like in vision_transformer.py
        self = nn.utils.weight_norm(self)
        self.weight_g.data.fill_(1)
        self.weight_g.requires_grad = False

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



###############################################################################
########################## PyTorch Pre-Release Code ###########################
###############################################################################
# Source: https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
import math
import warnings

import torch
from torch import Tensor


# These no_grad_* functions are necessary as wrappers around the parts of these
# functions that use `with torch.no_grad()`. The JIT doesn't support context
# managers, so these need to be implemented as builtins. Using these wrappers
# lets us keep those builtins small and re-usable.
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor: Tensor, mean: float = 0., std: float = 1., a: float = -2., b: float = 2.) -> Tensor:
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
