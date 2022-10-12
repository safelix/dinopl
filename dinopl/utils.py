from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision

## Recursive shape for debugging multicrop
def recshape(curr, recprefix='', prefix=''):
    t = type(curr)

    out = '\n'
    if t == list or t==tuple:
        out += (f'{recprefix}{prefix}{t.__name__} of length {len(curr)}')
        
        for idx, child in enumerate(curr, 1):
            out += recshape(child, recprefix=f'{recprefix}  ', prefix=f'{idx}. ') 
    
    elif t == dict:
        out += (f'{recprefix}{prefix}{t.__name__} of length {len(curr)}')
        
        for name, child in curr.items():
            out += recshape(child, recprefix=f'{recprefix}  ', prefix=f'\'{name}\': ') 
    
    elif t == torch.Tensor or t==np.ndarray:
        if len(curr.shape) == 1:
            out += (f'{recprefix}{prefix}{t.__name__} of shape {curr.shape}: {curr}')
        else:
            out += (f'{recprefix}{prefix}{t.__name__} of shape {curr.shape}')
    
    else:
        out += (f'{recprefix}{prefix}object of type {t.__name__}')
    
    return out


## Losses and Metrics
@torch.jit.script
def entropy(prob:torch.Tensor, log_prob:torch.Tensor):
    return torch.sum(prob * -log_prob, dim=-1)

@torch.jit.script
def cross_entropy(log_pred:torch.Tensor, targ:torch.Tensor):
    return torch.sum(targ * -log_pred, dim=-1)

@torch.jit.script
def kl_divergence(log_pred:torch.Tensor, targ:torch.Tensor, log_targ:torch.Tensor,):
    #return cross_entropy(log_pred, targ) - entropy(targ, log_targ)
    return torch.sum(targ * -(log_pred - log_targ), dim=-1)

## torch.nn utilities
def is_bias(n:str, p:nn.Parameter): 
    return n.endswith('bias') or len(p.shape)==1

def module_to_vector(module:nn.Module, grad=False):
    vec = []
    for param in module.parameters():
        if grad and param.grad is not None:
            vec.append(param.grad.view(-1))
        elif grad:  # fill non gradient parameters with 0 entries
            vec.append(torch.zeros_like(param.data, requires_grad=False).view(-1))
        else:
            vec.append(param.data.view(-1)) 

    if len(vec) == 0:
        return torch.empty(0)
    return torch.cat(vec)


# Source: https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/resnet_hacks.py
def modify_resnet_for_tiny_input(model:nn.Module, *, cifar_stem:bool=True, v1:bool=True) -> nn.Module:
    """Modifies some layers of a given torchvision resnet model to
    match the one used for the CIFAR-10 experiments in the SimCLR paper.
    Parameters
    ----------
    model : ResNet
        Instance of a torchvision ResNet model.
    cifar_stem : bool
        If True, adapt the network stem to handle the smaller CIFAR images, following
        the SimCLR paper. Specifically, use a smaller 3x3 kernel and 1x1 strides in the
        first convolution and remove the max pooling layer.
    v1 : bool
        If True, modify some convolution layers to follow the resnet specification of the
        original paper (v1). torchvision's resnet is v1.5 so to revert to v1 we switch the
        strides between the first 1x1 and following 3x3 convolution on the first bottleneck
        block of each of the 2nd, 3rd and 4th layers.
    Returns
    -------
    Modified ResNet model.
    """
    assert isinstance(model, torchvision.models.ResNet), "model must be a ResNet instance"
    if cifar_stem:
        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
        model.conv1 = conv1
        model.maxpool = nn.Identity()
    if v1:
        for l in range(2, 5):
            layer = getattr(model, "layer{}".format(l))
            block = list(layer.children())[0]
            if isinstance(block,  torchvision.models.resnet.Bottleneck):
                assert block.conv1.kernel_size == (1, 1) and block.conv1.stride == (
                    1,
                    1,
                )
                assert block.conv2.kernel_size == (3, 3) and block.conv2.stride == (
                    2,
                    2,
                )
                assert block.conv2.dilation == (
                    1,
                    1,
                ), "Currently, only models with dilation=1 are supported"
                block.conv1.stride = (2, 2)
                block.conv2.stride = (1, 1)
    return model


## MLP Utilities
def mlp_layer(in_dim:int, out_dim:int, act_fn:str = 'GELU', use_bn:bool = False):
    sublayers = OrderedDict()
    sublayers['lin'] = nn.Linear(in_dim, out_dim)

    if use_bn:
        sublayers['bn'] = nn.BatchNorm1d(out_dim)
    
    if act_fn.lower() == 'gelu':
        sublayers['act'] = nn.GELU()
    elif act_fn.lower() == 'relu':
        sublayers['act'] = nn.ReLU()
    else:
        raise RuntimeError('Unkown activation function.')
    
    return nn.Sequential(sublayers)

class L2Bottleneck(nn.Sequential):
    def __init__(self, in_dim:int, mid_dim:int, out_dim:int):
        sublayers = OrderedDict()
        sublayers['lin'] = nn.Linear(in_dim, mid_dim)
        sublayers['featurenorm'] = LpNormalizeFeatures(p=2, dim=-1)
        sublayers['weightnorm'] = WeightNormalizedLinear(mid_dim, out_dim, bias=False)
        super().__init__(sublayers) 

class LpNormalizeFeatures(nn.Module):
    def __init__(self, p:float = 2, dim:int = -1):
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=-1)

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
