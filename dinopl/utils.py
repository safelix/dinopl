from collections import OrderedDict
from typing import Optional, Iterator, List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import argparse

from torch.utils import _pytree # TODO: use for crops? or nested tensors?


## Losses and Metrics
@torch.jit.script
def entropy(prob:torch.Tensor, log_prob:torch.Tensor):
    return torch.sum(prob * -log_prob, dim=-1)

@torch.jit.script
def cross_entropy(log_pred:torch.Tensor, targ:torch.Tensor):
    return torch.sum(targ * -log_pred, dim=-1)

@torch.jit.script
def kl_divergence(log_pred:torch.Tensor, targ:torch.Tensor, log_targ:torch.Tensor):
    #return cross_entropy(log_pred, targ) - entropy(targ, log_targ)
    return torch.sum(targ * -(log_pred - log_targ), dim=-1)

@torch.jit.script
def mean_squared_error(pred:torch.Tensor, targ:torch.Tensor):
    #return cross_entropy(log_pred, targ) - entropy(targ, log_targ)
    return torch.mean(torch.square(targ - pred), dim=-1)

## torch.nn utilities
def is_bias(n:str, p:nn.Parameter): 
    return n.endswith('bias') or len(p.shape)==1

def module_to_vector(module:nn.Module, grad=False) -> torch.Tensor:
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

def vector_to_module(vec:torch.Tensor, module:nn.Module) -> None:
    if vec.dim() != 1:
        raise ValueError('Vector needs to be of dim==1')

    idx_start, idx_end = 0,0
    for param in module.parameters():
        idx_end = idx_start + param.numel()
        if idx_end > len(vec):
            raise ValueError('Module contains more parameters than the vector. ')

        param.data = vec[idx_start:idx_end].reshape_as(param)
        idx_start = idx_end 
    
    if idx_end != len(vec):
        raise ValueError('Module contains less parameters than the vector.')


def vector_as_params(vec:torch.Tensor, module:nn.Module) -> List[nn.Parameter]:
    if vec.dim() != 1:
        raise ValueError('Vector needs to be of dim==1')

    params = []
    idx_start, idx_end = 0,0
    for param in module.parameters():
        idx_end = idx_start + param.numel()
        if idx_end > len(vec):
            raise ValueError('Module contains more parameters than the vector. ')

        params.append(vec[idx_start:idx_end].view_as(param))
        idx_start = idx_end 
    
    if idx_end != len(vec):
        raise ValueError('Module contains less parameters than the vector.')

    return params

# Source: https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/resnet_hacks.py
def modify_resnet_for_tiny_input(model:nn.Module, *, cifar_stem:bool=True, v1:bool=True) -> nn.Module:
    '''Modifies some layers of a given torchvision resnet model to
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
    '''
    assert isinstance(model, torchvision.models.ResNet), 'model must be a ResNet instance'
    if cifar_stem:
        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
        model.conv1 = conv1
        model.maxpool = nn.Identity()
    if v1:
        for l in range(2, 5):
            layer = getattr(model, 'layer{}'.format(l))
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
                ), 'Currently, only models with dilation=1 are supported'
                block.conv1.stride = (2, 2)
                block.conv2.stride = (1, 1)
    return model


def bool_parser(s):
    '''
    Parse boolean arguments from the command line.
    '''
    FALSY_STRINGS = {'off', 'false', '0'}
    TRUTHY_STRINGS = {'on', 'true', '1'}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError('invalid value for a boolean flag')

def floatint_parser(s):
    '''
    Parse float or integer arguments from the command line.
    '''
    try:
        return float(s) if '.' in s else int(s)
    except ValueError:
        argparse.ArgumentTypeError('Invalid value for an integer or float')

def pick_single_gpu() -> int:
    for i in range(torch.cuda.device_count()):

        # Try to allocate on device:
        device = torch.device(f'cuda:{i}')
        try:
            torch.ones(1).to(device=device)
            torch.cuda.synchronize(device=device)
        except RuntimeError:
            continue
        return i

    raise RuntimeError('No GPUs available.')


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
        if len(curr.shape) <= 1:
            out += (f'{recprefix}{prefix}{t.__name__} of shape {curr.shape}: {curr}')
        else:
            out += (f'{recprefix}{prefix}{t.__name__} of shape {curr.shape}')
    
    else:
        out += (f'{recprefix}{prefix}object of type {t.__name__}')
    
    return out

def recprint(x):
    print(recshape(x))

def numparams(module:nn.Module):
    return sum(p.numel() for p in module.parameters())
