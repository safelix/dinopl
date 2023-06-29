from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F
from . import init

__all__ = [
    'Linear'
]

class Linear(nn.Linear):
    '''A custom Linear Layer:
    - the weight and bias can be modified in the forwardpass.
    - the reset_parameters takes method and a generator and as arguments.'''
    def forward(self, input: torch.Tensor, weight:torch.Tensor=None, bias:torch.Tensor=None) -> torch.Tensor:
        weight = self.weight if weight is None else weight
        bias = self.bias if bias is None else bias
        return F.linear(input, self.weight, self.bias)
    
    def reset_parameters(self, method='default', generator:torch.Generator=None):
        if method == 'default':
            #m.reset_parameters() is equal to:
            bound = 1 / sqrt(self.in_features)
            init.uniform_(self.weight, -bound, bound, generator=generator)
            if self.bias is not None:
                init.uniform_(self.bias, -bound, bound, generator=generator)

        if method == 'trunc_normal':
            init.trunc_normal_(self.weight, std=.02, generator=generator)
            if self.bias is not None:
                init.constant_(self.bias, 0)
