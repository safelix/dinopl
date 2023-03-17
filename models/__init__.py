# models/__init__.py
from torch.nn import Module
class Encoder(Module):
    embed_dim:int
    fc:Module
    def reset_parameters(self) -> None:
        raise NotImplementedError()

from .flatten import *
from .mlp import *
from .convnet import *
from .vgg import *
from .resnet import *
from .vit import *
__all__ = ( 
    flatten.__all__ +
    mlp.__all__ +
    convnet.__all__ +
    vgg.__all__ +
    resnet.__all__ +
    vit.__all__
)