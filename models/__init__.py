# models/__init__.py
from torch.nn import Module
class Encoder(Module):
    embed_dim:int
    fc:Module
    def reset_parameters(self) -> None:
        raise NotImplementedError()

# merge __all__ from subdirectories
from .flatten import __all__
from .mlp import __all__
from .convnet import __all__
from .vgg import __all__
from .resnet import __all__
from .vit import __all__
__all__ = ( 
    flatten.__all__ +
    mlp.__all__ +
    convnet.__all__ +
    vgg.__all__ +
    resnet.__all__ +
    vit.__all__
)

# load subdirectories
from .flatten import *
from .mlp import *
from .convnet import *
from .vgg import *
from .resnet import *
from .vit import *