# datasets/__init__.py

from .base import *
from .mnist import *
from .cifar import *
from .tinyimagenet import *
from . import augmentation
from . import targetnoise
from . import stratifiedsubset

__all__ = ( 
    #base.__all__ + # we don't want that, since it is is used in config
    mnist.__all__ +
    cifar.__all__ +
    tinyimagenet.__all__ 
)
