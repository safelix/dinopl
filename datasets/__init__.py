# datasets/__init__.py

from .mnist import *
from .cifar import *
__all__ = ( 
    mnist.__all__ +
    cifar.__all__
)