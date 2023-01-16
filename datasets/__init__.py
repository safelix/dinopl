# datasets/__init__.py

from .mnist import *
from .cifar import *
from .stl import *
from .tinyimagenet import *

__all__ = mnist.__all__ + cifar.__all__ + stl.__all__ + tinyimagenet.__all__
