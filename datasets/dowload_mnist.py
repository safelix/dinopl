import os
import pathlib
import torch
import torchvision
import torchvision.datasets as datasets


cwd = pathlib.Path().resolve()
path_to_data = cwd.joinpath('../../data')
path_to_data.mkdir(exist_ok=True) 

mnist_trainset = datasets.MNIST(root=path_to_data, 
                                download=True)




