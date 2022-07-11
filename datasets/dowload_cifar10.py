import os
import torchvision.datasets as datasets


path_to_data = os.environ['DINO_DATA']
os.makedirs(path_to_data, exist_ok=True) 

mnist_trainset = datasets.CIFAR10(root=path_to_data, 
                                download=True)




