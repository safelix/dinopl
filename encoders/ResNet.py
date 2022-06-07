from torch import nn
from torchvision.models import resnet18

from configuration import Configuration

class ResNet18(nn.Module):
    def __init__(self, config:Configuration):
        super().__init__()

        self = resnet18() # instanciate resnet18

        # Don't use fully connected embedding layer
        self.embed_dim = self.fc.in_features
        config.embed_dim = self.fc.in_features
        self.fc = nn.Identity
