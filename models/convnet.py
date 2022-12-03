# Copy from Sotiris Anagnostidis
import torch.nn as nn
import torch.nn.functional as F
import torch


class ConvNet(nn.Module):
    def __init__(self, n_layers=None, n_filters=16, depth=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, out_channels=n_filters, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.pool = nn.MaxPool2d(2, 2)

        module_list = []
        for _ in range(depth - 2):
            module_list.append(
                nn.Conv2d(
                    n_filters, out_channels=n_filters, kernel_size=3, padding="same"
                )
            )
            module_list.append(nn.BatchNorm2d(n_filters))
            module_list.append(nn.ReLU())

        module_list.append(
            nn.Conv2d(n_filters, out_channels=n_filters, kernel_size=3, padding="same")
        )
        self.block = nn.Sequential(*module_list)

        # self.bn2 = norm_layer(16)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))  # nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_filters * 4, n_filters * 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.avgpool(self.block(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc(x)
        return x


def conv_net(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ConvNet(**kwargs)