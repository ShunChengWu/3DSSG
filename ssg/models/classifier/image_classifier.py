import torch
import torch.nn as nn
from torchvision import models


class VGG16(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, replace: bool = True, pretrained: bool = True):
        super().__init__()
        if in_channels != 25088:
            raise RuntimeError('input channel must be 25088')

        vgg16 = models.vgg16(pretrained=pretrained)
        network = vgg16.classifier
        if not replace and out_channels != 1000:
            raise RuntimeError(
                'replace must be true when the output channel != 1000')
        elif replace:
            network[-1] = nn.Linear(4096, out_channels)
        self.network = network

    def forward(self, x):
        return self.network(x)


class basic(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, replace: bool = True, pretrained: bool = True):
        super().__init__()
        self.network = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.network(x)


if __name__ == '__main__':
    # try:
    #     net = VGG16(512,False,True)
    # except RuntimeError:
    #     print('correct')
    net = VGG16(512, True, True)
    net = VGG16(4096, True, True)
