import torch
import torch.nn as nn
from torchvision import models
from codeLib.common import normalize_imagenet


class ConvEncoder(nn.Module):
    r''' Simple convolutional encoder network.

    It consists of 5 convolutional layers, each downsampling the input by a
    factor of 2, and a final fully-connected layer projecting the output to
    c_dim dimensions.

    Args:
        c_dim (int): output dimension of latent embedding
    '''

    def __init__(self, c_dim=128):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 32, 3, stride=2)
        self.conv1 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2)
        self.fc_out = nn.Linear(512, c_dim)
        self.actvn = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        net = self.conv0(x)
        net = self.conv1(self.actvn(net))
        net = self.conv2(self.actvn(net))
        net = self.conv3(self.actvn(net))
        net = self.conv4(self.actvn(net))
        net = net.view(batch_size, 512, -1).mean(2)
        out = self.fc_out(self.actvn(net))

        return out


# class Resnet18(nn.Module):
#     r''' ResNet-18 encoder network for image input.
#     Args:
#         c_dim (int): output dimension of the latent embedding
#         normalize (bool): whether the input images should be normalized
#         use_linear (bool): whether a final linear layer should be used
#     '''

#     def __init__(self, c_dim, normalize=True, use_linear=True):
#         super().__init__()
#         self.normalize = normalize
#         self.use_linear = use_linear
#         self.features = models.resnet18(pretrained=True)
#         self.features.fc = nn.Sequential()
#         if use_linear:
#             self.fc = nn.Linear(512, c_dim)
#         elif c_dim == 512:
#             self.fc = nn.Sequential()
#         else:
#             raise ValueError('c_dim must be 512 if use_linear is False')

#     def forward(self, x):
#         if self.normalize:
#             x = normalize_imagenet(x)
#         net = self.features(x)
#         out = self.fc(net)
#         return out

class Resnet18(nn.Module):
    r''' ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_local=False, use_linear=True, hook_layers=['layer1','layer2','layer3','layer4']):
        super().__init__()
        self.normalize = normalize
        self.use_local = use_local
        resnet18 = models.resnet18(pretrained=True)
        self.layers = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4
        )
        if use_linear:
            self.fc = nn.Conv2d(512, c_dim, kernel_size=(1,1))
        else:
            self.fc = nn.Sequential()
        
        # elif c_dim == 512:
        # else:
        #     raise ValueError('c_dim must be 512 if use_linear is False')
            
        
        self.layers_output = None
        self.hook_layers_size = 0
        self.hook_layers_sizes = None
        if use_local:
            self.layers_output = list()
            self.hook_layers_sizes=list()
            self.fhooks = []
            hook_layers_size = 0
            for i, name in enumerate(list(self.features._modules.keys())):
                if name in hook_layers:
                    layer = getattr(self.features, name)
                    self.fhooks.append(layer.register_forward_hook(self.forward_hook(name)))
                    
                    self.hook_layers_sizes.append(layer[-1].conv2.out_channels)
                    hook_layers_size += layer[-1].conv2.out_channels
            self.hook_layers_size = hook_layers_size
            self.hook_layers_sizes = torch.tensor(self.hook_layers_sizes)
                
    def forward_hook(self, layer_name):
        def hook(module, data_in, data_out):
            self.layers_output.append(data_out)
        return hook

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        if self.use_local: self.layers_output.clear()
        net = self.layers(x)
        out = self.fc(net)
        return out


class Vgg16(nn.Module):
    r''' Vgg-16 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
        
    Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace=True)
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU(inplace=True)
      (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): ReLU(inplace=True)
      (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU(inplace=True)
      (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): ReLU(inplace=True)
      (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): ReLU(inplace=True)
      (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (18): ReLU(inplace=True)
      (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (20): ReLU(inplace=True)
      (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (22): ReLU(inplace=True)
      (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (25): ReLU(inplace=True)
      (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (27): ReLU(inplace=True)
      (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (29): ReLU(inplace=True)
      (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True, use_local=False, hook_layers=['0','2']):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.use_local = use_local
        model = models.vgg16(pretrained=True)
        self.features = model.features#[:-1] # skip the last maxpool
        # self.features.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.features.classifier = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')
    #     self.layers_output = None
    #     self.hook_layers_size = 0
    #     self.hook_layers_sizes = None
    #     if use_local:
    #         self.layers_output = list()
    #         self.hook_layers_sizes=list()
    #         self.fhooks = []
    #         hook_layers_size = 0
    #         last_output_channel = 0
    #         for i, name in enumerate(list(self.features.features._modules.keys())):
    #             layer = getattr(self.features.features, name)
    #             if hasattr(layer, 'out_channels'):
    #                 last_output_channel = layer.out_channels
    #                 # print(layer.__class__, layer.out_channels)
    #             # if isinstance(layer, nn.Conv2d):
    #             #     last_output_channel = layer.out_channels
                
    #             if name in hook_layers:
    #                 layer = getattr(self.features.features, name)
    #                 self.fhooks.append(layer.register_forward_hook(self.forward_hook(name)))
                    
    #                 self.hook_layers_sizes.append(last_output_channel)
    #                 hook_layers_size += last_output_channel
                    
    #                 # print('hook',i,name,layer.__class__, last_output_channel)
    #                 # if isinstance(layer, nn.Conv2d):
    #                 #     self.hook_layers_sizes.append(layer.out_channels)
    #                 #     hook_layers_size += layer.out_channels
    #                 # elif isinstance(layer, nn.MaxPool2d):
    #                 #     tmp_layer = self.features.features[i-1]
    #                 #     hook_layers_size += tmp_layer.out_channels
    #                 # elif isinstance(layer, nn.Sequential):
    #                 #     self.hook_layers_sizes.append(layer[-1].conv2.out_channels)
    #                 #     hook_layers_size += layer[-1].conv2.out_channels
    #         self.hook_layers_size = hook_layers_size
    #         self.hook_layers_sizes = torch.tensor(self.hook_layers_sizes)
                
    # def forward_hook(self, layer_name):
    #     def hook(module, data_in, data_out):
    #         self.layers_output.append(data_out)
    #     return hook

    def forward(self, x):
        if self.normalize: x = normalize_imagenet(x)
        if self.use_local: self.layers_output.clear()
        net = self.features(x)
        out = self.fc(net)
        return out
    

class Resnet34(nn.Module):
    r''' ResNet-34 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet34(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class Resnet50(nn.Module):
    r''' ResNet-50 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(2048, c_dim)
        elif c_dim == 2048:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 2048 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class Resnet101(nn.Module):
    r''' ResNet-101 encoder network.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(2048, c_dim)
        elif c_dim == 2048:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 2048 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out
    
    
if __name__ == '__main__':
    resnet = Resnet18(512,False)
    x = torch.rand(1,3,480,640)
    y = resnet(x)
    print(y.shape)