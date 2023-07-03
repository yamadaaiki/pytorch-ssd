import torch.nn as nn

# base network
def prpposed_detect_metastases(cfg, norm=False):
    layers = []
    in_channels = 3 # if use gray scale, change from 3 to 1. 
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if norm == "batchnorm":
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            elif norm == "groupnorm":
                layers =+ [conv2d, nn.GroupNorm(8, v), nn.ReLU(inplace=True)]
            else :
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    conv4_1 = nn.Conv2d(256, 1024, kernel_size=3, padding=1)
    conv4_2 = nn.Conv2d(1024, 1024, kernel_size=1)
    conv4_3 = nn.Conv2d(1024, 256, kernel_size=1)
    layers += [conv4_1, nn.ReLU(inplace=False),
               conv4_2, nn.ReLU(inplace=False),
               conv4_3, nn.ReLU(inplace=False)]
    return layers