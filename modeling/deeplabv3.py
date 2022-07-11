import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class DeepLabv3(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(DeepLabv3, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.block = nn.Sequential(
                                    nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,bias=False),
                                    BatchNorm(256),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv2d(256,num_classes,kernel_size=1,stride=1)
                                   )

        self._init_weight()

    def forward(self, x):
        x = self.block(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# def build_decoder(num_classes, backbone, BatchNorm):
#     return Decoder(num_classes, backbone, BatchNorm)


def build_deeplabv3(num_classes, backbone, BatchNorm):
    return DeepLabv3(num_classes, backbone, BatchNorm)