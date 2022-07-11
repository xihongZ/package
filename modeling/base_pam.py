import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.dense_attention import my_pam,PAM_Module


class Base(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Base, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        in_channels = 2048
        inter_channels = in_channels // 4
        #self.block = nn.Conv2d(2048,num_classes,kernel_size=1,stride=1)
        self.my_conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    BatchNorm(inter_channels),
                                    nn.ReLU())
        self.my_sa = my_pam(inter_channels)  #my module pam
        #self.da_sa = PAM_Module(inter_channels)  # DANet module pam
        self.my_conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1,bias=False),
                                    BatchNorm(inter_channels),
                                    nn.ReLU())
        self.my_block = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, num_classes, 1))
        self._init_weight()

    def forward(self, x):
        x = self.my_conv5a(x)
        x = self.my_sa(x)
        #x = self.da_sa(x)
        x = self.my_conv51(x)
        x = self.my_block(x)

        #x = self.block(x)
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



def build_Base_mypam(num_classes, backbone, BatchNorm):
    return Base(num_classes, backbone, BatchNorm)