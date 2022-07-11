import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.dense_attention import my_pam,PAM_Module
from modeling.MPP import CNBlock
from modeling.customize import PyramidPooling, StripPooling


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

        # in_channels = 2048
        # inter_channels = in_channels // 4
        # #self.block = nn.Conv2d(2048,num_classes,kernel_size=1,stride=1)
        # self.my_conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                             BatchNorm(inter_channels),
        #                             nn.ReLU())
        self.my_lastblock = nn.Conv2d(512, num_classes, 1)
        # self.my_mpp4 = CNBlock(256, 256)
        # self.my_conv4 = nn.Sequential(nn.Conv2d(256, 2048, 3, padding=1, bias=False),
        #                                BatchNorm(inter_channels),
        #                                nn.ReLU())
        self.my_mpp3 = CNBlock(512, 512)
        self.my_conv3 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1, bias=False),
                                      BatchNorm(512),
                                      nn.ReLU())
        self.my_mpp2 = CNBlock(1024, 1024)
        self.my_conv2 = nn.Sequential(nn.Conv2d(1024, 512, 3, padding=1, bias=False),
                                      BatchNorm(512),
                                      nn.ReLU())
        self.my_mpp1 = CNBlock(2048, 2048)
        self.my_conv1 = nn.Sequential(nn.Conv2d(2048, 512, 3, padding=1, bias=False),
                                      BatchNorm(512),
                                      nn.ReLU())
        # self.strip_pool1 = StripPooling(2048, (20, 12), BatchNorm)
        # self.strip_pool2 = StripPooling(2048, (20, 12), BatchNorm)

        self.my_lastblock = nn.Conv2d(512, num_classes, 1)

        self._init_weight()

    def forward(self, x1, x2, x3, x4):
        _, _, h, w = x4.size()
        # x1 = self.my_mpp4(x1)
        # x1 = self.my_conv4(x1)
        # x1 = F.interpolate(x1, size=(h, w), mode='bilinear')

        x2 = self.my_mpp3(x2)
        x2 = self.my_conv3(x2)
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear')

        x3 = self.my_mpp2(x3)
        x3 = self.my_conv2(x3)
        x3 = F.interpolate(x3, size=(h, w), mode='bilinear')

        x4 = self.my_mpp1(x4)
        x4 = self.my_conv1(x4)
        x4 = F.interpolate(x4, size=(h, w), mode='bilinear')

        x = (x2 + x3 + x4)
        # x = self.strip_pool1(x)
        # x = self.strip_pool2(x)
        # x = self.my_conv5a(x)
        x = self.my_lastblock(x)
        # x = self.my_mpp2(x)
        # x = self.my_mpp1(x)
        # x = self.strip_pool1(x)
        # x = self.strip_pool2(x)

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


def build_Base_mypam_cam(num_classes, backbone, BatchNorm):
    return Base(num_classes, backbone, BatchNorm)