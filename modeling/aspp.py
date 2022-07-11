import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
#from modeling.nn.attention import PAM_Module

import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

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

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)




        self.global_avg_pool = nn.Sequential(
                                             nn.AdaptiveAvgPool2d((1,1)),
                                             nn.Conv2d(256, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU()
                                             )
        self.global_avg_pool1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)

        self.conv2 = nn.Conv2d(2560,256,1,1,bias=False)
        self.gamma = Parameter(torch.zeros(1))

        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self._init_weight()


    def forward(self, x):
        x1 = self.aspp1(x)

        x6 = self.global_avg_pool(x1)

        x2 = self.aspp2(x)

        x7 = self.global_avg_pool(x2)

        x3 = self.aspp3(x)

        x8 = self.global_avg_pool(x3)

        x4 = self.aspp4(x)

        x9 = self.global_avg_pool(x4)

        x5 = self.global_avg_pool1(x)



        x10 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x11 = torch.cat((x1, x2, x3, x4, x10), dim=1)
        x13 = torch.cat((x1, x2, x3, x4, x10), dim=1)
        x11 = self.conv1(x11)

        #x = self.bn1(x)
        # x = self.relu(x)
        # return self.dropout(x)

        x12 = self.global_avg_pool(x11)

        a6 = torch.cat((a1, a2, a3, a4, a5), dim=1)
        #a6 = self.conv1(a6)

        a6 = torch.cat((x13,a6),dim=1)
        #a6 = self.conv1(a6)
#        a6 = self.conv2(a6)
#        a6 = self.gamma*a6+x11
#        a6 = x11+a6
        a6 = self.conv2(a6)
        a6 = self.bn1(a6)
        a6 = self.relu(a6)
        return self.dropout(a6)
#        return a6

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)

class M_ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, scale,padding, BatchNorm):
        super(M_ASPPModule, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding,  bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        self.scale = scale

        self._init_weight()

    def forward(self, x):
        b, c, h, w = x.size()
        x = F.interpolate(x, size=(int(h/self.scale), int(w/self.scale)), mode='bilinear', align_corners=True)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

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

# x=torch.rand(2,2048,32,32).cuda()
# a=ASPP('resnet',16,nn.BatchNorm2d).cuda()
# a.eval()
# B=a(x)
# print(B.size())