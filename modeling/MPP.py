import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class CNBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=nn.BatchNorm2d):
        super(CNBlock, self).__init__()
        self.s1 = H1_Module(inplanes, outplanes, norm_layer=nn.BatchNorm2d)
        self.s2 = H2_Module(inplanes, outplanes, norm_layer=nn.BatchNorm2d)
        self.s3 = H3_Module(inplanes, outplanes, norm_layer=nn.BatchNorm2d)
        self.s4 = H4_Module(inplanes, outplanes, norm_layer=nn.BatchNorm2d)

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.s1(x)
        x2 = self.s2(x)
        x3 = self.s3(x)
        x4 = self.s4(x)
        x = (x1 + x2 + x3 + x4)

        return x

class H1_Module(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None):
        super(H1_Module, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(outplanes)
        self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(outplanes)
        self.conv3 = nn.Conv2d(outplanes, outplanes, kernel_size=1, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))  # H*1
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))  # 1*W
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = x1.expand(-1, -1, h, w)
        # x1 = F.interpolate(x1, (h, w))

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)
        # x2 = F.interpolate(x2, (h, w))

        x3 = (x1 + x2)
        x3 = self.conv3(x3).sigmoid()  # 到底需不需要这一行呢
        x = x * x3
        return x


class H2_Module(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None):
        super(H2_Module, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(outplanes)
        self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(outplanes)
        self.conv1_2 = nn.Conv2d(outplanes, outplanes, kernel_size=(3, 1), stride=1, padding=(2, 0), dilation=(2, 1))
        self.bn3 = norm_layer(outplanes)
        self.conv2_1 = nn.Conv2d(outplanes, outplanes, kernel_size=(1, 3), stride=1, padding=(0, 2), dilation=(1, 2))
        self.bn4 = norm_layer(outplanes)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.conv1_2(x1)
        x1 = self.bn3(x1)
        x1 = x1.expand(-1, -1, h, w)

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.conv2_1(x2)
        x2 = self.bn4(x2)
        x2 = x2.expand(-1, -1, h, w)

        x3 = (x1 + x2)
        x = x * x3
        return x


class H3_Module(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None):
        super(H3_Module, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(outplanes)
        self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(outplanes)
        self.pool1 = nn.AdaptiveAvgPool2d((20, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, 20))

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = x1.expand(-1, -1, 20, 20)
        x1 = F.interpolate(x1, size=(h, w), mode='bilinear')

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, 20, 20)
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear')

        x3 = (x1 + x2)
        x = x * x3
        return x


class H4_Module(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None):
        super(H4_Module, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(outplanes)
        self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(outplanes)
        self.pool1 = nn.AdaptiveAvgPool2d((12, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, 12))

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = x1.expand(-1, -1, 12, 12)
        x1 = F.interpolate(x1, size=(h, w), mode='bilinear')

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, 12, 12)
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear')

        x3 = (x1 + x2)
        x = x * x3
        return x