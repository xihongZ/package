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



def Euclidean_Distance(a,b):
    a_batchsize, a_C, height, width = a.size()
    a = a.view(a_batchsize,a_C)
    b_batchsize, b_C, height, width = b.size()
    b = b.view(b_batchsize,b_C)
    distance = F.pairwise_distance(a,b)
    for i in range(distance.size()[0]):
        d = 1/(1+distance[i])
        distance[i] = d
    return distance.view(a_batchsize,1,1,1).detach()


# def Euclidean_Distance(a,b,c,d,e,f):
#     a_batchsize, a_C, height, width = a.size()
#     a = a.view(a_batchsize,a_C)
#     b_batchsize, b_C, height, width = b.size()
#     b = b.view(b_batchsize,b_C)
#     c_batchsize, c_C, height, width = c.size()
#     c = c.view(c_batchsize,c_C)
#     d_batchsize, d_C, height, width = d.size()
#     d = d.view(d_batchsize,d_C)
#     e_batchsize, e_C, height, width = e.size()
#     e = e.view(e_batchsize,e_C)
#     f_batchsize, f_C, height, width = f.size()
#     f = f.view(f_batchsize,f_C)
#     distance1 = F.pairwise_distance(a,f)
#     distance2 = F.pairwise_distance(b,f)
#     distance3 = F.pairwise_distance(c,f)
#     distance4 = F.pairwise_distance(d,f)
#     distance5 = F.pairwise_distance(e,f)

    # for i in range(distance1.size()[0]):
    #     g = 1/(1+distance1[i])
    #     distance1[i] = g
    # for i in range(distance2.size()[0]):
    #     g = 1/(1+distance2[i])
    #     distance2[i] = g
    # for i in range(distance3.size()[0]):
    #     g = 1/(1+distance3[i])
    #     distance3[i] = g
    # for i in range(distance4.size()[0]):
    #     g = 1/(1+distance4[i])
    #     distance4[i] = g
    # for i in range(distance5.size()[0]):
    #     g = 1/(1+distance5[i])
    #     distance5[i] = g
    # # h = np.vstack((distance1.data.numpy(),distance2.data.numpy(),distance3.data.numpy(),distance4.data.numpy(),distance5.data.numpy()))
    # # h = Variable(torch.from_numpy(h))
    # h = torch.cat((distance1.unsqueeze(0),distance2.unsqueeze(0),distance3.unsqueeze(0),distance4.unsqueeze(0),distance5.unsqueeze(0)),dim=0)
    # k = nn.Softmax(dim=0)
    # h = k(h)
    # h0 = h[0,:].view(a_batchsize,1,1,1).detach()
    # h1 = h[1,:].view(a_batchsize,1,1,1).detach()
    # h2 = h[2,:].view(a_batchsize,1,1,1).detach()
    # h3 = h[3,:].view(a_batchsize,1,1,1).detach()
    # h4 = h[4,:].view(a_batchsize,1,1,1).detach()
    # return h0,h1,h2,h3,h4


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
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(1280, 256, 3,bias=False),
        #     BatchNorm(256),
        #     nn.ReLU()
        # )

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

        # e1,e2,e3,e4,e5 = Euclidean_Distance(x6,x7,x8,x9,x5,x12)
        # a1 = x1 * e1
        # a2 = x2 * e2
        # a3 = x3 * e3
        # a4 = x4 * e4
        # a5 = x10 * e5

        a1 = x1*Euclidean_Distance(x6,x12)
        a2 = x2*Euclidean_Distance(x7,x12)
        a3 = x3*Euclidean_Distance(x8,x12)
        a4 = x4*Euclidean_Distance(x9,x12)
        a5 = x10*Euclidean_Distance(x5,x12)



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