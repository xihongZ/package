###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module']


class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.query_conv_1 = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=3,padding=1,dilation=1)
        self.a = Parameter(torch.ones(1))

        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv_1 = Conv2d(in_channels=in_dim, out_channels=in_dim // 8,  kernel_size=3,padding=1,dilation=1)
        self.b = Parameter(torch.ones(1))

        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.max = MaxPool2d(stride=1, kernel_size=3, padding=1)
        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.max(self.query_conv(x)).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.max(self.key_conv(x)).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class CCM(Module):
    def __init__(self,h,w):
        super(CCM, self).__init__()
        self.con_h=nn.Conv2d(h,h,kernel_size=1,stride=1)
        self.con_w = nn.Conv2d(w, w, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, input):
        b, c, h, w = input.size()
        permute1=input.permute(0,2,1,3)#b,h,c,w
        con_h=self.con_h(permute1).permute(0,2,1,3)
        #print(con_h.size())

        permute2 = input.permute(0, 3, 1, 2)  # b,w,c,h
        con_w=self.con_w(permute2).permute(0,2,3,1)
        #print(con_w.size())
        con=(con_h+con_w).view(b,c,-1)
        attention = self.softmax(con).view(b,c,h,w)
        out=attention.mul(input) * self.gamma+input
        return out
