import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding

class my_pam(nn.Module):
    def __init__(self, in_channel):
        super(my_pam,self).__init__()
        self.conv_h = Conv2d(in_channel, in_channel//8, kernel_size=(1,33))
        self.conv_v = Conv2d(in_channel, in_channel//8, kernel_size=(33, 1))
        self.gamma = Parameter(torch.zeros(1))
        #self.softmax = Softmax(-1)
        self.sigmiod = Sigmoid()


    def forward(self, x):
        b, c, h, w = x.size()
        c_h = self.conv_h(x).view(b,-1,h).permute(0,2,1)
        #print(c_h.size())
        c_v = self.conv_v(x).view(b,-1,w)
        energy = torch.bmm(c_h,c_v).view(b,-1,h,w)
        energy = (64**-.5) * energy
        attention = self.sigmiod(energy)
        x_attention = x*attention
        out = x + self.gamma*x_attention
        return out


class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
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
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out





if __name__ == "__main__":
    input = torch.rand(2, 512, 4, 4)
    #print(input)
    My_pam = my_pam(512,4,4)
    output = My_pam(input)
    print(output.size())
    #print(output)