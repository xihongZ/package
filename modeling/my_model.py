#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:chenxiaolong
# datetime:2019/6/26 16:01
# software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F



class LAM(nn.Module):
    def __init__(self, in_dim,size=3):
        super(LAM, self).__init__()
        if size%2==0:
            size=size+1
        self.size = size
        self.max1 = nn.MaxPool2d(stride=1, kernel_size=(self.size, 1))
        self.max2 = nn.MaxPool2d(stride=1, kernel_size=(1, self.size))
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.ones(1))

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.max3 = nn.MaxPool2d(stride=1, kernel_size=(self.size, self.size),padding=self.size // 2)
    def forward(self, input):
        b, c, h, w = input.size()
        x1 = self.query_conv(input)
        x1 = F.pad(input, (0, 0, self.size // 2, self.size // 2))
        x1 = self.max1(x1)

        x2 = self.key_conv(input)
        x2 = F.pad(input, (self.size // 2, self.size // 2, 0, 0))
        x2 = self.max2(x2)

        # x = (x1 + x2)

        x = torch.cat((x1, x2), dim=-2).view(b, 2 * c, h, w).permute(0, 2, 3, 1).view(b, h, w, c, 2)
        x = torch.max(x, dim=-1)[0].permute(0, 3, 1, 2)


        # energy_new = torch.max(x, -1, keepdim=True)[0].expand_as(x) - x
        softmax = self.softmax(x.view(b, -1, h * w)).view(b, c, h, w)
        out = self.gamma * softmax.mul(input) + input
        return out

class CAM(nn.Module):
    def __init__(self,in_chanel, size=272):
        super(CAM, self).__init__()

        self.size = size
        self.con=nn.Conv2d(in_chanel*in_chanel,in_chanel,kernel_size=1,stride=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        b, c, h, w = input.size()
        # x1=input.view(b, -1, h * w)
        # maxk = max((1,self.size))
        # _,pred=x1.topk(maxk,dim=-1,largest=True,sorted=False)
        # mask,a=torch.sort(pred,dim=-1)
        # pred=torch.gather(_,dim=-1,index=a)
        # pred1=pred.permute(0,2,1)
        # energy = torch.bmm(pred, pred1)


        proj_query = input.view(b, c, -1)
        proj_key = input.view(b, c, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = input.view(b, c, -1)

        # energy=energy.view(b,-1,1,1)
        # out=self.con(energy)
        #
        # scale = torch.sigmoid(out).expand_as(input)
        # print(energy.size())

        # softmax = self.softmax(x.view(b, -1, h * w)).view(b, c, h, w)
        # out = self.gamma * softmax + input
        # return input*scale+input
        out = torch.bmm(attention, proj_value)
        out = out.view(b, c, h, w)

        out = self.gamma*out + input
        return out


class HighDivModule(nn.Module):
    def __init__(self, in_channels, order=1):
        super(HighDivModule, self).__init__()

        self.gamma = nn.Parameter(torch.zeros(1))
        self.order = order
        self.inter_channels = in_channels // 8 * 2
        for j in range(self.order):
            for i in range(j + 1):
                name = 'order' + str(self.order) + '_' + str(j + 1) + '_' + str(i + 1)
                setattr(self, name, nn.Sequential(nn.Conv2d(in_channels, self.inter_channels, 1, padding=0, bias=False))
                        )
        for i in range(self.order):
            name = 'convb' + str(self.order) + '_' + str(i + 1)
            setattr(self, name, nn.Sequential(nn.Conv2d(self.inter_channels, in_channels, 1, padding=0, bias=False),
                                              nn.Sigmoid()
                                              )
                    )

    def forward(self, x):
        y = []
        for j in range(self.order):
            for i in range(j + 1):
                name = 'order' + str(self.order) + '_' + str(j + 1) + '_' + str(i + 1)
                layer = getattr(self, name)
                y.append(layer(x))
        y_ = []
        cnt = 0
        for j in range(self.order):
            y_temp = 1
            for i in range(j + 1):
                y_temp = y_temp * y[cnt]
                cnt += 1
            y_.append(F.relu(y_temp))

        # y_ = F.relu(y_)
        y__ = 0
        for i in range(self.order):
            name = 'convb' + str(self.order) + '_' + str(i + 1)
            layer = getattr(self, name)
            y__ += layer(y_[i])
        out = x * y__ / self.order
        out = out * self.gamma + x
        return out  # , y__/ self.order

# class AA_Module(nn.Module):
#     """ Position attention module"""
#     #Ref from SAGAN
#     def __init__(self, in_dim):
#         super(AA_Module, self).__init__()
#         self.chanel_in = in_dim
#
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = nn.Parameter(torch.ones(1))
#
#         self.pool=nn.AdaptiveAvgPool2d((8,8))
#
#         self.aphal = nn.Parameter(torch.ones(1))
#         self.softmax = nn.Softmax(dim=-1)
#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X (HxW) X (HxW)
#         """
#         m_batchsize, C, height, width = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
#         proj_key = self.pool(self.key_conv(x)).view(m_batchsize, -1, 64)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(self.aphal * energy)
#         out = torch.bmm(proj_key, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, height, width)
#
#         out = self.gamma*out + x
#         return out


if __name__ == "__main__":
    model = LAM(64,3)
    # model.eval()
    input = torch.rand(1, 64, 256,256)
    output = model(input)
    print(output.size())