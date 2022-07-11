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
    def __init__(self,in_chanel, size=10):
        super(CAM, self).__init__()

        self.size = size
        self.in_chanel=in_chanel
        self.con=nn.Conv2d(in_chanel*in_chanel,in_chanel,kernel_size=1,stride=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.weight = nn.Parameter(torch.rand(1,self.in_chanel,self.in_chanel))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, input):
        b, c, h, w = input.size()
        # x1=input.view(b, -1, h * w)
        # maxk = max((1,self.size))
        # _,pred=x1.topk(maxk,dim=-1,largest=True,sorted=False)
        # mask,a=torch.sort(pred,dim=-1)
        # pred=torch.gather(_,dim=-1,index=a)

        # pred=x1.mean(dim=-1,keepdim=True)

        x=(input*self.avg_pool(input)).view(b,c,-1)
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)
        pred=torch.cat((mean,std),-1)

        pred1=pred.permute(0,2,1)
        energy = torch.bmm(pred, pred1)

        attention = self.softmax(energy)

        out = torch.bmm(attention, input.view(b, c, -1))
        out = out.view(b, c, h, w)

        out = self.gamma * out + input
        return out
        # proj_query = input.view(b, c, -1)
        # proj_key = input.view(b, c, -1).permute(0, 2, 1)
        # energy = torch.bmm(proj_query, proj_key)
        #
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        # attention = self.softmax(energy_new)
        # proj_value = input.view(b, c, -1)

        # energy = energy.view(b,-1,1,1)
        # out = self.con(energy)
        # #
        # scale = torch.sigmoid(out).expand_as(input)
        #print(energy.size())

        # scale = self.weight.expand(b,self.in_chanel,self.in_chanel).mul(energy).sum(dim=-1,keepdim=True).view(b,c,1,1).expand_as(input)
        # print(scale.size())

        # softmax = self.softmax(x.view(b, -1, h * w)).view(b, c, h, w)
        # out = self.gamma * softmax + input
        # return input*scale+input
        # out = torch.bmm(attention, proj_value)
        # out = out.view(b, c, h, w)

        # out = self.gamma*out + input
        # return out

class EN_CAM(nn.Module):
    def __init__(self,in_dim, scale=2):
        super(EN_CAM, self).__init__()

        self.scale = scale
        in_dim=in_dim*scale**2
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        x_permuted = x.permute(0, 3, 2, 1)
        x_permuted = x_permuted.contiguous().view((b, w, int(h / (self.scale)) ,c * self.scale))

        x_permuted = x_permuted.permute(0, 2, 1, 3)
        x_permuted = x_permuted.contiguous().view((b, int(h / (self.scale)), int(w / (self.scale)), c * self.scale * self.scale))
        x_permuted = x_permuted.permute(0, 3, 1, 2)

        m_batchsize, C, height, width = x_permuted.size()
        proj_query = self.query_conv(x_permuted).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x_permuted).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x_permuted).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x_permuted



        x_permuted = out.permute(0, 2, 3, 1)
        x_permuted = x_permuted.contiguous().view((m_batchsize, height, width * self.scale, int(C / (self.scale))))
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        x_permuted = x_permuted.contiguous().view((m_batchsize, width * self.scale, height * self.scale, int(C / (self.scale * self.scale))))
        x = x_permuted.permute(0, 3, 2, 1)
        return x
# class SpatialGroupEnhance(nn.Module):
#     def __init__(self, batch,groups = 32):
#         super(SpatialGroupEnhance, self).__init__()
#         self.batch=batch
#         self.groups   = groups
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.weight   = nn.Parameter(torch.zeros(1, groups, 1, 1))
#         self.bias     = nn.Parameter(torch.ones(1, groups, 1, 1))
#         self.sig      = nn.Sigmoid()
#         self.threshold = nn.Parameter(torch.rand(self.batch*groups,1))
#         self.layer_weight = nn.Parameter(torch.rand(self.batch * groups, 1))
#         self.pro=nn.Parameter(torch.ones(1))
#     def forward(self, x): # (b, c, h, w)
#         b, c, h, w = x.size()
#         input=x
#         x = x.view(b * self.groups, -1, h, w)
#         xn = x * self.avg_pool(x)
#         xn = xn.sum(dim=1, keepdim=True)
#         t = xn.view(b * self.groups, -1)
#         t = t - t.mean(dim=1, keepdim=True)
#         std = t.std(dim=1, keepdim=True) + 1e-5
#         t = t / std
#         threshold=torch.abs(self.pro*self.sig(self.threshold))
#         # print(t.size(),threshold.size())
#         if t.size(0)!=threshold.size(0):
#             return input
#         less=torch.lt(threshold.expand_as(t),t)#(t<threshold)
#         more=torch.gt((-threshold).expand_as(t),t)#(t>(-threshold))
#         select=less&more
#
#         ww=torch.ones(b * self.groups,h*w).cuda()*self.layer_weight
#         t[select]=ww[select]
#
#         t = t.view(b, self.groups, h, w)
#         t = t + self.bias# t * self.weight + self.bias
#         t = t.view(b * self.groups, 1, h, w)
#         x = x * t #self.sig(t)
#         x = x.view(b, c, h, w)
#         return x+input
# class SpatialGroupEnhance(nn.Module):
#     def __init__(self, groups = 32):
#         super(SpatialGroupEnhance, self).__init__()
#         self.groups   = groups
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.weight   = nn.Parameter(torch.zeros(1, groups, 1, 1))
#         self.bias     = nn.Parameter(torch.ones(1, groups, 1, 1))
#         self.sig      = nn.Sigmoid()
#         self.threshold = nn.Parameter(torch.rand(1,groups))
#         self.layer_weight = nn.Parameter(torch.rand(1, groups))
#         self.pro=nn.Parameter(torch.ones(1))
#     def forward(self, x): # (b, c, h, w)
#         b, c, h, w = x.size()
#         input=x
#         x = x.view(b * self.groups, -1, h, w)
#         xn = x * self.avg_pool(x)
#         xn = xn.sum(dim=1, keepdim=True)
#         t = xn.view(b * self.groups, -1)
#         t = t - t.mean(dim=1, keepdim=True)
#         std = t.std(dim=1, keepdim=True) + 1e-5
#         t = t / std
#         expand_threshold=self.threshold.expand(b,self.groups)
#         expand_threshold1=expand_threshold.reshape(b*self.groups,1)
#         threshold=torch.abs(self.pro*self.sig(expand_threshold1))
#         # print(t.size(),threshold.size())
#         # if t.size(0)!=threshold.size(0):
#         #     return input
#         less=torch.lt(threshold.expand_as(t),t)#(t<threshold)
#         more=torch.gt((-threshold).expand_as(t),t)#(t>(-threshold))
#         select=less&more
#
#         layer_weight=self.layer_weight.expand(b,self.groups)
#         layer_weight1=   layer_weight.reshape(b*self.groups,1)
#         ww=torch.ones(b * self.groups,h*w).cuda()*layer_weight1
#         t[select]=ww[select]
#
#         t = t.view(b, self.groups, h, w)
#         t = t * self.weight + self.bias#t + self.bias
#         t = t.view(b * self.groups, 1, h, w)
#         x = x * t #self.sig(t)
#         x = x.view(b, c, h, w)
#         return x+input
class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups = 64):
        super(SpatialGroupEnhance, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight   = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias     = nn.Parameter(torch.ones(1, groups, 1, 1))
        self.sig      = nn.Sigmoid()

    def forward(self, x): # (b, c, h, w)
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x

class CCM(nn.Module):
    def __init__(self, h, w):
        super(CCM, self).__init__()
        self.con_h = nn.Conv2d(h, h, kernel_size=1, stride=1)
        self.con_w = nn.Conv2d(w, w, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, input):
        b, c, h, w = input.size()
        permute1 = input.permute(0, 2, 1, 3)  # b,h,c,w
        con_h = self.con_h(permute1).permute(0, 2, 1, 3)
        # print(con_h.size())

        permute2 = input.permute(0, 3, 1, 2)  # b,w,c,h
        con_w = self.con_w(permute2).permute(0, 2, 3, 1)
        # print(con_w.size())
        con = (con_h + self.gamma * con_w).view(b, c, -1)
        attention = self.softmax(con).view(b, c, h, w)
        out = attention.mul(input) + input
        return out

class HighDivModule(nn.Module):
    def __init__(self, in_channels, order=1):
        super(HighDivModule, self).__init__()
        self.order = order
        self.inter_channels = in_channels // 8 * 2
        self.gamma = nn.Parameter(torch.zeros(1))
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

class AA_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, base=8):
        super(AA_Module, self).__init__()
        self.chanel_in = in_dim
        self.base = base
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))

        self.pool=nn.AdaptiveAvgPool2d((self.base, self.base))

        self.aphal = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
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
        proj_key = self.pool(self.key_conv(x)).view(m_batchsize, -1, self.base*self.base)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(self.aphal * energy)
        out = torch.bmm(proj_key, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class AA2_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(AA2_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.gamma1 = nn.Parameter(torch.ones(1))

        self.pool=nn.AdaptiveAvgPool2d((8,8))
        self.pool1 = nn.AdaptiveAvgPool2d((8, 8))

        self.aphal = nn.Parameter(torch.ones(1))
        self.aphal1 = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
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
        proj_key = self.pool(self.key_conv(x)).view(m_batchsize, -1, 64)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(self.aphal * energy)
        out = torch.bmm(proj_key, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        proj_key1 = self.pool1(self.key_conv1(x)).view(m_batchsize, -1, 64)
        energy1 = torch.bmm(proj_query, proj_key1)
        attention1 = self.softmax(self.aphal1 * energy1)
        out1 = torch.bmm(proj_key1, attention1.permute(0, 2, 1))
        out2 = out1.view(m_batchsize, C, height, width)

        out = self.gamma*out + x+self.gamma1*out2
        return out

# class AA_Module(nn.Module):
#     """ Position attention module"""
#     #Ref from SAGAN
#     def __init__(self, in_dim):
#         super(AA_Module, self).__init__()
#         self.chanel_in = in_dim
#
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         # self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = nn.Parameter(torch.ones(1))
#
#         # self.pool=nn.AdaptiveMaxPool2d((8,8))
#         self.in_dim=in_dim
#         self.base = nn.Parameter(torch.ones(1,in_dim, 64))
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
#         proj_key = self.base.expand(m_batchsize,self.in_dim,64)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(self.aphal * energy)
#         out = torch.bmm(proj_key, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, height, width)
#
#         out = self.gamma*out + x
#         return out

class CCAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim,channel=64):
        super(CCAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.channel = channel
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.channel, kernel_size=1)
        self.aphal = nn.Parameter(torch.ones(1))
        self.martx= nn.Parameter(torch.FloatTensor(1,64*64,64))
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        # print(x.size())
        proj_query = x.view(m_batchsize, C, -1)
        proj_key1 =self.martx.expand(m_batchsize,64*64,64)
        # print(self.martx)
        proj_key = proj_key1.permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key1)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax( self.aphal * energy_new)


        out = torch.bmm(attention, proj_key)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class CAMM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAMM_Module, self).__init__()
        self.chanel_in = in_dim

        self.pool = nn.AdaptiveAvgPool2d((8,8))
        self.gamma = nn.Parameter(torch.ones(1))
        self.aphal = nn.Parameter(torch.ones(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x1):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        x = self.pool(x1)
        m_batchsize, C, height, width = x1.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax( self.aphal * energy_new)
        proj_value = x1.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x1
        return out






if __name__ == "__main__":
    model = LAM(64,3)
    # model.eval()
    input = torch.rand(1, 64, 256,256)
    output = model(input)
    print(output.size())