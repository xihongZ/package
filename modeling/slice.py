
from torch.autograd import Variable
import torchvision.models as models
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import numpy as np



def init_bilinear(arr):
    weight = np.zeros(np.prod(arr.size()), dtype='float32')
    shape = arr.size()
    f = np.ceil(shape[3] / 2.)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(np.prod(shape)):
        x = i % shape[3]
        y = (i / shape[3]) % shape[2]
        weight[i] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
    return torch.from_numpy(weight.reshape(shape))

def set_require_grad_to_false(m):
    for param in m.parameters():
        param.requires_grad = False

class SliceLayer(nn.Module):

    def __init__(self):
        super(SliceLayer, self).__init__()

    def forward(self, input_data):
        """
        slice into several single piece in a specific dimension. Here for dim=1
        """
        sliced_list = []
        for idx in range(input_data.size()[1]):
            sliced_list.append(input_data[:, idx, :, :].unsqueeze(1))

        return sliced_list


class ConcatLayer(nn.Module):

    def __init__(self):
        super(ConcatLayer, self).__init__()

    def forward(self, input_data_list, dim):
        concat_feats = torch.cat((input_data_list), dim=dim)
        return concat_feats


class Slice(nn.Module):
    def __init__(self,num_classes, backbone, BatchNorm):
        super(Slice, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24

        self.slice_layer = SliceLayer()
        self.concat_layer = ConcatLayer()

        self.conv0 = nn.Sequential(
                                    nn.Conv2d(64, 128,3,padding=1,bias=False),
                                    BatchNorm(128),
                                    nn.ReLU(),
                                    nn.Conv2d(128,64,3,padding=1,bias=False),
                                    BatchNorm(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 1, 1,bias=True)
                                  )

        self.conv1 = nn.Sequential(
                                    nn.Conv2d(256,128,3,padding=1,bias=False),
                                    BatchNorm(128),
                                    nn.ReLU(),
                                    nn.Conv2d(128,64,3,padding=1,bias=False),
                                    BatchNorm(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 1, 1, bias=True)
                                    )

        #self.conv1 = nn.Conv2d(256,1,1,bias=True)
        # self.conv2 = nn.Sequential(
        #                             nn.Conv2d(512,1,1,bias=False),
        #                             BatchNorm(1),
        #                             nn.ReLU())
        self.conv2 = nn.Sequential(
                                    nn.Conv2d(512,128,5,padding=2,bias=False),
                                    BatchNorm(128),
                                    nn.ReLU(),
                                    nn.Conv2d(128,64,3,padding=1,bias=False),
                                    BatchNorm(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64,1,1,bias=True)
                                )
        #self.conv2 = nn.Conv2d(256,1,1,bias=True)
        # self.conv3 = nn.Sequential(nn.Conv2d(1024,1,1,bias=False),
        #                            BatchNorm(1),
        #                            nn.ReLU())
        self.conv3 = nn.Sequential(
                                    nn.Conv2d(1024,512,5,padding=2,bias=False),
                                    BatchNorm(512),
                                    nn.ReLU(),
                                    nn.Conv2d(512,64,3,padding=1,bias=False),
                                    BatchNorm(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64,1, 1, bias=True)
        )
        #self.conv3 = nn.Conv2d(512,1,1,bias=True)
        # self.conv4 = nn.Sequential(
        #                             nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,bias=False),
        #                             BatchNorm(256),
        #                             nn.ReLU(),
        #                             #nn.Dropout(0.5),
        #                             nn.Conv2d(256,num_classes,kernel_size=1,stride=1)
        #                            )
        # self.conv4 = nn.Sequential(
        #                             nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #                             BatchNorm(64),
        #                             nn.ReLU(),
        #                             #nn.Dropout(0.1),
        #                             nn.Conv2d(64, num_classes, kernel_size=1, stride=1,bias=True)
        #)
        self.conv4 = nn.Conv2d(256,num_classes,1,1,bias=True)
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(256, 64, 3, bias=False),
        #     BatchNorm(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, num_classes, 1, bias=True)
        # )
        self.ce_fusion = nn.Conv2d(num_classes * 3, num_classes, kernel_size=1, groups=num_classes, bias=True)
        #self.ce_fusion = nn.Conv2d(num_classes * 3, num_classes, 1,1, bias=True)
        #self._init_weight()

    def forward(self, x,size,low_level0,low_level1,low_level2,low_level3):
        low_level0 = self.conv0(low_level0)
        low_level0 = F.interpolate(low_level0, size, mode='bilinear', align_corners=True)
        low_level1 = self.conv1(low_level1)
        low_level1 = F.interpolate(low_level1, size, mode='bilinear', align_corners=True)
        #low_level1 = set_require_grad_to_false(low_level1)
        #low_level2 = self.conv2(low_level2)
        #low_level2 = F.interpolate(low_level2, size, mode='bilinear', align_corners=True)
        #low_level2 = set_require_grad_to_false(low_level2)
        #low_level3 = self.conv3(low_level3)
        #low_level3 = F.interpolate(low_level3, size, mode='bilinear', align_corners=True)
        #low_level3 = set_require_grad_to_false(low_level3)
        x = self.conv4(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        #x = set_require_grad_to_false(x)
        sliced_list = self.slice_layer(x)
        # sliced_list1 = self.slice_layer(low_level1)
        # sliced_list2 = self.slice_layer(low_level2)
        # sliced_list3 = self.slice_layer(low_level3)
        final_sliced_list = []
        for i in range(len(sliced_list)):
            final_sliced_list.append(sliced_list[i])
            # final_sliced_list.append(sliced_list1[i])
            # final_sliced_list.append(sliced_list2[i])
            # final_sliced_list.append(sliced_list3[i])
            final_sliced_list.append(low_level0)
            final_sliced_list.append(low_level1)
            #final_sliced_list.append(low_level2)
            #final_sliced_list.append(low_level3)
        concat_feats = self.concat_layer(final_sliced_list,dim=1)
        fused_feats = self.ce_fusion(concat_feats)
        return fused_feats

def build_slice(num_classes, backbone, BatchNorm):
    return Slice(num_classes, backbone, BatchNorm)



