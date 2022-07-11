import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.easpp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone


from modeling.base import build_Base
from modeling.base_pam import build_Base_mypam
from modeling.base_pam_cam import build_Base_mypam_cam


# from modeling.deeplabv3 import build_deeplabv3
# from modeling.slice import build_slice

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
 #       self.decoder = build_decoder(num_classes, backbone, BatchNorm)

 #       self.decoder = build_Base(num_classes, backbone, BatchNorm) #Restnet101
 #       self.decoder = build_Base_mypam(num_classes, backbone, BatchNorm)
        self.decoder = build_Base_mypam_cam(num_classes, backbone, BatchNorm)

#        self.decoder = build_deeplabv3(num_classes, backbone, BatchNorm)





         # if freeze_bn:
         #     self.freeze_bn()
         # ##########################################
         # self.conv2 = nn.Conv2d(2048, 512, 1, bias=False)
         # self.bn2 = BatchNorm(512)
         # self.relu2 = nn.ReLU()


    def forward(self, input):
        # x, low_level_feat = self.backbone(input)
        x1,x2,x3, x4 = self.backbone(input)
        #x = self.backbone(input)

        #x = self.aspp(x)
        #print(x.size())
        # x = self.decoder(x)
        x = self.decoder(x1,x2,x3,x4)
        #head = F.interpolate(head, size=input.size()[2:], mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x
        #return head,x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        #modules = [self.aspp, self.decoder]
        modules = [self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab(backbone='resnet', output_stride=16)
    model.eval()
    input = torch.rand(2, 3, 513, 513)
#    output = model(input)
#    print(output.size())
    output = model(input)
    print(output.size())
#    print(low_level_feat.size())

