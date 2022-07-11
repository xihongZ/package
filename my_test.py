import modeling.models.danet as danet
import torch

model = danet.get_danet(backbone='resnet50', pretrained=False,
        root='./pretrain_models')
# model.eval()
# torch.manual_seed(1)
input = torch.rand(2, 3,512,512)
output = model(input)
print(output.size())