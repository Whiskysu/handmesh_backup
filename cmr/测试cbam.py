from cmr.resnet import *
import torch

# 测试使用torchwatch
import sys
# import tensorwatch as tw
# import torchvision.models
#
# alexnet_model = torchvision.models.alexnet()
# tw.draw_model(alexnet_model, [1, 3, 224, 224])


# 测试使用tensorboard
from tensorboardX import SummaryWriter
net = resnet18()#网络模型
writer = SummaryWriter('graph_exp')
inputs = torch.randn(1,3,224,224)#生成输入的张量
writer.add_graph(net,inputs)#讲模型和输入记录到Summary中
writer.close()
print(net)
input = torch.rand(1, 3, 224, 224)


model = resnet18(pretrained=False)



out = model(input)
print(input.size())





