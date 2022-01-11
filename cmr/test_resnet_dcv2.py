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
path = '/home/su/文档/code/HandMeshV2（另一个复件）/cmr/model_save/resnet18_v1.pth'



out = model(input)
torch.save(model.state_dict(), path)                       #存储整个模型


new_model = torch.load(path)


old_net_path = '/home/su/文档/code/HandMeshV2（另一个复件）/cmr/model_save/resnet18_v1.pth'
new_net_path = '/home/su/文档/code/HandMeshV2（另一个复件）/cmr/model_save/resnet18_v1.onnx'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

# 导入模型
net = model.to(device)
net.load_state_dict(torch.load(old_net_path, map_location=device))
net.eval()

input = torch.randn(1, 3, 224, 224).to(device)   # BCHW  其中Batch必须为1，因为测试时一般为1，尺寸HW必须和训练时的尺寸一致
torch.onnx.export(net, input, new_net_path, verbose=False)


