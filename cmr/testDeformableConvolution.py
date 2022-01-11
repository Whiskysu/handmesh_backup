import torch
import torchvision.ops
from torch import nn


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size * kernel_size,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = max(h, w) / 4.

        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator
                                          )
        return x


class MNISTClassifier(nn.Module):
    def __init__(self,
                 deformable=False):
        super(MNISTClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        conv = nn.Conv2d if deformable == False else DeformableConv2d
        self.conv4 = conv(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = conv(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)  # [14, 14]
        x = torch.relu(self.conv2(x))
        x = self.pool(x)  # [7, 7]
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = self.gap(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x

input = torch.rand(28, 28)
model = MNISTClassifier()
out = model(input)
print(input.size())


#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import math
from torch import nn
from torch.nn.modules.utils import _pair

from functions.modulated_dcn_func import ModulatedDeformConvFunction
from functions.modulated_dcn_func import DeformRoIPoolingFunction

class ModulatedDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, deformable_groups=1, no_bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups
        self.no_bias = no_bias

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()
        if self.no_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        func = ModulatedDeformConvFunction(self.stride, self.padding, self.dilation, self.deformable_groups)
        return func(input, offset, mask, self.weight, self.bias)


class ModulatedDeformConvPack(ModulatedDeformConv):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, deformable_groups=1, no_bias=False):
        super(ModulatedDeformConvPack, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, deformable_groups, no_bias)

        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
                                          kernel_size=self.kernel_size,
                                          stride=(self.stride, self.stride),
                                          padding=(self.padding, self.padding),
                                          bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        func = ModulatedDeformConvFunction(self.stride, self.padding, self.dilation, self.deformable_groups)
        return func(input, offset, mask, self.weight, self.bias)


class DeformRoIPooling(nn.Module):

    def __init__(self,
                 spatial_scale,
                 pooled_size,
                 output_dim,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0):
        super(DeformRoIPooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.output_dim = output_dim
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std
        self.func = DeformRoIPoolingFunction(self.spatial_scale,
                             self.pooled_size,
                             self.output_dim,
                             self.no_trans,
                             self.group_size,
                             self.part_size,
                             self.sample_per_part,
                             self.trans_std)

    def forward(self, data, rois, offset):

        if self.no_trans:
            offset = data.new()
        return self.func(data, rois, offset)

class ModulatedDeformRoIPoolingPack(DeformRoIPooling):

    def __init__(self,
                 spatial_scale,
                 pooled_size,
                 output_dim,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0,
                 deform_fc_dim=1024):
        super(ModulatedDeformRoIPoolingPack, self).__init__(spatial_scale,
                                         pooled_size,
                                         output_dim,
                                         no_trans,
                                         group_size,
                                         part_size,
                                         sample_per_part,
                                         trans_std)

        self.deform_fc_dim = deform_fc_dim

        if not no_trans:
            self.func_offset = DeformRoIPoolingFunction(self.spatial_scale,
                                                    self.pooled_size,
                                                    self.output_dim,
                                                    True,
                                                    self.group_size,
                                                    self.part_size,
                                                    self.sample_per_part,
                                                    self.trans_std)
            self.offset_fc = nn.Sequential(
                nn.Linear(self.pooled_size * self.pooled_size * self.output_dim, self.deform_fc_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_dim, self.deform_fc_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_dim, self.pooled_size * self.pooled_size * 2)
            )
            self.offset_fc[4].weight.data.zero_()
            self.offset_fc[4].bias.data.zero_()
            self.mask_fc = nn.Sequential(
                nn.Linear(self.pooled_size * self.pooled_size * self.output_dim, self.deform_fc_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_dim, self.pooled_size * self.pooled_size * 1),
                nn.Sigmoid()
            )
            self.mask_fc[2].weight.data.zero_()
            self.mask_fc[2].bias.data.zero_()

    def forward(self, data, rois):
        if self.no_trans:
            offset = data.new()
        else:
            n = rois.shape[0]
            offset = data.new()
            x = self.func_offset(data, rois, offset)
            offset = self.offset_fc(x.view(n, -1))
            offset = offset.view(n, 2, self.pooled_size, self.pooled_size)
            mask = self.mask_fc(x.view(n, -1))
            mask = mask.view(n, 1, self.pooled_size, self.pooled_size)
            feat = self.func(data, rois, offset) * mask
            return feat
        return self.func(data, rois, offset)