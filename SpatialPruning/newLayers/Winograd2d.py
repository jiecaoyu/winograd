#!/usr/bin/env python2
from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn as nn
import torch
import numpy
import math
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd + '/../')
import utils


KernelSize2InputTileSize = {
        3 : 6,
        5 : 8,
        }

class Winograd2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
            padding=0, groups=1, bias=True):
        super(Winograd2d, self).__init__()
        assert(stride == 1), 'Only stride = 1 is supported'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.need_bias = bias

        assert((in_channels % groups == 0)), 'in_channels % groups != 0'
        self.weight = nn.Parameter(torch.FloatTensor(
            out_channels, int(in_channels/groups),
            KernelSize2InputTileSize[kernel_size],
            KernelSize2InputTileSize[kernel_size]).normal_(0, 0.01))
        if self.need_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).normal_(1, 0.01))

        # register buffers for parameter with no grad
        # use .float() to make sure tensors are 32-bit float numbers
        # self.register_buffer('G', torch.from_numpy(G_4x4_5x5).float())
        if kernel_size == 5:
            BT = utils.para.BT_4x4_5x5
            AT = utils.para.AT_4x4_5x5
            G = torch.from_numpy(utils.para.G_4x4_5x5).float()
        else:
            BT = utils.para.BT_4x4_3x3
            AT = utils.para.AT_4x4_3x3
            G = torch.from_numpy(utils.para.G_4x4_3x3).float()
        self.register_buffer('BT', torch.from_numpy(BT).float())
        self.register_buffer('AT', torch.from_numpy(AT).float())

        # determine the size
        self.input_tile_size = self.BT.shape[0]
        self.output_tile_size = self.AT.shape[0]

        # initialization
        n = in_channels * kernel_size * kernel_size
        stdv = 1. / math.sqrt(n)
        weight_normal = torch.zeros(
                [out_channels, int(in_channels / groups), kernel_size, kernel_size],
                dtype=torch.float32).uniform_(-stdv, stdv)
        weight_t = weight_normal.view(int(out_channels * in_channels / groups),
                kernel_size, kernel_size)
        weight_t = torch.bmm(G.unsqueeze(0).expand(weight_t.size(0), *G.size()),
                weight_t)
        GT = G.transpose(0, 1)
        weight_t = torch.bmm(weight_t,
                GT.unsqueeze(0).expand(weight_t.size(0), *GT.size()))
        weight_t = weight_t.view(out_channels, int(in_channels / groups),
                BT.shape[0], BT.shape[1])
        self.weight.data.copy_(weight_t)

        if self.need_bias:
            self.bias.data.uniform_(-stdv, stdv)
        del weight_normal, weight_t
        return

    def forward(self, x):
        # add padding
        if self.padding != 0:
            x = torch.nn.functional.pad(x, [self.padding] * 4, 'constant', 0)

        s = x.size()
        # generate output tensor shape
        output_width = s[2] - self.kernel_size + 1
        # if(self.kernel_size == 5):
        #     assert(output_width % 4 == 0, 'with kernel_size = 5, width % 4 should be 0.')
        if output_width % self.output_tile_size != 0:
            additinal_padding = True
            additinal_padding_size = self.output_tile_size - \
                    (output_width % self.output_tile_size)
            x = torch.nn.functional.pad(x,
                    [0, additinal_padding_size, 0, additinal_padding_size], 'constant', 0)
        else:
            additinal_padding = False

        # prepare the weights // use torch.bmm()
        weight_size = list(self.weight.shape)
        weight_size = [self.groups, int(self.out_channels / self.groups)] \
                + weight_size[1:]
        weight_t = self.weight.view(weight_size)

        # prepare the inputs
        x = x.unfold(dimension=2, size=self.input_tile_size, step=self.output_tile_size)\
                .unfold(dimension=3, size=self.input_tile_size, step=self.output_tile_size)
        x_size = x.size()
        x = x.contiguous().view(-1,
                self.input_tile_size, self.input_tile_size)
        x_t = torch.bmm(self.BT.unsqueeze(0).expand(x.size()[0], *self.BT.size()), x)
        B = self.BT.transpose(0, 1)
        x_t = torch.bmm(x_t, B.unsqueeze(0).expand(x.size()[0], *B.size()))

        # prepare the shape of the inputs and weights
        x_t = x_t.view(
                [x_size[0], self.groups, int(x_size[1]/self.groups),\
                        x_size[2], x_size[3], x_size[4], x_size[5]])

        # calculate the output
        # this computation strategy destory the memory usage and need to optimize
        weight_t = weight_t.permute(0, 3, 4, 1, 2)
        weight_t = weight_t.contiguous().view(
                self.groups * self.input_tile_size * self.input_tile_size,
                -1, int(self.in_channels / self.groups))
        x_t = x_t.permute(1, 5, 6, 2, 0, 3, 4)
        x_t = x_t.contiguous().view(
                self.groups * self.input_tile_size * self.input_tile_size,
                int(self.in_channels / self.groups), -1)

        y_t = torch.bmm(weight_t, x_t)
        y_t = y_t.view(self.groups, self.input_tile_size, self.input_tile_size,
                int(self.out_channels / self.groups), x_size[0],
                x_size[2], x_size[3])
        y_t = y_t.permute(4, 0, 3, 5, 6, 1, 2).contiguous()
        y_t_size = y_t.size()
        y_t = y_t.view(-1, 
                self.input_tile_size, self.input_tile_size)

        y = torch.bmm(self.AT.unsqueeze(0).expand(y_t.size()[0], *self.AT.size()), y_t)
        A = self.AT.transpose(0, 1)
        y = torch.bmm(y, A.unsqueeze(0).expand(y_t.size()[0], *A.size()))

        y = y.view(y_t_size[0], y_t_size[1] * y_t_size[2], y_t_size[3], y_t_size[4],
                self.output_tile_size, self.output_tile_size)
        y = y.permute(0,1,2,4,3,5).contiguous().view(y_t_size[0], y_t_size[1] * y_t_size[2],
                y_t_size[3] * self.output_tile_size, y_t_size[3] * self.output_tile_size)

        if self.need_bias:
            y = y.add(self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3))

        if additinal_padding:
            y = y[:, :, 0:output_width, 0:output_width]
        y = y.contiguous()
        return y
