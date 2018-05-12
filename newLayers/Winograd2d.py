from __future__ import print_function
from torch.autograd import Variable

import torch.nn as nn
import torch
import numpy


# set the kernels

## kernel_size == 5
G_4x4_5x5 = numpy.array([
    [    1.,      0.,     0.,      0.,      0.],
    [-2./9.,  -2./9., -2./9.,  -2./9.,  -2./9.],
    [-2./9.,   2./9., -2./9.,   2./9.,  -2./9.],
    [1./90.,  1./45., 2./45.,  4./45.,  8./45.],
    [1./90., -1./45., 2./45., -4./45.,  8./45.],
    [4./45.,  2./45., 1./45.,  1./90., 1./180.],
    [4./45., -2./45., 1./45., -1./90., 1./180.],
    [    0.,      0.,     0.,      0.,      1.]]
    )

BT_4x4_5x5 = numpy.array([
    [1.,   0.  ,  -21./4.,    0.  ,  21./4. ,    0. ,   -1.,  0.],
    [0. ,  1.   ,   1.    ,-17./4. , -17./4. ,   1.  ,  1.  , 0.],
    [0. ,  -1.  ,   1.    ,17./4.  , -17./4. ,  -1.  ,  1.  , 0.],
    [0. , 1./2. ,   1./4. ,  -5./2.,   -5./4.,     2.,    1.,   0.],
    [0. , -1./2.,   1./4. ,   5./2.,   -5./4.,    -2.,    1.,   0.],
    [0. ,  2.   ,   4.    ,-5./2.  ,  -5.    , 1./2. ,  1.  , 0.],
    [0. ,  -2.  ,   4.    , 5./2.  ,  -5.    ,-1./2. ,  1.  , 0.],
    [0. ,  -1.  ,   0.    ,21./4.  ,   0.    ,-21./4.,  0.  , 1.]]
    )

AT_4x4_5x5 = numpy.array([
    [1.,  1. , 1. ,  1.,  1. ,  8. , 8. ,  0.],
    [0.,  1. , -1.,  2.,  -2.,  4. , -4.,  0.],
    [0.,  1. , 1. ,  4.,  4. ,  2. , 2. ,  0.],
    [0.,  1. , -1.,  8.,  -8.,  1. , -1.,  1.]]
    )

## kernel_size == 3
G_4x4_3x3 = numpy.array([
    [ 1./4.,      0.,     0.],
    [-1./6.,  -1./6., -1./6.],
    [-1./6.,   1./6., -1./6.],
    [1./24.,  1./12.,  1./6.],
    [1./24., -1./12.,  1./6.],
    [    0.,      0.,    1.]]
    )

BT_4x4_3x3 = numpy.array([
    [4.,  0., -5.,  0., 1., 0.],
    [0., -4., -4.,  1., 1., 0.],
    [0.,  4., -4., -1., 1., 0.],
    [0., -2., -1.,  2., 1., 0.],
    [0.,  2., -1., -2., 1., 0.],
    [0.,  4.,  0., -5., 0., 1.]]
    )

AT_4x4_3x3 = numpy.array([
    [1., 1.,  1., 1.,  1., 0.],
    [0., 1., -1., 2., -2., 0.],
    [0., 1.,  1., 4.,  4., 0.],
    [0., 1., -1., 8., -8., 1.]]
    )

KernelSize2InputTileSize = {
        3 : 6,
        5 : 8,
        }

class Winograd2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
            padding=0,
            initialize_std = 0.05):
        super(Winograd2d, self).__init__()
        assert(stride == 1, 'Only stride = 1 is supported')
        assert(kernel_size == 5, 'Only kernel_size = 5 is supported')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.weight = nn.Parameter(torch.FloatTensor(out_channels, in_channels,
            KernelSize2InputTileSize[kernel_size],
            KernelSize2InputTileSize[kernel_size]).normal_(0, 0.01))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels).normal_(1, 0.01))

        # register buffers for parameter with no grad
        # use .float() to make sure tensors are 32-bit float numbers
        # self.register_buffer('G', torch.from_numpy(G_4x4_5x5).float())
        if kernel_size == 5:
            BT = BT_4x4_5x5
            AT = AT_4x4_5x5
            G = torch.from_numpy(G_4x4_5x5).float()
        else:
            BT = BT_4x4_3x3
            AT = AT_4x4_3x3
            G = torch.from_numpy(G_4x4_3x3).float()
        self.register_buffer('BT', torch.from_numpy(BT).float())
        self.register_buffer('AT', torch.from_numpy(AT).float())

        # determine the size
        self.input_tile_size = self.BT.shape[0]
        self.output_tile_size = self.AT.shape[0]

        # initialization
        weight_normal = torch.zeros([out_channels, in_channels, kernel_size, kernel_size],
                dtype=torch.float32).normal_(0, initialize_std)
        weight_t = weight_normal.view(out_channels * in_channels,
                kernel_size, kernel_size)
        weight_t = torch.bmm(G.unsqueeze(0).expand(weight_t.size(0), *G.size()),
                weight_t)
        GT = G.transpose(0, 1)
        weight_t = torch.bmm(weight_t,
                GT.unsqueeze(0).expand(weight_t.size(0), *GT.size()))
        weight_t = weight_t.view(out_channels, in_channels,
                BT.shape[0], BT.shape[1])
        self.weight.data.copy_(weight_t)
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
        weight_t = self.weight.unsqueeze(0).unsqueeze(3).unsqueeze(4)

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
        x_t = x_t.view(x_size).unsqueeze(1)

        # calculate the output
        y_t = x_t.mul(weight_t)
        y_t = y_t.sum(2)
        y_t_size = y_t.size()
        y_t = y_t.view(-1, 
                self.input_tile_size, self.input_tile_size)

        y = torch.bmm(self.AT.unsqueeze(0).expand(y_t.size()[0], *self.AT.size()), y_t)
        A = self.AT.transpose(0, 1)
        y = torch.bmm(y, A.unsqueeze(0).expand(y_t.size()[0], *A.size()))

        y = y.view(y_t_size[0], y_t_size[1], y_t_size[2], y_t_size[3],
                self.output_tile_size, self.output_tile_size)
        y = y.permute(0,1,2,4,3,5).contiguous().view(y_t_size[0], y_t_size[1],
                y_t_size[2] * self.output_tile_size, y_t_size[2] * self.output_tile_size)

        y = y.add(self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3))

        if additinal_padding:
            y = y[:, :, 0:output_width, 0:output_width]
        return y
