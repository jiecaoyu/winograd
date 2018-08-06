#!/usr/bin/env python2

#
# Please ignore the code here and refer to
#   WinogradPruning/utils/grad_compute.py instead.
#

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import pickle
from torch.autograd import Variable
from . import para
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd + '/../')
import newLayers

class GradOptimizer():
    def __init__(self, model, spatial_mask, prune_list=None):
        self.grad_target = []
        self.opt_operator = []
        self.kernel_size = []
        self.input_tile_size = []
        for m in model.modules():
            if isinstance(m, newLayers.Winograd2d.Winograd2d):
                self.grad_target.append(m.weight)
                self.kernel_size.append(m.kernel_size)
                self.input_tile_size.append(m.input_tile_size)
        self.spatial_mask = spatial_mask
        self.prune_list = prune_list
        if spatial_mask:
            fp_spatial_mask = open('spatial_mask.pth.tar', 'r')
            self.spatial_mask = torch.load(fp_spatial_mask)
            fp_spatial_mask.close()
        return

    def step(self):
        count = 0
        for index in range(len(self.grad_target)):
            if self.kernel_size[index] == 5:
                G = torch.from_numpy(para.G_4x4_5x5).float().cuda()
            elif self.kernel_size[index] == 3:
                G = torch.from_numpy(para.G_4x4_3x3).float().cuda()
            input_tile_size = self.input_tile_size[index]
            grad_target = self.grad_target[index].grad.data
            GT = G.transpose(0, 1)
            if not self.spatial_mask:
                G = torch.matmul(G, GT)
                s = grad_target.shape
                grad_target = grad_target.view(-1, input_tile_size, input_tile_size)
                grad_target = torch.bmm(G.unsqueeze(0).expand(grad_target.size(0), *G.size()),
                        grad_target)
                grad_target = torch.bmm(grad_target,
                        G.unsqueeze(0).expand(grad_target.size(0), *G.size()))
                grad_target = grad_target.view(s[0],s[1], input_tile_size, input_tile_size)
            else:
                s = grad_target.shape
                grad_target = grad_target.view(-1, input_tile_size, input_tile_size)
                grad_target = torch.bmm(GT.unsqueeze(0).expand(grad_target.size(0), *GT.size()),
                        grad_target)
                grad_target = torch.bmm(grad_target,
                        G.unsqueeze(0).expand(grad_target.size(0), *G.size()))
                tmp_spatial_mask = self.spatial_mask[self.prune_list[count]]
                tmp_spatial_mask = tmp_spatial_mask.view(-1,
                        tmp_spatial_mask.shape[2], tmp_spatial_mask.shape[3])
                grad_target = grad_target.mul(1.0 - tmp_spatial_mask)
                grad_target = torch.bmm(G.unsqueeze(0).expand(grad_target.size(0), *G.size()),
                        grad_target)
                grad_target = torch.bmm(grad_target,
                        GT.unsqueeze(0).expand(grad_target.size(0), *GT.size()))
                grad_target = grad_target.view(s[0],s[1], input_tile_size, input_tile_size)
            self.grad_target[index].grad.data = grad_target
            count += 1
        return
