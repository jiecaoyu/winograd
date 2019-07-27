#!/usr/bin/env python2
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import pickle
from torch.autograd import Variable
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd + '/../')
import newLayers
from . import para

class GradOptimizer():
    def __init__(self, model, spatial_mask=False, prune_list=None):
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
            if self.kernel_size[index] == 3:
                mask_multi = torch.from_numpy(para.mask_multi_2x2_3x3).float().cuda()
            mask_multi.div_(mask_multi.min())
            grad_target = self.grad_target[index].grad.data
            grad_target.div_(mask_multi.pow(1.5).unsqueeze(0).unsqueeze(1))
            self.grad_target[index].grad.data = grad_target
            count += 1
        return
