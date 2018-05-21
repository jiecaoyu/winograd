from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd+'/../')
import numpy
from torch.autograd import Variable
from newLayers import *
from .mask_para import *

class GradOptimizer():
    def __init__(self, model):
        self.grad_target = []
        self.opt_operator = []
        self.kernel_size = []
        self.input_tile_size = []
        for m in model.modules():
            if isinstance(m, Winograd2d):
                self.grad_target.append(m.weight)
                self.kernel_size.append(m.kernel_size)
                self.input_tile_size.append(m.input_tile_size)
        return

    def step(self):
        for index in range(len(self.grad_target)):
            if self.kernel_size[index] == 5:
                G = torch.from_numpy(G_4x4_5x5).float().cuda()
            elif self.kernel_size[index] == 3:
                G = torch.from_numpy(G_4x4_3x3).float().cuda()
            input_tile_size = self.input_tile_size[index]
            grad_target = self.grad_target[index].grad.data
            GT = G.transpose(0, 1)
            G = torch.matmul(G, GT)
            s = grad_target.shape
            grad_target = grad_target.view(-1, input_tile_size, input_tile_size)
            grad_target = torch.bmm(G.unsqueeze(0).expand(grad_target.size(0), *G.size()),
                    grad_target)
            grad_target = torch.bmm(grad_target,
                    G.unsqueeze(0).expand(grad_target.size(0), *G.size()))
            grad_target = grad_target.view(s[0],s[1], input_tile_size, input_tile_size)
            self.grad_target[index].grad.data = grad_target
        return
    
    def step_prune(self, mask):
        for index in range(len(self.grad_target)):
            input_tile_size = self.input_tile_size[index]
            if self.kernel_size[index] == 5:
                S = torch.from_numpy(S_4x4_5x5).float()
            elif self.kernel_size[index] == 3:
                S = torch.from_numpy(S_4x4_3x3).float()
            else:
                raise Exception ('Currently not supported')
            if self.grad_target[index].grad.data.is_cuda:
                S = S.cuda()
            # need to optimize memory usage
            S = S.unsqueeze(0)
            tmp_mask = mask.mask[index].float().mul(-1.0).add(1.0)
            mask_shape = tmp_mask.shape
            tmp_mask = tmp_mask.view(mask_shape[0] * mask_shape[1], -1)
            tmp_mask = tmp_mask.unsqueeze(2).mul(S)
            tmp_mask = torch.bmm(tmp_mask, tmp_mask.transpose(1,2))
            grad_target = self.grad_target[index].grad.data
            s = grad_target.shape
            grad_target = grad_target.view(-1, input_tile_size ** 2, 1)
            grad_target = torch.bmm(tmp_mask, grad_target)
            grad_target = grad_target.view(s[0],s[1], input_tile_size, input_tile_size)
            self.grad_target[index].grad.data = grad_target
        return
