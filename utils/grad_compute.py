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
from torchvision import datasets, transforms
from torch.autograd import Variable
from newLayers import *

class GradOptimizer():
    def __init__(self, model):
        self.grad_target = []
        self.opt_operator = []
        for m in model.modules():
            if isinstance(m, Winograd2d):
                self.grad_target.append(m.weight)
                if m.kernel_size == 5:
                    continue
                else:
                    raise Exception ('Kernel size of ' + str(m.kernel_size) + " is not supported.")
        return

    def step(self):
        for index in range(len(self.grad_target)):
            grad_target = self.grad_target[index].grad.data
            G = torch.from_numpy(G_4x4_5x5).float().cuda()
            GT = G.transpose(0, 1)
            G = torch.matmul(G, GT)
            s = grad_target.shape
            grad_target = grad_target.view(-1, 8, 8)
            grad_target = torch.bmm(G.unsqueeze(0).expand(grad_target.size(0), *G.size()),
                    grad_target)
            grad_target = torch.bmm(grad_target,
                    G.unsqueeze(0).expand(grad_target.size(0), *G.size()))
            grad_target = grad_target.view(s[0],s[1],8,8)
            self.grad_target[index].grad.data = grad_target
        return
