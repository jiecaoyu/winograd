#!/usr/bin/env python2
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy
import argparse
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd + '/../../')
import newLayers


class conv_pool_cnn_c_winograd(nn.Module):
    def __init__(self):
        super(conv_pool_cnn_c_winograd, self).__init__()
        self.classifer = nn.Sequential(
                nn.Dropout(0.2),
                newLayers.Winograd2d.Winograd2d(3, 96, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(96, momentum=0.003),
                nn.ReLU(inplace=True),

                newLayers.Winograd2d.Winograd2d(96, 96, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(96, momentum=0.003),
                nn.ReLU(inplace=True),

                newLayers.Winograd2d.Winograd2d(96,  96, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(96, momentum=0.003),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                newLayers.Winograd2d.Winograd2d(96, 192, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(192, momentum=0.003),
                nn.ReLU(inplace=True),

                newLayers.Winograd2d.Winograd2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(192, momentum=0.003),
                nn.ReLU(inplace=True),

                newLayers.Winograd2d.Winograd2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(192, momentum=0.003),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                newLayers.Winograd2d.Winograd2d(192, 192, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(192, momentum=0.003),
                nn.ReLU(inplace=True),

                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(192, momentum=0.003),
                nn.ReLU(inplace=True),

                nn.Conv2d(192,  100, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=6, stride=1, padding=0),

                )

    def forward(self, x):
        x = self.classifer(x)
        x = x.view(x.size(0), 100)
        return x
