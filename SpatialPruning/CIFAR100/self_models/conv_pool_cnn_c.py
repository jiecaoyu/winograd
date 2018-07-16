#!/usr/bin/env python2
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import cPickle as pickle
import numpy
import argparse


class conv_pool_cnn_c(nn.Module):
    def __init__(self):
        super(conv_pool_cnn_c, self).__init__()
        self.classifer = nn.Sequential(
                nn.Dropout(0.2),
                nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),

                nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),

                nn.Conv2d(96,  96, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),

                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),

                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),

                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),

                nn.Conv2d(192,  100, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),

                )

    def forward(self, x):
        x = self.classifer(x)
        x = x.view(x.size(0), 100)
        return x
