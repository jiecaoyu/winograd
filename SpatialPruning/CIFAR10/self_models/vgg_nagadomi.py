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


class vgg_nagadomi(nn.Module):
    def __init__(self):
        super(vgg_nagadomi, self).__init__()
        self.feature = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                )
        self.fc0        = nn.Linear(256 * 4 * 4, 1024)
        self.relu_fc0   = nn.ReLU(inplace=True)

        self.dropout3   = nn.Dropout(0.5)

        self.fc1        = nn.Linear(1024, 1024)
        self.relu_fc1   = nn.ReLU(inplace=True)

        self.dropout4   = nn.Dropout(0.5)
        self.fc2        = nn.Linear(1024, 10)
        return

    def forward(self, x):
        out = self.feature(x)

        out = out.view(out.size(0), 256 * 4 * 4)

        out = self.fc0(out)
        out = self.relu_fc0(out)

        out = self.dropout3(out)

        out = self.fc1(out)
        out = self.relu_fc1(out)

        out = self.dropout4(out)
        out = self.fc2(out)
        return out
