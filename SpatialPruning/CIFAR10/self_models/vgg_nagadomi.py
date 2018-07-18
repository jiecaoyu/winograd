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
        self.conv0      = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu_conv0 = nn.ReLU(inplace=True)
        self.conv1      = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu_conv1 = nn.ReLU(inplace=True)

        self.pool0      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout0   = nn.Dropout(0.25)

        self.conv2      = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu_conv2 = nn.ReLU(inplace=True)
        self.conv3      = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu_conv3 = nn.ReLU(inplace=True)

        self.pool1      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout1   = nn.Dropout(0.25)

        self.conv4      = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu_conv4 = nn.ReLU(inplace=True)
        self.conv5      = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu_conv5 = nn.ReLU(inplace=True)
        self.conv6      = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu_conv6 = nn.ReLU(inplace=True)
        self.conv7      = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu_conv7 = nn.ReLU(inplace=True)

        self.pool2      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout2   = nn.Dropout(0.25)

        self.fc0        = nn.Linear(256 * 4 * 4, 1024)
        self.relu_fc0   = nn.ReLU(inplace=True)

        self.dropout3   = nn.Dropout(0.5)

        self.fc1        = nn.Linear(1024, 1024)
        self.relu_fc1   = nn.ReLU(inplace=True)

        self.dropout4   = nn.Dropout(0.5)
        self.fc2        = nn.Linear(1024, 10)
        return

    def forward(self, x):
        out = self.conv0(x)
        out = self.relu_conv0(out)
        out = self.conv1(out)
        out = self.relu_conv1(out)

        out = self.pool0(out)
        out = self.dropout0(out)

        out = self.conv2(out)
        out = self.relu_conv2(out)
        out = self.conv3(out)
        out = self.relu_conv3(out)

        out = self.pool1(out)
        out = self.dropout1(out)

        out = self.conv4(out)
        out = self.relu_conv4(out)
        out = self.conv5(out)
        out = self.relu_conv5(out)
        out = self.conv6(out)
        out = self.relu_conv6(out)
        out = self.conv7(out)
        out = self.relu_conv7(out)

        out = self.pool2(out)
        out = self.dropout2(out)

        out = out.view(out.size(0), 256 * 4 * 4)

        out = self.fc0(out)
        out = self.relu_fc0(out)

        out = self.dropout3(out)

        out = self.fc1(out)
        out = self.relu_fc1(out)

        out = self.dropout4(out)
        out = self.fc2(out)
        return out
