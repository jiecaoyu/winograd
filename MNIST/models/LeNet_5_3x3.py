from __future__ import print_function
import torch
import torch.nn as nn
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd+'../')
from newLayers import *

class LeNet_5_3x3(nn.Module):
    def __init__(self):
        super(LeNet_5_3x3, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.relu_conv1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=3, stride=1, padding=1)
        self.relu_conv2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(50, 50, kernel_size=3, stride=1, padding=1)
        self.relu_conv3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(50, 10, kernel_size=3, stride=1, padding=1)
        self.relu_conv4 = nn.ReLU(inplace=True)
        self.pool4 = nn.AvgPool2d(kernel_size=7, stride=1)

        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu_conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu_conv3(x)

        x = self.conv4(x)
        x = self.pool4(x)

        x = x.view(x.size()[0], 10)

        return x

class LeNet_5_3x3_Winograd(nn.Module):
    def __init__(self):
        super(LeNet_5_3x3_Winograd, self).__init__()
        self.conv1 = Winograd2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.relu_conv1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Winograd2d(20, 50, kernel_size=3, stride=1, padding=1)
        self.relu_conv2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = Winograd2d(50, 50, kernel_size=3, stride=1, padding=1)
        self.relu_conv3 = nn.ReLU(inplace=True)

        self.conv4 = Winograd2d(50, 10, kernel_size=3, stride=1, padding=1)
        self.relu_conv4 = nn.ReLU(inplace=True)
        self.pool4 = nn.AvgPool2d(kernel_size=7, stride=1)

        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu_conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu_conv3(x)

        x = self.conv4(x)
        x = self.pool4(x)

        x = x.view(x.size()[0], 10)

        return x
