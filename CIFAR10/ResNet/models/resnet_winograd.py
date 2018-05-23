from __future__ import print_function
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd+'/../../')
from newLayers import *

def conv3x3_winograd(in_planes, out_planes, stride=1):
    if stride == 1:
        return Winograd2d(in_planes, out_planes, kernel_size=3, stride=stride,
                padding=1, bias=False)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3_winograd(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_winograd(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        return

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet_winograd(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet_winograd, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, Winograd2d):
                if m.kernel_size == 5:
                    BT = BT_4x4_5x5
                    G = torch.from_numpy(G_4x4_5x5).float()
                else:
                    BT = BT_4x4_3x3
                    G = torch.from_numpy(G_4x4_3x3).float()
                n = m.kernel_size * m.kernel_size * m.out_channels
                weight_normal = torch.zeros(
                        [m.out_channels, m.in_channels / m.groups, m.kernel_size, m.kernel_size],
                        dtype=torch.float32).normal_(0, math.sqrt(2. / n))
                weight_t = weight_normal.view(m.out_channels * m.in_channels / m.groups,
                        m.kernel_size, m.kernel_size)
                weight_t = torch.bmm(G.unsqueeze(0).expand(weight_t.size(0), *G.size()),
                        weight_t)
                GT = G.transpose(0, 1)
                weight_t = torch.bmm(weight_t,
                        GT.unsqueeze(0).expand(weight_t.size(0), *GT.size()))
                weight_t = weight_t.view(m.out_channels, m.in_channels / m.groups,
                        BT.shape[0], BT.shape[1])
                m.weight.data.copy_(weight_t)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                    )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet18_winograd():
    model = ResNet_winograd(BasicBlock, [3, 3, 3])
    return model
