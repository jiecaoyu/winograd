#!/usr/bin/env python2
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import torch
import cPickle as pickle
import numpy
import torchvision.transforms as transforms

class dataset():
    def __init__(self, root=None, train=True):
        self.root = root
        self.train = train
        self.transform = transforms.ToTensor()
        if self.train:
            train_data_path = os.path.join(root, 'train_data')
            train_labels_path = os.path.join(root, 'train_labels')
            self.train_data = numpy.load(open(train_data_path, 'r'))
            self.train_data = torch.from_numpy(self.train_data.astype('float32'))
            self.train_labels = numpy.load(open(train_labels_path, 'r')).astype('int')
        else:
            test_data_path = os.path.join(root, 'test_data')
            test_labels_path = os.path.join(root, 'test_labels')
            self.test_data = numpy.load(open(test_data_path, 'r'))
            self.test_data = torch.from_numpy(self.test_data.astype('float32'))
            self.test_labels = numpy.load(open(test_labels_path, 'r')).astype('int')

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]

            indicator = numpy.random.uniform(0.0, 1.0)
    
            img_np = img.numpy()
            if indicator > 0.5:
                img_np = numpy.flip(img_np, 2).copy()

            padding = 2

            img_np = numpy.pad(img_np,
                    ((0,0), (padding, padding), (padding, padding)), 'constant', constant_values=0.0)
            new_x = int(numpy.random.uniform(0.0, padding * 2.0 + 1.0))
            new_y = int(numpy.random.uniform(0.0, padding * 2.0 + 1.0))
            img_np = img_np[:, new_x:new_x+32, new_y:new_y+32]

            img = torch.from_numpy(img_np)
        else:
            img, target = self.test_data[index], self.test_labels[index]


        return img, target
