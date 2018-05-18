#!/usr/bin/env python2
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import numpy
import torch
from newLayers import *
from .mask_para import *
cwd = os.getcwd()
sys.path.append(cwd+'/../')

mask_multi_4x4_3x3 = numpy.array([
    [ 33.2565783 ,   48.74423043,  53.44155686, 133.60389216, 133.60389216, 29.15475947],
    [ 48.74423043,   65.96969001,  96.16652224, 240.4163056 , 240.4163056 ,48.74423043],
    [ 53.44155686,   96.16652224,  96.16652224, 240.4163056 , 240.4163056 ,53.44155686],
    [133.60389216,  240.4163056 , 240.4163056 , 601.04076401, 601.04076401, 133.60389216],
    [133.60389216,  240.4163056 , 240.4163056 , 601.04076401, 601.04076401, 133.60389216],
    [ 29.15475947,   48.74423043,  53.44155686, 133.60389216, 133.60389216, 29.15475947]])

mask_multi_4x4_5x5 = numpy.array([
    [  40.40585122,   67.72947106,   67.70732051,  178.50781233,  178.50781233, 357.01562466 , 357.01562466 ,  40.39347488],
    [  67.72947106,   68.0661443 ,  113.49063838,  275.45558853,  290.88448094, 550.91117705 , 581.76896187 ,  67.70732051],
    [  67.70732051,  113.49063838,  113.49063838,  299.2138136 ,  299.2138136 ,598.4276272   ,598.4276272   , 67.70732051],
    [ 178.50781233,  275.45558853,  299.2138136 ,  774.42329853,  783.69729189, 1548.84659707, 1567.39458378,  178.50781233],
    [ 178.50781233,  290.88448094,  299.2138136 ,  783.69729189,  787.0091774 ,1567.39458378 ,1574.0183548  , 178.50781233],
    [ 357.01562466,  550.91117705,  598.4276272 , 1548.84659707, 1567.39458378, 3097.69319414, 3134.78916755,  357.01562466],
    [ 357.01562466,  581.76896187,  598.4276272 , 1567.39458378, 1574.0183548 ,3134.78916755 ,3148.0367096  , 357.01562466],
    [  40.39347488,   67.70732051,   67.70732051,  178.50781233,  178.50781233, 357.01562466 , 357.01562466 ,  40.39347488]
    ])

class Mask():
    def __init__(self, model, threshold, gamma=0):
        self.target = []
        self.mask = []
        self.gamma = gamma
        index = 0
        for m in model.modules():
            if isinstance(m, Winograd2d):
                self.target.append(m)
                if index == 0:
                    # ignore the first layer
                    index += 1
                    self.mask.append(m.weight.data.clone().abs().lt(-1.0))
                else:
                    if m.kernel_size == 5:
                        threshold_tensor = torch.from_numpy(mask_multi_4x4_5x5).float()
                    else:
                        raise Exception ('kernel_size currently not supported')
                    if m.weight.data.is_cuda:
                        threshold_tensor = threshold_tensor.cuda()
                    self.mask.append(m.weight.data.clone().abs().lt(threshold_tensor.pow(-1.0).mul(threshold)))
        self.compute_sparse_grad_transfer()
        return

    def print_info(self):
        print('-------------------------------------')
        for i in range(len(self.mask)):
            mask = self.mask[i]
            print('[{}]: {} / {} ( {:.2f}% )'.format(i, mask.sum(), mask.nelement(), 100. * float(mask.sum()) / mask.nelement()))
        print('-------------------------------------')
        return

    def apply(self):
        for i in range(len(self.mask)):
            self.target[i].weight.data[self.mask[i]] = 0.0
        return

    def mask_grad(self):
        for i in range(len(self.mask)):
            self.target[i].weight.grad.data[self.mask[i]] = 0.0
        return

    def compute_sparse_grad_transfer(self):
        self.sparse_grad_transfer = []
        for i in range(len(self.mask)):
            tmp_mask = self.mask[i]
            s = tmp_mask.shape
            tmp_mask = tmp_mask.view(s[0] * s[1], s[2] * s[3]).float()
            if self.target[i].kernel_size == 5:
                A = torch.from_numpy(A_4x4_5x5).float()
                I = torch.eye(5 * 5)
            else:
                raise Exception ('kernel_size currently not supported')
            if self.target[i].weight.is_cuda:
                A = A.cuda()
                I = I.cuda()

            B = tmp_mask.unsqueeze(2).mul(A.unsqueeze(0))
            BT_mul_B = B.transpose(1, 2).bmm(B)
            transfer_matrix = I + BT_mul_B.mul(self.gamma)
            transfer_matrix = [t.inverse() for t in torch.functional.unbind(transfer_matrix)]
            transfer_matrix = torch.stack(transfer_matrix)
            self.sparse_grad_transfer.append(transfer_matrix)
        return
