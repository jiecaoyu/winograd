#!/usr/bin/env python2
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
from . import para
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd + '/../')
import newLayers
import pickle

class Mask():
    def __init__(self, model, threshold_multi=0.0,
            prune_list=None, winograd_structured=False, winograd_domain=False,
            percentage=0.0, generate_mask=False):
        '''
        initialize the mask
        '''
        if prune_list == None:
            raise Exception ('Pruning requires a prune_list of targeting layers')
        self.model = model
        self.prune_list = prune_list
        self.threshold_multi = threshold_multi
        self.percentage = percentage
        if not winograd_domain:
            raise Exception ('Here only winograd pruning is supported')
        else:
            if self.percentage == 0.0:
                self.mask_list = self.mask_winograd_domain(model, threshold_multi, prune_list)
            else:
                self.mask_list = self.mask_winograd_domain_percentage(model, percentage, prune_list)
        self.print_mask_info()
        return

    def mask_winograd_domain(self, model, threshold_multi, prune_list):
        '''
        generate mask for pruning in winograd domain
        '''
        mask_list = {}
        count = 0
        print('Perform normal pruning in spatial domain ...')
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, newLayers.Winograd2d.Winograd2d):
                if (count in prune_list) and isinstance(m, newLayers.Winograd2d.Winograd2d):
                    threshold = para.resnet18_threshold_dict_winograd[count] * threshold_multi
                    print(threshold)
                    if m.kernel_size == 5:
                        threshold_tensor = torch.from_numpy(para.mask_multi_4x4_5x5).float()
                    elif m.kernel_size == 3:
                        threshold_tensor = torch.from_numpy(para.mask_multi_4x4_3x3).float()
                    else:
                        raise Exception ('kernel_size currently not supported')
                    if m.weight.data.is_cuda:
                        threshold_tensor = threshold_tensor.cuda()
                    threshold_tensor = threshold_tensor / threshold_tensor.min()
                    tmp_mask = m.weight.data.clone().abs()\
                            .lt(threshold_tensor.pow(-1.0).mul(threshold)).float()
                    mask_list[count] = tmp_mask
                count += 1
        return mask_list

    def mask_winograd_domain_percentage(self, model, percentage, prune_list):
        '''
        generate mask for pruning in winograd domain
        '''
        mask_list = {}
        self.threshold_list = {}
        count = 0
        print('Perform normal pruning in spatial domain ...')
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, newLayers.Winograd2d.Winograd2d):
                if (count in prune_list) and isinstance(m, newLayers.Winograd2d.Winograd2d):
                    left = 0.0
                    right = m.weight.data.abs().max()
                    tmp_percentage = -1.0
                    while True:
                        threshold = (left + right) / 2.0
                        tmp_weight = m.weight.data.abs()
                        tmp_mask = tmp_weight.lt(-1.0)
                        if m.kernel_size == 5:
                            threshold_tensor = torch.from_numpy(para.mask_multi_4x4_5x5).float()
                        elif m.kernel_size == 3:
                            threshold_tensor = torch.from_numpy(para.mask_multi_4x4_3x3).float()
                        else:
                            raise Exception ('kernel_size currently not supported')
                        if m.weight.data.is_cuda:
                            threshold_tensor = threshold_tensor.cuda()
                        threshold_tensor = threshold_tensor / threshold_tensor.min()
                        tmp_mask = m.weight.data.clone().abs()\
                                .lt(threshold_tensor.pow(-1.0).mul(threshold)).float()
                        pruned = tmp_mask.sum()
                        total = tmp_mask.nelement()
                        tmp_percentage = float(pruned) / total
                        if abs(percentage - tmp_percentage) < 0.0001:
                            break
                        elif tmp_percentage > percentage:
                            right = threshold
                        else:
                            left = threshold
                        print(tmp_percentage)
                    mask_list[count] = tmp_mask
                    self.threshold_list[count] = threshold
                count += 1
        return mask_list

    def apply(self):
        '''
        Apply the mask onto targeting layers
        '''
        count = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, newLayers.Winograd2d.Winograd2d):
                if count in self.prune_list:
                    m.weight.data.mul_(1.0 - self.mask_list[count])
                count += 1
        return

    def mask_grad(self):
        '''
        Apply the mask onto grad of targeting layers
        '''
        count = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, newLayers.Winograd2d.Winograd2d):
                if count in self.prune_list:
                    m.weight.grad.data.mul_(1.0 - self.mask_list[count])
                count += 1
        return

    def print_mask_info(self):
        '''
        Print the info of the masks
        '''
        print('-'*50)
        print('Mask info:')
        count = 0
        pruned_acc = 0
        total_acc = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, newLayers.Winograd2d.Winograd2d):
                if count in self.prune_list:
                    pruned = self.mask_list[count].sum()
                    total = self.mask_list[count].nelement()
                    if float(pruned) / total > 0.05:
                        pruned_acc += pruned
                        total_acc += total
                    print('CONV {} : {:8d}/{:8d} -- {:.4f}'\
                            .format(count, int(pruned), total, float(pruned) / total))
                else:
                    print('CONV ' + str(count) + ' : not pruned')
                count += 1
        
        # avoid divided by zero error
        if total_acc == 0:
            total_acc = 1
        print('\nOverall of Pruned Layers {:8d}/{:8d} -- {:.4f}'.format(int(pruned_acc), int(total_acc), float(pruned_acc) / total_acc))
        print('-'*50)
        return
