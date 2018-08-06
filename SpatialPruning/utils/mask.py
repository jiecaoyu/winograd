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
            if self.percentage == 0.0:
                self.mask_list = self.mask_winograd_structured(model, threshold_multi, prune_list)
            else:
                self.mask_list =\
                        self.mask_winograd_structured_percentage(model, percentage, prune_list)
            # save the mask_list if required
            if generate_mask:
                mask_name = 'spatial_mask.pth.tar'
                # pickle.dump(self.mask_list, open(mask_name, 'w'))
                torch.save(self.mask_list, mask_name)
        else:
            raise Exception('Winograd-domain pruning is not supported here')
        self.print_mask_info()
        if not winograd_domain:
            self.print_mask_info_winograd()
        return

    def mask_winograd_structured(self, model, threshold_multi, prune_list):
        '''
        generate mask for normal weights but structured for winograd domain
        '''
        mask_list = {}
        count = 0
        print('Perform winograd-driven structured pruning in spatial domain ...')
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if count in prune_list:
                    threshold = para.resnet18_threshold_dict[count] * threshold_multi
                    tmp_weight = m.weight.data.abs()
                    tmp_mask = tmp_weight.lt(-1.0)
                    if tmp_weight.shape[2] == 5:
                        S = torch.from_numpy(para.S_4x4_5x5).float()
                    elif tmp_weight.shape[2] == 3:
                        S = torch.from_numpy(para.S_4x4_3x3).float()
                    else:
                        raise Exception ('The kernel size is not supported.')
                    if tmp_weight.is_cuda:
                        S = S.cuda()
                    for i in range(S.shape[0]):
                        S_piece = S[i].view(tmp_weight.shape[2],
                                tmp_weight.shape[2])
                        mask_piece = S_piece.abs().gt(0.0)
                        tmp_weight_masked = tmp_weight.mul(
                                mask_piece.unsqueeze(0).unsqueeze(0).float())
                        tmp_weight_masked = tmp_weight_masked.view(
                                tmp_weight_masked.shape[0],
                                tmp_weight_masked.shape[1],
                                -1)
                        tmp_weight_masked,_ = torch.max(
                                tmp_weight_masked, dim=2)
                        tmp_weight_masked = tmp_weight_masked.lt(threshold)
                        mask_piece = tmp_weight_masked.unsqueeze(2).unsqueeze(3)\
                                .mul(mask_piece.unsqueeze(0).unsqueeze(1))
                        tmp_mask = tmp_mask | mask_piece
                    mask_list[count] = tmp_mask.float()
                count += 1
        return mask_list

    def mask_winograd_structured_percentage(self, model, percentage, prune_list):
        '''
        generate mask for normal weights but structured for winograd domain
        the threshold is determined based on specified pruning percentage
        '''
        mask_list = {}
        count = 0
        print('Perform winograd-driven structured pruning in spatial domain ...')
        assert(percentage >= 0), 'percentage must be positive or zero.'
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if count in prune_list:
                    left = 0.0
                    right = m.weight.data.abs().max()
                    tmp_percentage = -1.0
                    while True:
                        threshold = (left + right) / 2.0
                        tmp_weight = m.weight.data.abs()
                        tmp_mask = tmp_weight.lt(-1.0)
                        if tmp_weight.shape[2] == 5:
                            S = torch.from_numpy(para.S_4x4_5x5).float()
                        elif tmp_weight.shape[2] == 3:
                            S = torch.from_numpy(para.S_4x4_3x3).float()
                        else:
                            raise Exception ('The kernel size is not supported.')
                        if tmp_weight.is_cuda:
                            S = S.cuda()
                        for i in range(S.shape[0]):
                            S_piece = S[i].view(tmp_weight.shape[2],
                                    tmp_weight.shape[2])
                            mask_piece = S_piece.abs().gt(0.0)
                            tmp_weight_masked = tmp_weight.mul(
                                    mask_piece.unsqueeze(0).unsqueeze(0).float())
                            tmp_weight_masked = tmp_weight_masked.view(
                                    tmp_weight_masked.shape[0],
                                    tmp_weight_masked.shape[1],
                                    -1)
                            tmp_weight_masked,_ = torch.max(
                                    tmp_weight_masked, dim=2)
                            tmp_weight_masked = tmp_weight_masked.lt(threshold)
                            mask_piece = tmp_weight_masked.unsqueeze(2).unsqueeze(3)\
                                    .mul(mask_piece.unsqueeze(0).unsqueeze(1))
                            tmp_mask = tmp_mask | mask_piece

                        # test winograd sparsity
                        tmp_weight = m.weight.data.clone()
                        tmp_weight = tmp_weight.mul(1.0 - tmp_mask.float())
                        if tmp_weight.shape[2] == 5:
                            G = torch.from_numpy(para.G_4x4_5x5).float()
                        elif tmp_weight.shape[2] == 3:
                            G = torch.from_numpy(para.G_4x4_3x3).float()
                        else:
                            raise Exception ('The kernel size is not supported.')
                        tmp_weight = tmp_weight.view(-1, tmp_weight.shape[2], tmp_weight.shape[3])
                        if tmp_weight.is_cuda:
                            G = G.cuda()
                        tmp_weight_t = torch.bmm(
                                G.unsqueeze(0).expand(tmp_weight.size(0), *G.size()), tmp_weight)
                        GT = G.transpose(0, 1)
                        tmp_weight_t = torch.bmm(
                                tmp_weight_t,
                                GT.unsqueeze(0).expand(tmp_weight_t.size(0), *GT.size()))
                        pruned = tmp_weight_t.eq(0.0).sum()
                        total = tmp_weight_t.nelement()
                        del tmp_weight
                        tmp_percentage = float(pruned) / total
                        if abs(percentage - tmp_percentage) < 0.0001:
                            break
                        elif tmp_percentage > percentage:
                            right = threshold
                        else:
                            left = threshold
                        print(tmp_percentage)
                    mask_list[count] = tmp_mask.float()
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
                    # here using element-wise mul to accelerate the computation
                    m.weight.data.mul_(1.0 - self.mask_list[count])
                count += 1
        return

    def mask_grad(self):
        '''
        Apply the mask onto grad of targeting layers
        *** Not used in current implementation ***
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
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, newLayers.Winograd2d.Winograd2d):
                if count in self.prune_list:
                    pruned = self.mask_list[count].sum()
                    total = self.mask_list[count].nelement()
                    print('CONV {} : {:8d}/{:8d} -- {:.4f}'\
                            .format(count, int(pruned), total, float(pruned) / total))
                else:
                    print('CONV ' + str(count) + ' : not pruned')
                count += 1

        print('-'*50)
        return
    
    def print_mask_info_winograd(self):
        '''
        Print the info of the masks in the winograd domain
        '''
        print('-'*50)
        print('Mask info in Winograd Domain:')
        count = 0
        weights_pruned = 0
        weights_total = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                if count in self.prune_list:
                    tmp_weight = m.weight.data.clone()
                    tmp_weight = tmp_weight.mul(1.0 - self.mask_list[count])
                    if tmp_weight.shape[2] == 5:
                        G = torch.from_numpy(para.G_4x4_5x5).float()
                    elif tmp_weight.shape[2] == 3:
                        G = torch.from_numpy(para.G_4x4_3x3).float()
                    else:
                        raise Exception ('The kernel size is not supported.')
                    tmp_weight = tmp_weight.view(-1, tmp_weight.shape[2], tmp_weight.shape[3])
                    if tmp_weight.is_cuda:
                        G = G.cuda()
                    tmp_weight_t = torch.bmm(
                            G.unsqueeze(0).expand(tmp_weight.size(0), *G.size()), tmp_weight)
                    GT = G.transpose(0, 1)
                    tmp_weight_t = torch.bmm(
                            tmp_weight_t,
                            GT.unsqueeze(0).expand(tmp_weight_t.size(0), *GT.size()))
                    pruned = tmp_weight_t.eq(0.0).sum()
                    total = tmp_weight_t.nelement()
                    if float(pruned) / total > 0.05:
                        weights_pruned += int(pruned)
                        weights_total += total
                    print('CONV {} : {:8d}/{:8d} -- {:.4f}'\
                            .format(count, int(pruned), total, float(pruned) / total))
                    del tmp_weight
                else:
                    print('CONV ' + str(count) + ' : not pruned')
                count += 1
        
        # avoid divided by zero error
        if weights_total == 0:
            weights_total = 1
        print('\nOverall of Pruned Layers {:8d}/{:8d} -- {:.4f}'.format(weights_pruned, weights_total, float(weights_pruned) / weights_total))
        print('-'*50)
        return
