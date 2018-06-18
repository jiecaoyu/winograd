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

class Mask():
    def __init__(self, model, percentage=0.0, winograd_domain=False):
        '''
        initialize the mask
        '''
        self.model = model
        self.percentage = percentage
        if winograd_domain:
            self.mask_list = self.mask_percentage_winograd_domain(model, percentage)
        else:
            self.mask_list = self.mask_winograd_structured_percentage(model, percentage)
        return

    def mask_winograd_structured_percentage(self, model, percentage):
        '''
        generate mask for normal weights but structured for winograd domain
        the threshold is determined based on specified pruning percentage
        '''
        mask_list = {}
        print('Perform winograd-driven structured pruning in spatial domain ...')
        assert(percentage >= 0), 'percentage must be positive or zero.'
        for m in model.modules():
            if isinstance(m, newLayers.Winograd2d.Winograd2d):
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
                        mask_piece = S_piece.gt(0.0)
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
                    tmp_weight_t = tmp_weight_t.view(m.weight.shape[0], m.weight.shape[1],
                            G.shape[0], G.shape[0])
                    pruned = tmp_weight_t.eq(0.0).sum()
                    total = tmp_weight_t.nelement()
                    del tmp_weight
                    tmp_percentage = float(pruned) / total
                    if abs(percentage - tmp_percentage) < 0.0001:
                        m.weight.data.mul_(1.0 - tmp_mask.float())
                        m.Mask.copy_(tmp_weight_t.eq(0.0).float())
                        break
                    elif tmp_percentage > percentage:
                        right = threshold
                    else:
                        left = threshold
                    print(tmp_percentage)
        return mask_list

    def mask_percentage_winograd_domain(self, model, percentage):
        mask_list = {}
        print('Perform winograd-driven structured pruning in spatial domain ...')
        assert(percentage >= 0), 'percentage must be positive or zero.'
        for m in model.modules():
            if isinstance(m, newLayers.Winograd2d.Winograd2d):
                # transfer weights into winograd domain
                tmp_weight = m.weight.data.clone()
                if tmp_weight.shape[2] == 5:
                    G = torch.from_numpy(para.G_4x4_5x5).float()
                    threshold_tensor = torch.from_numpy(para.mask_multi_4x4_5x5).float()
                elif tmp_weight.shape[2] == 3:
                    G = torch.from_numpy(para.G_4x4_3x3).float()
                    threshold_tensor = torch.from_numpy(para.mask_multi_4x4_3x3).float()
                else:
                    raise Exception ('The kernel size is not supported.')
                tmp_weight = tmp_weight.view(-1, tmp_weight.shape[2], tmp_weight.shape[3])
                if tmp_weight.is_cuda:
                    G = G.cuda()
                    threshold_tensor = threshold_tensor.cuda()
                threshold_tensor = threshold_tensor.div(threshold_tensor.min()).pow(-1.0)
                tmp_weight_t = torch.bmm(
                        G.unsqueeze(0).expand(tmp_weight.size(0), *G.size()), tmp_weight)
                GT = G.transpose(0, 1)
                tmp_weight_t = torch.bmm(
                        tmp_weight_t,
                        GT.unsqueeze(0).expand(tmp_weight_t.size(0), *GT.size()))
                tmp_weight_t = tmp_weight_t.view(m.weight.shape[0], m.weight.shape[1],
                        G.shape[0], G.shape[0])

                # determine the threshold
                left = 0.0
                right = m.weight.data.max() / threshold_tensor.min()
                while True:
                    threshold = (left + right) / 2.0
                    tmp_threshold_tensor = threshold_tensor.mul(threshold).unsqueeze(0).unsqueeze(0)
                    tmp_mask = tmp_weight_t.abs().lt(tmp_threshold_tensor)
                    pruned = float(tmp_mask.sum())
                    total = float(tmp_mask.nelement())
                    tmp_percentage = pruned / total
                    if abs(percentage - tmp_percentage) < 0.0001:
                        m.Mask.copy_(tmp_mask.float())
                        break
                    elif tmp_percentage > percentage:
                        right = threshold
                    else:
                        left = threshold
                    print(tmp_percentage, float(threshold))
        return mask_list

    def print_mask_info_winograd(self):
        '''
        Print the info of the masks in the winograd domain
        '''
        print('-'*50)
        print('Mask info in Winograd Domain:')
        count = 0
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
                    print('CONV {} : {:8d}/{:8d} -- {:.4f}'\
                            .format(count, int(pruned), total, float(pruned) / total))
                    del tmp_weight
                else:
                    print('CONV ' + str(count) + ' : not pruned')
                count += 1

        print('-'*50)
        return

    def regularize_grad(self):
        for m in self.model.modules():
            if isinstance(m, newLayers.Winograd2d.Winograd2d):
                m.regularize_grad()
        return
