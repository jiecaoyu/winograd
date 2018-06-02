#!/usr/bin/env python2
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import numpy
import math
from torch.optim.optimizer import Optimizer, required
from . import para

class winogradAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay)
        super(winogradAdam, self).__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('winogradAdam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if len(grad.shape) > 2 and grad.shape[2] == 8:
                    mask_multi = torch.from_numpy(para.mask_multi_4x4_5x5).float().cuda()
                    mask_multi = mask_multi.pow(2.0)
                    mask_multi = mask_multi.pow(-1.0).mul(mask_multi[0][0])
                    update = denom.pow(-1.0).mul(mask_multi)
                    update = update.mul(exp_avg)
                    if group['weight_decay'] != 0:
                        update = update.add(group['weight_decay'], p.data)
                    update = update.mul(-step_size)
                    p.data.add_(update)
                elif len(grad.shape) > 2 and grad.shape[2] == 6:
                    mask_multi = torch.from_numpy(para.mask_multi_4x4_3x3).float().cuda()
                    mask_multi = mask_multi.pow(2.0)
                    mask_multi = mask_multi.pow(-1.0).mul(mask_multi[0][0])
                    update = denom.pow(-1.0).mul(mask_multi)
                    update = update.mul(exp_avg)
                    if group['weight_decay'] != 0:
                        update = update.add(group['weight_decay'], p.data)
                    update = update.mul(-step_size)
                    p.data.add_(update)
                else:
                    update = exp_avg.div(denom)
                    if group['weight_decay'] != 0:
                        update = update.add(group['weight_decay'], p.data)
                    update = update.mul(-step_size)
                    p.data.add_(update)

        return loss
