#!/usr/bin/env python2
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.optim.optimizer import Optimizer, required
from . import para

class SGD_newWD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
            weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_newWD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_newWD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            if 'winograd' in group.keys():
                winograd = group['winograd']
            else:
                winograd = False


            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                    if winograd:
                        weight = p.data
                        assert(len(weight.shape) == 4), 'targeting conv layers only.'
                        if weight.shape[2] == 5:
                            G = torch.from_numpy(para.G_4x4_5x5).float()
                        elif weight.shape[2] == 3:
                            G = torch.from_numpy(para.G_4x4_3x3).float()
                        else:
                            raise Exception ('kernel size not supported.')
                        if p.data.is_cuda:
                            G = G.cuda()
                        G = torch.matmul(G.transpose(0, 1), G)
                        weight = weight.view(-1, weight.shape[2], weight.shape[3])
                        weight = torch.bmm(G.unsqueeze(0).expand(weight.shape[0], *G.size()),
                                weight)
                        weight = torch.bmm(weight,
                                G.unsqueeze(0).expand(weight.shape[0], *G.size()))
                        weight = weight.view(p.data.shape)
                        d_p.add_(weight_decay, weight)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
