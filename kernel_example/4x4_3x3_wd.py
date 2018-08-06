#!/usr/bin/env python2
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import numpy

stdv = 1.0
kernel = torch.zeros([1000000,3,3]).normal_(0, stdv).float()

G = numpy.array([
    [ 1./4.,      0.,     0.],
    [-1./6.,  -1./6., -1./6.],
    [-1./6.,   1./6., -1./6.],
    [1./24.,  1./12.,  1./6.],
    [1./24., -1./12.,  1./6.],
    [    0.,      0.,    1.]]
    )

G = torch.from_numpy(G).float()
kernel_t = torch.bmm(G.unsqueeze(0).expand(kernel.size(0), *G.size()), kernel)

GT = G.transpose(0, 1)
kernel_t = torch.bmm(kernel_t, GT.unsqueeze(0).expand(kernel_t.size(0), *GT.size()))

print(kernel_t.std(0))
print(kernel_t.mean(0))
