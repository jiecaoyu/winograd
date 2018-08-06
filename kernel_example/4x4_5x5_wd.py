#!/usr/bin/env python2
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import numpy

stdv = 1.0
kernel = torch.zeros([1000000,5,5]).normal_(0, stdv).float()

G = numpy.array([
    [ 1.   ,   0.    , 0.     , 0.      ,0.  ],
    [-2./9.,  -2./9. ,  -2./9.,  -2./9. ,  -2./9. ],
    [-2./9.,   2./9. ,  -2./9.,   2./9. ,  -2./9. ],
    [1./90.,  1./45. ,  2./45.,  4./45. ,  8./45. ],
    [1./90.,  -1./45.,  2./45.,  -4./45.,  8./45. ],
    [4./45.,  2./45. ,  1./45.,  1./90. ,  1./180.],
    [4./45.,  -2./45.,  1./45.,  -1./90.,  1./180.],
    [ 0.   ,   0.    , 0.     , 0.      ,1.  ]]
    )

G = torch.from_numpy(G).float()
kernel_t = torch.bmm(G.unsqueeze(0).expand(kernel.size(0), *G.size()), kernel)

GT = G.transpose(0, 1)
kernel_t = torch.bmm(kernel_t, GT.unsqueeze(0).expand(kernel_t.size(0), *GT.size()))

print(kernel_t.std(0))
print(kernel_t.mean(0))
