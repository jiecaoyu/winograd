#!/usr/bin/env python2
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy


# kernel_size = 3
G_2x2_3x3 = numpy.array([
    [   1.,     0.,    0.],
    [1./2.,  1./2., 1./2.],
    [1./2., -1./2., 1./2.],
    [   0.,     0.,    1.]]
    )

BT_2x2_3x3 = numpy.array([
    [1.,  0., -1., 0.],
    [0.,  1.,  1., 0.],
    [0., -1.,  1., 0.],
    [0., -1.,  0., 1.]]
    )

AT_2x2_3x3 = numpy.array([
    [1., 1.,  1., 0.],
    [0., 1., -1., 1.]]
    )

S_2x2_3x3 = numpy.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1/2, 1/2, 1/2, 0, 0, 0, 0, 0, 0],
    [1/2, -1/2, 1/2, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1/2, 0, 0, 1/2, 0, 0, 1/2, 0, 0],
    [1/4, 1/4, 1/4, 1/4, 1/4, 1/4, 1/4, 1/4, 1/4],
    [1/4, -1/4, 1/4, 1/4, -1/4, 1/4, 1/4, -1/4, 1/4],
    [0, 0, 1/2, 0, 0, 1/2, 0, 0, 1/2],
    [1/2, 0, 0, -1/2, 0, 0, 1/2, 0, 0],
    [1/4, 1/4, 1/4, -1/4, -1/4, -1/4, 1/4, 1/4, 1/4],
    [1/4, -1/4, 1/4, -1/4, 1/4, -1/4, 1/4, -1/4, 1/4],
    [0, 0, 1/2, 0, 0, -1/2, 0, 0, 1/2],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1/2, 1/2, 1/2],
    [0, 0, 0, 0, 0, 0, 1/2, -1/2, 1/2],
    [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])


mask_multi_2x2_3x3 = numpy.array([
    [2.        , 2.82842712, 2.82842712, 2.        ],
    [2.82842712, 4.        , 4.        , 2.82842712],
    [2.82842712, 4.        , 4.        , 2.82842712],
    [2.        , 2.82842712, 2.82842712, 2.        ]
    ])