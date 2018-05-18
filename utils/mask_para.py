#!/usr/bin/env python2
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.linalg import inv
import re
import numpy

A_4x4_5x5 = numpy.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-2/9, -2/9, -2/9, -2/9, -2/9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-2/9, 2/9, -2/9, 2/9, -2/9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1/90, 1/45, 2/45, 4/45, 8/45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1/90, -1/45, 2/45, -4/45, 8/45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4/45, 2/45, 1/45, 1/90, 1/180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4/45, -2/45, 1/45, -1/90, 1/180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-2/9, 0, 0, 0, 0, -2/9, 0, 0, 0, 0, -2/9, 0, 0, 0, 0, -2/9, 0, 0, 0, 0, -2/9, 0, 0, 0, 0],
    [4/81, 4/81, 4/81, 4/81, 4/81, 4/81, 4/81, 4/81, 4/81, 4/81, 4/81, 4/81, 4/81, 4/81, 4/81, 4/81, 4/81, 4/81, 4/81, 4/81, 4/81, 4/81, 4/81, 4/81, 4/81],
    [4/81, -4/81, 4/81, -4/81, 4/81, 4/81, -4/81, 4/81, -4/81, 4/81, 4/81, -4/81, 4/81, -4/81, 4/81, 4/81, -4/81, 4/81, -4/81, 4/81, 4/81, -4/81, 4/81, -4/81, 4/81],
    [-1/405, -2/405, -4/405, -8/405, -16/405, -1/405, -2/405, -4/405, -8/405, -16/405, -1/405, -2/405, -4/405, -8/405, -16/405, -1/405, -2/405, -4/405, -8/405, -16/405, -1/405, -2/405, -4/405, -8/405, -16/405],
    [-1/405, 2/405, -4/405, 8/405, -16/405, -1/405, 2/405, -4/405, 8/405, -16/405, -1/405, 2/405, -4/405, 8/405, -16/405, -1/405, 2/405, -4/405, 8/405, -16/405, -1/405, 2/405, -4/405, 8/405, -16/405],
    [-8/405, -4/405, -2/405, -1/405, -1/810, -8/405, -4/405, -2/405, -1/405, -1/810, -8/405, -4/405, -2/405, -1/405, -1/810, -8/405, -4/405, -2/405, -1/405, -1/810, -8/405, -4/405, -2/405, -1/405, -1/810],
    [-8/405, 4/405, -2/405, 1/405, -1/810, -8/405, 4/405, -2/405, 1/405, -1/810, -8/405, 4/405, -2/405, 1/405, -1/810, -8/405, 4/405, -2/405, 1/405, -1/810, -8/405, 4/405, -2/405, 1/405, -1/810],
    [0, 0, 0, 0, -2/9, 0, 0, 0, 0, -2/9, 0, 0, 0, 0, -2/9, 0, 0, 0, 0, -2/9, 0, 0, 0, 0, -2/9],
    [-2/9, 0, 0, 0, 0, 2/9, 0, 0, 0, 0, -2/9, 0, 0, 0, 0, 2/9, 0, 0, 0, 0, -2/9, 0, 0, 0, 0],
    [4/81, 4/81, 4/81, 4/81, 4/81, -4/81, -4/81, -4/81, -4/81, -4/81, 4/81, 4/81, 4/81, 4/81, 4/81, -4/81, -4/81, -4/81, -4/81, -4/81, 4/81, 4/81, 4/81, 4/81, 4/81],
    [4/81, -4/81, 4/81, -4/81, 4/81, -4/81, 4/81, -4/81, 4/81, -4/81, 4/81, -4/81, 4/81, -4/81, 4/81, -4/81, 4/81, -4/81, 4/81, -4/81, 4/81, -4/81, 4/81, -4/81, 4/81],
    [-1/405, -2/405, -4/405, -8/405, -16/405, 1/405, 2/405, 4/405, 8/405, 16/405, -1/405, -2/405, -4/405, -8/405, -16/405, 1/405, 2/405, 4/405, 8/405, 16/405, -1/405, -2/405, -4/405, -8/405, -16/405],
    [-1/405, 2/405, -4/405, 8/405, -16/405, 1/405, -2/405, 4/405, -8/405, 16/405, -1/405, 2/405, -4/405, 8/405, -16/405, 1/405, -2/405, 4/405, -8/405, 16/405, -1/405, 2/405, -4/405, 8/405, -16/405],
    [-8/405, -4/405, -2/405, -1/405, -1/810, 8/405, 4/405, 2/405, 1/405, 1/810, -8/405, -4/405, -2/405, -1/405, -1/810, 8/405, 4/405, 2/405, 1/405, 1/810, -8/405, -4/405, -2/405, -1/405, -1/810],
    [-8/405, 4/405, -2/405, 1/405, -1/810, 8/405, -4/405, 2/405, -1/405, 1/810, -8/405, 4/405, -2/405, 1/405, -1/810, 8/405, -4/405, 2/405, -1/405, 1/810, -8/405, 4/405, -2/405, 1/405, -1/810],
    [0, 0, 0, 0, -2/9, 0, 0, 0, 0, 2/9, 0, 0, 0, 0, -2/9, 0, 0, 0, 0, 2/9, 0, 0, 0, 0, -2/9],
    [1/90, 0, 0, 0, 0, 1/45, 0, 0, 0, 0, 2/45, 0, 0, 0, 0, 4/45, 0, 0, 0, 0, 8/45, 0, 0, 0, 0],
    [-1/405, -1/405, -1/405, -1/405, -1/405, -2/405, -2/405, -2/405, -2/405, -2/405, -4/405, -4/405, -4/405, -4/405, -4/405, -8/405, -8/405, -8/405, -8/405, -8/405, -16/405, -16/405, -16/405, -16/405, -16/405],
    [-1/405, 1/405, -1/405, 1/405, -1/405, -2/405, 2/405, -2/405, 2/405, -2/405, -4/405, 4/405, -4/405, 4/405, -4/405, -8/405, 8/405, -8/405, 8/405, -8/405, -16/405, 16/405, -16/405, 16/405, -16/405],
    [1/8100, 1/4050, 1/2025, 2/2025, 4/2025, 1/4050, 1/2025, 2/2025, 4/2025, 8/2025, 1/2025, 2/2025, 4/2025, 8/2025, 16/2025, 2/2025, 4/2025, 8/2025, 16/2025, 32/2025, 4/2025, 8/2025, 16/2025, 32/2025, 64/2025],
    [1/8100, -1/4050, 1/2025, -2/2025, 4/2025, 1/4050, -1/2025, 2/2025, -4/2025, 8/2025, 1/2025, -2/2025, 4/2025, -8/2025, 16/2025, 2/2025, -4/2025, 8/2025, -16/2025, 32/2025, 4/2025, -8/2025, 16/2025, -32/2025, 64/2025],
    [2/2025, 1/2025, 1/4050, 1/8100, 1/16200, 4/2025, 2/2025, 1/2025, 1/4050, 1/8100, 8/2025, 4/2025, 2/2025, 1/2025, 1/4050, 16/2025, 8/2025, 4/2025, 2/2025, 1/2025, 32/2025, 16/2025, 8/2025, 4/2025, 2/2025],
    [2/2025, -1/2025, 1/4050, -1/8100, 1/16200, 4/2025, -2/2025, 1/2025, -1/4050, 1/8100, 8/2025, -4/2025, 2/2025, -1/2025, 1/4050, 16/2025, -8/2025, 4/2025, -2/2025, 1/2025, 32/2025, -16/2025, 8/2025, -4/2025, 2/2025],
    [0, 0, 0, 0, 1/90, 0, 0, 0, 0, 1/45, 0, 0, 0, 0, 2/45, 0, 0, 0, 0, 4/45, 0, 0, 0, 0, 8/45],
    [1/90, 0, 0, 0, 0, -1/45, 0, 0, 0, 0, 2/45, 0, 0, 0, 0, -4/45, 0, 0, 0, 0, 8/45, 0, 0, 0, 0],
    [-1/405, -1/405, -1/405, -1/405, -1/405, 2/405, 2/405, 2/405, 2/405, 2/405, -4/405, -4/405, -4/405, -4/405, -4/405, 8/405, 8/405, 8/405, 8/405, 8/405, -16/405, -16/405, -16/405, -16/405, -16/405],
    [-1/405, 1/405, -1/405, 1/405, -1/405, 2/405, -2/405, 2/405, -2/405, 2/405, -4/405, 4/405, -4/405, 4/405, -4/405, 8/405, -8/405, 8/405, -8/405, 8/405, -16/405, 16/405, -16/405, 16/405, -16/405],
    [1/8100, 1/4050, 1/2025, 2/2025, 4/2025, -1/4050, -1/2025, -2/2025, -4/2025, -8/2025, 1/2025, 2/2025, 4/2025, 8/2025, 16/2025, -2/2025, -4/2025, -8/2025, -16/2025, -32/2025, 4/2025, 8/2025, 16/2025, 32/2025, 64/2025],
    [1/8100, -1/4050, 1/2025, -2/2025, 4/2025, -1/4050, 1/2025, -2/2025, 4/2025, -8/2025, 1/2025, -2/2025, 4/2025, -8/2025, 16/2025, -2/2025, 4/2025, -8/2025, 16/2025, -32/2025, 4/2025, -8/2025, 16/2025, -32/2025, 64/2025],
    [2/2025, 1/2025, 1/4050, 1/8100, 1/16200, -4/2025, -2/2025, -1/2025, -1/4050, -1/8100, 8/2025, 4/2025, 2/2025, 1/2025, 1/4050, -16/2025, -8/2025, -4/2025, -2/2025, -1/2025, 32/2025, 16/2025, 8/2025, 4/2025, 2/2025],
    [2/2025, -1/2025, 1/4050, -1/8100, 1/16200, -4/2025, 2/2025, -1/2025, 1/4050, -1/8100, 8/2025, -4/2025, 2/2025, -1/2025, 1/4050, -16/2025, 8/2025, -4/2025, 2/2025, -1/2025, 32/2025, -16/2025, 8/2025, -4/2025, 2/2025],
    [0, 0, 0, 0, 1/90, 0, 0, 0, 0, -1/45, 0, 0, 0, 0, 2/45, 0, 0, 0, 0, -4/45, 0, 0, 0, 0, 8/45],
    [4/45, 0, 0, 0, 0, 2/45, 0, 0, 0, 0, 1/45, 0, 0, 0, 0, 1/90, 0, 0, 0, 0, 1/180, 0, 0, 0, 0],
    [-8/405, -8/405, -8/405, -8/405, -8/405, -4/405, -4/405, -4/405, -4/405, -4/405, -2/405, -2/405, -2/405, -2/405, -2/405, -1/405, -1/405, -1/405, -1/405, -1/405, -1/810, -1/810, -1/810, -1/810, -1/810],
    [-8/405, 8/405, -8/405, 8/405, -8/405, -4/405, 4/405, -4/405, 4/405, -4/405, -2/405, 2/405, -2/405, 2/405, -2/405, -1/405, 1/405, -1/405, 1/405, -1/405, -1/810, 1/810, -1/810, 1/810, -1/810],
    [2/2025, 4/2025, 8/2025, 16/2025, 32/2025, 1/2025, 2/2025, 4/2025, 8/2025, 16/2025, 1/4050, 1/2025, 2/2025, 4/2025, 8/2025, 1/8100, 1/4050, 1/2025, 2/2025, 4/2025, 1/16200, 1/8100, 1/4050, 1/2025, 2/2025],
    [2/2025, -4/2025, 8/2025, -16/2025, 32/2025, 1/2025, -2/2025, 4/2025, -8/2025, 16/2025, 1/4050, -1/2025, 2/2025, -4/2025, 8/2025, 1/8100, -1/4050, 1/2025, -2/2025, 4/2025, 1/16200, -1/8100, 1/4050, -1/2025, 2/2025],
    [16/2025, 8/2025, 4/2025, 2/2025, 1/2025, 8/2025, 4/2025, 2/2025, 1/2025, 1/4050, 4/2025, 2/2025, 1/2025, 1/4050, 1/8100, 2/2025, 1/2025, 1/4050, 1/8100, 1/16200, 1/2025, 1/4050, 1/8100, 1/16200, 1/32400],
    [16/2025, -8/2025, 4/2025, -2/2025, 1/2025, 8/2025, -4/2025, 2/2025, -1/2025, 1/4050, 4/2025, -2/2025, 1/2025, -1/4050, 1/8100, 2/2025, -1/2025, 1/4050, -1/8100, 1/16200, 1/2025, -1/4050, 1/8100, -1/16200, 1/32400],
    [0, 0, 0, 0, 4/45, 0, 0, 0, 0, 2/45, 0, 0, 0, 0, 1/45, 0, 0, 0, 0, 1/90, 0, 0, 0, 0, 1/180],
    [4/45, 0, 0, 0, 0, -2/45, 0, 0, 0, 0, 1/45, 0, 0, 0, 0, -1/90, 0, 0, 0, 0, 1/180, 0, 0, 0, 0],
    [-8/405, -8/405, -8/405, -8/405, -8/405, 4/405, 4/405, 4/405, 4/405, 4/405, -2/405, -2/405, -2/405, -2/405, -2/405, 1/405, 1/405, 1/405, 1/405, 1/405, -1/810, -1/810, -1/810, -1/810, -1/810],
    [-8/405, 8/405, -8/405, 8/405, -8/405, 4/405, -4/405, 4/405, -4/405, 4/405, -2/405, 2/405, -2/405, 2/405, -2/405, 1/405, -1/405, 1/405, -1/405, 1/405, -1/810, 1/810, -1/810, 1/810, -1/810],
    [2/2025, 4/2025, 8/2025, 16/2025, 32/2025, -1/2025, -2/2025, -4/2025, -8/2025, -16/2025, 1/4050, 1/2025, 2/2025, 4/2025, 8/2025, -1/8100, -1/4050, -1/2025, -2/2025, -4/2025, 1/16200, 1/8100, 1/4050, 1/2025, 2/2025],
    [2/2025, -4/2025, 8/2025, -16/2025, 32/2025, -1/2025, 2/2025, -4/2025, 8/2025, -16/2025, 1/4050, -1/2025, 2/2025, -4/2025, 8/2025, -1/8100, 1/4050, -1/2025, 2/2025, -4/2025, 1/16200, -1/8100, 1/4050, -1/2025, 2/2025],
    [16/2025, 8/2025, 4/2025, 2/2025, 1/2025, -8/2025, -4/2025, -2/2025, -1/2025, -1/4050, 4/2025, 2/2025, 1/2025, 1/4050, 1/8100, -2/2025, -1/2025, -1/4050, -1/8100, -1/16200, 1/2025, 1/4050, 1/8100, 1/16200, 1/32400],
    [16/2025, -8/2025, 4/2025, -2/2025, 1/2025, -8/2025, 4/2025, -2/2025, 1/2025, -1/4050, 4/2025, -2/2025, 1/2025, -1/4050, 1/8100, -2/2025, 1/2025, -1/4050, 1/8100, -1/16200, 1/2025, -1/4050, 1/8100, -1/16200, 1/32400],
    [0, 0, 0, 0, 4/45, 0, 0, 0, 0, -2/45, 0, 0, 0, 0, 1/45, 0, 0, 0, 0, -1/90, 0, 0, 0, 0, 1/180],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2/9, -2/9, -2/9, -2/9, -2/9],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2/9, 2/9, -2/9, 2/9, -2/9],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/90, 1/45, 2/45, 4/45, 8/45],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/90, -1/45, 2/45, -4/45, 8/45],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4/45, 2/45, 1/45, 1/90, 1/180],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4/45, -2/45, 1/45, -1/90, 1/180],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])