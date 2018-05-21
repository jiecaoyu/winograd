#!/usr/bin/env python2
from __future__ import absolute_import, division, print_function, unicode_literals
from sympy import *
from numpy.linalg import inv
import re
import numpy


for i in range(3):
    for j in range(3):
        locals()['w_'+str(i)+'_'+str(j)] = Symbol('w_'+str(i)+'_'+str(j))

W = Matrix([
    [w_0_0, w_0_1, w_0_2],
    [w_1_0, w_1_1, w_1_2],
    [w_2_0, w_2_1, w_2_2],
    ])

G = Matrix([
    [Rational(' 1./4.'), Rational('     0.'), Rational('    0.')],
    [Rational('-1./6.'), Rational(' -1./6.'), Rational('-1./6.')],
    [Rational('-1./6.'), Rational('  1./6.'), Rational('-1./6.')],
    [Rational('1./24.'), Rational(' 1./12.'), Rational(' 1./6.')],
    [Rational('1./24.'), Rational('-1./12.'), Rational(' 1./6.')],
    [Rational('    0.'), Rational('     0.'), Rational('   1. ')]]
    )

A = G * W * (G.transpose())

A_list = []
for s in range(6):
    for t in range(6):
        row_index = s * 6 + t
        # initialize all entries
        row_list = []
        for i in range(3):
            for j in range(3):
                col_index = i * 3 + j
                locals()['q_'+str(row_index)+'_'+str(col_index)] = Rational('0')
                row_list.append(locals()['q_'+str(row_index)+'_'+str(col_index)])
        A_list.append(row_list)

        out = str(A[row_index])
        out = out.replace(' ','')
        while len(out) > 0:
            item = re.search('[+,-]*[0-9]*\**w_[0-9]*_[0-9]*/*[0-9]*', out).group(0)
            negative = False
            if item[0] == '-':
                negative = True
            numerator = re.search('[0-9]*(?:\*)', item)
            if numerator != None:
                numerator = numerator.group(0)
                numerator = int(numerator[0:len(numerator)-1])
            else:
                numerator = 1
            denominator = re.search('/[0-9]*', item)
            if denominator != None:
                denominator = denominator.group()
                denominator = int(denominator[1:])
            else:
                denominator = 1
            # calculate the index
            weight = re.search('w_[0-9]*_[0-9]*', item).group(0)
            weight_indexes = weight.split('_')
            col_index = int(weight_indexes[1]) * 3 + int(weight_indexes[2])
            num_str = str(numerator)+'/'+str(denominator)
            if negative:
                num_str = '-' + num_str
            A_list[row_index][col_index] = Rational(num_str)
            out = out[len(item):]
        
A = Matrix(A_list)
print(A)

# verify A
kernel = numpy.random.rand(3,3)
kernel_t = numpy.matmul(numpy.array(G).astype(numpy.float64), kernel)
kernel_t = numpy.matmul(kernel_t, numpy.transpose(numpy.array(G).astype(numpy.float64)))

kernel_t_new = kernel.reshape(9)
kernel_t_new = numpy.matmul(numpy.array(A).astype(numpy.float64), kernel_t_new)
kernel_t_new = kernel_t_new.reshape(6, 6)
diff = abs(kernel_t_new - kernel_t) > 1e-10
assert(diff.sum() == 0)

# calculate (I + \lambda * BT * B) ^ -1
m = numpy.random.rand(36) > 0.5
diag = numpy.diag(m.astype(numpy.float64))

B = numpy.matmul(diag, numpy.array(A).astype(numpy.float64))

LAMBDA = 0.5
C = numpy.identity(9) + LAMBDA * (numpy.matmul(numpy.transpose(B), B))

C_inv = inv(C)
