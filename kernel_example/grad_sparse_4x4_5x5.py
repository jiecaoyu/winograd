#!/usr/bin/env python2
from __future__ import absolute_import, division, print_function, unicode_literals
from sympy import *
from numpy.linalg import inv
import re
import numpy


for i in range(5):
    for j in range(5):
        locals()['w_'+str(i)+'_'+str(j)] = Symbol('w_'+str(i)+'_'+str(j))

W = Matrix([
    [w_0_0, w_0_1, w_0_2, w_0_3, w_0_4],
    [w_1_0, w_1_1, w_1_2, w_1_3, w_1_4],
    [w_2_0, w_2_1, w_2_2, w_2_3, w_2_4],
    [w_3_0, w_3_1, w_3_2, w_3_3, w_3_4],
    [w_4_0, w_4_1, w_4_2, w_4_3, w_4_4],
    ])

G = Matrix([
    [Rational(' 1.   '),  Rational(' 0.    '), Rational('0.     '), Rational('0.      '), Rational('0.       ')],
    [Rational('-2./9.'),  Rational('-2./9. '), Rational(' -2./9.'), Rational(' -2./9. '), Rational('  -2./9. ')],
    [Rational('-2./9.'),  Rational(' 2./9. '), Rational(' -2./9.'), Rational('  2./9. '), Rational('  -2./9. ')],
    [Rational('1./90.'),  Rational('1./45. '), Rational(' 2./45.'), Rational(' 4./45. '), Rational('  8./45. ')],
    [Rational('1./90.'),  Rational('-1./45.'), Rational(' 2./45.'), Rational(' -4./45.'), Rational('  8./45. ')],
    [Rational('4./45.'),  Rational('2./45. '), Rational(' 1./45.'), Rational(' 1./90. '), Rational('  1./180.')],
    [Rational('4./45.'),  Rational('-2./45.'), Rational(' 1./45.'), Rational(' -1./90.'), Rational('  1./180.')],
    [Rational(' 0.   '),  Rational(' 0.    '), Rational('0.     '), Rational('0.      '), Rational('1.       ')]]
    )

A = G * W * (G.transpose())

A_list = []
for s in range(8):
    for t in range(8):
        row_index = s * 8 + t
        # initialize all entries
        row_list = []
        for i in range(5):
            for j in range(5):
                col_index = i * 5 + j
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
            col_index = int(weight_indexes[1]) * 5 + int(weight_indexes[2])
            num_str = str(numerator)+'/'+str(denominator)
            if negative:
                num_str = '-' + num_str
            A_list[row_index][col_index] = Rational(num_str)
            out = out[len(item):]
        
A = Matrix(A_list)
print(A)

# verify A
kernel = numpy.random.rand(5,5)
kernel_t = numpy.matmul(numpy.array(G).astype(numpy.float64), kernel)
kernel_t = numpy.matmul(kernel_t, numpy.transpose(numpy.array(G).astype(numpy.float64)))

kernel_t_new = kernel.reshape(25)
kernel_t_new = numpy.matmul(numpy.array(A).astype(numpy.float64), kernel_t_new)
kernel_t_new = kernel_t_new.reshape(8, 8)
diff = abs(kernel_t_new - kernel_t) > 1e-10
assert(diff.sum() == 0)

# calculate (I + \lambda * BT * B) ^ -1
m = numpy.random.rand(64) > 0.5
diag = numpy.diag(m.astype(numpy.float64))

B = numpy.matmul(diag, numpy.array(A).astype(numpy.float64))

LAMBDA = 0.5
C = numpy.identity(25) + LAMBDA * (numpy.matmul(numpy.transpose(B), B))

C_inv = inv(C)
