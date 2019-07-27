#!/usr/bin/env python2
from __future__ import absolute_import, division, print_function, unicode_literals
from sympy import *
import re
import numpy

for i in range(4):
    for j in range(4):
        locals()['w_'+str(i)+'_'+str(j)] = Symbol('w_'+str(i)+'_'+str(j))

W = Matrix([
    [w_0_0, w_0_1, w_0_2, w_0_3],
    [w_1_0, w_1_1, w_1_2, w_1_3],
    [w_2_0, w_2_1, w_2_2, w_2_3],
    [w_3_0, w_3_1, w_3_2, w_3_3],
    ])
# pprint(W)

BT = Matrix([
    [Rational(1.), Rational( 0.), Rational(-1.), Rational(0.)],
    [Rational(0.), Rational( 1.), Rational( 1.), Rational(0.)],
    [Rational(0.), Rational(-1.), Rational( 1.), Rational(0.)],
    [Rational(0.), Rational(-1.), Rational( 0.), Rational(1.)]]
    )
# pprint(BT)

for i in range(4):
    for j in range(4):
        locals()['i_'+str(i)+'_'+str(j)] = Symbol('i_'+str(i)+'_'+str(j))

I = Matrix([
    [i_0_0, i_0_1, i_0_2, i_0_3],
    [i_1_0, i_1_1, i_1_2, i_1_3],
    [i_2_0, i_2_1, i_2_2, i_2_3],
    [i_3_0, i_3_1, i_3_2, i_3_3],
    ])
# pprint(I)

out_tmp = W.multiply_elementwise(BT * I * (BT.transpose()))

AT = Matrix([
    [Rational(1.), Rational(1.), Rational( 1.), Rational(0.)],
    [Rational(0.), Rational(1.), Rational(-1.), Rational(1.)]]
    )

out = AT * out_tmp * (AT.transpose())
out = Matrix(out.expand())
# pprint(out)

D = numpy.zeros([4,4])
C = numpy.zeros([4,4])
for i in range(2 * 2):
    tmp_out = str(out[i])
    tmp_out = tmp_out.replace(' ','')
    weights = {}
    while len(tmp_out) > 0:
        item = re.search('[+,-]*[0-9]*\**i_[0-9]*_[0-9]*\*w_[0-9]*_[0-9]*', tmp_out).group(0)
        numerator = re.search('[+,-]*[0-9]*(?:\*i)', item)
        if numerator != None:
            numerator = numerator.group(0)
            numerator = int(numerator[0:len(numerator)-2])
        else:
            if item[0] == '-':
                numerator = -1
            else:
                numerator = 1
        input_name = re.search('i_[0-9]*_[0-9]*', item).group(0)
        weight_name = re.search('w_[0-9]*_[0-9]*', item).group(0)
        if weight_name in weights.keys():
            if input_name in weights[weight_name].keys():
                weights[weight_name][input_name] += numerator
            else:
                weights[weight_name][input_name] = numerator
        else:
            weights[weight_name] = {input_name: numerator}
        tmp_out = tmp_out[len(item):]

    for weight_key in weights.keys():
        for input_key in weights[weight_key].keys():
            # D
            x = int(weight_key.split('_')[1])
            y = int(weight_key.split('_')[2])
            D[x][y] += (weights[weight_key][input_key] ** 2.0)

    for weight_key in weights.keys():
        for input_key1 in weights[weight_key].keys():
            for input_key2 in weights[weight_key].keys():
                if input_key1 == input_key2:
                    continue
                x = int(weight_key.split('_')[1])
                y = int(weight_key.split('_')[2])
                C[x][y] += (weights[weight_key][input_key1] * weights[weight_key][input_key2])
print(D ** 0.5)
# print(C)
