#!/usr/bin/env python2
from __future__ import absolute_import, division, print_function, unicode_literals
from sympy import *
import re
import numpy

for i in range(8):
    for j in range(8):
        locals()['w_'+str(i)+'_'+str(j)] = Symbol('w_'+str(i)+'_'+str(j))

W = Matrix([
    [w_0_0, w_0_1, w_0_2, w_0_3, w_0_4, w_0_5, w_0_6, w_0_7],
    [w_1_0, w_1_1, w_1_2, w_1_3, w_1_4, w_1_5, w_1_6, w_1_7],
    [w_2_0, w_2_1, w_2_2, w_2_3, w_2_4, w_2_5, w_2_6, w_2_7],
    [w_3_0, w_3_1, w_3_2, w_3_3, w_3_4, w_3_5, w_3_6, w_3_7],
    [w_4_0, w_4_1, w_4_2, w_4_3, w_4_4, w_4_5, w_4_6, w_4_7],
    [w_5_0, w_5_1, w_5_2, w_5_3, w_5_4, w_5_5, w_5_6, w_5_7],
    [w_6_0, w_6_1, w_6_2, w_6_3, w_6_4, w_6_5, w_6_6, w_6_7],
    [w_7_0, w_7_1, w_7_2, w_7_3, w_7_4, w_7_5, w_7_6, w_7_7],
    ])
pprint(W)

BT = Matrix([
    [Rational(1.), Rational(    0.), Rational(-21./4.), Rational(     0.), Rational( 21./4.), Rational(     0.), Rational(-1.), Rational(0.)],
    [Rational(0.), Rational(    1.), Rational(     1.), Rational(-17./4.), Rational(-17./4.), Rational(     1.), Rational( 1.), Rational(0.)],
    [Rational(0.), Rational(   -1.), Rational(     1.), Rational( 17./4.), Rational(-17./4.), Rational(    -1.), Rational( 1.), Rational(0.)],
    [Rational(0.), Rational( 1./2.), Rational(  1./4.), Rational( -5./2.), Rational( -5./4.), Rational(     2.), Rational( 1.), Rational(0.)],
    [Rational(0.), Rational(-1./2.), Rational(  1./4.), Rational(  5./2.), Rational( -5./4.), Rational(    -2.), Rational( 1.), Rational(0.)],
    [Rational(0.), Rational(    2.), Rational(     4.), Rational( -5./2.), Rational(    -5.), Rational(  1./2.), Rational( 1.), Rational(0.)],
    [Rational(0.), Rational(   -2.), Rational(     4.), Rational(  5./2.), Rational(    -5.), Rational( -1./2.), Rational( 1.), Rational(0.)],
    [Rational(0.), Rational(   -1.), Rational(     0.), Rational( 21./4.), Rational(     0.), Rational(-21./4.), Rational( 0.), Rational(1.)]]
    )
pprint(BT)

for i in range(8):
    for j in range(8):
        locals()['i_'+str(i)+'_'+str(j)] = Symbol('i_'+str(i)+'_'+str(j))

I = Matrix([
    [i_0_0, i_0_1, i_0_2, i_0_3, i_0_4, i_0_5, i_0_6, i_0_7],
    [i_1_0, i_1_1, i_1_2, i_1_3, i_1_4, i_1_5, i_1_6, i_1_7],
    [i_2_0, i_2_1, i_2_2, i_2_3, i_2_4, i_2_5, i_2_6, i_2_7],
    [i_3_0, i_3_1, i_3_2, i_3_3, i_3_4, i_3_5, i_3_6, i_3_7],
    [i_4_0, i_4_1, i_4_2, i_4_3, i_4_4, i_4_5, i_4_6, i_4_7],
    [i_5_0, i_5_1, i_5_2, i_5_3, i_5_4, i_5_5, i_5_6, i_5_7],
    [i_6_0, i_6_1, i_6_2, i_6_3, i_6_4, i_6_5, i_6_6, i_6_7],
    [i_7_0, i_7_1, i_7_2, i_7_3, i_7_4, i_7_5, i_7_6, i_7_7],
    ])
pprint(I)

out_tmp = W.multiply_elementwise(BT * I * (BT.transpose()))

AT = Matrix([
    [Rational(1.), Rational(1.), Rational( 1.), Rational(1.), Rational( 1.), Rational(8.), Rational( 8.), Rational(0.)],
    [Rational(0.), Rational(1.), Rational(-1.), Rational(2.), Rational(-2.), Rational(4.), Rational(-4.), Rational(0.)],
    [Rational(0.), Rational(1.), Rational( 1.), Rational(4.), Rational( 4.), Rational(2.), Rational( 2.), Rational(0.)],
    [Rational(0.), Rational(1.), Rational(-1.), Rational(8.), Rational(-8.), Rational(1.), Rational(-1.), Rational(1.)]]
    )

out = AT * out_tmp * (AT.transpose())
out = Matrix(out.expand())

D = numpy.zeros([8,8])
C = numpy.zeros([8,8])
for i in range(4 * 4):
    tmp_out = str(out[i])
    tmp_out = tmp_out.replace(' ','')
    weights = {}
    while len(tmp_out) > 0:
        item = re.search('[+,-]*[0-9]*\**i_[0-9]*_[0-9]*\*w_[0-9]*_[0-9]*\/*[0-9]*', tmp_out).group(0)
        numerator = re.search('[+,-]*[0-9]*(?:\*i)', item)
        if numerator != None:
            numerator = numerator.group(0)
            numerator = int(numerator[0:len(numerator)-2])
        else:
            if item[0] == '-':
                numerator = -1
            else:
                numerator = 1
        denominator = re.search('/[0-9]*', item)
        if denominator != None:
            denominator = denominator.group()
            denominator = int(denominator[1:])
        else:
            denominator = 1
        input_name = re.search('i_[0-9]*_[0-9]*', item).group(0)
        weight_name = re.search('w_[0-9]*_[0-9]*', item).group(0)
        number = float(numerator) / denominator
        if weight_name in weights.keys():
            if input_name in weights[weight_name].keys():
                weights[weight_name][input_name] += number
            else:
                weights[weight_name][input_name] = number
        else:
            weights[weight_name] = {input_name: number}
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
