from sympy import * 
import numpy
import re


for i in range(6):
    for j in range(6):
        locals()['w_'+str(i)+'_'+str(j)] = Symbol('w_'+str(i)+'_'+str(j))

W = Matrix([
    [w_0_0, w_0_1, w_0_2, w_0_3, w_0_4, w_0_5],
    [w_1_0, w_1_1, w_1_2, w_1_3, w_1_4, w_1_5],
    [w_2_0, w_2_1, w_2_2, w_2_3, w_2_4, w_2_5],
    [w_3_0, w_3_1, w_3_2, w_3_3, w_3_4, w_3_5],
    [w_4_0, w_4_1, w_4_2, w_4_3, w_4_4, w_4_5],
    [w_5_0, w_5_1, w_5_2, w_5_3, w_5_4, w_5_5],
    ])

for i in range(6):
    for j in range(6):
        locals()['x_'+str(i)+'_'+str(j)] = Symbol('x_'+str(i)+'_'+str(j))

X = Matrix([
    [x_0_0, x_0_1, x_0_2, x_0_3, x_0_4, x_0_5],
    [x_1_0, x_1_1, x_1_2, x_1_3, x_1_4, x_1_5],
    [x_2_0, x_2_1, x_2_2, x_2_3, x_2_4, x_2_5],
    [x_3_0, x_3_1, x_3_2, x_3_3, x_3_4, x_3_5],
    [x_4_0, x_4_1, x_4_2, x_4_3, x_4_4, x_4_5],
    [x_5_0, x_5_1, x_5_2, x_5_3, x_5_4, x_5_5],
    ])

BT = Matrix([
    [4.,  0., -5.,  0., 1., 0.],
    [0., -4., -4.,  1., 1., 0.],
    [0.,  4., -4., -1., 1., 0.],
    [0., -2., -1.,  2., 1., 0.],
    [0.,  2., -1., -2., 1., 0.],
    [0.,  4.,  0., -5., 0., 1.]]
    )


X_t = BT * X
X_t = X_t * (BT.transpose())


O_t = W.multiply_elementwise(X_t)

AT = Matrix([
    [1., 1.,  1., 1.,  1., 0.],
    [0., 1., -1., 2., -2., 0.],
    [0., 1.,  1., 4.,  4., 0.],
    [0., 1., -1., 8., -8., 1.]]
    )

O = AT * O_t
O = O * (AT.transpose())

O = Matrix(Matrix(O).expand())

threshold = numpy.zeros([6,6])
for index in range(16):
    out = str(O[index])
    items = re.split(' + | - ', out)
    for item in items:
        l = item.split('*')
        weight = float(l[0]) ** 2.0
        w_index = l[1].split('_')
        threshold[int(w_index[1])][int(w_index[2])] += weight
print threshold**0.5
