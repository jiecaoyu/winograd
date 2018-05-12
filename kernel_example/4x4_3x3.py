import numpy
import basic_conv

input = numpy.random.rand(6,6)
kernel = numpy.random.rand(3,3)

output_base = basic_conv.conv(input, kernel)
BT = numpy.array([
    [4.,  0., -5.,  0., 1., 0.],
    [0., -4., -4.,  1., 1., 0.],
    [0.,  4., -4., -1., 1., 0.],
    [0., -2., -1.,  2., 1., 0.],
    [0.,  2., -1., -2., 1., 0.],
    [0.,  4.,  0., -5., 0., 1.]]
    )

G = numpy.array([
    [ 1./4.,      0.,     0.],
    [-1./6.,  -1./6., -1./6.],
    [-1./6.,   1./6., -1./6.],
    [1./24.,  1./12.,  1./6.],
    [1./24., -1./12.,  1./6.],
    [    0.,      0.,    1.]]
    )

AT = numpy.array([
    [1., 1.,  1., 1.,  1., 0.],
    [0., 1., -1., 2., -2., 0.],
    [0., 1.,  1., 4.,  4., 0.],
    [0., 1., -1., 8., -8., 1.]]
    )

input_t = numpy.matmul(BT, input)
input_t = numpy.matmul(input_t, numpy.transpose(BT))


kernel_t = numpy.matmul(G, kernel)
kernel_t = numpy.matmul(kernel_t, numpy.transpose(G))

output_t = input_t * kernel_t

output_win = numpy.matmul(AT, output_t)
output_win = numpy.matmul(output_win, numpy.transpose(AT))
print 'original:'
print output_base
print '-------------------------------------------------'
print 'winograd:'
print output_win
