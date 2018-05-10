import numpy
import basic_conv

input = numpy.random.rand(8,8)
kernel = numpy.random.rand(5,5)

output_base = basic_conv.conv(input, kernel)
BT = numpy.array([
    [1.,   0.  ,  -21./4.,    0.  ,  21./4. ,    0. ,   -1.,  0.],
    [0. ,  1.   ,   1.    ,-17./4. , -17./4. ,   1.  ,  1.  , 0.],
    [0. ,  -1.  ,   1.    ,17./4.  , -17./4. ,  -1.  ,  1.  , 0.],
    [0. , 1./2. ,   1./4. ,  -5./2.,   -5./4.,     2.,    1.,   0.],
    [0. , -1./2.,   1./4. ,   5./2.,   -5./4.,    -2.,    1.,   0.],
    [0. ,  2.   ,   4.    ,-5./2.  ,  -5.    , 1./2. ,  1.  , 0.],
    [0. ,  -2.  ,   4.    , 5./2.  ,  -5.    ,-1./2. ,  1.  , 0.],
    [0. ,  -1.  ,   0.    ,21./4.  ,   0.    ,-21./4.,  0.  , 1.]]
    )

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

AT = numpy.array([
    [1.,  1. , 1. ,  1.,  1. ,  8. , 8. ,  0.],
    [0.,  1. , -1.,  2.,  -2.,  4. , -4.,  0.],
    [0.,  1. , 1. ,  4.,  4. ,  2. , 2. ,  0.],
    [0.,  1. , -1.,  8.,  -8.,  1. , -1.,  1.]]
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
