import numpy

def conv(input, kernel):
    size = input.shape[0] - kernel.shape[0] + 1
    output = numpy.zeros([size, size])
    kernel_size = kernel.shape[0]
    for x in range(size):
        for y in range(size):
            for i in range(kernel_size):
                for j in range(kernel_size):
                    output[x][y] += kernel[i][j] * input[i + x][j + y]

    return output
