try:
    import sys
    import time
    import numpy as np

    import numba as nb
except ImportWarning:
    import os
    from ..utils.utils import printf

    printf('Downloading Dependencies')
    os.system('pip install numpy')

filter_type = [int, list, tuple]

union_reshape = [int, tuple, list, object]
print_union = [str, int, list, tuple, bool]


def __reshape__(inp: union_reshape) -> union_reshape:
    """
    :rtype: object
    """
    inp = inp.reshape(inp.shape[2], inp.shape[0], inp.shape[1])
    return inp


class Conv:
    def __init__(self,
                 channels: filter_type = 3,
                 kernel_size: filter_type = 3,
                 stride: filter_type = 2,
                 padding: filter_type = 0,
                 ):
        super(Conv, self).__init__()
        self.last_input = None
        self.inputs = None
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel = np.random.randn(self.channels, self.kernel_size, self.kernel_size) / (
                self.kernel_size * self.kernel_size)

    def loop(self, inputs):

        ch, cw, cc = inputs.shape
        self.inputs = inputs
        for c in range(self.channels):
            for h in range(((ch - self.kernel_size) / self.stride) + self.padding):
                for w in range(((cw - self.kernel_size) / self.stride) + self.padding):
                    iteration = inputs[c, h:h + self.kernel_size, w:w + self.kernel_size]
                    yield iteration, c, h, w

    def forward(self, inputs):
        self.last_input = inputs
        cc, ch, cw = inputs.shape
        prediction = np.zeros((self.channels, ((ch - self.kernel_size) / self.stride) + self.padding,
                               ((cw - self.kernel_size) / self.stride) + self.padding))
        for iteration, c, h, w in self.loop(inputs):
            prediction[c, h, w] = np.sum(self.kernel * iteration, axis=(1, 2))

        return prediction

    def backward(self, error, lr):
        update_parameters = np.zeros(self.kernel.shape)
        for iteration, i, j in self.loop(self.last_input):
            for k in range(self.kernel_size):
                update_parameters[k] += error[i, j, k] * iteration

        self.kernel -= lr * update_parameters

        return None
