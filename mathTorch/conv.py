try:
    import sys
    import time
    import numpy as np
    import yaml
    import cv2 as cv
except OSError:
    import os

    os.system('pip install numpy')
    os.system('pip install yaml')

filter_type = [int, list, tuple]


class Conv:
    def __init__(self,
                 channels: filter_type = 3,
                 kernel_size: filter_type = 3,
                 stride: filter_type = 2,
                 padding: filter_type = 0,
                 ):
        super(Conv, self).__init__()
        self.inputs = None
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel = np.random.randn(self.channels, self.kernel_size, self.kernel_size) / (self.kernel_size * self.kernel_size)

    def loop(self, inputs):
        ch, cw, cc = inputs.shape
        self.inputs = inputs
        for i in range(ch - self.kernel_size + self.padding):
            for j in range(cw - self.kernel_size + self.padding):
                iteration = inputs[i:i + self.kernel_size, j:j + self.kernel_size]
                yield iteration, i, j

    def forward(self, inputs):
        ch, cw, cc = inputs.shape
        prediction = np.zeros((ch - self.kernel_size, cw - self.kernel_size, self.channels))

        for iteration, i, j in self.loop(inputs):
            prediction[i, j] = np.sum(iteration * self.kernel, axis=(1, 2))

        return prediction

    def backward(self, error, lr):
        error_parameters = np.zeros(self.kernel.shape)
        for iteration, i, j in self.loop(self.inputs):
            for k in range(self.channels):
                error_parameters[k] = iteration * error[i, j, k]
        self.kernel -= error_parameters * lr
        return error_parameters


if __name__ == '__main__':
    nn = Conv(channels=3)
    x = np.random.randn(300, 300, 3)
    out = nn.forward(x)
    print(out.shape)
    while True:
        cv.imshow('out', out)
        cv.waitKey(1)
