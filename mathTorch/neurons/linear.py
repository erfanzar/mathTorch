import numpy as np

from .layer import Layer


class Linear(Layer):
    def __init__(self, in_dim, out_dim):
        super(Linear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights = np.random.randn(in_dim, out_dim) - 0.5  #
        self.bias = np.random.randn(1, out_dim) - 0.5  #
        self.input = None
        self.out = None

    def forward(self, x, *args):
        self.input = x
        self.out = np.dot(self.input, self.weights) + self.bias
        return self.out

    def backward(self, output_error, lr):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= lr * weights_error
        self.bias -= lr * output_error

        return input_error
