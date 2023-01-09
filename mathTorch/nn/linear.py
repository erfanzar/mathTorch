import numpy as np

try:
    from .layer import Layer
except:
    from layer import Layer


class Linear(Layer):
    def __init__(self, in_dim, out_dim):
        super(Linear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights = np.random.randn(in_dim, out_dim)  #
        self.bias = np.random.randn(1, out_dim)  #
        self.input = None
        self.out = None

    def forward(self, x, *args):
        self.input = x
        return np.dot(self.input, self.weights) + self.bias

    def __str__(self):
        return f'Linear({self.in_dim}, {self.out_dim})'

    def backward(self, output_error, lr):
        weight_grad = np.dot(output_error, self.input.T)
        input_grad = np.dot(self.weights.T, output_error)

        self.weights -= lr * weight_grad
        self.bias -= lr * input_grad

        return input_grad
