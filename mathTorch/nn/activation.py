try:
    import sys
    import time
    import numpy as np
    import yaml

    try:

        from .layer import Layer
    except:
        from layer import Layer
except OSError:
    import os

    os.system('pip install numpy')
    os.system('pip install yaml')

nodes_union = [tuple, int]


class Softmax(Layer):
    def __init__(self, input_len: int, bs: nodes_union):
        super(Softmax, self).__init__()
        self.last_shape = None
        self.last_input = None
        self.oot = None
        self.input_len = input_len
        self.nodes = bs
        self.w = np.random.randn(input_len, bs) / input_len
        self.b = np.zeros(bs)

    def forward(self, x):
        self.last_shape = x.shape
        x = x.flatten()
        self.last_input = x
        oot = x.T.dot(self.w) + self.b
        self.oot = oot
        exp = np.exp(oot)
        return exp / np.sum(exp, axis=1)

    def backward(self, error, lr):
        for i, gradiant in enumerate(error):
            if gradiant == 0:
                continue

            texp = np.exp(self.oot)
            s = np.sum(texp)

    def __str__(self):
        return 'Softmax()'
class ActivationLayer(Layer):
    def __init__(self, use_act: str = 'relu'):
        super(ActivationLayer, self).__init__()
        if use_act == 'relu':
            self.activation = relu
            self.activation_prime = relu

        if use_act == 'mse':
            self.activation = mse
            self.activation_prime = mse_prime

        if use_act == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        self.x = None
        self.out = None
        self.use_act = use_act
        self.weights = "ReLU"
        self.bias = "ReLU"
        self.in_dim = "ReLU"
        self.out_dim = "ReLU"

    def forward(self, x, *args):
        self.x = x
        self.out = self.activation(self.x)
        return self.out

    def backward(self, output_error, lr):
        return self.activation_prime(self.x) * output_error

    def __str__(self):
        return f'ActivationLayer({self.use_act})'

def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def relu(x):
    return max(0, x.all())


class MSE(Layer):
    def __init__(self):
        super().__init__()
        self.y = None
        self.y_hat = None

    def __str__(self):
        return f'MSE()'

    def forward(self, x, *args):
        self.y = x
        self.y_hat = args[0]

        return np.mean(np.power(self.y - self.y_hat, 2))

    def backward(self, output_error, lr):
        return 2 * (self.y_hat - self.y) / self.y.size


class ReLU(Layer):
    def __init__(self):
        super(ReLU, self).__init__()
        self.out = None

    def forward(self, x, *args):
        self.out = max(0, x.all())
        return self.out

    def __str__(self):
        return f'ReLU()'

    def backward(self, output_error, lr):
        return self.out * output_error * lr
