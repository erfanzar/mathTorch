try:
    import sys
    import time
    import numpy as np
    import yaml
except OSError:
    import os

    os.system('pip install numpy')
    os.system('pip install yaml')

nodes_union = [tuple, int]


class Softmax:
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
