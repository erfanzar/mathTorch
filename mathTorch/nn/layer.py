import numpy as np


class Layer:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(x=args[0])

    def forward(self, x, *args):
        return NotImplementedError

    def backward(self, output_error, lr):
        return NotImplementedError

    def mt(self):
        return NotImplementedError

    def ckpt(self, name: str = 'save'):
        return NotImplementedError

    @classmethod
    def dim(cls, array, shape):
        array = np.array(array)
        if len(array.shape) == 0 or len(array.shape) <= 1:
            return array.reshape(shape)
        else:
            return array


class Sequential(Layer):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        self.layers = [] if len(args) == 0 else [v for v in args]

        self.output = None
        self.act = None
        self.error = None
        self.d_error = None

    def add(self, layer):
        self.layers.append(layer)

    def tst(self, x):
        return self.__class__(x)

    def __str__(self):
        if len(self.layers) != 0:
            print('Sequential : (')
            for v in self.layers:
                print(f'\t\033[1;36m{v} ,')
            print(')')
            return ''
        else:
            print('\033[4;32m No Layer is Initialized Yet')
            return ''

    def multi_add(self, **kwargs):
        for args in kwargs:
            self.layers.append(args)

    def set_act(self, act):
        self.act = act

    def mt(self, name):
        data = [
            {
                f'{idx}': [
                    {'weights': ls.weights.tolist()},
                    {'bias': ls.bias.tolist()},
                    {'in_dim': f'{ls.in_dim}'},
                    {'out_dim': f'{ls.out_dim}'}
                ]
            } for idx, ls in enumerate(self.layers)]
        print(data)
        with open(f'{name}.mt', 'wb') as w:
            w.write(bytes(f"{data}"))

    def forward(self, x, *args):
        if len(self.layers) > 0:
            for layer in self.layers:
                x = layer.forward(x)

            self.output = x
            return x

    def backpropagation(self, lr, true_y):
        if self.act is None:
            self.act = lambda y, y_hat: np.sum(y - y_hat)
        if len(self.layers) > 0:
            self.error = np.array(self.act(true_y, self.output))
            self.d_error = np.sum(true_y - self.output) ** 2
            if len(self.error.shape) <= 1:
                self.error = self.error.reshape((1, self.error.shape[0]))
            for layer in self.layers[::-1]:
                self.error = layer.backward(self.error, lr)
