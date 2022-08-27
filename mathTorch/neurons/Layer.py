import yaml


class Layer:
    def __init__(self):
        pass

    def forward(self, x):
        return NotImplementedError

    def backward(self, lr):
        return NotImplementedError

    def ckpt(self, name: str = 'save'):
        return NotImplementedError


class Network(Layer):
    def __init__(self):
        super(Network, self).__init__()
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        if len(self.layers) > 0:
            for layer in self.layers:
                x = layer.forward(x)

        return x

    def ckpt(self, name: str = 'save'):
        with open(f'{name}.yaml', 'w') as w:
            yaml.dump([{'bias': {ls.bias}, 'weights': {ls.weights}} for ls in self.layers], w)

    def backward(self, lr):

        if len(self.layers) > 0:
            for layer in self.layers[::-1]:
                x = layer.backward(lr)
