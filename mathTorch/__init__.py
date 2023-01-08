from mathTorch.neurons.activation import ActivationLayer, ReLU, Mse, Softmax
from mathTorch.neurons.conv import Conv
from mathTorch.neurons.layer import Layer, Network
from mathTorch.neurons.linear import Linear


class nn:
    Linear = Linear
    Conv = Conv
    Network = Network
    Layer = Layer
    Softmax = Softmax
    Mse = Mse
    ReLU = ReLU
    ActivationLayer = ActivationLayer


del Softmax, ReLU, Mse, ActivationLayer, Layer, Network, Conv, Layer
