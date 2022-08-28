from .activation import mse_prime, mse, tanh_prime, tanh, relu, Softmax, ActivationLayer, ReLU, Mse
from .conv import Conv
from .layer import Network
from .linear import Linear


class MathTorch:
    """
    init for the functions or the classes
    inside the main class
    iv done this with this method cause this way was faster to work with
    and easier to use
    :except None
    """
    ReLU = ReLU
    Mse = Mse
    Conv = Conv
    Linear = Linear
    Network = Network
    mse = mse
    mse_prime = mse_prime
    tanh = tanh
    tanh_prime = tanh_prime
    relu = relu
    Softmax = Softmax
    act = ActivationLayer
