try:
    from .activation import ActivationLayer, ReLU, MSE, Softmax, mse, mse_prime, tanh_prime, tanh, relu
    from .networking import Layer, Conv, Linear, Sequential

except :

    print('\033[1;36m RUNNING FROM INSIDE PACKAGE')
    from activation import ActivationLayer, ReLU, MSE, Softmax, mse, mse_prime, tanh_prime, tanh, relu
    from networking import Layer, Conv, Linear, Sequential
