import numpy as np

from mathTorch.neurons.MathTorch import MathTorch as Mt

"""
Test For The Linear neuralNetwork IN Library 
"""

x = [
    [[1, 3]], [[6, 1]], [[7, 2]], [[9, 1]], [[1, 6]], [[7, 6]], [[1, 4]], [[3, 4]]
]

y = [
    [[4]], [[7]], [[9]], [[10]], [[7]], [[13]], [[5]], [[7]]
]

if __name__ == "__main__":
    x = np.array(x).reshape(8, 2)
    y = np.array(y).reshape(8, 1)

    net = Mt.Network()
    net.add(Mt.Linear(2, 2))
    net.add(Mt.Linear(2, 1))


    net.set_act(Mt.mse_prime)

    for i in range(1000):
        err = 0
        lp = 0
        ly = 0
        for idx, (xi, yi) in enumerate(zip(x, y)):
            xi = xi.reshape(1, 2)
            y_hat = net.forward(xi)
            lp = y_hat
            ly = yi
            net.backpropagation(0.003, yi)
            err += net.d_error
        print(f'{err / len(x)} {ly} / {lp}')

    net.ckpt('save')
