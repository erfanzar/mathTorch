try : 
    import sys
    import time
    import numpy as np
    import yaml
except:
    import os
    os.system('pip install numpy')
    os.system('pip install yaml')
    
    
filt_type = [float, int, list, tuple]


class Linear:
    def __init__(
            self,
            input_size: int,
            target_size: int,
            train: bool = True,
            save: bool = False
    ):
        self.inputs = None
        self.output = None
        self.train = train
        self.targets = None
        self.times = 0
        self.error = None
        self.accuracy = 0
        self.input_size = input_size
        self.target_size = target_size
        self.w = np.random.randn(self.input_size, self.target_size)
        self.b = 1
        self.lr = np.array([3e-4])

    def forward(self,
                inputs: filt_type,
                targets: filt_type,
                ):
        self.times += 1
        self.inputs = np.array(inputs, dtype=np.float64)
        self.targets = np.array(targets, dtype=np.float64)
        self.output = self.w.T.dot(self.inputs) + self.b
        if self.train:
            self.backward()
        self.accuracy += 1 if self.output == self.targets else 0
        self.accuracy /= self.times
        self.print_function()
        return self.output

    def print_function(self):
        sys.stdout.write(f'\r prediction : {self.output} / {self.targets} accuracy : {self.accuracy} ')
        sys.stdout.flush()

    def load_point(self, file: str):
        try:
            with open(f'{file}', 'r') as rd:
                read = yaml.full_load(rd)
                if self.input_size == read[0]["input_size"] and self.target_size == read[1]["target_size"]:
                    self.w = np.array(read[2]['weights'])
                    self.b = np.array(read[3]['biases'])
                    self.lr = np.array(read[4]['lr'])
                    print(f'Read Wights : {self.w}')
                    time.sleep(5)
                else:
                    sys.stdout.write('neuron with saved weights are not same size \n'
                                     f'excepted input size : {self.input_size} found {read[0]["input_size"]}\n'
                                     f'excepted targets size : {self.target_size} found {read[1]["target_size"]}\n'
                                     )
                    time.sleep(5)

        except OSError:
            print('No file found')

    def backward(self):
        self.error = self.targets - self.output
        perhaps = self.inputs * self.error * self.lr
        perhaps = np.resize(perhaps, self.w.shape)
        self.w += perhaps
        self.b += self.error * self.lr

    def save_point(self, name: str = 'save'):
        with open(f'{name}.yaml', 'a') as sv:
            point = [
                {'input_size': self.input_size},
                {'target_size': self.target_size},
                {'weights': self.w.tolist()},
                {'biases': self.b.tolist()},
                {'lr': self.lr.tolist()}
            ]
            writer = yaml.dump(point, sv)
