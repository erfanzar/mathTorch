import sys
import time

try:
    from interfaces import colorp
    import numpy as np
    from random import random

except OSError:
    import os
    from interfaces import colorp

    colorp('Downloading Dependencies', 255, 100, 100)
    os.system('pip install numpy')

union_reshape = [int, tuple, list, object]


def colorp(r, b, g, *args):
    return f'\r \033[38;2;{r};{b};{g}m{args}\033[38;2;255;255;255m'


class Tensor:

    def shape(self, *dn):
        sh = [h for h in dn]

        return self.shape_loop(sh)

    def shape_loop(self, sh):
        sp_2 = 0
        sp_3 = 0
        sp_1 = 0
        for p in sh:
            sp_1 += 1
            for k in p:
                sp_2 += 1
                for j in k:
                    sp_3 += 1

            return [sp_1, sp_2, sp_3]

    def reshape(self, *shape):
        list_n = [v for v in shape]
        return self.loop(list_n)

    def loop(self, o) -> list:
        ...
        lop = []
        if len(o) > 1:
            for b in range(len(o)):
                lop.append(self.loop(o[1:]))
            return lop
        else:
            for i in range(o[0]):
                v = random()
                lop.append(v)
            return lop


if __name__ == "__main__":
    tpa = Tensor()
    v = tpa.reshape(6,6,6)
    shape = tpa.shape(v)
    print(shape)
