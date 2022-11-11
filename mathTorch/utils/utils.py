import os
import numba as nb
import sys
import numpy as np

Any = [list, dict, int, float, str]


class Cp:
    Type = 1
    BLACK = f'\033[{Type};30m'
    RED = f'\033[{Type};31m'
    GREEN = f'\033[{Type};32m'
    YELLOW = f'\033[{Type};33m'
    BLUE = f'\033[{Type};34m'
    MAGENTA = f'\033[{Type};35m'
    CYAN = f'\033[{Type};36m'
    WHITE = f'\033[{Type};1m'
    RESET = f"\033[{Type};39m"


def printf(*args, end: bool = False):
    sys.stdout.write('\033[1;36m'.join(f'{a}' for a in args))
    if end:
        print('\n')


def attr_exist_check_(attr, index):
    try:
        s = attr[index]
    except IndexError:
        s = []
    return s


@nb.jit(forceobj=True, fastmath=True)
def fast_reader(path, names, total: int = None, sep: str = ' '):
    tta = []
    if total is None:
        total = len(names)
    for i in range(total):
        ba = np.roll(
            np.loadtxt(f'{path}/{names[i][:-4]}.txt', delimiter=sep, ndmin=2, ), 4,
            axis=1).tolist()
        tta.append(ba)
    return tta
