try:
    from interfaces import colorp
    import numpy
except OSError:
    import os
    from interfaces import colorp

    colorp('Downloading Dependencies', 255, 100, 100)
    os.system('pip install numpy')

union_reshape = [int, tuple, list, object]


def __reshape__(inp: union_reshape) -> union_reshape:
    """
    :rtype: object
    """
    inp = inp.reshape(inp[-1], inp[0], inp[1])
    return inp
