# import numpy as np
# from erutils.command_line_interface import fprint
#
# from activation import ReLU
# from networking import Linear, Sequential
#
# if __name__ == "__main__":
#     fprint('Imported Successfully')
#
#     x = np.arange(0, 30)
#     y = [3, 4, 5, 7, 10, 8, 9, 10, 10, 23, 27, 44, 50, 63, 67, 60, 62, 70, 75, 88, 81, 87, 95, 100, 108, 135, 151, 160,
#          169, 179]
#
#     fprint('Data Created ')
#
#     seq = Sequential(
#         Linear(1, 2), ReLU(), Linear(2, 4), ReLU(), Linear(4, 8), ReLU(), Linear(8, 1)
#     )
#
#     for x1 in x:
#
#         x1 = seq(x1)
#         print(x1)
