import numpy as np
from lib import N, smax, D, L, v


def S(x, offset=0.15):
    return (smax - offset)*np.sin(np.pi*x/D)+offset


def time(k, direction):
    dx = N / D
    dy = L / D
    g = dy / dx
    x = dx*k
    current = S(x + dx / 2)
    match direction:
        case 0:  # straight
            denominator = 1 - current**2
        case 1:  # diagonal up
            denominator = np.sqrt(1 + g**2 - current**2) - g * current
        case -1:  # diagonal down
            denominator = np.sqrt(1 + g**2 - current**2) + g * current
        case _:
            raise ValueError("Invalid direction")
    return dx / v * (1 + g**2) / denominator
