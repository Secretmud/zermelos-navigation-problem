import numpy as np
from lib import N, smax, D, L, v


def S(x, offset=0.15):
    return smax*np.exp(-(x-D/2)**4/5000) 
    # return (smax - offset)*np.sin(np.pi*x/D)+offset


def time(k, direction):
    dx = D / N
    dy = L / N
    g = dy / dx
    x = dx*k
    current = S(x + dx / 2)
    print(f"{dx=}, {dy=}, {g=}, {x=}, {current=}")
    match direction:
        case 0:  # straight
            frac = 1 / (1 - current**2)
        case 1:  # diagonal up
            frac = (1 + g**2) / (np.sqrt(1 + g**2 - current**2) - g * current)
        case -1:  # diagonal down
            frac = (1 + g**2) / (np.sqrt(1 + g**2 - current**2) + g * current)
        case _:
            raise ValueError("Invalid direction")
    return dx / v * frac
