import numpy as np
from lib import smax, D, v


def S(x):
    # return smax*np.exp(-(x-D/2)**4/5000)
    return smax*np.exp(-(x-D/3)**4/4000)

scale = 100


def ntime(k, N):
    dx = D / N
    dy = dx / scale
    g = dy / dx
    x = dx * k
    current = S(x + dx / 2)

    t_up = dx / v * ((1 + g**2) / (np.sqrt(1 + g**2 - current**2) - g * current))
    t_straight = dx / v * (1 / (1 - current**2))
    t_down = dx / v * ((1 + g**2) / (np.sqrt(1 + g**2 - current**2) + g * current))

    # Return in order matching direction codes: 0 (straight), 1 (up), -1 (down)
    # print(t_up, t_straight, t_down)
    return np.array([t_up, t_straight, t_down])



def time(k, direction, N):
    dx = D / N
    dy = dx / scale
    g = dy / dx
    x = dx*k
    current = S(x + dx / 2)
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
