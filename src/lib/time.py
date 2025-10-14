import numpy as np
from lib import smax, D, L, v


def S(x, offset=0.15):
    # return smax*np.exp(-(x-D/2)**4/5000) 
    # return (smax - offset)*np.sin(np.pi*x/D)+offset
    x0 = 0.2*D
    sigma = 0.14*D
    return smax*np.exp(-0.5*((x-x0)/sigma)**2)    


def time(k, direction, N):
    dx = D / N
    dy = L / N
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
