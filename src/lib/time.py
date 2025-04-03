import numpy as np
from lib import L, N, smax, dx, v
# S to be used for the time
def S(x): return smax*np.exp(-np.power(x-L/2, 4)/5000)


def timef(x):
    return dx*(S(x+dx/2)/(v**2-S(x+dx/2)**2))


# time = np.fromfunction(timef, (3**N,), dtype='float32')
time = [1, 2, 3, 4, 3, 2, 1, 1]
