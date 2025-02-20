import numpy as np
from lib import L, N, smax, dx, v
# S to be used for the time
S = lambda x: smax*np.exp(-np.power(x-L/2, 4)/5000)
def timef(x):
    return dx*(S(x+dx/2)/(v**2-S(x+dx/2)**2))
#time = np.fromfunction(timef, (3**N,), dtype='float32')
time = np.array([1, 3, 2])