import numpy as np
from lib import Z

def sigma(N, k, matrix=Z):
    return np.kron(np.kron(np.identity(3**k), matrix), np.identity(3**(N-k-1)))
