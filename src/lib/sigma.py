import numpy as np
from lib import Z


def sigma(N, k, matrix=Z):
    return np.kron(np.kron(np.identity(3**k), matrix), np.identity(3**(N-k-1)))


def sigma_test(N, k, matrix=np.array([1, 0, -1])):
    return np.kron(np.kron(np.ones(3**k), matrix), np.ones(3**(N-k-1)))
