import numpy as np
from lib import Z


def sigma(N, k, matrix=Z, basis=3):
    return np.kron(np.kron(np.identity(basis**k), matrix), np.identity(basis**(N-k-1)))


def sigma_vec(N, k, matrix=np.array([1, 0, -1]), basis=3):
    return np.kron(np.kron(np.ones(basis**k), matrix), np.ones(basis**(N-k-1)))
