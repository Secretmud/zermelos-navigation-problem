import numpy as np
from joblib import Memory

from lib.sigma import sigma, sigma_vec
from lib.time import ntime, time
from lib import Z, float_type

# Set up joblib memory for caching
memory = Memory(location=".joblib_cache", verbose=0)



def H_B(N):
    costmatrix = np.zeros((3**N, 3**N), dtype=float_type)
    for k in range(N):
        c_mat = np.diag(ntime(k, N))
        costmatrix += sigma(N, k, matrix=c_mat)
    return costmatrix


def H_P(N):
    penalty_matrix = np.zeros((3**N, 3**N), dtype=float_type)
    for k in range(N):
        penalty_matrix += sigma(N, k)

    return np.matmul(penalty_matrix, penalty_matrix)


@memory.cache
def H_D(N, matrix):
    d_matrix = np.zeros((3**N, 3**N), dtype=float_type)
    for k in range(N):
        d_matrix += sigma(N, k, matrix=matrix)

    return d_matrix
