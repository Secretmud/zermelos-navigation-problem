import numpy as np
from lib.sigma import sigma, sigma_test
from joblib import Memory
from lib.time import time
from lib import Z

# Set up joblib memory for caching
memory = Memory(location=".joblib_cache", verbose=0)


@memory.cache
def H_B(N):
    costmatrix = np.zeros((3**N, 3**N), dtype='float64')
    # times = [1, 3, 2]
    for k in range(N):
        mat = Z
        for i in range(N):
            c_time = time(k, mat[i][i])
            mat[i][i] = c_time
        s_mat = sigma(N, k, matrix=mat)
        
        # c_time = times[k]
        costmatrix += s_mat

    return costmatrix


@memory.cache
def H_B_test(N):
    costmatrix = np.zeros(3**N, dtype='float64')
    for k in range(N):
        costmatrix += time[k]*sigma_test(N, k)

    return costmatrix


@memory.cache
def H_P(N):
    penalty_matrix = np.zeros((3**N, 3**N), dtype='float64')
    for k in range(N):
        penalty_matrix += sigma(N, k)

    return np.matmul(penalty_matrix, penalty_matrix)


@memory.cache
def H_D(N, matrix):
    d_matrix = np.zeros((3**N, 3**N), dtype='float64')
    for k in range(N):
        d_matrix += sigma(N, k, matrix=matrix)

    return d_matrix


@memory.cache
def H_P_test(N):
    penalty_matrix = np.zeros(3**N, dtype='float64')
    for k in range(N):
        penalty_matrix += sigma_test(N, k)

    penalty_matrix = np.square(penalty_matrix)

    return penalty_matrix


def Hamiltonian(alpha, beta, N):
    # if beta == 0:
    return H_B(N) + alpha * H_P(N)
