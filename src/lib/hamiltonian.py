import numpy as np
from lib.sigma import sigma, sigma_test
from joblib import Memory
from lib.time import time
from lib import Z, float_type
import copy

# Set up joblib memory for caching
memory = Memory(location=".joblib_cache", verbose=0)


@memory.cache
def H_B(N):
    costmatrix = np.zeros((3**N, 3**N), dtype='float64')
    sigma_trav = np.copy(Z) 
    for k in range(N):
        tmp_mat = np.zeros((3, 3), dtype='float64')
        for i in range(3):
            t = time(k, sigma_trav[i,i], N)
            print(t)
            tmp_mat[i, i] = t

        s_mat = sigma(N, k, matrix=tmp_mat)
        costmatrix += s_mat

    return costmatrix


@memory.cache
def H_B_test(N):
    costmatrix = np.zeros(3**N, dtype='float64')
    sigma_trav = np.array([1, 0, -1])  # Just to get the right shape
    for k in range(N):
        tmp_mat = np.zeros(3, dtype='float64')
        for i in range(3):
            t = time(k, sigma_trav[i], N)
            tmp_mat[i] = t
        s_mat = sigma(N, k, matrix=tmp_mat)
        costmatrix += s_mat

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
