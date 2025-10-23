import numpy as np
from joblib import Memory

from lib.sigma import sigma, sigma_test
from lib.time import ntime, time
from lib import Z, float_type

# Set up joblib memory for caching
memory = Memory(location=".joblib_cache", verbose=0)


"""
@memory.cache
def H_B(N):
    costmatrix = np.zeros((3**N, 3**N), dtype='float64')
    time = [1, 3, 2, 1]
    for k in range(N):
        costmatrix += time[k]*sigma(N, k)

    return costmatrix


"""

@memory.cache
def H_B2(N):
    costmatrix = np.zeros((3**N, 3**N), dtype='float64')
    times = [1, 3, 2]
    for k in range(N):
        mat = np.copy(Z)
        for i in range(N):
            c_time = time(k, mat[i][i], N)
            print(c_time, end=" ")
            mat[i][i] = times[i]
        print()
        s_mat = sigma(N, k, matrix=mat)
        
        costmatrix += s_mat

    return costmatrix

def H_B(N):
    costmatrix = np.zeros((3**N, 3**N), dtype=float_type)
    times = None# [1, 3, 2, 1]
    for k in range(N):
        if not times:
            c_mat = np.diag(ntime(k, N))
            costmatrix += sigma(N, k, matrix=c_mat)
        else:
            costmatrix += sigma(N, k)*times[k]
    return costmatrix
"""
#@memory.cache
def H_B(N):
    costmatrix = np.zeros((3**N, 3**N), dtype=float_type)
    sigma_trav = np.copy(Z)
    for k in range(N):
        tmp_mat = np.zeros((3, 3), dtype=float_type)
        for i in range(3):
            t = time(k, sigma_trav[i, i], N)
            print(f"{t:.6f}", end=" ")
            tmp_mat[i, i] = t
        print()

        s_mat = sigma(N, k, matrix=tmp_mat)
        # s_mat = sigma(N, k) * c_time[k]
        costmatrix += s_mat

    return costmatrix
"""
#@memory.cache
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


@memory.cache
def H_P_test(N):
    penalty_matrix = np.zeros(3**N, dtype=float_type)
    for k in range(N):
        penalty_matrix += sigma_test(N, k)

    penalty_matrix = np.square(penalty_matrix)

    return penalty_matrix
