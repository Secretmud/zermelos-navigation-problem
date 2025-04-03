import numpy as np
from lib.sigma import sigma, sigma_test
from lib.time import time
from lib import X, N


def H_B():
    costmatrix = np.zeros((3**N, 3**N), dtype='float64')
    for k in range(N):
        costmatrix += time[k]*sigma(N, k)

    return costmatrix


def H_B_test():
    costmatrix = np.zeros(3**N, dtype='float64')
    for k in range(N):
        costmatrix += time[k]*sigma_test(N, k)

    return costmatrix


def H_P():
    penalty_matrix = np.zeros((3**N, 3**N), dtype='float64')
    for k in range(N):
        penalty_matrix += sigma(N, k)

    penalty_matrix = np.matmul(penalty_matrix, penalty_matrix)

    return penalty_matrix


def H_D():
    d_matrix = np.zeros((3**N, 3**N), dtype='float64')
    for k in range(N):
        d_matrix += sigma(N, k, matrix=X)

    return d_matrix


def H_P_test():
    penalty_matrix = np.zeros(3**N, dtype='float64')
    for k in range(N):
        penalty_matrix += sigma_test(N, k)

    penalty_matrix = np.square(penalty_matrix)

    return penalty_matrix
