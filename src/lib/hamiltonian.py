import numpy as np
from lib.sigma import sigma, sigma_test
from joblib import Memory

# Set up joblib memory for caching
memory = Memory(location=".joblib_cache", verbose=0)


@memory.cache
def H_B(N, time):
    costmatrix = np.zeros((3**N, 3**N), dtype='float64')
    for k in range(N):
        costmatrix += time[k]*sigma(N, k)

    return costmatrix


@memory.cache
def H_B_test(N, time):
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

def Hamiltonian(alpha, beta, N, time):
    #if beta == 0:
    #   print("H = H_B + alpha*H_P")
    return H_B(N, time) + alpha * H_P(N) 
    #return H_B() + alpha * H_P() + beta * H_D()
