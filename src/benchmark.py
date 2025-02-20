import timeit
import numpy as np

# Assuming N, X, Z, and the sigma function are defined elsewhere in the code
Z = np.array([[1,0,0],[0,0,0],[0,0,-1]])
X = np.array([[0,1,0],[1,0,1],[0,1,0]])
N = 9

def sigma(k, matrix):
    ret_mat = np.identity(3) if k != 0 else matrix

    for x in range(1, N):
        if k == x:
            ret_mat = np.kron(ret_mat, matrix)
        else:
            ret_mat = np.kron(ret_mat, np.identity(3))

    return ret_mat

def sigma_test(k, matrix=Z): 
    return np.kron(np.kron(np.identity(3**k), matrix), np.identity(3**(N-k-1)))

def sigma_test_aux(k, matrix=Z):
    aux = np.kron(np.identity(3**k), matrix)
    return np.kron(aux, np.identity(3**(N-k-1)))

# Wrapper functions for timeit
def benchmark_sigma():
    for i in range(N):
        sigma(i, matrix=X)

def benchmark_sigma_test():
    for i in range(N):
        sigma_test(i, matrix=X)
        
def benchmark_sigma_test_aux():
    for i in range(N):
        sigma_test_aux(i, matrix=X)

# Benchmarking
time_sigma = timeit.timeit(benchmark_sigma, number=20)
time_sigma_test = timeit.timeit(benchmark_sigma_test, number=20)
time_sigma_test_aux = timeit.timeit(benchmark_sigma_test_aux, number=20)

print(f"Time for sigma: {time_sigma:.6f} seconds")
print(f"Time for sigma_test: {time_sigma_test:.6f} seconds")
print(f"Time for sigma_test_aux: {time_sigma_test_aux:.6f} seconds")

# Verification
def verify_results():
    for i in range(N):
        result_sigma = sigma(i, matrix=X)
        result_sigma_test = sigma_test(i, matrix=X)
        if not np.allclose(result_sigma, result_sigma_test):
            print(f"Mismatch found at k={i}")
            return
    print("All results match.")

verify_results()