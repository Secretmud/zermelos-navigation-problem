from manim import *
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='Plot the Hamiltonian of a quantum system.')
parser.add_argument('-n', help='Number of qubits', default=2)
parser.add_argument('-l', help='Length of the chain', default=20)
args = parser.parse_args()

# Define Pauli matrices
Z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
X = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

N = int(args.n)
L = float(args.l)
dx = L / N
smax = 0.5
v = 1

# Initialize s as (-1, -1, ..., -1) of length N
s = tuple([-1] * N)

def S(x):
    return smax * np.exp(-np.power(x - L/2, 4) / 5000)

def costf(x):
    return dx * (S(x+dx/2) / ( v**2 - S(x+dx/2)**2))

# Generate cost values
x_vals = np.linspace(0, L, 3**N)
#cost = np.array([costf(x) for x in x_vals])
cost = np.array([1, 3, 2, 1])
# Sigma function for tensor products
def sigma(k, t=Z):
    ret_mat = np.identity(3) if k != 0 else t
    for x in range(1, N):
        if k == x:
            ret_mat = np.kron(ret_mat, t)
        else:
            ret_mat = np.kron(ret_mat, np.identity(3))
    return ret_mat

# Hamiltonians
def H_B():
    return sum(costf(i)*sigma(i) for i in range(N))

def H_P():
    penalty_matrix = sum(sigma(i) for i in range(N))
    return np.matmul(penalty_matrix, penalty_matrix)

def H_D():
    return sum(sigma(i, t=X) for i in range(N))



# Parameter sweep
alpha_vals = np.linspace(0, 5, 50)
beta = 0.1
eigenvalues = []

for alpha in alpha_vals:
    H = H_B() + alpha * H_P()
    eigvals = np.sort(np.linalg.eigvalsh(H))
    eigenvalues.append(eigvals)

eigenvalues = np.array(eigenvalues).T

for eig in eigenvalues:
    plt.plot(alpha_vals, eig)
plt.xlabel("Alpha")
plt.ylabel("H")
plt.grid()
plt.show()