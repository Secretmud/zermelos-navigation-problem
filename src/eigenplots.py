"""
Eigenplots.py

This file lets you run the eigenplots for crossings, 
avoided crossings and also the basis states.
"""
import numpy as np
import matplotlib
import itertools

import matplotlib.pyplot as plt

from lib.hamiltonian import H_B, H_P, H_D
from lib import N, X

alpha = np.linspace(0, 20, 1500)
beta = 0.1

eigvals = []
eigvecs = []
for a in alpha:
    H = H_B(N) + a*H_P(N)
    vals, vecs = np.linalg.eig(H)
    eigvals.append(vals)
    eigvecs.append(vecs)
    del H

eigvals = np.array(eigvals).T
eigvecs = np.array(eigvecs)

# Plot eigenvalues
plt.figure(figsize=(8, 6))
for eig in eigvals:
    plt.plot(alpha, eig)
plt.xlabel(r"$ \alpha $")
plt.ylabel("Eigenenergies")
plt.grid()
plt.title(r"$H = H_B + \alpha*(H_P)^2$")
plt.show()

del eigvals, eigvecs


eigvals = []
eigvecs = []
scale = 1
for a in alpha:
    H = scale*H_B(N) + a*H_P(N) + beta*H_D(N, X)
    vals, vecs = np.linalg.eigh(H)
    eigvals.append(vals)
    eigvecs.append(vecs)
    del H

eigvals = np.array(eigvals).T

eigvecs = np.array(eigvecs)

# Plot eigenvalues
plt.figure(figsize=(8, 6))
for eig in eigvals:
    plt.plot(alpha, eig)
plt.xlabel(r"$ \alpha $")
plt.ylabel("Eigenenergies")
plt.grid()
plt.title(r" $H = H_B + \alpha*(H_P)^2 + \beta*H_D$")
plt.show()

del eigvals, eigvecs


alpha_values = [5, 7.5, 10, 12.5, 15, 20]
beta = 0.1

eigvecs = []


def normalize(vec):
    return vec / np.sum(vec) if np.sum(vec) != 0 else vec


for a in alpha_values:
    H = scale*H_B(N) + a*H_P(N) + beta*H_D(N, X)
    _, vecs = np.linalg.eigh(H)
    eigvecs.append(normalize(np.abs(vecs[:, 0])**2))
    del H

eigvecs = np.array(eigvecs)

# Plot eigenstates as bar charts in a 2x3 layout
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
indices = np.arange(eigvecs.shape[1])
ids = []
for i, ax in enumerate(axes.flat):
    ax.bar(indices, eigvecs[i], alpha=0.7)
    for j in range(len(eigvecs[i])):
        if eigvecs[i][j] > 10e-6:
            ids.append(j)
    ax.set_title(rf"$\alpha = {alpha_values[i]}$")
    ax.set_xlabel("Path Index")
    ax.set_ylabel("Population")

plt.tight_layout()
plt.show()

def all_move_sequences(N):
    """
    Returns all possible sequences of 'up', 'straight', 'down' for N steps.
    Each sequence is a tuple of length N.
    """
    moves = ("up", "straight", "down")
    return list(itertools.product(moves, repeat=N))

sequences = all_move_sequences(N)
print(f"All possible move sequences for N={N}:")
for idx in ids:
    print(f"Index {idx}: {sequences[idx]}")