"""
Eigenplots.py

This file lets you run the eigenplots for crossings, 
avoided crossings and also the basis states.
"""
import numpy as np
import matplotlib
import itertools
import joblib

import matplotlib.pyplot as plt

from lib.hamiltonian import H_B, H_P, H_D
from lib import N, X


joblib_memory = joblib.Memory(location=".joblib_cache", verbose=0)

@joblib_memory.cache
def all_move_sequences(N):
    """
    Returns all possible sequences of 'up', 'straight', 'down' for N steps.
    Each sequence is a tuple of length N.
    """
    moves = ("up", "straight", "down")
    return list(itertools.product(moves, repeat=N))

def normalize(vec):
    return vec / np.sum(vec) if np.sum(vec) != 0 else vec

@joblib_memory.cache
def Eigenvalues(N, beta, alpha, driver = False, vec_normalize = False):
    eigvals = []
    eigvecs = []
    for a in alpha:
        print(f"Calculating for alpha={a}")
        if driver:
            H = H_B(N) + a*H_P(N) + beta*H_D(N, X)
        else:
            H = H_B(N) + a*H_P(N)
        vals, vecs = np.linalg.eigh(H)
        eigvals.append(vals)
        if vec_normalize:
            eigvecs.append(normalize(np.abs(vecs[:, 0])**2))
        else:
            eigvecs.append(vecs)
        del H
    
    return np.array(eigvals).T, np.array(eigvecs)

beta = 0.1
driver = True
alpha_values = np.linspace(0, 3, 200)
eigvals, eigvecs = Eigenvalues(N=N, beta=beta, alpha=alpha_values, driver=True, vec_normalize=True)
# Plot eigenvalues
"""
plt.figure(figsize=(8, 6))
for eig in eigvals:
    plt.plot(alpha_values, eig)
plt.xlabel(r"$ \alpha $")
plt.ylabel("Eigenenergies")
plt.grid()
title = r"$H = H_B + \alpha*(H_P)^2 + \beta*H_D$" if driver == True else r"$H = H_B + \alpha*(H_P)^2$"
plt.title(title)
plt.tight_layout()
plt.show()

"""
fig, axes = plt.subplots(2, 2)
indices = np.arange(eigvecs.shape[1])
ids = {}

# Let us pick 4 alpha indices evenly spaced from alpha_values array
alpha_indices = np.linspace(0, len(alpha_values)-1, 4, dtype=int)
selected_alpha_values = [alpha_values[i] for i in alpha_indices]
selected_eigvecs = [eigvecs[i] for i in alpha_indices]
for i, (alpha_val, eigvec) in enumerate(zip(selected_alpha_values, selected_eigvecs)):
    ax = axes.flat[i]
    ax.bar(indices, eigvec, linewidth=1)
    ids[alpha_val] = []
    ax_ids = []
    for j in range(len(eigvec)):
        if eigvec[j] > 10e-3:
            data = {"eigvec": eigvec[j], "path": j, "id": i}
            ids[alpha_val].append(data)
            ax_ids.append(j)
    ax.set_title(rf"$\alpha = {alpha_val}$")
    ax.set_xlabel("Path Index")
    ax.set_ylabel("Population")
    ax.set_xticks(ax_ids)
    ax.set_xticklabels(ax_ids, rotation=90)

sequences = all_move_sequences(N)
print(f"All possible move sequences for N={N}:")
for k, v in ids.items():
    print(f"For subplot with alpha {k}:")
    for data in v:
        print(f"  Eigenvector: {data['eigvec']}")
        print(f"  Index {data['path']}: {sequences[data['path']]}")
plt.tight_layout()
plt.show()


