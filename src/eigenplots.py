"""
Eigenplots.py

This file lets you run the eigenplots for crossings, 
avoided crossings and also the basis states.
"""
import numpy as np
import scipy as sp
import matplotlib
import itertools
import joblib

import matplotlib.pyplot as plt

from lib.hamiltonian import H_B, H_B2,  H_P, H_D
from lib import N, X, D


joblib_memory = joblib.Memory(location=".joblib_cache", verbose=0)


@joblib_memory.cache
def all_move_sequences(N):
    moves = (1, 0, -1)
    return list(itertools.product(moves, repeat=N))


def normalize(vec):
    return vec / np.sum(vec) if np.sum(vec) != 0 else vec


@joblib_memory.cache
def Eigenvalues(N, beta, alpha, driver=False, vec_normalize=False, bench_hamiltonian=False):
    eigvals = []
    eigvecs = []
    for a in alpha:
        if driver:
            if bench_hamiltonian:
                H = H_B2(N) + a*H_P(N) + beta*H_D(N, X)
            else:
                H = H_B(N) + a*H_P(N) + beta*H_D(N, X)
        else:
            if bench_hamiltonian:
                H = H_B2(N) + a*H_P(N)
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


beta = 0.2
driver = True
alpha_values = np.linspace(0, 20, 1000)
eigvals, eigvecs = Eigenvalues(
    N=N, beta=beta, alpha=alpha_values, driver=True, vec_normalize=True)
eigvals_bench, eigvecs_bench = Eigenvalues(
    N=N, beta=beta, alpha=alpha_values, driver=False, vec_normalize=True)
# Plot eigenvalues
plt.figure(figsize=(8, 6))
for eig_avoided_crossing, eig_crossing in zip(eigvals, eigvals_bench):
    plt.plot(alpha_values, eig_avoided_crossing, color="black", alpha=0.5)
    plt.plot(alpha_values, eig_crossing, color="red",
             alpha=0.5, linestyle="dashed")
plt.xlabel(r"$ \alpha $")
plt.ylabel("Eigenenergies")
plt.grid()
title = "Avoided crossings (black) vs Crossings (red dashed)"
# title = r"$H = H_B + \alpha*(H_P)^2 + \beta*H_D$" if driver == True else r"$H = H_B + \alpha*(H_P)^2$"
plt.title(title)
plt.tight_layout()
plt.show()

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

plt.tight_layout()
plt.show()

sequences = all_move_sequences(N)
print(f"All possible move sequences for N={N}:")
plt.figure(figsize=(10, 6))
x = np.linspace(0, D, N+1)
for k, v in ids.items():
    print(f"For subplot with alpha {k}:")
    for data in v:
        path = np.cumsum(sequences[data['path']])
        path = np.insert(path, 0, 0, axis=0)
        print(f"  Index {data['path']}: {sequences[data['path']]}")
        plt.plot(x, path, label=f"α={k}, id={data['id']}")

plt.xlabel("Step")
plt.ylabel("Position")
plt.title("Significant Paths for Selected α Values")
plt.legend()
plt.grid()
plt.show()


"""

beta = 0.1

print("H_B")
eig, _ = sp.linalg.eigh(H_B(N))
print(eig)
print("H_B2")
eig, _ = sp.linalg.eigh(H_B2(N))
print(eig)
"""
