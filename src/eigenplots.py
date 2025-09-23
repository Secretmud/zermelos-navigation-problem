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

from lib.hamiltonian import H_B, H_P, H_D
from lib.time import S
from lib import X, D, N


memory = joblib.Memory(location=".joblib_cache", verbose=0)


@memory.cache
def all_move_sequences(N):
    moves = ("up", "straight", "down")
    return list(itertools.product(moves, repeat=N))


def normalize(vec):
    return vec / np.sum(vec) if np.sum(vec) != 0 else vec


@memory.cache
def Eigenvalues(N, beta, alpha, driver=False, vec_normalize=False, bench_hamiltonian=False):
    eigvals = []
    eigvecs = []
    print(f"Calculating Eigenvalues for N={N}, beta={beta}, driver={driver}")
    for a in alpha:
        print(f"Calculating for alpha={a:2f}", end="\r")
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

    gaps = np.min(np.diff(eigvals[-1], axis=0), axis=0)

    return np.array(eigvals).T, np.array(eigvecs), gaps


beta = 0.1
driver = True
alpha_values = np.linspace(0, 12, 2000)
ns = np.arange(2, N)
gap = []
for N in ns:
    eigvals_avoided_crossing, eigvecs, gaps = Eigenvalues(N=N, beta=beta, alpha=alpha_values, driver=True, vec_normalize=True)
    eigvals_crossing, _, _ = Eigenvalues(N=N, beta=beta, alpha=alpha_values, driver=False, vec_normalize=True)
    eig = np.sort(eigvals_avoided_crossing[-1])
    lowest_eigenvalues = eig[:2]
    gap.append(np.min(np.diff(lowest_eigenvalues)))  # Minimum gap at the last alpha value

print(f"Gaps for N={ns}: {gap}")
plt.figure(figsize=(8, 6))
plt.plot(ns, gap, marker='o')
plt.xlabel("N")
plt.ylabel("Minimum Energy Gap")
plt.yscale("log")
plt.title(r"Minimum Energy Gap vs N for $H = H_B + \alpha*(H_P)^2 + \beta*H_D$")
plt.grid()
plt.tight_layout()
plt.show()

eigvals_avoided_crossing, eigvecs, _ = Eigenvalues(N=N, beta=beta, alpha=alpha_values, driver=True, vec_normalize=True)
eigvals_crossing, _, _ = Eigenvalues(N=N, beta=beta, alpha=alpha_values, driver=False, vec_normalize=True)
# Plot eigenvalues

plt.figure(figsize=(8, 6))
for eig_avoided_crossing, eig_crossing in zip(eigvals_avoided_crossing, eigvals_crossing):
    plt.plot(alpha_values, eig_avoided_crossing, color="black", alpha=0.5)
    plt.plot(alpha_values, eig_crossing, color="red",
             alpha=0.5, linestyle="dashed")
plt.xlabel(r"$ \alpha $")
plt.ylabel("Eigenenergies")
plt.grid()
title = "Avoided crossings (black) vs Crossings (red dashed)"
plt.title(title)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2)
indices = np.arange(eigvecs.shape[1])
ids = {}

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
print(f"Top moving sequences for N={N}:")
for k, v in ids.items():
    print(f"For subplot with alpha {k}:")
    for data in v:
        print(f"\tEigenvector: {data['eigvec']}")
        print(f"\tIndex {data['path']}: {sequences[data['path']]}")

plt.tight_layout()
plt.show()