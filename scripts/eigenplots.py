"""
eigenplots.py
-------------
Produces three figures for a given system size n:
  1. Avoided crossings (with H_D) vs bare crossings (without H_D)
     as a function of the penalty coupling alpha.
  2. Ground-state population bar charts at four selected alpha values.
  3. Paths in the river grid corresponding to the dominant ground-state
     components at the final alpha value.

Saves to <out_dir>/:
  crossing.pdf, populations.pdf, paths.pdf

Run from the project root:
  python scripts/eigenplots.py
"""

import itertools
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from adiabatic_computation import H_B, H_P, H_D, X, D, time

memory = joblib.Memory(location=".joblib_cache", verbose=0)


@memory.cache
def all_move_sequences(N):
    moves = (1, 0, -1)
    return list(itertools.product(moves, repeat=N))


def normalize(vec):
    return vec / np.sum(vec) if np.sum(vec) != 0 else vec


@memory.cache
def compute_eigenvalues(N, beta, alpha_values, driver=False, vec_normalize=False):
    eigvals, eigvecs = [], []
    print(f"Computing eigenvalues for N={N}, beta={beta}, driver={driver}")
    for a in alpha_values:
        if driver:
            H = H_B(N) + a * H_P(N) + beta * H_D(N, X)
            vals, vecs = np.linalg.eigh(H)
        else:
            H = H_B(N) + a * H_P(N)
            vals, vecs = np.linalg.eig(H)
        eigvals.append(vals)
        eigvecs.append(normalize(np.abs(vecs[:, 0]) ** 2) if vec_normalize else vecs)
    return np.array(eigvals).T, np.array(eigvecs)


def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    beta         = 0.1
    alpha_values = np.linspace(0, 1, 3000)
    N            = 4

    eigvals_avoided, eigvecs = compute_eigenvalues(
        N=N, beta=beta, alpha_values=alpha_values, driver=True, vec_normalize=True)
    eigvals_crossing, _ = compute_eigenvalues(
        N=N, beta=beta, alpha_values=alpha_values, driver=False)

    # Figure 1: crossing vs avoided crossing
    fig, ax = plt.subplots(figsize=(8, 6))
    for ev_av, ev_cr in zip(eigvals_avoided, eigvals_crossing):
        ax.plot(alpha_values, ev_av, color="black", alpha=0.5)
        ax.plot(alpha_values, ev_cr, color="red", alpha=0.5, linestyle="dashed")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Eigenenergies")
    ax.set_title("Avoided crossings (black) vs crossings (red dashed)")
    ax.grid()
    fig.tight_layout()
    p1 = out_dir / "crossing.pdf"
    fig.savefig(p1)
    plt.close(fig)
    print(f"Saved {p1}")

    # Figure 2: ground-state population at four alpha values
    fig, axes = plt.subplots(2, 2)
    alpha_indices    = np.linspace(0, len(alpha_values) - 1, 4, dtype=int)
    selected_alpha   = [alpha_values[i] for i in alpha_indices]
    selected_eigvecs = [eigvecs[i] for i in alpha_indices]
    indices          = np.arange(eigvecs.shape[1])
    ids              = {}

    for i, (alpha_val, eigvec) in enumerate(zip(selected_alpha, selected_eigvecs)):
        ax = axes.flat[i]
        ax.bar(indices, eigvec, linewidth=1)
        if i == len(selected_alpha) - 1:
            ids[alpha_val] = [
                {"eigvec": eigvec[j], "path": j}
                for j in range(len(eigvec)) if eigvec[j] > 1e-2
            ]
        ax_ids = [j for j in range(len(eigvec)) if eigvec[j] > 1e-2]
        ax.set_title(rf"$\alpha = {alpha_val:.3f}$")
        ax.set_xlabel("Path Index")
        ax.set_ylabel("Population")
        ax.set_xticks(ax_ids)
        ax.set_xticklabels(ax_ids, rotation=90)

    fig.tight_layout()
    p2 = out_dir / "populations.pdf"
    fig.savefig(p2)
    plt.close(fig)
    print(f"Saved {p2}")

    # Figure 3: dominant paths in the river grid
    sequences = all_move_sequences(N)
    x         = np.linspace(0, D, N + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for alpha_val, entries in ids.items():
        for data in entries:
            t_total = sum(time(i, sequences[data["path"]][i], N) for i in range(N))
            path    = np.insert(np.cumsum(sequences[data["path"]]), 0, 0)
            print(f"  alpha={alpha_val:.3f}, index={data['path']}, "
                  f"seq={sequences[data['path']]}, time={t_total:.4f}")
            ax.plot(x, path, label=f"α={alpha_val:.2f}, idx={data['path']}")

    ax.set_xlabel("Step")
    ax.set_ylabel("Position")
    ax.set_title(r"Dominant paths for selected $\alpha$ values")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    p3 = out_dir / "paths.pdf"
    fig.savefig(p3)
    plt.close(fig)
    print(f"Saved {p3}")


if __name__ == "__main__":
    main(Path("plots/eigenplots"))
