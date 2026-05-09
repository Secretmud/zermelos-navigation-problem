"""
energy_gap.py
-------------
Computes the minimal spectral gap (Delta E_min = E_1 - E_0) as a function
of system size n for the Landau-Zener Hamiltonian H(t) = H_B + beta(t)*H_P + alpha(t)*H_D.

Run from the project root:
  python scripts/energy_gap.py
"""

from pathlib import Path

import joblib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from adiabatic_computation import H_B, H_P, H_D, X, alpha as a, beta as b

steps  = 100
memory = joblib.Memory(location=".joblib_cache", verbose=0)


@memory.cache
def avoided(n, Ts, T_x, T_pen):
    HB = H_B(n); HP = H_P(n); HX = H_D(n, X)
    beta_max  = 1 / n
    alpha_max = 0.05 / n
    eigvals = []
    for t in Ts:
        H = HB + b(t, T_x, T_pen, beta_max) * HP + a(t, T_x, T_pen, alpha_max) * HX
        vals, _ = np.linalg.eigh(H)
        eigvals.append(vals)
    return np.array(eigvals)


def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    n_values     = range(2, 7)
    T_pen_values = [100, 400, 800, 1600, 3200, 6400, 10000]

    results = {T_pen: {} for T_pen in T_pen_values}

    for n in tqdm.tqdm(n_values, desc="n"):
        for T_pen in T_pen_values:
            T_x = T_pen / 5
            T_f = 2 * T_x + T_pen
            Ts  = np.linspace(0.1, T_f, steps)

            eigvals = avoided(n, Ts, T_x, T_pen)
            gap     = eigvals[:, 1] - eigvals[:, 0]
            results[T_pen][n] = {"min_gap": float(gap.min())}

    ns   = sorted(n_values)
    gaps = [results[T_pen_values[0]][n]["min_gap"] for n in ns]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(ns, gaps, marker="o", color="steelblue", linewidth=2)
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$\Delta E_{\min}$")
    ax.set_title("Minimal energy gap vs $n$")
    ax.set_xticks(ns)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / "minimal_energy_gap.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main(Path("plots/energy_gap"))
