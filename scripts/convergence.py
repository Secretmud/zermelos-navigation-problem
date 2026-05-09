"""
convergence.py
--------------
Convergence study for the Yves TDSE integrator: sweeps over time-step
sizes (nsteps) and measures how the fidelity curve and threshold time
converge as dt -> 0. The finest run is used as the reference solution.

Produces:
  <out_dir>/convergence_tthr_vs_dt.pdf
  <out_dir>/convergence_curve_error_vs_dt.pdf

Run from the project root:
  python scripts/convergence.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from adiabatic_computation import H_B, H_P, H_D, X, YvesConfig, yves_TDSE

# ── Parameters ────────────────────────────────────────────────────────────────
T_0   = 15.0
T_f   = 1500.0
beta  = -1
F_thr = 0.9

plt.rcParams.update({
    "figure.figsize":    (6.2, 3.8),
    "savefig.dpi":       300,
    "font.size":         10,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})


def compute_dense_fidelity(n: int, pen: int, nsteps: int):
    H_f    = H_B(n) + pen * H_P(n)
    H_i    = beta * H_D(n, X)
    ts     = np.linspace(T_0, T_f, 500)
    gs_idx = np.argmin(np.diagonal(H_f))
    fs     = []
    for t in tqdm(ts, leave=False, desc=f"n={n} pen={pen} nsteps={nsteps}"):
        config = YvesConfig(H_f=H_f, H_i=H_i, n=n, t=t)
        psi    = yves_TDSE(config, steps=nsteps)
        fs.append(float(np.abs(psi[gs_idx]) ** 2))
    F = np.asarray(fs, dtype=float)
    if not np.all(np.isfinite(F)):
        raise ValueError("Non-finite values in fidelity curve.")
    return ts, F


def first_crossing_time(ts, F, thr):
    idx = np.where(F >= thr)[0]
    if len(idx) == 0:
        return np.nan
    k = int(idx[0])
    if k == 0:
        return float(ts[0])
    t0, t1 = ts[k - 1], ts[k]
    f0, f1 = F[k - 1], F[k]
    return float(t0 + (thr - f0) * (t1 - t0) / (f1 - f0)) if f1 != f0 else float(t1)


def curve_error(ts_ref, F_ref, ts, F):
    return float(np.max(np.abs(np.interp(ts_ref, ts, F) - F_ref)))


def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    cases       = [(5, 30), (3, 30), (2, 30)]
    nsteps_list = [1000, 3000, 5000, 10000, 15000, 25000, 35000]

    fig,  ax  = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax.set_xlabel(r"Timestep $\Delta t$")
    ax.set_ylabel(r"Threshold time $t_{\mathrm{thr}}$ for $F_{\mathrm{thr}}=0.9$")
    ax.set_xscale("log")
    ax2.set_xlabel(r"Timestep $\Delta t$")
    ax2.set_ylabel(r"$\|F_{\Delta t}-F_{\mathrm{ref}}\|_\infty$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    for n, pen in cases:
        ts_ref, F_ref = compute_dense_fidelity(n, pen, max(nsteps_list))
        dts, t_thrs, errs = [], [], []
        for nsteps in nsteps_list:
            ts, F  = compute_dense_fidelity(n, pen, nsteps)
            dt     = (T_f - T_0) / (nsteps - 1)
            t_thr  = first_crossing_time(ts, F, F_thr)
            err    = curve_error(ts_ref, F_ref, ts, F)
            dts.append(dt); t_thrs.append(t_thr); errs.append(err)
            print(f"  n={n} pen={pen} nsteps={nsteps} t_thr={t_thr:.2f}")

        label = rf"$n={n},\,p={pen}$"
        ax.plot(dts, t_thrs, marker="o", label=label)
        ax2.plot(dts, errs,  marker="o", label=label)

    ax.legend(loc="best", frameon=True)
    p1 = out_dir / "convergence_tthr_vs_dt.pdf"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p1}")

    ax2.legend(loc="best", frameon=True)
    p2 = out_dir / "convergence_curve_error_vs_dt.pdf"
    fig2.savefig(p2, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {p2}")


if __name__ == "__main__":
    main(Path("plots/convergence"))
