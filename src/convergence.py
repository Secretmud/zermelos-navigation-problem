from lib.hamiltonian import H_B, H_D, H_P
from lib.schrodinger import yves_TDSE
from lib import X, yvesData

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Fixed parameters
T_0 = 15.0
T_f = 1500.0
beta = -1
F_thr = 0.9

OUT = Path("figures/convergence")
OUT.mkdir(parents=True, exist_ok=True)

def compute_dense_fidelity(n: int, pen: int, nsteps: int):
    """Return ts, F(ts) for a dense run."""
    Hf = H_B(n) + pen * H_P(n)
    Hi = beta * H_D(n, X)
    ts = np.linspace(T_0, T_f, nsteps)
    gs_idx = np.argmin(np.diagonal(Hf))
    fs = []
    for t in tqdm(ts):
        args = yvesData(Hf=Hf, Hi=Hi, n=n, t=t)
        F = yves_TDSE(args, steps=nsteps)  # should return array-like, same length as ts
        fs.append(np.abs(F[gs_idx])**2)

    F = np.asarray(fs, dtype=float)
    if F.shape[0] != ts.shape[0]:
        raise ValueError(f"Expected F to have length {len(ts)}, got {F.shape}")
    if np.any(~np.isfinite(F)):
        raise ValueError("Non-finite values encountered in fidelity curve (NaN/Inf).")

    return ts, F

def first_crossing_time(ts, F, thr):
    """
    First time where F(t) crosses thr (linear interpolation).
    Returns np.nan if never reaches.
    """
    idx = np.where(F >= thr)[0]
    if len(idx) == 0:
        return np.nan
    k = int(idx[0])
    if k == 0:
        return float(ts[0])
    # interpolate between k-1 and k
    t0, t1 = ts[k-1], ts[k]
    f0, f1 = F[k-1], F[k]
    if f1 == f0:
        return float(t1)
    return float(t0 + (thr - f0) * (t1 - t0) / (f1 - f0))

def curve_error_against_reference(ts_ref, F_ref, ts, F):
    """
    L-infinity error against a reference curve, comparing on the reference grid.
    """
    F_on_ref = np.interp(ts_ref, ts, F)
    return float(np.max(np.abs(F_on_ref - F_ref)))

def convergence_study(cases, nsteps_list):
    """
    cases: list of (n, pen)
    """
    plt.rcParams.update({
        "figure.figsize": (6.2, 3.8),
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots()
    ax.set_xlabel(r"Timestep $\Delta t$")
    ax.set_ylabel(r"Threshold time $t_{\mathrm{thr}}$ for $F_{\mathrm{thr}}=0.9$")
    ax.set_xscale("log")

    # Optional second figure for curve error
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel(r"Timestep $\Delta t$")
    ax2.set_ylabel(r"$\|F_{\Delta t}-F_{\mathrm{ref}}\|_\infty$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    for (n, pen) in cases:
        # Use finest run as reference
        nsteps_ref = max(nsteps_list)
        ts_ref, F_ref = compute_dense_fidelity(n, pen, nsteps_ref)

        dts = []
        t_thrs = []
        errs = []

        for Ns in nsteps_list:
            ts, F = compute_dense_fidelity(n, pen, Ns)
            dt = (T_f - T_0) / (Ns - 1)
            tthr = first_crossing_time(ts, F, F_thr)
            err = curve_error_against_reference(ts_ref, F_ref, ts, F)

            dts.append(dt)
            t_thrs.append(tthr)
            errs.append(err)

        label = rf"$n={n},\,p={pen}$"
        ax.plot(dts, t_thrs, marker="o", label=label)
        ax2.plot(dts, errs, marker="o", label=label)

    ax.legend(loc="best", frameon=True)
    fig.savefig(OUT / "convergence_tthr_vs_dt.pdf", bbox_inches="tight")
    plt.close(fig)

    ax2.legend(loc="best", frameon=True)
    fig2.savefig(OUT / "convergence_curve_error_vs_dt.pdf", bbox_inches="tight")
    plt.close(fig2)


# Choose convergence test cases (tweak as needed)
cases = [
    #(3, 30),  # your “suspect” case
    (2, 30),  # fast/easy case for contrast
]

nsteps_list = [1000, 3000, 5000, 10000]
convergence_study(cases, nsteps_list)
print("Saved convergence figures to figures/convergence/")
