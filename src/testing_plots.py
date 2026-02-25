from lib.hamiltonian import H_B, H_D, H_P
from lib.schrodinger import yves_TDSE
from lib.solvers import yield_bisection
from lib import X, yvesData

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# -----------------------------
# Experiment parameters
# -----------------------------
nsteps = 2000
T_0 = 15
T_f = 1500
beta = -1
F_thr = 0.9

runtime = {
    1:  [2, 3],
    5:  [2, 3],
    10: [2, 3],
    15: [2, 3],
    20: [2, 3],
    30: [2, 3],
}

# Pick representative penalties for the “mechanism” plots
REP_PENS = (1, 10, 30)

# -----------------------------
# Output folders
# -----------------------------
OUT = Path("figures")
OUT_MAIN = OUT / "main"
OUT_TRACES = OUT / "traces"
OUT_DIAG = OUT / "diagnostics"
OUT_DATA = OUT / "data"

for d in [OUT_MAIN, OUT_TRACES, OUT_DIAG, OUT_DATA]:
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Thesis-friendly Matplotlib defaults
# -----------------------------
plt.rcParams.update({
    "figure.figsize": (6.2, 3.7),
    "figure.dpi": 150,
    "savefig.dpi": 300,

    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,

    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,

    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# -----------------------------
# Core computation
# -----------------------------
def solve_points(n: int, pen: int):
    """
    Returns:
        ts: full time grid used by TDSE (for metadata)
        points: sorted list of (time, fidelity) from bisection evaluations
        sol_t, sol_f: final solution point (last yielded)
    """
    Hf = H_B(n) + pen * H_P(n)
    Hi = beta * H_D(n, X)
    ts = np.linspace(T_0, T_f, nsteps)
    args = yvesData(Hf=Hf, Hi=Hi, n=n, ts=ts)

    points = []
    sol_t, sol_f = None, None
    for time, fid in yield_bisection(yves_TDSE, args, f_thr=F_thr):
        points.append((float(time), float(fid)))
        sol_t, sol_f = float(time), float(fid)

    if not points:
        raise ValueError("No points returned from yield_bisection().")

    points.sort(key=lambda x: x[0])
    return ts, points, sol_t, sol_f

# -----------------------------
# Plot helpers (reader-first)
# -----------------------------
def annotate_solution(ax, sol_t, sol_f):
    ax.axhline(F_thr, linestyle="--", linewidth=1.2)
    ax.axvline(sol_t, linestyle="--", linewidth=1.2)
    ax.plot(sol_t, sol_f, marker="o", linestyle="None", zorder=5)
    ax.text(
        0.02, 0.06,
        rf"$T_f={sol_t:.2f}$, $F(T_f)={sol_f:.3f}$",
        transform=ax.transAxes,
        va="bottom", ha="left"
    )

def plot_diagnostic(points, sol_t, sol_f, n, pen, path):
    """Appendix: show evaluation points + dashed connect (solver-facing)."""
    p = np.array(points)
    x, y = p[:, 0], p[:, 1]

    fig, ax = plt.subplots()
    ax.set_xlim(T_0, T_f)
    ax.set_ylim(0.0, 1.0)

    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Fidelity $F(t)$")
    ax.set_title(rf"Diagnostics: $n={n}$, penalty $p={pen}$")

    ax.axhline(F_thr, linestyle="--", linewidth=1.2, label=rf"$F_\text{{thr}}={F_thr}$")

    ax.plot(x, y, linestyle="--", linewidth=1.1, label="Bisection samples")
    ax.plot(x, y, linestyle="None", marker="o", label="Evaluations")

    ax.axvline(sol_t, linestyle="--", linewidth=1.2, label=rf"$T_f={sol_t:.2f}$")
    ax.plot(sol_t, sol_f, marker="o", linestyle="None", zorder=5, label="Final point")

    ax.legend(loc="lower right", frameon=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_reader_trace(points, sol_t, sol_f, n, pen, path):
    """
    Main-text trace: emphasize the “story” not the solver.
    """
    p = np.array(points)
    x, y = p[:, 0], p[:, 1]

    fig, ax = plt.subplots()
    ax.set_xlim(T_0, T_f)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Fidelity $F(t)$")
    ax.set_title(rf"$n={n}$, penalty $p={pen}$")

    ax.plot(x, y, linestyle="-", linewidth=1.8)
    ax.plot(x, y, linestyle="None", marker="o", alpha=0.35)

    annotate_solution(ax, sol_t, sol_f)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Step 1: run all, save CSV, save diagnostics
# -----------------------------
rows = []
for pen, ns in runtime.items():
    print(f"Penalty {pen}")
    for n in ns:
        print(f"  n={n}")
        try:
            ts, points, sol_t, sol_f = solve_points(n=n, pen=pen)

            diag_name = f"diag_n{n}_p{pen}_F{F_thr}_N{nsteps}_T{T_0}-{T_f}.pdf"
            plot_diagnostic(points, sol_t, sol_f, n, pen, OUT_DIAG / diag_name)

            rows.append({
                "n": n,
                "penalty": pen,
                "F_thr": F_thr,
                "nsteps": nsteps,
                "T_0": T_0,
                "T_f": T_f,
                "t_star": sol_t,
                "F_at_t_star": sol_f,
                "n_evals": len(points),
            })

            if pen in REP_PENS:
                trace_name = f"trace_n{n}_p{pen}_F{F_thr}_T{T_0}-{T_f}.pdf"
                plot_reader_trace(points, sol_t, sol_f, n, pen, OUT_TRACES / trace_name)

        except Exception as e:
            # More informative during development; feel free to revert to ValueError later
            print(f"    FAILED for (n={n}, penalty={pen}): {type(e).__name__}: {e}")

cols = [
    "n", "penalty", "F_thr", "nsteps", "T_0", "T_f",
    "t_star", "F_at_t_star", "n_evals"
]
df = pd.DataFrame(rows, columns=cols)

if df.empty:
    raise RuntimeError(
        "No results were produced (rows is empty). "
        "Scroll up for FAILED messages to see which (n, penalty) cases failed."
    )

df = df.sort_values(["n", "penalty"])
csv_path = OUT_DATA / f"results_F{F_thr}_N{nsteps}_T{T_0}-{T_f}.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved results to: {csv_path}")

# -----------------------------
# Step 2: Main thesis figure — t*(p) for each n
# -----------------------------
def make_main_summary_figure(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.set_xlabel("Penalty $p$")
    ax.set_ylabel(r"Threshold time $T_f$")
    ax.set_title(rf"Time to reach $F_\text{{thr}}={F_thr}$")

    for n, g in df.groupby("n"):
        g = g.sort_values("penalty")
        ax.plot(g["penalty"], g["t_star"], marker="o", label=rf"$n={n}$")

    ax.legend(loc="lower right", frameon=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

main_fig_path = OUT_MAIN / f"tstar_vs_penalty_F{F_thr}_T{T_0}-{T_f}.pdf"
make_main_summary_figure(df, main_fig_path)
print(f"Saved main figure to: {main_fig_path}")

# -----------------------------
# Step 3: Representative multi-panel figure per n (1×3)
# -----------------------------
def make_rep_panel_for_n(n: int, pens, path: Path):
    fig, axes = plt.subplots(1, len(pens), figsize=(6.6, 2.5), sharex=True, sharey=True)
    if len(pens) == 1:
        axes = [axes]

    for ax, pen in zip(axes, pens):
        row = df[(df["n"] == n) & (df["penalty"] == pen)]
        if row.empty:
            ax.set_axis_off()
            continue

        t_star = float(row["t_star"].iloc[0])
        f_star = float(row["F_at_t_star"].iloc[0])

        # Reload points (keeps CSV light)
        _, points, _, _ = solve_points(n=n, pen=pen)
        p_arr = np.array(points)
        x, y = p_arr[:, 0], p_arr[:, 1]

        ax.plot(x, y, linewidth=1.6)
        ax.axhline(F_thr, linestyle="--", linewidth=1.1)
        ax.axvline(t_star, linestyle="--", linewidth=1.1)
        ax.plot(t_star, f_star, marker="o", linestyle="None", zorder=5)

        ax.set_title(rf"$p={pen}$")
        ax.text(0.04, 0.06, rf"$T_f={t_star:.2f}$", transform=ax.transAxes)

    axes[0].set_ylabel("Fidelity $F(t)$")
    for ax in axes:
        ax.set_xlabel("Time $t$")
        ax.set_xlim(T_0, T_f)
        ax.set_ylim(0.0, 1.0)

    fig.suptitle(rf"Representative fidelity traces for $n={n}$", y=1.02)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

for n in sorted(df["n"].unique()):
    rep_path = OUT_MAIN / f"rep_traces_n{n}_pens{'-'.join(map(str, REP_PENS))}_F{F_thr}.pdf"
    make_rep_panel_for_n(n, REP_PENS, rep_path)
    print(f"Saved representative panel for n={n} to: {rep_path}")
