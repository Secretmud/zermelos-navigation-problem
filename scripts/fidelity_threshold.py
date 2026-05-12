"""
fidelity_threshold.py
---------------------
Runs the bisection search for each (n, penalty) combination using the
conventional (Yves) annealing scheme and produces:

  <out_dir>/main/     — summary t*(p) plot + per-n representative panels
  <out_dir>/traces/   — one trace PDF per (n, representative-penalty)
  <out_dir>/diagnostics/ — one diagnostic PDF per (n, penalty) showing all bisection evals
  <out_dir>/data/     — CSV with one row per (n, penalty): t*, F(t*), n_evals

Run from the project root:
  python scripts/fidelity_threshold.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from adiabatic_computation import (
    H_B,
    H_P,
    H_D,
    X,
    YvesConfig,
    yves_TDSE,
    yield_bisection,
)

# ── Parameters ────────────────────────────────────────────────────────────────
nsteps = 2000
T_0 = 15
T_f = 1500
beta = -1
F_thr = 0.9

runtime = {
    1: [2, 3, 4, 5],
    5: [2, 3, 4, 5],
    10: [2, 3, 4, 5],
    15: [2, 3, 4, 5],
    20: [2, 3, 4, 5],
    30: [2, 3, 4, 5],
}

REP_PENS = (1, 10, 30)

plt.rcParams.update(
    {
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
    }
)


def solve_points(n: int, pen: int):
    H_f = H_B(n) + pen * H_P(n)
    H_i = beta * H_D(n, X)
    t_range = np.linspace(T_0, T_f, nsteps)
    config = YvesConfig(H_f=H_f, H_i=H_i, n=n, t_range=t_range)

    points = []
    sol_t, sol_f = None, None
    for t, fid in yield_bisection(yves_TDSE, config, f_thr=F_thr):
        points.append((float(t), float(fid)))
        sol_t, sol_f = float(t), float(fid)

    if not points:
        raise ValueError("No points returned from yield_bisection().")

    points.sort(key=lambda x: x[0])
    return points, sol_t, sol_f


def annotate_solution(ax, sol_t, sol_f):
    ax.axhline(F_thr, linestyle="--", linewidth=1.2)
    ax.axvline(sol_t, linestyle="--", linewidth=1.2)
    ax.plot(sol_t, sol_f, marker="o", linestyle="None", zorder=5)
    ax.text(
        0.02,
        0.06,
        rf"$T_f={sol_t:.2f}$, $F(T_f)={sol_f:.3f}$",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
    )


def plot_diagnostic(points, sol_t, sol_f, n, pen, path):
    p = np.array(points)
    fig, ax = plt.subplots()
    ax.set_xlim(T_0, T_f)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Fidelity $F(t)$")
    ax.set_title(rf"Diagnostics: $n={n}$, penalty $p={pen}$")
    ax.axhline(F_thr, linestyle="--", linewidth=1.2, label=rf"$F_\text{{thr}}={F_thr}$")
    ax.plot(p[:, 0], p[:, 1], linestyle="--", linewidth=1.1, label="Bisection samples")
    ax.plot(p[:, 0], p[:, 1], linestyle="None", marker="o", label="Evaluations")
    ax.axvline(sol_t, linestyle="--", linewidth=1.2, label=rf"$T_f={sol_t:.2f}$")
    ax.plot(sol_t, sol_f, marker="o", linestyle="None", zorder=5, label="Final point")
    ax.legend(loc="lower right", frameon=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_reader_trace(points, sol_t, sol_f, n, pen, path):
    p = np.array(points)
    fig, ax = plt.subplots()
    ax.set_xlim(T_0, T_f)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Fidelity $F(t)$")
    ax.set_title(rf"$n={n}$, penalty $p={pen}$")
    ax.plot(p[:, 0], p[:, 1], linestyle="-", linewidth=1.8)
    ax.plot(p[:, 0], p[:, 1], linestyle="None", marker="o", alpha=0.35)
    annotate_solution(ax, sol_t, sol_f)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(out_dir: Path):
    out_main = out_dir / "main"
    out_traces = out_dir / "traces"
    out_diag = out_dir / "diagnostics"
    out_data = out_dir / "data"
    for d in [out_main, out_traces, out_diag, out_data]:
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for pen, ns in runtime.items():
        print(f"Penalty {pen}")
        for n in ns:
            print(f"  n={n}")
            try:
                points, sol_t, sol_f = solve_points(n=n, pen=pen)

                diag_name = f"diag_n{n}_p{pen}_F{F_thr}_N{nsteps}_T{T_0}-{T_f}.pdf"
                plot_diagnostic(points, sol_t, sol_f, n, pen, out_diag / diag_name)

                rows.append(
                    {
                        "n": n,
                        "penalty": pen,
                        "F_thr": F_thr,
                        "nsteps": nsteps,
                        "T_0": T_0,
                        "T_f": T_f,
                        "t_star": sol_t,
                        "F_at_t_star": sol_f,
                        "n_evals": len(points),
                    }
                )

                if pen in REP_PENS:
                    trace_name = f"trace_n{n}_p{pen}_F{F_thr}_T{T_0}-{T_f}.pdf"
                    plot_reader_trace(
                        points, sol_t, sol_f, n, pen, out_traces / trace_name
                    )

            except Exception as e:
                print(f"    FAILED (n={n}, penalty={pen}): {type(e).__name__}: {e}")

    df = pd.DataFrame(
        rows,
        columns=[
            "n",
            "penalty",
            "F_thr",
            "nsteps",
            "T_0",
            "T_f",
            "t_star",
            "F_at_t_star",
            "n_evals",
        ],
    ).sort_values(["n", "penalty"])

    if df.empty:
        raise RuntimeError("No results produced — check FAILED messages above.")

    csv_path = out_data / f"results_F{F_thr}_N{nsteps}_T{T_0}-{T_f}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}")

    # t*(p) summary figure
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.set_xlabel("Penalty $p$")
    ax.set_ylabel(r"Threshold time $T_f$")
    ax.set_title(rf"Time to reach $F_\text{{thr}}={F_thr}$")
    for n, g in df.groupby("n"):
        ax.plot(g["penalty"], g["t_star"], marker="o", label=rf"$n={n}$")
    ax.legend(loc="lower right", frameon=True)
    main_fig_path = out_main / f"tstar_vs_penalty_F{F_thr}_T{T_0}-{T_f}.pdf"
    fig.savefig(main_fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved main figure to: {main_fig_path}")

    # Per-n representative panels
    def make_rep_panel(n: int, pens, path: Path):
        fig, axes = plt.subplots(
            1, len(pens), figsize=(6.6, 2.5), sharex=True, sharey=True
        )
        if len(pens) == 1:
            axes = [axes]
        for ax, pen in zip(axes, pens):
            row = df[(df["n"] == n) & (df["penalty"] == pen)]
            if row.empty:
                ax.set_axis_off()
                continue
            t_star = float(row["t_star"].iloc[0])
            f_star = float(row["F_at_t_star"].iloc[0])
            points, _, _ = solve_points(n=n, pen=pen)
            p_arr = np.array(points)
            ax.plot(p_arr[:, 0], p_arr[:, 1], linewidth=1.6)
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
        rep_path = (
            out_main
            / f"rep_traces_n{n}_pens{'-'.join(map(str, REP_PENS))}_F{F_thr}.pdf"
        )
        make_rep_panel(n, REP_PENS, rep_path)
        print(f"Saved representative panel for n={n} to: {rep_path}")


if __name__ == "__main__":
    main(Path("plots/fidelity"))
