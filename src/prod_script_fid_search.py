#%matplotlib widget
from lib.initial_states import initialState
from lib.hamiltonian import H_B, H_D, H_P
from lib.schrodinger import yves_TDSE
from lib.solvers import yield_bisection
from lib import X, D, yvesData
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import pathlib

nsteps = 2000
T_0 = 15
T_f = 1500
beta = -1
F_thr = 0.9


runtime = {
    1:  [2, 3, 4, 5],
    5:  [2, 3, 4, 5],
    10: [2, 3, 4, 5],
    15: [2, 3, 4, 5],
    20: [2, 3, 4, 5],
    30: [2, 3, 4, 5],
}

for pen in runtime.keys():
    print(f"Producing data for penalty: {pen}")
    for n in runtime[pen]:
        print(f"Producing for {n=}")
        Hf = H_B(n) + pen * H_P(n)
        Hi = beta*H_D(n, X)
        ts = np.linspace(T_0, T_f, nsteps)
        args = yvesData(Hf=Hf, Hi=Hi, n=n, ts=ts)
        try:
            plt.xlim(T_0, T_f)
            plt.ylim(0, 1)
            plt.axhline(y=F_thr, color='b', linestyle='--', zorder=1)
            sol_t, sol_f = 0, 0

            p = []
            for time, fid in yield_bisection(yves_TDSE, args, f_thr=F_thr):
                p.append([time, fid])
                print(p[-1])
                plt.plot(time, fid, 'ro', zorder=2)
                sol_t, sol_f = time, fid


            p = sorted(p, key=lambda x: x[0])

            p = np.array(p)

            x = p[:, 0]
            y = p[:, 1]

            plt.plot(x, y, '--', zorder=1)
            plt.plot(sol_t, sol_t, 'go', zorder=3)
            plt.axvline(sol_t, color='b', linestyle='--', label=f'{pen=} Fidelity: {sol_f:.3f} at Time: {sol_t:.3f}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{n}_{pen}_{F_thr}_{nsteps}_{T_0}_{T_f}.pdf")
            plt.clf()
        except ValueError as e:
            print(str(e))
