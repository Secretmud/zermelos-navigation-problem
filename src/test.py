import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from lib import X, yvesData
from lib.solvers import yield_bisection
from lib.schrodinger import yves_TDSE
from lib.hamiltonian import H_B, H_P, H_D

n = 2
pen = 10 
beta = -1
T_0 = 15
T_f = 1500
nsteps = 5000
F_thr = 0.8

Hf = H_B(n) + pen * H_P(n)
Hi = beta*H_D(n, X)
ts = np.linspace(T_0, T_f, nsteps)
args = yvesData(Hf=Hf, Hi=Hi, n=n, ts=ts)

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
plt.plot(sol_t, sol_t, 'o', zorder=3, label=f"{pen=}")
plt.axvline(sol_t, color='b', linestyle='--', label=f'{pen=} Fidelity: {sol_f:.3f} at Time: {sol_t:.3f}')
plt.legend()
plt.tight_layout()
plt.savefig(f"{n}_{pen}_{F_thr}_{nsteps}_{T_0}_{T_f}.pdf")


gs_idx = np.argmin(np.diagonal(Hf))
k = 20
t = float(args.ts[k])
t2 = np.nextafter(t, np.inf)  # tiniest float bigger than t

F1 = np.abs(yves_TDSE(yvesData(Hf=args.Hf, Hi=args.Hi, n=args.n, t=t))[gs_idx])**2
F2 = np.abs(yves_TDSE(yvesData(Hf=args.Hf, Hi=args.Hi, n=args.n, t=t2))[gs_idx])**2

print(t, t2, F1, F2, F2-F1)
