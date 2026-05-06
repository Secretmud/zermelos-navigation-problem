import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from lib.hamiltonian import H_B, H_P, H_D
from lib.schedulers import alpha as a, beta as b
from lib import X

steps = 100
memory = joblib.Memory(location=".joblib_cache", verbose=0)

@memory.cache
def avoided(n, Ts, T_x, T_pen):
    HB = H_B(n)
    HP = H_P(n)
    HX = H_D(n, X)
    beta_max  = 1 / n
    alpha_max = 0.05 / n
    eigvals, eigvecs = [], []
    for t in Ts:
        H = HB + b(t, T_x, T_pen, beta_max) * HP + a(t, T_x, T_pen, alpha_max) * HX
        vals, vecs = np.linalg.eigh(H)
        eigvals.append(vals)
        eigvecs.append(vecs)
    return np.array(eigvals), np.array(eigvecs)

n_values     = range(2, 7)
T_pen_values = [100, 400, 800, 1600, 3200, 6400, 10000]

cmap   = cm.viridis
colors = {T_pen: cmap(i / (len(T_pen_values) - 1))
          for i, T_pen in enumerate(T_pen_values)}

# results[T_pen][n] = dict of quantities
results = {T_pen: {} for T_pen in T_pen_values}
import tqdm
for n in tqdm.tqdm(n_values):
    for T_pen in T_pen_values:
        T_x = T_pen / 5
        T_f = 2 * T_x + T_pen
        Ts  = np.linspace(0.1, T_f, steps)
        dt  = Ts[1] - Ts[0]

        eigvals, eigvecs = avoided(n, Ts, T_x, T_pen)
        eigvals_T = eigvals.T          # (num_levels, steps)
        f, s      = eigvals_T[0], eigvals_T[1]
        gap       = s - f
        argmin    = np.argmin(gap)
        min_gap   = gap[argmin]


        results[T_pen][n] = {
            "min_gap":      min_gap,
        }

ns = sorted(n_values)
fig, ax = plt.subplots(1, 1, figsize=(15, 5))

# --- Plot 1: min gap vs n (T_pen independent, so one line) ---
gaps = [results[T_pen_values[0]][n]["min_gap"] for n in ns]
ax.plot(ns, gaps, marker='o', color='steelblue', linewidth=2)
ax.set_xlabel('$n$')
ax.set_ylabel('$\\Delta E_{\\min}$')
ax.set_title('Minimal energy gap vs $n$')
ax.set_xticks(ns)
ax.grid(True, alpha=0.3)
plt.savefig("minimal_energy_gap.pdf")