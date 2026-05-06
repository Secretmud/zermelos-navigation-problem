import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import PercentFormatter
from lib.hamiltonian import H_B, H_P, H_D
from lib.schedulers import alpha as a, beta as b
from lib.schrodinger import schrodinger_split
from lib import X
import tqdm

memory = joblib.Memory(location=".joblib_cache", verbose=0)

ns    = [2, 3, 4, 5]
pens  = {2: 5000, 3: 10000, 4: 20000, 5: 30000}

@memory.cache
def target_ground_state(n):
    """Ground state of H_cost + beta_max * H_pen, the t -> inf limit."""
    beta_max = 1 / n
    HB = H_B(n)
    HP = H_P(n)
    H_final = HB + beta_max * HP
    vals, vecs = np.linalg.eigh(H_final)
    return vecs[:, 0]

for n in ns:
    beta_max  = 1 / n
    alpha_max = 0.05 / n

    gs = target_ground_state(n)  # fixed target, independent of T_pen

    T_pen_values = np.linspace(100, pens[n], 500)
    fidelities   = []

    for T_pen in tqdm.tqdm(T_pen_values, desc=f"n={n}"):
        T_x = T_pen / 5
        psi = schrodinger_split(
            T_pen=T_pen,
            N=n,
            T_x=T_x,
            alpha_max=alpha_max,
            beta_max=beta_max,
            n_steps=pens[n] // 2,
        )
        fidelity = np.abs(gs.conj() @ psi) ** 2
        fidelities.append(fidelity)

    plt.figure()
    plt.plot(T_pen_values, fidelities)
    plt.xlabel("Anneal time $T_{pen}$")
    plt.ylabel("Fidelity")
    plt.title(f"$n={n}$")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(f"{n}.pdf")
    plt.close()