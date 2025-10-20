import numpy as np
import matplotlib.pyplot as plt
from joblib import Memory

from lib import beta, X
from lib.hamiltonian import H_D, H_B, H_P
from lib.schrodinger import schrodinger
from lib.time import time
# System parameters
memory = Memory(location=".joblib_cache", verbose=0)
subplot = False
plot_groups = {
    2: "blue",
    # 3: "orange",
    # 4: "green",
    # 5: "pink",
    # 6: "lightblue",
    # 7: "black"
}


# Function to calculate fidelity for a single a_0 value
def calculate_fidelity(a_0, N, beta, psi_0):
    # Function to get the evolved state for given alpha
    psi = schrodinger(a_0, N, beta)
    psi /= np.linalg.norm(psi)
    fidelity = np.abs(np.vdot(psi_0, psi))**2  # Fidelity calculation
    return fidelity


@memory.cache
def quickLZ(N, a_0_values, beta):
    P0 = 1
    for i in range(N, 0, -1):
        P0 *= (1-np.exp(-2*np.pi*np.abs(beta)**2/a_0_values *
               (np.abs(1/np.sqrt(2))**2/((i)**2 - (i-1)**2))))

    return P0


if subplot:
    fig, ax = plt.subplots(len(plot_groups), 1, figsize=(8, 5), sharex=True)
else:
    fig = plt.figure()

plt.tight_layout()
a_0_values = np.linspace(0.0001, 1, 100)

for N, color in plot_groups.items():
    plot = N - 2
    P0 = quickLZ(N, a_0_values, beta)

    alpha = 10
    H = H_B(N) + alpha*H_P(N) + beta*H_D(N, X)
    _, eigvecs = np.linalg.eigh(H)
    psi_0 = eigvecs[:, 0]  # Ground state
    phi_0 = psi_0 / np.linalg.norm(psi_0)  # Normalize

    fidelities = [0] * len(a_0_values)

    for index, a_0 in enumerate(a_0_values):
        fidelity = calculate_fidelity(a_0, N, beta, phi_0)
        fidelities[index] = fidelity
        print(f"{index+1:>4d}/{len(a_0_values)} completed for N={N}", end="\r")
    print()
    if subplot:
        ax[plot].semilogx(a_0_values, fidelities, linestyle='-',
                          label=f"{N=} - Fidelity(Schrodinger)", color=color)
        ax[plot].semilogx(a_0_values, P0, label=f"{N=} - Fidelity(LZ)", color=color, linestyle='--')
        ax[plot].grid(True, which="both", linestyle="--",
                      linewidth=0.5)  # Grid for better readability
        ax[plot].legend()
    else:
        plt.semilogx(a_0_values, fidelities, linestyle='-',
                     label=f"{N=} - Fidelity(Schrodinger)", color=color)
        plt.semilogx(a_0_values, P0, label=f"{N=} - Fidelity(LZ)", color=color, linestyle='--')
        plt.xlabel(r"$\alpha_0$")
        plt.ylabel("Fidelity")
        plt.title(r"Fidelity vs. $\alpha_0$ ")
        # Grid for better readability
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend()

plt.show()
