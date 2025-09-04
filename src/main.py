import numpy as np
import matplotlib.pyplot as plt
from joblib import Memory 

from lib import beta, X
from lib.hamiltonian import Hamiltonian, H_D, H_B, H_P, H_P_test, H_B_test
from lib.schrodinger import schrodinger
from lib.time import time
# System parameters
memory = Memory(location=".joblib_cache", verbose=0)
time = [1, 3, 2, 1, 1, 1]
subplot = True
# time = [1, 2, 3, 4, 3, 2,1]
plot_groups = {
    2: "blue",
    3: "orange",
    # 4: "green",
    # 5: "pink",
    # 6: "lightblue",
    # 7: "black"
}


# Function to calculate fidelity for a single a_0 value
def calculate_fidelity(a_0, N, beta, time, psi_0):
    psi = schrodinger(a_0, N, beta, time)  # Function to get the evolved state for given alpha
    psi /= np.linalg.norm(psi)
    fidelity = np.abs(np.vdot(psi_0, psi))**2  # Fidelity calculation
    return fidelity


@memory.cache
def quickLZ(N, a_0_values, beta):
    P0 = 1
    for i in range(N, 0, -1):
        print(f"{i**2} -> {(i-1)**2}")
        P0 *= (1-np.exp(-2*np.pi*np.abs(beta)**2/a_0_values*(np.abs(1/np.sqrt(2))**2/((i)**2 - (i-1)**2))))

    return P0


if subplot:
    fig, ax = plt.subplots(len(plot_groups), 1, figsize=(8, 5), sharex=True)
else:
    fig = plt.figure()

plt.tight_layout()
a_0_values = 10**np.linspace(-3.5, 0, 1000)

for N, color in plot_groups.items():
    plot = N - 2
    print(f"{N=}")
    # P0 = LZ(search_space, slopes, beta, a_0_values)
    P0 = quickLZ(N, a_0_values, beta)

    # P_1 = (1-np.exp(-2*np.pi*np.abs(beta)**2/a_0_values*(np.abs(1/np.sqrt(2))**2/(4-1))))
    # P_2 = P_1 * (1-np.exp(-2*np.pi*np.abs(beta)**2/a_0_values*(np.abs(1/np.sqrt(2))**2/(1))))

    time_c = [time[i] for i in range(N)]

    alpha = 3
    H = Hamiltonian(alpha, beta, N, time_c)
    _, eigvecs = np.linalg.eigh(H)
    psi_0 = eigvecs[:, 0]  # Ground state
    phi_0 = psi_0 / np.linalg.norm(psi_0)  # Normalize

    fidelities = [0] * len(a_0_values)

    for index, a_0 in enumerate(a_0_values):
        fidelity = calculate_fidelity(a_0, N, beta, time, phi_0)
        fidelities[index] = fidelity
        print(f"{index}: Fidelity for a_0={a_0_values[index]}: {fidelity}")
f
    if subplot:
        ax[plot].semilogx(a_0_values, fidelities, linestyle='-', label=f"{N=} - Fidelity(Schrodinger)", color=color)
        ax[plot].semilogx(a_0_values, P0, label=f"{N=} - Fidelity(LZ)", color=color, linestyle='--')
        ax[plot].grid(True, which="both", linestyle="--", linewidth=0.5)  # Grid for better readability
        ax[plot].legend()
    else:
        plt.semilogx(a_0_values, fidelities, linestyle='-', label=f"{N=} - Fidelity(Schrodinger)", color=color)
        plt.semilogx(a_0_values, P0, label=f"{N=} - Fidelity(LZ)", color=color, linestyle='--')
        plt.xlabel(r"$\alpha_0$")
        plt.ylabel("Fidelity")
        plt.title(r"Fidelity vs. $\alpha_0$ ")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Grid for better readability
        plt.legend()

plt.show()
