import numpy as np
from scipy.linalg import eigh, expm
import matplotlib.pyplot as plt
from joblib import Memory

from lib import X
from lib.hamiltonian import H_D, H_B, H_P

# Set up joblib memory for caching
memory = Memory(location=".joblib_cache", verbose=0)


@memory.cache
def hamiltonian_energies(N):
    energies = []
    gamma_vals = []
    t = 0.0
    beta = 1
    alpha = 3

    Hi = beta * H_D(N, X)
    Hf = H_B(N) + alpha * H_P(N)

    Tf = 3
    gamma = 0
    dt = 0.05
    tvals = np.arange(0, Tf+dt, dt)

    for t in tvals:
        gamma = t/Tf
        print(f"{gamma}")
        H = gamma * Hf + (1 - gamma) * Hi
        eigvals, _ = eigh(H)
        energies.append(eigvals)
        gamma_vals.append(gamma)

    return np.array(energies), gamma_vals


@memory.cache
def yves_TDSE(N, Hi, Hf, T=3.0, steps=1900):
    dt = T / steps
    psi = np.zeros(3**N, dtype=complex)
    psi[-1] = 1.0
    psi /= np.linalg.norm(psi)

    # Precompute exponentials once
    eA0 = expm(-1j * Hf * dt**2 / (4*T))
    MA  = expm(-1j * Hf * dt**2 / (2*T))
    eB0 = expm(-1j * Hi * dt / T + 1j * Hi * dt**2 / (2*T))
    MB  = expm(1j * Hi * dt**2 / T)

    eA = eA0.copy()
    eB = eB0.copy()

    gamma_vals = []
    fidelities = []

    t = 0.0
    for _ in range(steps):
        gamma = t / T
        psi = eA @ eB @ eA @ psi
        eA = MA @ eA
        eB = MB @ eB

        H = (1-gamma) * Hi + gamma * Hf
        eigvals, eigvecs = np.linalg.eigh(H)

        overlaps = np.abs(eigvecs.conj().T @ psi)**2
        fidelities.append(overlaps)

        gamma_vals.append(gamma)
        t += dt

    return np.array(fidelities), gamma_vals

"""
for i in range(2, 7):
    energies, gamma_vals = hamiltonian_energies(i)
    plt.figure(figsize=(8, 6))
    N = i
    for i in range(energies.shape[1]):
        plt.plot(gamma_vals, energies[:, i])
    plt.xlabel("γ")
    plt.ylabel("Energy")
    plt.title(f"Energy spectrum {N=}")
    plt.pause(0.1)
plt.show()
"""
beta = 1
alpha = 3
N = 3

Hi = beta * H_D(N, X)
Hf = H_B(N) + alpha * H_P(N)

fidelities, gamma_vals = yves_TDSE(N, Hi, Hf)

plt.figure(figsize=(8,6))
for i in range(fidelities.shape[1]):
    plt.plot(gamma_vals, fidelities[:, i])
plt.xlabel("γ")
plt.ylabel("Fidelity")
plt.title(f"TDSE fidelities for N={N}")
plt.show()