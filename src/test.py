import numpy as np
from lib.hamiltonian import H_D, H_B, H_P
from lib import X
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply  # optional faster method

import numpy as np
from scipy.linalg import expm

def yves_TDSE_split(N, Hi, Hf, T=1.0, steps=20000, psi0=None, normalize_each=True):
    """
    Split-operator propagation for the quantum-annealing Hamiltonian:
        H(t) = γ(t) Hf + [1 - γ(t)] Hi,
    with γ(t) = t / T.

    Based on equations (47)–(61) of the provided reference.

    Parameters
    ----------
    N : int
        System size (used only for psi0 initialization if not provided).
    Hi, Hf : ndarray
        Initial and final Hamiltonians.
    T : float
        Total evolution time.
    steps : int
        Number of time steps.
    psi0 : ndarray or None
        Initial wavefunction. If None, uses |...00> (last basis state).
    normalize_each : bool
        Renormalize wavefunction each step to avoid numerical drift.

    Returns
    -------
    psi : ndarray
        Final wavefunction.
    """
    dt = T / steps
    γ0 = 0.05

    # Initial state
    if psi0 is None:
        psi = np.zeros(3**N, dtype=complex)
        psi[-1] = 1.0
    else:
        psi = psi0.copy()
    psi /= np.linalg.norm(psi)

    # Precompute matrix exponentials (eqs. 58–61)
    eA0 = expm(-1j * γ0 * Hf * dt**2 / 2)
    MA  = expm(-1j * γ0 * Hf * dt**2)
    eB0 = expm(-1j * Hi * dt + 1j * γ0 * Hi * dt**2 / 2)
    MB  = expm(1j * γ0 * Hi * dt**2)

    # Initialize propagators
    eA = eA0.copy()
    eB = eB0.copy()

    # Time evolution
    for _ in range(steps):
        psi = eA @ eB @ eA @ psi
        eA = MA @ eA
        eB = MB @ eB
        if normalize_each:
            psi /= np.linalg.norm(psi)

    return psi


import matplotlib.pyplot as plt

beta = 0.1
N = 3
H = H_B(N) + 3 * H_P(N)
_, eigvecs = np.linalg.eigh(H)
psi_target = eigvecs[:, 0]
psi_target /= np.linalg.norm(psi_target)

Hi = beta * H_D(N, X)

alpha_values = np.linspace(0.001, 1, 200)
fidelities = []

# Precompute target ground states for each alpha if you want final ground state fidelity
for alpha in alpha_values:
    Hf = H_B(N) + alpha * H_P(N)

    # choose T and steps appropriately — larger T means more adiabatic
    T = 3.0         # tune this
    steps = 2000     # tune this (tradeoff accuracy / speed)

    psi_final = yves_TDSE_split(N, Hi, Hf, T=T, steps=steps)

    psi_final /= np.linalg.norm(psi_final)
    fidelity = np.abs(np.vdot(psi_target, psi_final))**2
    fidelities.append(fidelity)

# Plot once
plt.figure(figsize=(8, 6))
plt.ylim(0, 1)
plt.xlim(alpha_values[0], alpha_values[-1])
plt.semilogx(alpha_values, fidelities)
plt.xlabel("α")
plt.ylabel("Fidelity")
plt.title(f"Fidelity vs α for N={N} (T={T}, steps={steps})")
plt.grid(True)
plt.show()
