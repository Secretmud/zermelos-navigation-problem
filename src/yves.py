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
def yves_2(t_0, N, Hi, Hf):
    T = 3
    dt = (T - t_0)/100
    psi = np.zeros(3**N, dtype=complex)
    psi[-1] = 1
    psi /= np.linalg.norm(psi)

    tau = dt/T
    exp_A = expm(-1j*(tau**2/(4*T))*Hf)
    MA = expm(-1j*(tau**2/(2*T))*Hf)
    exp_B = expm(-1j*(1-tau/(2*T))*Hi*tau)
    MB = expm(1j*(tau**2/T)*Hi)
    t = 0
    while t < T:
        psi = exp_A @ exp_B @ exp_A @ psi
        exp_A = MA @ exp_A
        exp_B = MB @ exp_B
        t += dt

    return psi

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

taus = np.linspace(0.001, 3, 10)
fidelities = []
for t in taus:
    H = Hi + (1-t)*Hf
    _, eigvecs = np.linalg.eigh(H)
    psi_0 = eigvecs[:, 0]  # Ground state
    phi_0 = psi_0 / np.linalg.norm(psi_0)  # Normalize
    psi = yves_2(t, N, Hi, Hf)
    psi /= np.linalg.norm(psi)
    fidelity = np.abs(np.vdot(phi_0, psi))**2  # Fidelity calculation
    fidelities.append(fidelity)

plt.figure()
plt.plot(taus, fidelities, linestyle='-', label=f"{N=} - Fidelity")
plt.show()
