import numpy as np
from scipy.linalg import eigh, expm
import matplotlib.pyplot as plt
import itertools
from joblib import Memory

from lib import X
from lib.hamiltonian import H_D, N_H_B as H_B, H_P

# Set up joblib memory for caching
memory = Memory(location=".joblib_cache", verbose=0)


@memory.cache
def hamiltonian_energies(N):
    energies = []
    gamma_vals = []
    vecs = []
    t = 0.0
    beta = 1
    alpha = 3

    Hi = beta * H_D(N, X)
    Hf = H_B(N) + alpha * H_P(N)

    Tf = 15
    gamma = 0
    dt = 0.05
    tvals = np.arange(0, Tf+dt, dt)

    for t in tvals:
        gamma = t/Tf
        H = gamma * Hf + (1 - gamma) * Hi
        eigvals, vec = eigh(H)
        energies.append(eigvals)
        vecs.append(vec)
        gamma_vals.append(gamma)

    return np.array(energies), vecs, gamma_vals


# @memory.cache
def yves_TDSE(a_0, N, Hi, Hf, T=3, steps=10000):
    dt = T / steps
    psi = np.zeros(3**N, dtype=complex)
    psi[-1] = 1.0
    psi /= np.linalg.norm(psi)

    # Precompute exponentials once
    eA0 = expm(-1j  * Hf * dt**2 / 4)
    MA  = expm(-1j  *Hf * dt**2 / 2)
    eB0 = expm(-1j * Hi * dt + 1j *  Hi * dt**2 / 2)
    MB  = expm(1j  * Hi * dt**2)

    eA = eA0.copy()
    eB = eB0.copy()
    del eA0, eB0
    gamma = a_0
    while gamma < T:
        psi = eA @ eB @ eA @ psi
        eA = MA @ eA
        eB = MB @ eB
        gamma += dt
    return psi


beta = 1
alpha = 2
N = 3
energies, vecs, gamma_vals = hamiltonian_energies(N)
Hf = H_B(N) + alpha * H_P(N)
_, vec = np.linalg.eigh(Hf)
print(vec)
psi_0 = vec[:, 0]  # Ground state
print(psi_0)
psi_0 /= np.linalg.norm(psi_0)  # Normalize
print(psi_0)

@memory.cache
def all_move_sequences(N):
    moves = (1, 0, -1)
    return list(itertools.product(moves, repeat=N))

plt.figure(figsize=(8, 6))
for i in range(energies.shape[1]):
    plt.plot(gamma_vals, energies[:, i])
plt.xlabel(r"$\alpha$")
plt.ylabel("Energy")
plt.title(f"Energy spectrum {N=}")
plt.show()
vecs = np.array([np.abs(v[:, 0])**2 for v in vecs])  # Ground state populations
fig, axes = plt.subplots(2, 2)
indices = np.arange(vecs.shape[1])
alpha_indices = np.linspace(0, len(vecs)-1, 4, dtype=int)
selected_eigvecs = [vecs[i] for i in alpha_indices]
moves = all_move_sequences(N)
for i, eigvec in enumerate(selected_eigvecs):
    ax = axes.flat[i]
    ax.bar(indices, eigvec, linewidth=1)  
    ax.set_xlabel("Path Index")
    ax.set_ylabel("Population")



plt.tight_layout()
plt.show()
alpha_values = np.linspace(10**(-3.5), 10**0, 500)
Hi = beta * H_D(N, X)

plt.figure(figsize=(8, 6))
plt.legend()
plt.xlim(alpha_values[0], alpha_values[-1])
plt.ylim(0, 1)
plt.axhline(y=0.9, color='r', linestyle='--', label='Ideal Fidelity')
plt.grid(True)
plt.ion()
fidelities = []
for i, alpha in enumerate(alpha_values):
    Hf = H_B(N) + alpha * H_P(N)
    psi = yves_TDSE(alpha, N, Hi, Hf)
    psi /= np.linalg.norm(psi)
    fidelity = np.abs(np.vdot(psi_0, psi))**2  # Fidelity calculation
    fidelities.append(fidelity)
    if i % 10 == 0:
        plt.plot(alpha, fidelity, 'bo')
        plt.pause(0.1)

plt.plot(alpha_values, fidelities)
plt.xlabel("α")
plt.ylabel("Fidelity")
plt.title(f"Fidelity vs α for N={N}")
plt.ioff()
plt.show()