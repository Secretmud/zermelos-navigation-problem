import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg

from lib.hamiltonian import H_B, H_P, H_D
from lib import N
from lib.schrodinger import schrodinger

# System parameters
dt = 0.01
T = 3.0  
hbar = 1  
alpha_values = [1, 0.1, 0.05, 0.01, 0.005, 0.001]
beta = 0.1

psis = []

for a_0 in alpha_values:
    psis.append(np.abs(schrodinger(a_0))**2)


# Plot probabilities
fig, ax = plt.subplots(3, 2, figsize=(12, 12))
for i, prob in enumerate(psis):
    ax[i % 3, i // 3].bar(range(3**N), prob)
    ax[i % 3, i // 3].set_title(r"$\alpha = $" + str(alpha_values[i]))

plt.show()
"""
fidelities = []
alpha_values = np.linspace(1, 0.001, 400)
psi = []
for alpha_t in alpha_values:
    psi_0 = np.zeros(3**N, dtype=complex)
    psi_0[-1] = 1j
    psi.append(psi_0)
    for t in range(timesteps):
        A = -1j * alpha_t * (t + dt/2) * H_P() * (dt / (2 * hbar))
        exp_A = scipy.linalg.expm(A)
        psi.append(exp_B @ exp_A @ exp_B @ psi[-1])


    # Compute fidelity
    fidelity = np.abs(psi)**2
    fidelities.append(fidelity)

# Plot fidelity
plt.figure(figsize=(8, 5))
plt.plot(alpha_values, fidelities, linestyle='-')
plt.xscale('log')
plt.xlabel(r"$\alpha$")
plt.yticks(np.linspace(0, 1, 11))
plt.ylabel("Fidelity")
plt.title("Fidelity vs. Alpha")
plt.show()
"""
