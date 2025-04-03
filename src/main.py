import numpy as np
import matplotlib.pyplot as plt

from lib import N, beta
from lib.hamiltonian import Hamiltonian, H_D, H_B, H_P, H_P_test, H_B_test
from lib.schrodinger import schrodinger
from lib.time import time
# System parameters

a_0_values = 10**np.linspace(-4, 1, 150)
eigvals = []
for a in a_0_values:
    H = H_B() + a*H_P()
    vals, _ = np.linalg.eig(H)
    eigvals.append(vals)
    del H

eig_first = H_B_test() + a_0_values[0]*H_P_test()
eig_last = H_B_test() + a_0_values[-1]*H_P_test()
slopes = (eig_last - eig_first)/a_0_values[-1]



def lowest_value(vals: list):
    """
    Find the lowest value and its index from a list of indices.

    arg:
        vals: list of indices

    return:
        tuple: tuple of index and value
    """
    index = vals[0]
    val = eig_first[index]
    for i in range(1, len(vals)):
        idx = vals[i]
        if eig_first[idx] < val:
            index = vals[i]
            val = eig_first[index]

    return (index, val)


# Let us assume that number of crossings coincides with N (number of qutrits).
# Thus, we would N + 1 values to calculate the crossings
groups = {}
for i, slope in enumerate(slopes):
    s = round(slope,5)
    if s not in groups.keys():
        groups[s] = []

    groups[s].append(i)

# Create the search space, including slope = 0. Since the ground state lies there.
search_space = []
for k,v in groups.items():
    search_space.append(lowest_value(v))
    if slope == 0:
        break


coupling = H_D()
P0 = 1
for i in range(len(search_space)-1):
    ii, ij = search_space[i][0], search_space[i+1][0]
    couple = coupling[ii][ij]
    print(f"{ii},{ij},{1/np.sqrt(2)},{couple},{slopes[ii]},{slopes[ij]}")
    if couple == 0:
        print(f"no coupling for {ii},{ij}")
    P0 *= (1-np.exp(-2*np.pi*np.abs(beta)**2/a_0_values*(np.abs(couple)**2/(slopes[ii] - slopes[ij]))))


P_1 = (1-np.exp(-2*np.pi*np.abs(beta)**2/a_0_values*(np.abs(1/np.sqrt(2))**2/(4-1))))
P_2 = P_1 * (1-np.exp(-2*np.pi*np.abs(beta)**2/a_0_values*(np.abs(1/np.sqrt(2))**2/(1))))

alpha = 3
H = Hamiltonian(alpha, beta)
_, eigvecs = np.linalg.eigh(H)
psi_0 = eigvecs[:, 0]  # Ground state
phi_0 = psi_0 / np.linalg.norm(psi_0)  # Normalize

fidelities = []
for a_0 in a_0_values:
    psi = schrodinger(a_0)  # Function to get the evolved state for given alpha
    psi /= np.linalg.norm(psi)
    fidelity = np.abs(np.vdot(psi_0, psi))**2  # Fidelity calculation
    fidelities.append(fidelity)
    print(f"{a_0=}")

plt.figure(figsize=(8, 5))
plt.semilogx(a_0_values, fidelities, marker="o", linestyle='-', label="Fidelity(Schrodinger)")
plt.semilogx(a_0_values, P_2, linestyle='-', label="Fidelity(LZ)")
plt.semilogx(a_0_values, P0, label='P0', color='blue')
plt.xlabel(r"$\alpha_0$")
plt.ylabel("Fidelity")
plt.title(r"Fidelity vs. $\alpha_0$ ")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Grid for better readability
plt.legend()

plt.show()

