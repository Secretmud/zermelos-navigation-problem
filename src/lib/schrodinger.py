import numpy as np
import scipy
from lib import X
from lib.hamiltonian import H_B, H_P, H_D
from joblib import Memory

# System parameters
T = 30

# Set up joblib memory for caching
memory = Memory(location=".joblib_cache", verbose=0)

def calculate_fidelity(N, beta, psi, phi_0):
    phi_0 = ground_state_final(N, beta)
    return np.abs(np.vdot(phi_0, psi))**2  # Fidelity calculation

@memory.cache
def schrodinger(a_0, N, beta, n_steps=400):
    dt = T/n_steps
    print(dt)
    H_p = H_P(N)
    psi = np.zeros(3**N, dtype=complex)
    psi[-1] = 1
    psi /= np.linalg.norm(psi)
    H_0 = H_B(N) + beta * H_D(N, X)
    B = -1j * H_0 * dt
    exp_B = scipy.linalg.expm(B)
    del H_0, B
    t = 0
    a = a_0
    print(f"Starting Schrodinger evolution with a_0={a_0}, N={N}, beta={beta}")
    while a <= T:
        a = a_0* (t + dt/2)  # Ensure a evolves correctly
        A = -1j * a * H_p * dt/2
        exp_A = scipy.linalg.expm(A)
        psi = exp_A @ exp_B @ exp_A @ psi
        fidelity = calculate_fidelity(N, beta, psi, None)
        if fidelity >= 0.8:
            print(f"\nEarly stopping at t={t:.4f} with fidelity={fidelity:.4f} alpha={a:.4f}")
            break

        t += dt

    return psi


    
def ground_state_final(N, beta, alpha=T):
    H_final = H_B(N) + beta * H_D(N, X) + alpha*H_P(N)
    _, eigvecs = np.linalg.eigh(H_final)
    gs = eigvecs[:, 0]
    return gs / np.linalg.norm(gs)
