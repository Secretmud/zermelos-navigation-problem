import numpy as np
import scipy
from lib import X
from lib.hamiltonian import H_B, H_P, H_D
from joblib import Memory

# System parameters
T = 3.0

# Set up joblib memory for caching
memory = Memory(location=".joblib_cache", verbose=0)


@memory.cache
def schrodinger(a_0, N, beta):
    """Solve the Schrodinger equation for given parameters."""
    dt = (T-a_0)/100
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
    while a <= T:
        a = a_0 * (t + dt/2)  # Ensure a evolves correctly
        A = -1j * a * H_p * dt/2
        exp_A = scipy.linalg.expm(A)
        psi = exp_A @ exp_B @ exp_A @ psi

        t += dt

    return psi
