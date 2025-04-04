import numpy as np
import scipy
from lib.hamiltonian import H_B, H_P, H_D
from lib import N, beta

# System parameters
T = 3.0

def choose_time_step(alpha_start, dt_min=1e-4, dt_max=0.1, alpha_stop=T):
    """Choose an appropriate time step based on the starting value of alpha."""
    dt = dt_min + (dt_max - dt_min) * (1 - alpha_start / alpha_stop)
    return max(dt, dt_min)  # Ensure dt is never below the minimum threshold


def schrodinger(a_0):
    dt = choose_time_step(a_0)
    H_p = H_P()
    psi = np.zeros(3**N, dtype=complex)
    psi[-1] = 1
    psi /= np.linalg.norm(psi)
    H_0 = H_B() + beta * H_D()
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
