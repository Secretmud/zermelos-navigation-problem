import numpy as np
import scipy
from lib import X
from lib.hamiltonian import H_B, H_P, H_D
from joblib import Memory

# System parameters
T = 4

# Set up joblib memory for caching
memory = Memory(location=".joblib_cache", verbose=0)
@memory.cache
def schrodinger(a_0, N, beta, n_steps=800):
    dt = (T - a_0)/n_steps
    a_s = np.arange(0, T+dt, dt)
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
    a = 0
    steps = 0
    for a in a_s:
        A = -1j * a*(t + dt/2) * H_p * dt/2
        exp_A = scipy.linalg.expm(A)
        psi = exp_A @ exp_B @ exp_A @ psi
        psi /= np.linalg.norm(psi)
        steps += 1
        t += dt

    # print(f"{a=} {steps=} {t=}")
    return psi
