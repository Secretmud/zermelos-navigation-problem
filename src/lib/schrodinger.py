import numpy as np
import scipy as sp
from lib import X
from lib.hamiltonian import H_B, H_P, H_D
from joblib import Memory


# Set up joblib memory for caching
memory = Memory(location=".joblib_cache", verbose=0)
@memory.cache
def schrodinger(a_0, N, beta, n_steps=800):
    dt = a_0/n_steps
    print(dt)
    H_p = H_P(N)
    psi = np.zeros(3**N, dtype=complex)
    psi[-1] = 1
    psi /= np.linalg.norm(psi)
    H_0 = H_B(N) + beta * H_D(N, X)
    B = -1j * H_0 * dt
    exp_B = sp.linalg.expm(B)
    del H_0, B
    for t in range(n_steps):
        a = a_0 * (t + dt/2)
        A = -1j * a * H_p * dt/2
        exp_A = scipy.linalg.expm(A)
        psi = exp_A @ exp_B @ exp_A @ psi
        psi /= np.linalg.norm(psi)

    # print(f"{a=} {steps=} {t=}")
    return psi


def calc_steps(T, steps):
    dt = T/steps
    while dt > 0.05:
        steps += 100
        dt = T/steps

    return dt


@memory.cache
def yves_TDSE(psi_init, Hi, Hf, T, steps=1000):
    dt = calc_steps(T, steps)

    psi = psi_init.copy()

    eB = sp.linalg.expm(-1j * ( 1 - dt / (2*T)) * Hi * dt)
    MB = sp.linalg.expm(1j * Hi * dt**2/T)
    t = 0
   
    while t < T:
        eA = sp.linalg.expm(-1j*(t + dt / 2)*dt/(2*T)*Hf)
        psi = eA @ eB @ eA @ psi
        # update eA and eB for next step using the recursion
        eB = MB @ eB
        t += dt

    return psi
    