import numpy as np
import scipy as sp
from lib import X
from lib import yvesData
from lib.hamiltonian import H_B, H_P, H_D
from lib.schedulers import alpha, beta
from lib.initial_states import initialState
from joblib import Memory


# Set up joblib memory for caching
memory = Memory(location=".joblib_cache", verbose=0)


@memory.cache
def schrodinger(a_0, N, beta, n_steps=800):
    dt = a_0 / n_steps
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
        a = a_0 * (t + dt / 2)
        A = -1j * a * H_p * dt / 2
        exp_A = sp.linalg.expm(A)
        psi = exp_A @ exp_B @ exp_A @ psi
        psi /= np.linalg.norm(psi)

    return psi

@memory.cache
def yves_TDSE(args, steps=25000):
    dt = 0.05
    psi = initialState(args.n)

    eB = sp.linalg.expm(-1j * ( 1 - dt / (2*args.t)) * args.Hi * dt)
    MB = sp.linalg.expm(1j * args.Hi * dt**2/args.t)
    t = 0
    Hf = np.diagonal(args.Hf)
    while t < args.t:
        eA = np.exp(-1j*(t + dt / 2)*dt/(2*args.t)*Hf)
        psi = eA * psi
        psi = np.matmul(eB, psi)
        psi = eA * psi
        eB = MB @ eB
        t += dt

    return psi