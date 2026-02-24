import numpy as np
import scipy as sp
from lib import X
from lib import yvesData
from lib.hamiltonian import H_B, H_P, H_D
from lib.initial_states import initialState
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

    return psi


@memory.cache
def yves_TDSE(args, steps=15000):
    T = float(args.t)
    dt = T/steps
    psi = initialState(args.n)

    eB = sp.linalg.expm(-1j * ( 1 - dt / (2*T)) * args.Hi * dt)
    MB = sp.linalg.expm(1j * args.Hi * dt**2/T)
    
    for k in range(steps):
       t = k * dt

       eA = sp.linalg.expm(-1j*(t + dt / 2)*dt/(2*T)*args.Hf)
       psi = eA @ eB @ eA @ psi
       eB = MB @ eB

    return psi
    
"""

@memory.cache
def yves_TDSE(args, steps=15000):
    T = float(args.t)
    dt = T / steps

    psi = initialState(args.n).astype(np.complex128).reshape(-1)

    # Extract diagonals as 1D arrays (works for sparse diagonal matrices too)
    hi = np.asarray(args.Hi.diagonal(), dtype=np.float64)
    hf = np.asarray(args.Hf.diagonal(), dtype=np.float64)

    # Precompute diagonal phase factors
    # eB0 = exp(-i * (1 - dt/(2T)) * Hi * dt)
    eB = np.exp(-1j * (1.0 - dt/(2.0*T)) * hi * dt)

    # MB = exp(+i * Hi * dt^2 / T)
    MB = np.exp(1j * hi * (dt**2) / T)

    for k in range(steps):
        t = k * dt
        # eA = exp(-i * (t + dt/2) * dt/(2T) * Hf)
        a = -1j * (t + 0.5*dt) * dt / (2.0*T)
        eA = np.exp(a * hf)

        # psi = eA @ eB @ eA @ psi  (all diagonal => elementwise)
        psi = (eA * eB * eA) * psi

        # eB = MB @ eB  (diagonal => elementwise)
        eB = MB * eB

    return psi.reshape(-1, 1)  # if you want column vector
"""
