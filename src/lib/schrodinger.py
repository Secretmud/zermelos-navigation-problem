import numpy as np
import scipy as sp
from lib import X
from lib import yvesData
from lib.hamiltonian import H_B, H_P, H_D
from lib.schedulers import alpha as a, beta as b
from lib.initial_states import initialState
from joblib import Memory

# Set up joblib memory for caching
memory = Memory(location=".joblib_cache", verbose=0)

@memory.cache
def schrodinger_split(
    T_pen,
    N,
    T_x,
    alpha_max,
    beta_max,
    n_steps=2500,
):
    dt = T_pen / n_steps
    H_cost = H_B(N)
    H_X    = H_D(N, X)
    H_pen  = H_P(N)

    cost_vec = np.diagonal(H_cost)
    pen_vec  = np.diagonal(H_pen)


    eigvals_X, U_X = np.linalg.eigh(H_X)
    U_X_dag = U_X.conj().T


    eigvals_A2, U_A2 = np.linalg.eigh(H_cost + alpha_max * H_X)
    U_A2_dag = U_A2.conj().T
    exp_A2   = np.exp(-1j * eigvals_A2 * dt / 2.0)

    def a(t):
        if t < T_x:
            return alpha_max * (t / T_x)
        elif t < T_x + T_pen:
            return alpha_max
        elif t < 2 * T_x + T_pen:
            return alpha_max * (1 - (t - (T_x + T_pen)) / T_x)
        else:
            return 0

    def b(t):
        if t < T_x:
            return 0
        elif t < T_x + T_pen:
            return beta_max * (t - T_x) / T_pen
        else:
            return beta_max

    psi = np.zeros(3**N, dtype=complex)
    psi[-1] = 1.0

    t = 0
    while t < 2 * T_x + T_pen:
        t_mid = t + 0.5 * dt

        if 0 < t_mid <= T_x:
            exp_A = np.exp(-1j * cost_vec * dt / 2.0)
            exp_B = np.exp(-1j * a(t_mid) * eigvals_X * dt)
            psi = exp_A * psi
            psi = U_X @ (exp_B * (U_X_dag @ psi))
            psi = exp_A * psi

        elif T_x < t_mid <= T_x + T_pen:
            exp_B = np.exp(-1j * b(t_mid) * pen_vec * dt)
            psi = U_A2 @ (exp_A2 * (U_A2_dag @ psi))
            psi = exp_B * psi
            psi = U_A2 @ (exp_A2 * (U_A2_dag @ psi))

        elif T_x + T_pen < t_mid <= 2 * T_x + T_pen:
            exp_A = np.exp(-1j * (cost_vec + beta_max * pen_vec) * dt / 2.0)
            exp_B = np.exp(-1j * a(t_mid) * eigvals_X * dt)
            psi = exp_A * psi
            psi = U_X @ (exp_B * (U_X_dag @ psi))
            psi = exp_A * psi

        t += dt

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