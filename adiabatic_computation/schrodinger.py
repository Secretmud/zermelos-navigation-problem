import numpy as np
import scipy as sp
from scipy.linalg import expm
from joblib import Memory

from adiabatic_computation import X, LZConfig, YvesConfig
from adiabatic_computation.hamiltonian import H_B, H_P, H_D
from adiabatic_computation.initial_states import initialState

memory = Memory(location=".joblib_cache", verbose=0)


@memory.cache
def SolveTDSE_phase1(Hcost, Hexplore, Hpenalty, alpha, Pen, Psi, Tmax, Nsteps, ProperGS):
    """Phase 1: ramp exploration term on from 0 to alpha over time Tmax."""
    dt = Tmax / Nsteps

    Ahalf = np.exp(-1j * Hcost * dt / 2)
    B = expm(-1j * alpha / (2 * Tmax) * Hexplore * dt**2)
    Bincrement = expm(-1j * alpha / Tmax * Hexplore * dt**2)

    for _ in range(Nsteps):
        Psi = Ahalf * Psi
        Psi = B @ Psi
        Psi = Ahalf * Psi
        B = B @ Bincrement

    if ProperGS:
        H_full = np.diag(Hcost) + alpha * Hexplore
        eigenvalues, eigenvectors = np.linalg.eigh(H_full)
        PsiGS = eigenvectors[:, 0]
        TempFid = abs(Psi.conj() @ PsiGS) ** 2
    else:
        TempFid = None

    return Psi, TempFid


@memory.cache
def SolveTDSE_phase2(Hcost, Hexplore, Hpenalty, alpha, Pen, Psi, Tmax, Nsteps, ProperGS):
    """Phase 2: ramp penalty term on from 0 to Pen over time Tmax."""
    dt = Tmax / Nsteps

    Ahalf = np.exp(-1j * Pen / (4 * Tmax) * Hpenalty * dt**2)
    AhalfIncrement = np.exp(-1j * Pen / (2 * Tmax) * Hpenalty * dt**2)
    B = expm(-1j * (np.diag(Hcost) + alpha * Hexplore) * dt)

    for _ in range(Nsteps):
        Psi = Ahalf * Psi
        Psi = B @ Psi
        Psi = Ahalf * Psi
        Ahalf = Ahalf * AhalfIncrement

    if ProperGS:
        H_full = np.diag(Hcost + Pen * Hpenalty) + alpha * Hexplore
        eigenvalues, eigenvectors = np.linalg.eigh(H_full)
        PsiGS = eigenvectors[:, 0]
        TempFid = abs(Psi.conj() @ PsiGS) ** 2
    else:
        TempFid = None

    return Psi, TempFid


@memory.cache
def SolveTDSE_phase3(Hcost, Hexplore, Hpenalty, alpha, Pen, Psi, Tmax, Nsteps, ProperGS):
    """Phase 3: ramp exploration term back off from alpha to 0 over time Tmax."""
    dt = Tmax / Nsteps

    Ahalf = np.exp(-1j * (Hcost + Pen * Hpenalty) * dt / 2)
    B = expm(-1j * alpha * (1 - dt / (2 * Tmax)) * Hexplore * dt)
    Bincrement = expm(+1j * alpha / Tmax * Hexplore * dt**2)

    for _ in range(Nsteps):
        Psi = Ahalf * Psi
        Psi = B @ Psi
        Psi = Ahalf * Psi
        B = B @ Bincrement

    if ProperGS:
        MinInd = np.argmin(Hcost + Pen * Hpenalty)
        TempFid = abs(Psi[MinInd]) ** 2
    else:
        TempFid = None

    return Psi, TempFid


def schrodinger_split(config: LZConfig) -> np.ndarray:
    """Run the full three-phase LZ anneal and return the final state vector."""
    explore_fraction = 1 / 5
    n_steps_explore = int(np.floor(config.n_steps * explore_fraction))

    H_cost = H_B(config.n).astype(np.float64)
    H_pen = H_P(config.n).astype(np.float64)
    cost_vec = np.diagonal(H_cost)
    pen_vec = np.diagonal(H_pen)
    H_X = H_D(config.n, X).astype(np.float64)

    psi = np.zeros(3**config.n, dtype=complex)
    psi[-1] = 1.0

    psi, _ = SolveTDSE_phase1(
        Hcost=cost_vec, Hexplore=H_X, Hpenalty=pen_vec,
        alpha=config.alpha_max, Pen=config.beta_max, Psi=psi,
        Tmax=config.T_x, Nsteps=n_steps_explore, ProperGS=False,
    )
    psi, _ = SolveTDSE_phase2(
        Hcost=cost_vec, Hexplore=H_X, Hpenalty=pen_vec,
        alpha=config.alpha_max, Pen=config.beta_max, Psi=psi,
        Tmax=config.T_pen, Nsteps=config.n_steps, ProperGS=False,
    )
    psi, _ = SolveTDSE_phase3(
        Hcost=cost_vec, Hexplore=H_X, Hpenalty=pen_vec,
        alpha=config.alpha_max, Pen=config.beta_max, Psi=psi,
        Tmax=config.T_x, Nsteps=n_steps_explore, ProperGS=False,
    )
    return psi


@memory.cache
def yves_TDSE(config: YvesConfig, steps=25000) -> np.ndarray:
    dt = 0.05
    psi = initialState(config.n)

    eB = sp.linalg.expm(-1j * (1 - dt / (2 * config.t)) * config.H_i * dt)
    MB = sp.linalg.expm(1j * config.H_i * dt**2 / config.t)
    t = 0
    H_f_diag = np.diagonal(config.H_f)
    while t < config.t:
        eA = np.exp(-1j * (t + dt / 2) * dt / (2 * config.t) * H_f_diag)
        psi = eA * psi
        psi = np.matmul(eB, psi)
        psi = eA * psi
        eB = MB @ eB
        t += dt

    return psi
