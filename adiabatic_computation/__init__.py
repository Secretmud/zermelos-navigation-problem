import numpy as np
from dataclasses import dataclass, field

# Physical constants for the Zermelo river problem
D = 1  # river width
v = 1  # boat speed
smax = 0.9  # maximum river current amplitude

float_type = "float64"

# Qutrit single-site operators
Z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
X = 1 / np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])


@dataclass
class LZConfig:
    """Parameters for the three-phase Landau-Zener anneal (schrodinger_split)."""

    n: int
    T_pen: float
    T_x: float
    alpha_max: float
    beta_max: float
    n_steps: int = 5000


@dataclass
class YvesConfig:
    """Parameters for Yves's single-sweep TDSE solver and bisection search."""

    H_i: np.ndarray  # driver Hamiltonian
    H_f: np.ndarray  # target (cost + penalty) Hamiltonian
    n: int
    t: float = None  # anneal time for a single run
    t_range: np.ndarray = None  # [t_min, t_max] for bisection
    gs_idx: int = None


# Public API — import after constants are defined to avoid circular imports
from adiabatic_computation.hamiltonian import H_B, H_P, H_D
from adiabatic_computation.schrodinger import (
    schrodinger_split,
    SolveTDSE_phase1,
    SolveTDSE_phase2,
    SolveTDSE_phase3,
    yves_TDSE,
)
from adiabatic_computation.river import S, ntime, time, RiverConfig
from adiabatic_computation.schedulers import alpha, beta
from adiabatic_computation.initial_states import initialState
from adiabatic_computation.solvers import fid_calc, yield_bisection
from adiabatic_computation.sigma import sigma
