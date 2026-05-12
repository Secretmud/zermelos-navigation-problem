import numpy as np
from joblib import Memory

from adiabatic_computation.sigma import sigma
from adiabatic_computation.river import ntime, RiverConfig
from adiabatic_computation import Z, float_type

memory = Memory(location=".joblib_cache", verbose=0)


def H_B(N, river: RiverConfig = None):
    """Cost (bias) Hamiltonian — diagonal matrix of river traversal times.

    Parameters
    ----------
    N : int
        Number of qutrits (grid steps).
    river : RiverConfig, optional
        River parameters and current profile.  Pass a custom ``RiverConfig``
        to use a different current formula or physical constants.
        Defaults to the standard Gaussian profile.

    Examples
    --------
    Default:

        H = H_B(4)

    Custom sinusoidal current:

        import numpy as np
        river = RiverConfig(speed_fn=lambda x: 0.9 * np.sin(np.pi * x))
        H = H_B(4, river=river)
    """
    costmatrix = np.zeros((3**N, 3**N), dtype=float_type)
    for k in range(N):
        c_mat = np.diag(ntime(k, N, river=river))
        costmatrix += sigma(N, k, matrix=c_mat)
    return costmatrix


def H_P(N):
    """Penalty Hamiltonian — (sum_k sigma_Z^k)^2, penalises off-axis paths."""
    penalty_matrix = np.zeros((3**N, 3**N), dtype=float_type)
    for k in range(N):
        penalty_matrix += sigma(N, k)
    return np.matmul(penalty_matrix, penalty_matrix)


@memory.cache
def H_D(N, matrix):
    """Driver (exploration) Hamiltonian — sum_k M^k for a single-site matrix M."""
    d_matrix = np.zeros((3**N, 3**N), dtype=float_type)
    for k in range(N):
        d_matrix += sigma(N, k, matrix=matrix)
    return d_matrix
