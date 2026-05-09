import numpy as np


def initialState(N) -> np.ndarray:
    """Returns the N-qutrit |+> state: tensor product of [1/2, 1/sqrt(2), 1/2]."""
    P = np.array([1/2, 1/np.sqrt(2), 1/2], dtype="complex")
    state = P.copy()
    for _ in range(N - 1):
        state = np.kron(state, P)
    return state
