import numpy as np

def initialState(N) -> np.array:
    P = np.array([1/2, 1/np.sqrt(2), 1/2], dtype="complex")
    initialState = P.copy()
    for _ in range(N-1):
        initialState = np.kron(initialState, P)

    return initialState