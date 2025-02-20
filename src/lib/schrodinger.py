import numpy as np
import scipy
from lib.hamiltonian import H_B, H_P, H_D
from lib import N

# System parameters
dt = 0.01
T = 3.0
beta = 0.1


def schrodinger(a_0):
    H_p = H_P()
    psi = np.zeros(3**N, dtype=complex)
    psi[-1] = 1
    H_0 = H_B() + beta * H_D()
    B = -1j * H_0 * dt
    exp_B = scipy.linalg.expm(B)
    del H_0, B
    
    t = 0
    runs = 0
    a = a_0
    while a < T:
        a = a_0 * t  # Ensure a evolves correctly
        A = -1j * a * H_p * dt
        exp_A = scipy.linalg.expm(A)
        psi = exp_A @ exp_B @ exp_A @ psi
        
        t += dt
        runs += 1

    print(f"{a_0=}\n\t{a=}\n\t{t}\n\t{runs=}")
    return psi
