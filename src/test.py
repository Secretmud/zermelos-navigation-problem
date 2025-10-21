from lib.hamiltonian import H_D, N_H_B as HB, H_B
from lib import X
import numpy as np
N = 2

print(HB(N) - H_B(N))  # should be ~0
