import numpy as np
from matplotlib import pyplot as plt
# copy your imports / params
# or set N=4 explicitly for quick tests
from lib.hamiltonian import H_B
from lib.time import S

# terminal output of numpy arrays is truncated for large arrays
np.set_printoptions(linewidth=200, precision=3, suppress=True)

D = 20

N = 2

x = np.linspace(0, D, 100)
print("H_B(N) = ", H_B(N))

print("S(x) = ", S(x))

plt.plot(x, S(x))
plt.show()
