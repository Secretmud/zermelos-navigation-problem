import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg

from lib.hamiltonian import H_P_test, H_B_test
from lib import N
# System parameters


a_0_values = 10**np.linspace(-4, 1, 150)


it = 0

print(H_P_test())

traversal = []

groups = H_P_test()

# Get unique values and the inverse indices
unique_values, inverse_indices = np.unique(groups, return_inverse=True)

# Group the indices by the unique values
grouped_indices = {value: np.where(inverse_indices == idx)[
    0] for idx, value in enumerate(unique_values)}

H = H_B_test()
lowest_values = {value: np.min(H[indices])
                 for value, indices in grouped_indices.items()}

beta = 0.1
values = []
for low in lowest_values.values():
    values.append(low)

P0 = 1
for low in range(len(values)-1):
    P0 *= (1 - np.exp(-2*np.pi*beta/a_0_values *
           np.abs(1/np.sqrt(2))**2/np.abs(values[low] - values[low+1])))


plt.figure(figsize=(10, 6))

plt.semilogx(a_0_values, P0, label='P0', color='blue')
plt.show()
