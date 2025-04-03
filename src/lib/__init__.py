import numpy as np

Z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
X = 1/np.sqrt(2)*np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
Y = 1/np.sqrt(2)*np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])


L = 20
N = 8
smax = 0.5
dx = L/N
v = 1
