import numpy as np
from dataclasses import dataclass


@dataclass
class yvesData():
    Hi: np.ndarray
    Hf: np.ndarray 
    n: np.int
    t: np.float = None
    ts: np.array = None
    gs_idx: np.int = None

Z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
X = 1/np.sqrt(2)*np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

D = 1
N = 2
smax = 0.9
v = 1
beta = 0.1
alpha = 0.5
hbar = 1

float_type = "float32"

