import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from lib import X, yvesData
from lib.solvers import secant 
from lib.schrodinger import yves_TDSE
from lib.hamiltonian import H_B, H_P, H_D

n = 2
pen = 10 
beta = -1
T_0 = 15
T_f = 1500
nsteps = 5000
F_thr = 0.8

Hf = H_B(n) + pen * H_P(n)
Hi = beta*H_D(n, X)
ts = np.linspace(T_0, T_f, nsteps)
args = yvesData(Hf=Hf, Hi=Hi, n=n, ts=ts)

time, fid = secant(yves_TDSE, args, f_thr=F_thr)


print(time, fid)
