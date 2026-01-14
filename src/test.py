from lib.hamiltonian import H_B, H_D, H_P
from lib.schrodinger import yves_TDSE
from lib import X, D
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from joblib import Memory
import itertools
import tqdm
import csv
import pathlib

memory = Memory(location=".joblib_cache", verbose=0)

nsteps = 100
T_0 = 15
T_f = 1500
beta = -1

threshold = 0.9

# Lets find threshold points quickly

for i in range(2, 6):
    mid = (T_f - T_0)/2
    
    if 