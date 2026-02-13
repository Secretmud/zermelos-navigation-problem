from lib.hamiltonian import H_B, H_D, H_P
from lib.schrodinger import yves_TDSE
from lib import X, D, yvesData
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

runtime = {
    10: [6]
}

"""
runtime = {
    1: [2, 3, 4, 5, 6],
    5: [2, 3, 4, 5, 6],
    10: [2, 3, 4, 5, 6],
    15: [2, 3, 4, 5, 6],
    20: [2, 3, 4, 5, 6],
    30: [2, 3, 4, 5, 6],
}
"""

for pen in runtime.keys():
    print(f"Producing data for penalty: {pen}")
    for N in runtime[pen]:
        print(f"for n = {N}")
        Hf = H_B(N) + pen * H_P(N)
        Hi = beta*H_D(N, X)
        
        ts = np.linspace(T_0, T_f, nsteps)
        args = yvesData(Hf=Hf, Hi=Hi, n=N, ts=ts)
        
        fidelities = []
        
        psi_f = np.diag(Hf)
        gs_idx = np.argmin(psi_f)
        
        for t in tqdm.tqdm(ts):
            args.t = t
            psi = yves_TDSE(args)
            fidelity = np.abs(psi[gs_idx])**2
            fidelities.append(fidelity)
        
        
        f_name = f"{nsteps}_{beta}_{T_0}_{T_f}_{pen}_fids.csv"
        
        data_dir = pathlib.Path("data")
        data_dir.mkdir(exist_ok=True, parents=True)
        
        file_path = data_dir / f_name
        file_path.touch(exist_ok=True)
        
        fidelities = np.array(fidelities)
        
        data = {N: fidelities.tolist()}
        
        with open(file_path, newline="") as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',')
            for d in data_reader:
                key = int(d[0])
                if key not in data:
                    data[key] = d[1]
        
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            for k, v in data.items():
                writer.writerow([k, v])