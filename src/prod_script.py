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


runtime = {
    1: [2, 3, 4, 5],
    5: [2, 3, 4, 5],
    10: [2, 3, 4, 5],
    15: [2, 3, 4, 5],
    20: [2, 3, 4, 5],
    30: [2, 3, 4, 5],
}


for pen in runtime.keys():
    print(f"Producing data for penalty: {pen}")
    for N in runtime[pen]:
        print(f"for n = {N}")
        Hf = H_B(N) + pen * H_P(N)
        Hi = beta*H_D(N, X)
        
        psi_f = np.diag(Hf)
        
        P = np.array([1/2, 1/np.sqrt(2), 1/2], dtype="complex")
        initialState = P.copy()
        for _ in range(N-1):
            initialState = np.kron(initialState, P)
        
        ts = np.linspace(T_0, T_f, nsteps)
        f_all = False
        
        
        def check_degrenecy(arr):
            min_value = np.min(arr)
            mask = (arr == min_value)
            return np.count_nonzero(mask)
        c = check_degrenecy(psi_f)
        
        if c > 1:
            print(f"Solution is degenerate, there are {c} occurences")
            idx = np.where(psi_f == np.min(psi_f))[0]
            print(f"They are located at indecies {idx}")
        fidelities = []
        gs_idx = np.argmin(psi_f)
        plot_all = False
        
        for t in tqdm.tqdm(ts):
            psi = yves_TDSE(initialState, Hi, Hf, t)
            if plot_all:
                fidelities.append(np.abs(psi))
            else:
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