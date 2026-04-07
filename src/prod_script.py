from lib.hamiltonian import H_B, H_D, H_P
from lib.schrodinger import yves_TDSE
from lib import X, yvesData
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from joblib import Memory
import itertools
import tqdm
import csv
import pathlib

memory = Memory(location=".joblib_cache", verbose=0)

nsteps = 10
T_0 = 15
T_f = 1500
beta = -1

"""
runtime = {
    1:  [2, 3, 4, 5, 6, 7],
    5:  [2, 3, 4, 5, 6, 7],
    10: [2, 3, 4, 5, 6, 7],
    15: [2, 3, 4, 5, 6, 7],
    20: [2, 3, 4, 5, 6, 7],
    30: [2, 3, 4, 5, 6, 7],
}
"""
runtime = {
    1:  [2, 3, 4, 5, 6, 7],
    10:  [2, 3, 4, 5, 6, 7],
    20: [2, 3, 4, 5, 6, 7],
    40: [2, 3, 4, 5, 6, 7],
    80: [2, 3, 4, 5, 6, 7],
    120: [2, 3, 4, 5, 6, 7],
}
for pen in runtime.keys():
    print(f"Producing data for penalty: {pen}")
    for n in runtime[pen]:
        if n < 4:
            nsteps = 800
        elif 4 <= n < 6:
            nsteps = 100
        else:
            nsteps = 10
        print(f"for n = {n}")
        n_pen = pen / n
        Hf = H_B(n) + n_pen * H_P(n)
        Hi = beta * H_D(n, X)

        ts = np.linspace(T_0, T_f, nsteps)
        args = yvesData(Hf=Hf, Hi=Hi, n=n, ts=ts)

        fidelities = []

        for t in tqdm.tqdm(ts):
            args.t = t
            psi = yves_TDSE(args)
            idx = np.argmax(np.abs(psi)**2)
            fidelity = np.abs(psi[idx])**2
            fidelities.append(fidelity)

        f_name = f"{nsteps}_{beta}_{T_0}_{T_f}_{pen}_fids.csv"

        data_dir = pathlib.Path("data/final2/fidelities")
        data_dir.mkdir(exist_ok=True, parents=True)

        file_path = data_dir / f_name
        file_path.touch(exist_ok=True)

        fidelities = np.array(fidelities)

        data = {n: fidelities.tolist()}

        with open(file_path, newline="") as csvfile:
            data_reader = csv.reader(csvfile, delimiter=",")
            for d in data_reader:
                key = int(d[0])
                if key not in data:
                    data[key] = d[1]

        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            for k, v in data.items():
                writer.writerow([k, v])
