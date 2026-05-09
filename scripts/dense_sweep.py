"""
dense_sweep.py
--------------
Evaluates the Yves TDSE at a dense grid of anneal times for each
(n, penalty) combination and writes fidelity curves to CSV.

Each CSV file is keyed by T_pen count so that multiple n values can
accumulate into the same file across runs.

Run from the project root:
  python scripts/dense_sweep.py
"""

import csv
import pathlib

import numpy as np
import tqdm

from adiabatic_computation import H_B, H_P, H_D, X, YvesConfig, yves_TDSE

# ── Parameters ────────────────────────────────────────────────────────────────
T_0  = 15
T_f  = 1500
beta = -1

runtime = {
    1:   [2, 3, 4, 5, 6, 7],
    10:  [2, 3, 4, 5, 6, 7],
    20:  [2, 3, 4, 5, 6, 7],
    40:  [2, 3, 4, 5, 6, 7],
    80:  [2, 3, 4, 5, 6, 7],
    120: [2, 3, 4, 5, 6, 7],
}

data_dir = pathlib.Path("src/data/final2/fidelities")
data_dir.mkdir(exist_ok=True, parents=True)

# ── Main loop ─────────────────────────────────────────────────────────────────
for pen, ns in runtime.items():
    print(f"Producing data for penalty: {pen}")
    for n in ns:
        if n < 4:
            nsteps = 800
        elif n < 6:
            nsteps = 100
        else:
            nsteps = 10

        print(f"  n={n}, nsteps={nsteps}")
        n_pen = pen / n
        H_f = H_B(n) + n_pen * H_P(n)
        H_i = beta * H_D(n, X)
        gs  = np.argmin(np.diagonal(H_f))

        ts = np.linspace(T_0, T_f, nsteps)
        fidelities = []

        for t in tqdm.tqdm(ts, leave=False):
            config = YvesConfig(H_f=H_f, H_i=H_i, n=n, t=t)
            psi    = yves_TDSE(config)
            fidelities.append(float(np.abs(psi[gs]) ** 2))

        f_name    = f"{nsteps}_{beta}_{T_0}_{T_f}_{pen}_fids.csv"
        file_path = data_dir / f_name
        file_path.touch(exist_ok=True)

        # Merge with any existing data in the CSV
        data = {n: fidelities}
        with open(file_path, newline="") as csvfile:
            for row in csv.reader(csvfile, delimiter=","):
                key = int(row[0])
                if key not in data:
                    data[key] = row[1]

        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            for k, v in data.items():
                writer.writerow([k, v])

print(f"\nData written to {data_dir}")
