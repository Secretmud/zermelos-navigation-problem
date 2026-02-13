#%matplotlib widget
from lib.initial_states import initialState
from lib.hamiltonian import H_B, H_D, H_P
from lib.schrodinger import yves_TDSE
from lib.solvers import yield_bisection
from lib import X, D, yvesData
import numpy as np
import pandas as pd
import csv
import pathlib

nsteps = 2000
T_0 = 15
T_f = 1500
beta = -1


runtime = {
    1:  [6],
    5:  [6],
    10: [6],
    15: [6],
    20: [6],
    30: [6],
}

for pen in runtime.keys():
    print(f"Producing data for penalty: {pen}")
    for n in runtime[pen]:
        print(f"Producing for {n=}")
        Hf = H_B(n) + pen * H_P(n)
        Hi = beta*H_D(n, X)
        ts = np.linspace(T_0, T_f, nsteps)
        args = yvesData(Hf=Hf, Hi=Hi, n=n, ts=ts)
        try:
            time = []
            fid = []
            for t, f in yield_bisection(yves_TDSE, args=args):
                print(t, f)
                time.append(t)
                fid.append(f)
    
            f_name = f"{nsteps}_{beta}_{T_0}_{T_f}_{pen}_fids.csv"

            data_dir = pathlib.Path("data/solver/bisection")
            data_dir.mkdir(exist_ok=True, parents=True)
            
            file_path = data_dir / f_name
            
            if file_path.exists() and file_path.stat().st_size > 0:
                df = pd.read_csv(file_path)
            else:
                df = pd.DataFrame(columns=["qutrits", "time", "fid"])
            
            new_df = pd.DataFrame({
                "qutrits": n,
                "time": time,
                "fid": fid
            })
            
            df = pd.concat([df, new_df], ignore_index=True)
            
            df.to_csv(file_path, index=False)

        except ValueError as e:
            print("No root: " + str(e))
