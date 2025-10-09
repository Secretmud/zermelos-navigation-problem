import numpy as np
import pandas as pd
from lib.hamiltonian import H_B, H_P, H_D
from lib import X

# Numpy print options
np.set_printoptions(precision=3, suppress=True)

Hd_coeff = 0.1


def prioritize(data):
    indexed = [(e, p, i) for i, (e, p) in enumerate(data)]
    indexed.sort(key=lambda x: x[0])
    result = []
    seen_p = set()
    zero_added = False

    for energy, p, idx in indexed:
        if p == 0.0:
            if not zero_added:
                result.append(idx)   # take the first p=0
                zero_added = True
        else:
            if p not in seen_p:
                result.append(idx)
                seen_p.add(p)
    return result


# Excel writer
with pd.ExcelWriter("hamiltonian_results.xlsx", engine="openpyxl") as writer:
    for N in range(2, 4):  # choose the values of N you want
        Hb = H_B(N)
        Hp = H_P(N)
        Hd = Hd_coeff * H_D(N, X)

        energies = np.diag(Hb)
        hp_diag = np.diag(Hp)

        flat_H_B = Hb.flatten()
        flat_H_P = Hp.flatten()

        data = [(flat_H_B[i], flat_H_P[i]) for i in range(len(flat_H_B))]

        selected_indices = prioritize(data)
        n = Hb.shape[0]
        del Hb, Hp  # free memory

        traversal_path = [idx // n for idx in selected_indices]

        # Collect step info
        rows = []
        for i in range(len(traversal_path)-1):
            a = traversal_path[i]
            b = traversal_path[i+1]
            rows.append({
                "Step": i+1,
                "From state": a,
                "To state": b,
                "Energy (from)": energies[a],
                "Energy (to)": energies[b],
                "Penalty (from)": hp_diag[a],
                "Penalty (to)": hp_diag[b],
                "Hd coupling": Hd[a, b]
            })
        del Hd  # free memory
        df = pd.DataFrame(rows)

        # Write to sheet
        sheet_name = f"N={N}"
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print("Saved results to hamiltonian_results.xlsx")
