import numpy as np
import joblib
import itertools
import pandas as pd
import matplotlib.pyplot as plt

from lib.hamiltonian import H_B, H_P, H_D
from lib import X
memory = joblib.Memory(location=".joblib_cache", verbose=0)


@memory.cache
def all_move_sequences(N):
    moves = (1, 0, -1)
    return list(itertools.product(moves, repeat=N))

def line_intersection(Ec, Pc, Ej, Pj):
    denom = (Pc - Pj)
    if denom == 0:
        return np.inf
    return (Ej - Ec) / denom

def next_ground_state(current_idx, states):
    Ec, Pc = states[current_idx][1], states[current_idx][2]
    best_alpha = np.inf
    best_idx = None
    for j, Ej, Pj in states:
        if j == current_idx:
            continue
        a = line_intersection(Ec, Pc, Ej, Pj)
        if a > 0 and Pc > Pj and a < best_alpha:
            best_alpha = a
            best_idx = j
    return best_alpha, best_idx

def traverse_until_zero(states):
    path = []
    current = min(states, key=lambda t: t[1])[0]
    alpha_pos = 0.0
    while True:
        i, Ei, Pi = states[current]
        path.append((alpha_pos, i, Ei, Pi))
        if Pi == 0.0:
            break
        a_next, next_idx = next_ground_state(current, states)
        if next_idx is None or not np.isfinite(a_next):
            break
        alpha_pos = a_next
        current = next_idx
    return path


"""
excel_file = "ground_state_progression.xlsx"
with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
    for N in range(2, 9):
        Hb = H_B(N)
        Hp = H_P(N)
        Hd = H_D(N, X)

        energies = np.diag(Hb).astype(float)
        slopes = np.diag(Hp).astype(float)
        states = [(i, energies[i], slopes[i]) for i in range(len(energies))]

        path = traverse_until_zero(states)
        path_states = [idx for _, idx, _, _ in path]

        rows = []
        for k, (alpha, idx, E, P) in enumerate(path):
            next_alpha = path[k+1][0] if k+1 < len(path) else None
            next_state = path_states[k+1] if k+1 < len(path_states) else None
            coupling = Hd[idx, next_state] if next_state is not None else None
            move_seq = all_move_sequences(N)[idx]
            rows.append({
                "N": N,
                "Step": k,
                "Alpha": alpha,
                "StateIndex": idx,
                "Energy": E,
                "Slope(Penalty)": P,
                "NextAlpha": next_alpha,
                "NextStateIndex": next_state,
                "CouplingToNext": coupling,
                "MoveSequence": move_seq
            })

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name=f"N={N}", index=False)

print(f"Wrote results to {excel_file}")
"""

N = 4
Hb = H_B(N)
Hp = H_P(N)
Hd = H_D(N, X)

energies = np.diag(Hb).astype(float)
slopes = np.diag(Hp).astype(float)
states = [(i, energies[i], slopes[i]) for i in range(len(energies))]

path = traverse_until_zero(states)
path_states = [idx for _, idx, _, _ in path]
state_tuples = [(path_states[i], path_states[i+1]) for i in range(len(path_states)-1)]

plt.figure(figsize=(8, 6))
plt.imshow(Hd, cmap='viridis', aspect='auto')
plt.colorbar(label="Hd value")
plt.title("Hamiltonian Hd with Ground-State Transitions")

if state_tuples:
    xs, ys = zip(*state_tuples)
    plt.scatter(xs, ys, label="Intersections", c='red', zorder=3)

    # Arrow between successive intersection points
    for (x0, y0), (x1, y1) in zip(state_tuples[:-1], state_tuples[1:]):
        plt.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="white", lw=1.2),
        )

plt.legend()
plt.tight_layout()
plt.show()