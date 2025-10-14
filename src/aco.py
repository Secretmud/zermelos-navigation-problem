import numpy as np
import matplotlib.pyplot as plt
import time as pytime  # to control animation speed

# --- Parameters ---
D = 20
L = 10
smax = 0.9
v = 1.0
N = 15
dy_levels = 5 * N
num_ants = 20
num_iterations = 20
penalty_factor = 1e3

dx = D / N
dy_step = 2*L / (dy_levels-1)
y_values = np.linspace(-L, L, dy_levels)
directions = [-1, 0, 1]

alpha = 0.5
beta = 3.0
rho = 0.1
Q = 1.0
explore_prob = 0.1

# --- Functions ---


def S(x, offset=0.15):
    return (smax - offset) * np.sin(np.pi * x / D) + offset


def time_edge(x, y, direction):
    g = dy_step / dx
    next_y = y + direction*dy_step
    current = S(x + dx/2)
    match direction:
        case 0:
            denominator = 1 - current**2
        case 1:
            denominator = np.sqrt(1 + g**2 - current**2) - g * current
        case -1:
            denominator = np.sqrt(1 + g**2 - current**2) + g * current
        case _:
            raise ValueError("Invalid direction")
    return dx / v * (1 + g**2) / denominator, next_y


# --- Build graph ---
graph = {}
for k in range(N):
    for yi, y in enumerate(y_values):
        edges = []
        for d in directions:
            next_y = y + d*dy_step
            if -L <= next_y <= L:
                t, next_y_val = time_edge(dx*k, y, d)
                next_y_idx = np.argmin(np.abs(y_values - next_y_val))
                edges.append((k+1, next_y_idx, t))
        graph[(k, yi)] = edges

# Initialize pheromones and heuristics
pheromone = {(node, (e[0], e[1])): 1.0 for node,
             edges in graph.items() for e in edges}
heuristic = {(node, (e[0], e[1])): 1.0 / e[2]
             for node, edges in graph.items() for e in edges}

# --- Dummy straight path ---
dummy_path = [(0, dy_levels//2)]
total_time_dummy = 0
for k in range(N):
    node = dummy_path[-1]
    edges = graph[node]
    next_node = None
    for e in edges:
        if e[1] == node[1]:
            next_node = (e[0], e[1])
            total_time_dummy += e[2]
            break
    if next_node is None:
        next_node = edges[0]
        total_time_dummy += edges[0][2]
    dummy_path.append(next_node)
final_y = y_values[dummy_path[-1][1]]
total_time_dummy += penalty_factor * abs(final_y)

# Boost pheromones along dummy path
for i in range(len(dummy_path)-1):
    edge = (dummy_path[i], dummy_path[i+1])
    pheromone[edge] += Q / total_time_dummy

heights_dummy = [y_values[n[1]] for n in dummy_path]
print("Dummy straight path total time:", total_time_dummy)

# --- Ant path function ---


def run_ant():
    path = [(0, dy_levels//2)]
    total_time = 0
    while path[-1][0] < N:
        node = path[-1]
        edges = graph[node]
        pher_vals = np.array([pheromone[(node, (e[0], e[1]))] for e in edges])
        heur_vals = np.array([heuristic[(node, (e[0], e[1]))] for e in edges])
        probs = (pher_vals ** alpha) * (heur_vals ** beta)
        probs /= probs.sum()
        if np.random.rand() < explore_prob:
            choice = np.random.choice(len(edges))
        else:
            choice = np.random.choice(len(edges), p=probs)
        next_node = (edges[choice][0], edges[choice][1])
        total_time += edges[choice][2]
        path.append(next_node)
    final_y = y_values[path[-1][1]]
    total_time += penalty_factor * abs(final_y)
    return path, total_time


# --- Live animation setup ---
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, D)
ax.set_ylim(-L-1, L+1)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("ACO Step-by-Step Animation")
ax.grid(True)

best_path = None
best_time = np.inf
fade_alpha = 0.3
x_positions = np.linspace(0, D, N+1)

for iteration in range(num_iterations):
    ants_results = [run_ant() for _ in range(num_ants)]

    # Evaporate pheromones
    for key in pheromone:
        pheromone[key] *= (1 - rho)

    # Find best
    iter_best_time = min(t for _, t in ants_results)
    iter_best_path = [path for path,
                      t in ants_results if t == iter_best_time][0]
    if iter_best_time < best_time:
        best_time = iter_best_time
        best_path = iter_best_path

    # Deposit pheromones along best
    for i in range(len(best_path)-1):
        edge = (best_path[i], best_path[i+1])
        pheromone[edge] += Q / best_time

    # --- Animate ants step-by-step ---
    for step in range(N+1):
        ax.clear()
        ax.set_xlim(0, D)
        ax.set_ylim(-L-1, L+1)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title(f"ACO Animation Iter {iteration+1} Step {step}")
        ax.grid(True)

        # Dummy path
        ax.plot(x_positions, heights_dummy, color='green',
                lw=2, linestyle='--', label='Dummy Path')

        # Best path so far
        heights_best = [y_values[n[1]] for n in best_path[:step+1]]
        ax.plot(x_positions[:step+1], heights_best, color='red',
                lw=2, label=f"Best Time: {best_time:.2f}")

        # All ants
        for path, _ in ants_results:
            heights = [y_values[n[1]] for n in path[:step+1]]
            ax.plot(x_positions[:step+1], heights,
                    color='blue', alpha=fade_alpha)

        ax.legend()
        plt.pause(0.05)  # speed of animation

plt.ioff()
plt.show()
print("Final best total time:", best_time)

