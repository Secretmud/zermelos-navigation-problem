import numpy as np
from adiabatic_computation import smax, D, v


def S(x):
    return smax * np.exp(-1 * (x - (D / np.pi))**2)


scale = 4  # dy = dx / scale, i.e. dy = 0.25 * D / N


def ntime(k, N):
    """Time costs (up, straight, down) for qutrit k in an N-step grid.

    k is 0-indexed (k = 0 … N-1); the x-coordinate is centred at (k+0.5)*dx.
    """
    dx = D / N
    dy = dx / scale
    x_k = (k + 1/2) * dx
    current = S(x_k)
    t_up = (dx**2 + dy**2) / (np.sqrt((dx**2 + dy**2)*v**2 - dx**2*current**2) - dy*current)
    t_straight = dx / np.sqrt(v**2 - current**2)
    t_down = (dx**2 + dy**2) / (np.sqrt((dx**2 + dy**2)*v**2 - dx**2*current**2) + dy*current)
    return np.array([t_up, t_straight, t_down])


def time(k, direction, N):
    """Travel time for a single move at step k (0-indexed) in direction {-1, 0, 1}."""
    dx = D / N
    dy = dx / scale
    g = dy / dx
    x = dx * k
    current = S(x + dx / 2)
    match direction:
        case 0:
            frac = 1 / (1 - current**2)
        case 1:
            frac = (1 + g**2) / (np.sqrt(1 + g**2 - current**2) - g * current)
        case -1:
            frac = (1 + g**2) / (np.sqrt(1 + g**2 - current**2) + g * current)
        case _:
            raise ValueError(f"Invalid direction {direction!r}: must be -1, 0, or 1")
    return dx / v * frac
