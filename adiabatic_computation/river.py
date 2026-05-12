from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from adiabatic_computation import smax as _default_smax, D as _default_D, v as _default_v


def _gaussian_current(x: float, smax: float, D: float) -> float:
    """Default river current profile: Gaussian centred at D/π."""
    return smax * np.exp(-1 * (x - D / np.pi) ** 2)


@dataclass
class RiverConfig:
    """Physical parameters and current profile for the Zermelo river problem.

    Parameters
    ----------
    D : float
        River width.
    v : float
        Boat speed (must satisfy v > smax everywhere for traversal to be possible).
    smax : float
        Maximum current amplitude used by the default Gaussian profile.
        Ignored when a custom speed_fn is supplied.
    speed_fn : callable, optional
        A function ``speed_fn(x: float) -> float`` that returns the river
        current at cross-river position *x*.  Defaults to the Gaussian profile
        ``smax * exp(-(x - D/π)²)``.

    Examples
    --------
    Default (Gaussian) current:

        river = RiverConfig()

    Sinusoidal current:

        import numpy as np
        river = RiverConfig(speed_fn=lambda x: 0.9 * np.sin(np.pi * x))

    Uniform current:

        river = RiverConfig(speed_fn=lambda x: 0.5)
    """
    D: float = _default_D
    v: float = _default_v
    smax: float = _default_smax
    speed_fn: Callable[[float], float] = None

    def __post_init__(self):
        if self.speed_fn is None:
            _smax, _D = self.smax, self.D
            self.speed_fn = lambda x: _gaussian_current(x, _smax, _D)

    def S(self, x: float) -> float:
        """Evaluate the current profile at position x."""
        return self.speed_fn(x)


_default_river = RiverConfig()

# Module-level S kept for backward compatibility
def S(x):
    return _default_river.speed_fn(x)


scale = 4  # dy = dx / scale, i.e. dy = 0.25 * D / N


def ntime(k: int, N: int, river: RiverConfig = None) -> np.ndarray:
    """Time costs (up, straight, down) for qutrit k in an N-step grid.

    k is 0-indexed (k = 0 … N-1); the x-coordinate is centred at (k+0.5)*dx.

    Parameters
    ----------
    k : int
        Qutrit index (0-indexed).
    N : int
        Total number of qutrits (grid steps).
    river : RiverConfig, optional
        River parameters and current profile.  Defaults to the standard
        Gaussian profile with the module-level D, v, smax constants.
    """
    if river is None:
        river = _default_river
    dx = river.D / N
    dy = dx / scale
    x_k = (k + 1 / 2) * dx
    current = river.S(x_k)
    t_up = (dx**2 + dy**2) / (
        np.sqrt((dx**2 + dy**2) * river.v**2 - dx**2 * current**2) - dy * current
    )
    t_straight = dx / np.sqrt(river.v**2 - current**2)
    t_down = (dx**2 + dy**2) / (
        np.sqrt((dx**2 + dy**2) * river.v**2 - dx**2 * current**2) + dy * current
    )
    return np.array([t_up, t_straight, t_down])


def time(k: int, direction: int, N: int, river: RiverConfig = None) -> float:
    """Travel time for a single move at step k (0-indexed) in direction {-1, 0, 1}.

    Parameters
    ----------
    k : int
        Qutrit index (0-indexed).
    direction : int
        Move direction: +1 (up), 0 (straight), -1 (down).
    N : int
        Total number of qutrits.
    river : RiverConfig, optional
        River parameters and current profile.  Defaults to the Gaussian profile.
    """
    if river is None:
        river = _default_river
    dx = river.D / N
    dy = dx / scale
    g = dy / dx
    x = dx * k
    current = river.S(x + dx / 2)
    match direction:
        case 0:
            frac = 1 / (1 - current**2)
        case 1:
            frac = (1 + g**2) / (np.sqrt(1 + g**2 - current**2) - g * current)
        case -1:
            frac = (1 + g**2) / (np.sqrt(1 + g**2 - current**2) + g * current)
        case _:
            raise ValueError(f"Invalid direction {direction!r}: must be -1, 0, or 1")
    return dx / river.v * frac
