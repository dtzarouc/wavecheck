"""Physical limits and consistency checks used as cheap monitors.

- Single-channel limit: one multipole channel of order n (electric or magnetic) of a passive
  sphere scatters at most Q_sca,n = 2 (2n + 1) / x^2. For the electric dipole (n = 1) this is
  6/x^2, i.e. sigma_max = 3 lambda^2 / (2 pi).
- Passivity per channel: Re(c) >= |c|^2 for every Mie coefficient c (no gain in any channel).
- Energy conservation: Q_ext = Q_sca + Q_abs, with Q_abs >= 0 and Q_abs = 0 for lossless spheres.

All inputs are dimensionless (x = k a, efficiencies Q = sigma / (pi a^2)).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def single_channel_limit(x: float, n: int = 1) -> float:
    """Maximum scattering efficiency of one multipole channel of order n: 2 (2n + 1) / x^2."""
    if n < 1:
        raise ValueError("multipole order n must be >= 1")
    return 2.0 * (2 * n + 1) / x**2


def dipole_limit(x: float) -> float:
    """Electric-dipole single-channel limit, 6 / x^2."""
    return single_channel_limit(x, n=1)


def violates_limit(q_sca: float, x: float, n: int = 1, rtol: float = 1e-9) -> bool:
    """True if q_sca exceeds the single-channel limit of order n by more than rtol."""
    return bool(q_sca > single_channel_limit(x, n) * (1.0 + rtol))


def passivity_residuals(coefficients: ArrayLike) -> np.ndarray:
    """Re(c) - |c|^2 for each Mie coefficient; every entry must be >= 0 for a passive sphere."""
    c = np.asarray(coefficients, dtype=complex)
    return c.real - np.abs(c) ** 2


def energy_residual(q_ext: float, q_sca: float, q_abs: float) -> float:
    """Q_ext - Q_sca - Q_abs (zero up to rounding when the bookkeeping is right)."""
    return float(q_ext - q_sca - q_abs)
