"""Exact Mie scattering by a homogeneous, non-magnetic sphere in vacuum.

Conventions (Bohren & Huffman, Absorption and Scattering of Light by Small Particles, ch. 4)
-------------------------------------------------------------------------------------------
- Time dependence exp(-i w t); passive media have Im(eps) >= 0.
- Relative refractive index m = sqrt(eps), principal branch with Im(m) >= 0.
- Size parameter x = k a = 2 pi a / lambda. Efficiencies Q = sigma / (pi a^2).
- Riccati-Bessel functions: psi_n(z) = z j_n(z), xi_n(z) = z h_n^(1)(z) = z [j_n(z) + i y_n(z)].
  scipy.special.spherical_jn / spherical_yn accept complex z and derivative=True.

    a_n = [m psi_n(mx) psi_n'(x) - psi_n(x) psi_n'(mx)]
          / [m psi_n(mx) xi_n'(x) - xi_n(x) psi_n'(mx)]
    b_n = [psi_n(mx) psi_n'(x) - m psi_n(x) psi_n'(mx)]
          / [psi_n(mx) xi_n'(x) - m xi_n(x) psi_n'(mx)]

    Q_sca = (2 / x^2) sum_n (2n + 1) (|a_n|^2 + |b_n|^2)
    Q_ext = (2 / x^2) sum_n (2n + 1) Re(a_n + b_n)
    Q_abs = Q_ext - Q_sca

TODO(Dimi): implement the functions below; tests/test_mie.py defines what "done" means.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Efficiencies:
    q_ext: float
    q_sca: float
    q_abs: float


def n_max_wiscombe(x: float) -> int:
    """Number of multipole orders for convergence (Wiscombe 1980): ceil(x + 4 x^(1/3) + 2)."""
    raise NotImplementedError


def mie_coefficients(
    eps: complex, x: float, n_max: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (n, a_n, b_n) for n = 1..n_max (default: n_max_wiscombe(x)).

    n is an int array; a_n and b_n are complex arrays of the same length.
    """
    raise NotImplementedError


def efficiencies(eps: complex, x: float, n_max: int | None = None) -> Efficiencies:
    """Extinction, scattering and absorption efficiencies of the sphere."""
    raise NotImplementedError


def channel_efficiencies(
    eps: complex, x: float, n_max: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel scattering efficiencies (electric, magnetic) for n = 1..n_max.

    Q_sca,n^E = 2 (2n + 1) |a_n|^2 / x^2 and Q_sca,n^M = 2 (2n + 1) |b_n|^2 / x^2.
    """
    raise NotImplementedError


def dipole_resonance_eps(x: float, bracket: tuple[float, float] = (-6.0, -1.05)) -> float:
    """Exact electric-dipole resonance permittivity of a lossless sphere of size x.

    For real eps, 1/a_1 = 1 - i M_1/N_1 with M_1/N_1 real, so the resonance (|a_1| = 1) is the
    root of Im(1/a_1) = 0. Use scipy.optimize.brentq on the bracket. Im(1/a_1) also changes sign
    through a pole where a_1 = 0 (near eps = 1 for small x), so the default bracket stops at -1.05.
    """
    raise NotImplementedError
