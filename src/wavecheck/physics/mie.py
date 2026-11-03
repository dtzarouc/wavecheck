"""Exact Mie scattering by a homogeneous, non-magnetic sphere in vacuum.

Conventions (Bohren & Huffman, Absorption and Scattering of Light by Small Particles, ch. 4)
-------------------------------------------------------------------------------------------
- Time dependence exp(-i w t); passive media have Im(eps) >= 0.
- Relative refractive index m = sqrt(eps), principal branch with Im(m) >= 0.
- Size parameter x = k a = 2 pi a / lambda. Efficiencies Q = sigma / (pi a^2).
- Riccati-Bessel functions: psi_n(z) = z j_n(z), xi_n(z) = z h_n^(1)(z) = z [j_n(z) + i y_n(z)].

    a_n = [m psi_n(mx) psi_n'(x) - psi_n(x) psi_n'(mx)]
          / [m psi_n(mx) xi_n'(x) - xi_n(x) psi_n'(mx)]
    b_n = [psi_n(mx) psi_n'(x) - m psi_n(x) psi_n'(mx)]
          / [psi_n(mx) xi_n'(x) - m xi_n(x) psi_n'(mx)]

    Q_sca = (2 / x^2) sum_n (2n + 1) (|a_n|^2 + |b_n|^2)
    Q_ext = (2 / x^2) sum_n (2n + 1) Re(a_n + b_n)
    Q_abs = Q_ext - Q_sca

Numerically we use the equivalent logarithmic-derivative form (B&H eq. 4.88),
D_n(z) = psi_n'(z) / psi_n(z), with D_n obtained by downward recurrence
D_{n-1} = n/z - 1/(D_n + n/z). This avoids overflow of j_n(mx) for strongly absorbing or
plasmonic spheres, where |Im(mx)| is large.

    a_n = [(D_n(mx)/m + n/x) psi_n(x) - psi_{n-1}(x)] / [(D_n(mx)/m + n/x) xi_n(x) - xi_{n-1}(x)]
    b_n = [(m D_n(mx) + n/x) psi_n(x) - psi_{n-1}(x)] / [(m D_n(mx) + n/x) xi_n(x) - xi_{n-1}(x)]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special


@dataclass(frozen=True)
class Efficiencies:
    q_ext: float
    q_sca: float
    q_abs: float


def n_max_wiscombe(x: float) -> int:
    """Number of multipole orders for convergence (Wiscombe 1980): ceil(x + 4 x^(1/3) + 2)."""
    return int(np.ceil(x + 4.0 * x ** (1.0 / 3.0) + 2.0))


def _refractive_index(eps: complex) -> complex:
    """m = sqrt(eps) on the branch Im(m) >= 0 (guards against a -0.0 imaginary part)."""
    m = complex(np.sqrt(complex(eps)))
    return -m if m.imag < 0 else m


def _log_derivative(z: complex, n_max: int) -> np.ndarray:
    """D_n(z) = psi_n'(z)/psi_n(z) for n = 0..n_max, by downward recurrence from D = 0."""
    n_start = int(max(n_max, abs(z)) + 16)
    d = 0j
    out = np.empty(n_max + 1, dtype=complex)
    for n in range(n_start, 0, -1):
        d = n / z - 1.0 / (d + n / z)  # this is D_{n-1}
        if n - 1 <= n_max:
            out[n - 1] = d
    return out


def mie_coefficients(
    eps: complex, x: float, n_max: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (n, a_n, b_n) for n = 1..n_max (default: n_max_wiscombe(x)).

    n is an int array; a_n and b_n are complex arrays of the same length.
    """
    if x <= 0:
        raise ValueError("size parameter x must be positive")
    n_max = n_max_wiscombe(x) if n_max is None else int(n_max)
    m = _refractive_index(eps)
    order = np.arange(0, n_max + 1)
    # Riccati-Bessel functions of the real argument x, orders 0..n_max.
    psi = x * special.spherical_jn(order, x)
    xi = psi + 1j * x * special.spherical_yn(order, x)
    d = _log_derivative(m * x, n_max)
    n = order[1:]
    ta = d[1:] / m + n / x
    tb = m * d[1:] + n / x
    a = (ta * psi[1:] - psi[:-1]) / (ta * xi[1:] - xi[:-1])
    b = (tb * psi[1:] - psi[:-1]) / (tb * xi[1:] - xi[:-1])
    return n, a, b


def efficiencies(eps: complex, x: float, n_max: int | None = None) -> Efficiencies:
    """Extinction, scattering and absorption efficiencies of the sphere."""
    n, a, b = mie_coefficients(eps, x, n_max)
    w = 2.0 * (2 * n + 1) / x**2
    q_ext = float(np.sum(w * (a.real + b.real)))
    q_sca = float(np.sum(w * (np.abs(a) ** 2 + np.abs(b) ** 2)))
    return Efficiencies(q_ext=q_ext, q_sca=q_sca, q_abs=q_ext - q_sca)


def channel_efficiencies(
    eps: complex, x: float, n_max: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel scattering efficiencies (electric, magnetic) for n = 1..n_max.

    Q_sca,n^E = 2 (2n + 1) |a_n|^2 / x^2 and Q_sca,n^M = 2 (2n + 1) |b_n|^2 / x^2.
    """
    n, a, b = mie_coefficients(eps, x, n_max)
    w = 2.0 * (2 * n + 1) / x**2
    return w * np.abs(a) ** 2, w * np.abs(b) ** 2


def _first_coefficient(eps: complex, x: float, channel: str) -> complex:
    """a_1 (channel 'E') or b_1 (channel 'M')."""
    _, a, b = mie_coefficients(eps, x, n_max=1)
    if channel == "E":
        return complex(a[0])
    if channel == "M":
        return complex(b[0])
    raise ValueError("channel must be 'E' or 'M'")


def dipole_resonance_eps(x: float, bracket: tuple[float, float] = (-6.0, -1.05)) -> float:
    """Exact electric-dipole resonance permittivity of a lossless sphere of size x.

    For real eps, 1/a_1 = 1 - i M_1/N_1 with M_1/N_1 real, so the resonance (|a_1| = 1) is the
    root of Im(1/a_1) = 0. Use scipy.optimize.brentq on the bracket. Im(1/a_1) also changes sign
    through a pole where a_1 = 0 (near eps = 1 for small x), so the default bracket stops at -1.05.
    """

    def f(e: float) -> float:
        return (1.0 / _first_coefficient(e, x, "E")).imag

    return float(optimize.brentq(f, *bracket, xtol=1e-14, rtol=1e-14, maxiter=500))


def magnetic_dipole_resonance_eps(x: float, n_grid: int = 4000) -> float:
    """Exact first magnetic-dipole resonance permittivity (|b_1| = 1) of a lossless sphere.

    As for a_1, Im(1/b_1) = 0 at a resonance, but it also changes sign through poles where b_1 = 0.
    We scan eps on a geometric grid from 1 to 2 (pi/x)^2 + 20 (the first resonance sits near
    -2 + (pi/x)^2 for small x), refine each sign change with brentq and return the lowest root
    with |b_1| = 1.
    """
    grid = np.geomspace(1.0 + 1e-6, 2.0 * (np.pi / x) ** 2 + 20.0, n_grid)

    def f(e: float) -> float:
        return (1.0 / _first_coefficient(e, x, "M")).imag

    values = np.array([f(e) for e in grid])
    for i in np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:])):
        root = optimize.brentq(f, grid[i], grid[i + 1], xtol=1e-13, rtol=1e-14, maxiter=500)
        if abs(abs(_first_coefficient(root, x, "M")) - 1.0) < 1e-6:
            return float(root)
    raise RuntimeError(f"no magnetic dipole resonance found for x = {x}")


def _solve_inverse_coefficient(x: float, channel: str, target: complex, guess: complex) -> complex:
    """Complex eps with 1/c(eps) = target, c = a_1 ('E') or b_1 ('M'); Newton-type root from guess.

    1/c varies smoothly with eps, whereas c itself is sharply resonant, so solving for 1/c is far
    better conditioned. Convergence is judged by the residual, because scipy's hybr may report
    "slow progress" once it is already at machine precision.
    """

    def residual(v: np.ndarray) -> list[float]:
        r = 1.0 / _first_coefficient(complex(v[0], v[1]), x, channel) - target
        return [r.real, r.imag]

    sol = optimize.root(residual, [guess.real, guess.imag], method="hybr", tol=1e-13)
    eps = complex(sol.x[0], sol.x[1])
    if abs(1.0 / _first_coefficient(eps, x, channel) - target) > 1e-9:
        raise RuntimeError(f"root 1/c = {target} not converged for x = {x}, channel {channel}")
    return eps


def max_absorption_eps(x: float, channel: str, guess: complex) -> complex:
    """Complex permittivity at which one dipole channel absorbs the most (critical coupling).

    The channel absorption efficiency is (6/x^2) (Re c - |c|^2) with c = a_1 or b_1. It is
    maximal, equal to (6/x^2)/4, exactly when c = 1/2, i.e. 1/c = 2. `guess` should be an
    approximate formula, e.g. from PRB 94, 140301(R).
    """
    return _solve_inverse_coefficient(x, channel, 2.0, guess)


def pole_eps(x: float, channel: str, guess: complex) -> complex:
    """Complex permittivity of the pole of a_1 ('E') or b_1 ('M'): 1/c(eps) = 0.

    With exp(-i w t) the pole lies in the active half-plane, Im(eps) < 0. Its real part is the
    natural "exact" counterpart of the complex Padé resonance conditions of PRB 94, 140301(R)
    (Eqs. 11, 13, 15), which are themselves poles of Padé approximants.
    """
    return _solve_inverse_coefficient(x, channel, 0.0, guess)
