"""Resonance and critical-coupling helpers of physics/mie.py, plus a lossy cross-check."""

import numpy as np
import pytest

from wavecheck.physics import mie

miepython = pytest.importorskip("miepython")


@pytest.mark.parametrize("eps", [-3 + 0.5j, -2.2 + 0.1j, 4 + 1j, -8 + 2j, 12 + 0.3j])
@pytest.mark.parametrize("x", [0.1, 1.0, 3.0])
def test_matches_miepython_for_absorbing_and_plasmonic_spheres(eps, x):
    """miepython uses m = n - i kappa, so we pass conj(m) to compare with our Im(m) >= 0."""
    m = np.sqrt(complex(eps))
    q_ext_ref, q_sca_ref, _, _ = miepython.efficiencies_mx(np.conj(m), x)
    q = mie.efficiencies(eps, x)
    assert q.q_ext == pytest.approx(float(q_ext_ref), rel=1e-6)
    assert q.q_sca == pytest.approx(float(q_sca_ref), rel=1e-6)


@pytest.mark.parametrize("x", [0.1, 0.4, 0.8])
def test_magnetic_resonance_has_unit_coefficient(x):
    _, _, b = mie.mie_coefficients(mie.magnetic_dipole_resonance_eps(x), x)
    assert abs(b[0]) == pytest.approx(1.0, abs=1e-8)


def test_magnetic_resonance_tends_to_the_first_zero_of_j0():
    """Small x: m x -> pi, i.e. eps x^2 -> pi^2."""
    x = 0.02
    assert mie.magnetic_dipole_resonance_eps(x) * x**2 == pytest.approx(np.pi**2, rel=1e-3)


@pytest.mark.parametrize(
    "channel,guess_fn",
    [("E", lambda x: -2 + 2j * x**3), ("M", lambda x: (np.pi / x) ** 2 - 2 + 2j * x)],
)
def test_max_absorption_is_critical_coupling(channel, guess_fn):
    x = 0.3
    eps = mie.max_absorption_eps(x, channel, guess_fn(x))
    _, a, b = mie.mie_coefficients(eps, x)
    c = a[0] if channel == "E" else b[0]
    assert abs(c - 0.5) < 1e-9
    assert eps.imag > 0  # passive


def test_electric_max_absorption_loss_tends_to_2x3():
    """PRB 94, 140301(R), Eq. (12): eps'' ~ 2 x^3 for the electric dipole."""
    x = 0.02
    eps = mie.max_absorption_eps(x, "E", -2 + 2j * x**3)
    assert eps.imag == pytest.approx(2 * x**3, rel=1e-2)


@pytest.mark.parametrize("x", [0.2, 0.6, 1.0])
def test_magnetic_pole_is_the_conjugate_of_critical_coupling(x):
    """For lossless-analytic c = 1/(1 - i R(eps)) with R real on the real axis, 1/c = 0 and
    1/c = 2 are mirror images: eps_pole = conj(eps_critical)."""
    guess = (np.pi / x) ** 2 - 2 - 2j * x
    p = mie.pole_eps(x, "M", guess)
    c = mie.max_absorption_eps(x, "M", np.conj(guess))
    assert p.imag < 0
    assert p == pytest.approx(np.conj(c), rel=1e-9)
