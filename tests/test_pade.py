"""Tests for the Padé-Mie expressions of PRB 94, 140301(R) in physics/quasistatic.py."""

import numpy as np
import pytest

from wavecheck.physics import mie
from wavecheck.physics import quasistatic as qs


@pytest.mark.parametrize("eps", [2.25, -3 + 0.5j, -1.5])
def test_pade_a1_matches_exact_mie_for_small_spheres(eps):
    x = 0.02
    _, a, _ = mie.mie_coefficients(eps, x)
    # |a_1^[3/3]| agrees with |a_1| far beyond the x^3 leading order
    assert abs(qs.pade_a1(eps, x)) == pytest.approx(abs(a[0]), rel=1e-5)


@pytest.mark.parametrize("x", [0.05, 0.2, 0.5])
def test_pade_a1_has_unit_modulus_at_the_pade_resonance(x):
    assert abs(qs.pade_a1(qs.pade_dipole_resonance(x), x)) == pytest.approx(1.0, abs=1e-12)


def test_pade_resonance_series_is_its_taylor_expansion():
    x = 1e-2
    assert qs.pade_dipole_resonance(x) - qs.pade_dipole_resonance_series(x) == pytest.approx(
        -36 / 25 * x**4, rel=1e-2
    )


def test_pade_resonance_rejects_sizes_beyond_its_pole():
    with pytest.raises(ValueError):
        qs.pade_dipole_resonance(1.3)


def _rel_err(value: float, x: float) -> float:
    exact = mie.magnetic_dipole_resonance_eps(x)
    return abs(value - exact) / exact


def test_eq15_converges_to_exact_magnetic_resonance():
    errs = [_rel_err(qs.magnetic_resonance_eq15(x).real, x) for x in (0.4, 0.2, 0.05)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-4


@pytest.mark.parametrize(
    "approx,limit",
    [
        (qs.magnetic_resonance_eq8, 10.5 / np.pi**2 - 1),
        (lambda x: qs.magnetic_resonance_eq13(x).real, (3.1614 / np.pi) ** 2 - 1),
    ],
)
def test_eq8_and_eq13_keep_a_constant_relative_offset_as_x_to_0(approx, limit):
    """Their leading term c/x^2 has c != pi^2, so the relative error tends to c/pi^2 - 1."""
    assert _rel_err(approx(0.01), 0.01) == pytest.approx(limit, rel=1e-2)


def test_eq15_term_selection():
    x = 0.5
    assert qs.magnetic_resonance_eq15(x, terms=1).real == pytest.approx((np.pi / x) ** 2)
    assert qs.magnetic_resonance_eq15(x, terms=2).real == pytest.approx((np.pi / x) ** 2 - 2)
