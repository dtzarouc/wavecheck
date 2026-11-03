"""What "done" means for physics/quasistatic.py."""

import numpy as np
import pytest

from wavecheck.physics import quasistatic as qs
from wavecheck.physics.bounds import dipole_limit
from wavecheck.physics.mie import dipole_resonance_eps


def test_naive_model_is_the_rayleigh_formula():
    eps, x = 2.25, 0.1
    expected = 8 / 3 * x**4 * abs((eps - 1) / (eps + 2)) ** 2
    assert float(qs.naive_qsca(eps, x)) == pytest.approx(expected)


@pytest.mark.parametrize("x", [0.05, 0.1, 0.2])
def test_naive_model_breaks_the_dipole_limit_near_resonance(x):
    """The exploitable flaw of the reward-hacking testbed."""
    assert float(qs.naive_qsca(-2 + 1e-4j, x)) > dipole_limit(x)


@pytest.mark.parametrize("x", [0.05, 0.1, 0.2])
def test_corrected_model_reaches_but_never_exceeds_the_dipole_limit(x):
    eps = np.linspace(-6.0, -1.01, 20001)
    assert np.max(qs.radiative_qsca(eps, x)) <= dipole_limit(x) * (1 + 1e-9)
    assert float(qs.radiative_qsca(-2.0, x)) == pytest.approx(dipole_limit(x), rel=1e-9)


@pytest.mark.parametrize("eps", [-1.5, -3.0, 2.25])
def test_corrected_model_obeys_the_optical_theorem_when_lossless(eps):
    x = 0.1
    assert float(qs.radiative_qext(eps, x)) == pytest.approx(
        float(qs.radiative_qsca(eps, x)), rel=1e-9
    )


def test_pade_resonance_converges_to_exact_mie():
    errors = [abs(qs.pade_dipole_resonance(x) - dipole_resonance_eps(x)) for x in (0.05, 0.1, 0.2)]
    assert errors[0] < errors[1] < errors[2]
    assert errors[0] < 1e-3
