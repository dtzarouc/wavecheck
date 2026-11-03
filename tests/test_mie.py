"""What "done" means for physics/mie.py: identities, limits and an independent implementation."""

import numpy as np
import pytest

from wavecheck.physics import mie

LOSSLESS = [2.25, 4.0, 12.0, -1.5, -3.0, -6.0]
SIZES = [0.05, 0.3, 1.0, 3.0, 5.0]


@pytest.mark.parametrize("eps", LOSSLESS)
@pytest.mark.parametrize("x", SIZES)
def test_lossless_sphere_conserves_energy(eps, x):
    q = mie.efficiencies(eps, x)
    assert abs(q.q_ext - q.q_sca) <= 1e-9 * max(1.0, q.q_ext)
    assert abs(q.q_abs) <= 1e-9 * max(1.0, q.q_ext)


@pytest.mark.parametrize("eps", [2.25, 10.0, -3 + 0.5j, 4 + 1j])
def test_small_sphere_matches_rayleigh(eps):
    x = 1e-3
    alpha = (eps - 1) / (eps + 2)
    rayleigh = 8 / 3 * x**4 * abs(alpha) ** 2
    assert mie.efficiencies(eps, x).q_sca == pytest.approx(rayleigh, rel=1e-4)


def test_passive_spheres_have_no_gain_in_any_channel():
    rng = np.random.default_rng(0)
    for _ in range(300):
        eps = complex(rng.uniform(-10, 10), rng.uniform(0, 5))
        x = rng.uniform(0.05, 5)
        _, a, b = mie.mie_coefficients(eps, x)
        assert np.all(a.real - np.abs(a) ** 2 >= -1e-12)
        assert np.all(b.real - np.abs(b) ** 2 >= -1e-12)


def test_no_channel_beats_its_single_channel_limit():
    rng = np.random.default_rng(1)
    for _ in range(100):
        eps = complex(rng.uniform(-10, 10), rng.uniform(0, 2))
        x = rng.uniform(0.05, 3)
        q_e, q_m = mie.channel_efficiencies(eps, x)
        n = np.arange(1, len(q_e) + 1)
        limit = 2 * (2 * n + 1) / x**2
        assert np.all(q_e <= limit * (1 + 1e-9))
        assert np.all(q_m <= limit * (1 + 1e-9))


def test_dipole_resonance_tends_to_the_frohlich_value():
    assert mie.dipole_resonance_eps(0.01) == pytest.approx(-2.0, abs=1e-3)


def test_dipole_resonance_redshifts_with_size():
    eps = [mie.dipole_resonance_eps(x) for x in (0.05, 0.1, 0.2, 0.3, 0.5)]
    assert all(e < -2.0 for e in eps)
    assert all(later < earlier for earlier, later in zip(eps, eps[1:], strict=False))


def test_lossless_dipole_resonance_has_unit_coefficient():
    x = 0.2
    _, a, _ = mie.mie_coefficients(mie.dipole_resonance_eps(x), x)
    assert abs(a[0]) == pytest.approx(1.0, abs=1e-6)
