"""What "done" means for physics/bounds.py."""

import numpy as np
import pytest

from wavecheck.physics import bounds


def test_dipole_limit_value():
    assert bounds.dipole_limit(0.1) == pytest.approx(600.0)


def test_single_channel_limit_for_higher_orders():
    assert bounds.single_channel_limit(0.5, n=2) == pytest.approx(2 * 5 / 0.25)


def test_violation_flag():
    assert bounds.violates_limit(601.0, 0.1)
    assert not bounds.violates_limit(599.0, 0.1)


def test_passivity_residuals():
    c = np.array([1.0, 0.5 + 0.5j, 0.2])
    np.testing.assert_allclose(bounds.passivity_residuals(c), [0.0, 0.0, 0.16], atol=1e-15)


def test_energy_residual():
    assert bounds.energy_residual(2.0, 1.5, 0.5) == pytest.approx(0.0)
