"""Cross-check against an independent implementation (miepython, installed with the dev extra)."""

import pytest

from wavecheck.physics import mie

miepython = pytest.importorskip("miepython")


@pytest.mark.parametrize("m", [1.33, 1.5, 2.0, 3.5])
@pytest.mark.parametrize("x", [0.1, 1.0, 5.0])
def test_matches_miepython_for_dielectric_spheres(m, x):
    """Real m only, so the two codes' sign conventions for absorption cannot disagree."""
    q_ext_ref, q_sca_ref, _, _ = miepython.efficiencies_mx(m, x)
    q = mie.efficiencies(m**2, x)
    assert q.q_ext == pytest.approx(float(q_ext_ref), rel=1e-6)
    assert q.q_sca == pytest.approx(float(q_sca_ref), rel=1e-6)
