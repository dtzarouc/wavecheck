"""Statistics helpers against textbook values."""

import pytest

from wavecheck.analysis.stats import mcnemar_exact, wilson_interval


def test_wilson_known_value():
    # 50/50 successes: lower bound 0.9287 (standard Wilson result)
    lo, hi = wilson_interval(50, 50)
    assert lo == pytest.approx(0.92865, abs=1e-4) and hi == pytest.approx(1.0)


def test_wilson_symmetric_at_half():
    lo, hi = wilson_interval(10, 20)
    assert 0.5 - lo == pytest.approx(hi - 0.5)


def test_mcnemar():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(2, 0) == pytest.approx(0.5)
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)
