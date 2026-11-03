"""Smoke and consistency tests for repro/prb2016.py on small grids (no files written)."""

import numpy as np

from wavecheck.repro import prb2016


def test_compute_curves_shapes_and_limits():
    c = prb2016.compute_curves(np.array([0.01, 0.2]))
    assert set(c) == {"x", "exact", "static", "pade", "pade_series"}
    assert abs(c["exact"][0] + 2.0) < 1e-3
    assert abs(c["pade"][1] - c["exact"][1]) < abs(c["static"][1] - c["exact"][1])


def test_magnetic_errors_reference_columns():
    d = prb2016.compute_magnetic_errors(np.array([0.1, 0.5]))
    assert (d["exact_pole_imag"] < 0).all()
    assert (d["err_eq15_pct"].abs() < d["err_eq8_pct"].abs()).all()


def test_max_absorption_is_passive():
    d = prb2016.compute_max_absorption(np.array([0.1, 0.5]))
    assert (d["exact_b1_eps_imag"] > 0).all() and (d["exact_a1_eps_imag"] > 0).all()


def test_summary_has_headline_quantities():
    xs = np.linspace(0.05, 1.0, 20)
    s = prb2016.summarise(
        __import__("pandas").DataFrame(prb2016.compute_curves(xs)),
        prb2016.compute_magnetic_errors(xs),
        prb2016.compute_max_absorption(xs),
    )
    assert "mag_max_abs_err_pct_eq15_x_le_0.8" in set(s["quantity"])
    assert s["value"].notna().sum() >= len(s) - 2
