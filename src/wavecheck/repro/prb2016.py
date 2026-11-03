"""Reproduce results of Tzarouchis, Ylä-Oijala & Sihvola, Phys. Rev. B 94, 140301(R) (2016).

Source: arXiv:1606.05523v4. Ground truth comes only from physics.mie (exact Mie coefficients).

Three reproductions, each written to a CSV in results/ and plotted in figures/:

1. Electric dipole (Eq. 7): resonance permittivity eps_res(x) of a lossless sphere from exact Mie
   (|a_1| = 1), the static Fröhlich value -2, the Padé [3/2]/[3/3] closed form and its series
   -2 - (12/5) x^2.                     -> results/repro_prb2016.csv, figures/fig1_repro.png
2. Paper Fig. 1: approximation error (%) of the first magnetic resonance conditions
   Eq. (8), (pi/x)^2, -2 + (pi/x)^2, Eq. (13) and Eq. (15) against exact Mie.
   The paper does not state how the exact value is extracted from Eq. (2). Primary reference:
   Re(eps) at the complex pole of the exact b_1 (1/b_1 = 0), the exact counterpart of the
   Padé pole conditions. Secondary reference (columns *_vs_real): the real eps of a lossless
   sphere at which |b_1| = 1.
   Output: results/repro_prb2016_fig1.csv, figures/fig1b_prb2016_fig1.png
3. Paper Fig. 2: eps'' required for maximum absorption of the b_1 channel versus x (Eqs. 13, 14,
   15 and exact Mie), and of the a_1 channel (Eq. 12: 2 x^3).
   ASSUMPTION (the paper does not define the procedure): "maximum absorption" = critical coupling,
   the complex eps at which the channel coefficient equals 1/2, where the channel absorption
   (6/x^2)(Re c - |c|^2) reaches its maximum 1.5/x^2. This eps is exactly the complex conjugate
   of the pole used in item 2 (tested in tests/test_mie_resonances.py). The paper writes the
   loss with a negative sign (exp(+j w t) convention); we compare magnitudes |eps''|.
   Output: results/repro_prb2016_fig2.csv, figures/fig1c_prb2016_fig2.png

Headline numbers go to results/repro_prb2016_summary.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wavecheck.config import ROOT
from wavecheck.physics import mie
from wavecheck.physics import quasistatic as qs

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def compute_curves(x_grid: np.ndarray) -> dict[str, np.ndarray]:
    """Electric dipole: {"x", "exact", "static", "pade", "pade_series"} on the given grid."""
    x_grid = np.asarray(x_grid, dtype=float)
    return {
        "x": x_grid,
        "exact": np.array([mie.dipole_resonance_eps(x, bracket=(-12.0, -1.05)) for x in x_grid]),
        "static": np.full_like(x_grid, -2.0),
        "pade": np.array([qs.pade_dipole_resonance(x) for x in x_grid]),
        "pade_series": np.array([qs.pade_dipole_resonance_series(x) for x in x_grid]),
    }


def compute_magnetic_errors(x_grid: np.ndarray) -> pd.DataFrame:
    """Paper Fig. 1: signed relative error (%) of each magnetic condition vs exact Mie.

    err_<k>_pct uses Re(eps_pole) of the exact b_1; err_<k>_pct_vs_real uses |b_1| = 1.
    Each pole is found from the Eq. (15) guess at that x (no continuation, so no branch jumps).
    """
    rows = []
    for x in np.asarray(x_grid, dtype=float):
        pole = mie.pole_eps(x, "M", qs.magnetic_resonance_eq15(x))
        exact = pole.real
        exact_real = mie.magnetic_dipole_resonance_eps(x)
        approx = {
            "eq8": qs.magnetic_resonance_eq8(x),
            "eq15_second_term": qs.magnetic_resonance_eq15(x, terms=1).real,
            "eq15_two_terms": qs.magnetic_resonance_eq15(x, terms=2).real,
            "eq13": qs.magnetic_resonance_eq13(x).real,
            "eq15": qs.magnetic_resonance_eq15(x).real,
        }
        row = {
            "x": x,
            "exact_pole_real": exact,
            "exact_pole_imag": pole.imag,
            "exact_real_resonance": exact_real,
        }
        for k, v in approx.items():
            row[k] = v
            row[f"err_{k}_pct"] = 100.0 * (v - exact) / exact
            row[f"err_{k}_pct_vs_real"] = 100.0 * (v - exact_real) / exact_real
        rows.append(row)
    return pd.DataFrame(rows)


def compute_max_absorption(x_grid: np.ndarray) -> pd.DataFrame:
    """Paper Fig. 2: |eps''| at critical coupling for b_1 (and a_1), exact vs the paper's formulas.

    Guesses at each x: conj(Eq. 15) for b_1 (passive branch), Padé resonance + 2 i x^3 for a_1.
    """
    rows = []
    for x in np.asarray(x_grid, dtype=float):
        eps_m = mie.max_absorption_eps(x, "M", np.conj(qs.magnetic_resonance_eq15(x)))
        eps_e = mie.max_absorption_eps(x, "E", complex(qs.pade_dipole_resonance(x), 2 * x**3))
        rows.append(
            {
                "x": x,
                "exact_b1_eps_real": eps_m.real,
                "exact_b1_eps_imag": eps_m.imag,
                "eq14_2x": 2 * x,
                "eq13_imag": abs(qs.magnetic_resonance_eq13(x).imag),
                "eq15_imag": abs(qs.magnetic_resonance_eq15(x).imag),
                "exact_a1_eps_real": eps_e.real,
                "exact_a1_eps_imag": eps_e.imag,
                "eq12_2x3": 2 * x**3,
            }
        )
    return pd.DataFrame(rows)


def summarise(elec: pd.DataFrame, mag: pd.DataFrame, absn: pd.DataFrame) -> pd.DataFrame:
    """Headline numbers, each traceable to a column of the CSVs above."""

    def max_abs_where(df: pd.DataFrame, col: str, xmax: float) -> float:
        return float(df.loc[df["x"] <= xmax + 1e-12, col].abs().max())

    elec_err_pade = (elec["pade"] - elec["exact"]).abs()
    elec_err_static = (elec["static"] - elec["exact"]).abs()
    out = [
        ("elec_max_abs_err_pade_x_le_0.5", float(elec_err_pade[elec["x"] <= 0.5].max())),
        ("elec_max_abs_err_static_x_le_0.5", float(elec_err_static[elec["x"] <= 0.5].max())),
        (
            "elec_max_abs_err_series_x_le_0.5",
            float((elec["pade_series"] - elec["exact"]).abs()[elec["x"] <= 0.5].max()),
        ),
        ("mag_max_abs_err_pct_eq15_x_le_0.8", max_abs_where(mag, "err_eq15_pct", 0.8)),
        ("mag_max_abs_err_pct_eq13_x_le_0.8", max_abs_where(mag, "err_eq13_pct", 0.8)),
        ("mag_max_abs_err_pct_eq8_x_le_0.8", max_abs_where(mag, "err_eq8_pct", 0.8)),
        ("mag_max_abs_err_pct_eq15_x_le_1.0", max_abs_where(mag, "err_eq15_pct", 1.0)),
        (
            "mag_x_where_eq15_err_first_exceeds_1pct",
            float(mag.loc[mag["err_eq15_pct"].abs() > 1.0, "x"].min())
            if (mag["err_eq15_pct"].abs() > 1.0).any()
            else float("nan"),
        ),
        (
            "mag_max_abs_err_pct_eq15_vs_real_x_le_0.8",
            max_abs_where(mag, "err_eq15_pct_vs_real", 0.8),
        ),
        (
            "mag_x_where_eq15_err_vs_real_first_exceeds_1pct",
            float(mag.loc[mag["err_eq15_pct_vs_real"].abs() > 1.0, "x"].min())
            if (mag["err_eq15_pct_vs_real"].abs() > 1.0).any()
            else float("nan"),
        ),
        (
            "absn_b1_max_rel_dev_eq15_imag_x_le_0.8",
            float(
                ((absn["eq15_imag"] - absn["exact_b1_eps_imag"]) / absn["exact_b1_eps_imag"])[
                    absn["x"] <= 0.8 + 1e-12
                ]
                .abs()
                .max()
            ),
        ),
    ]
    sel = absn[(absn["x"] >= 0.4) & (absn["x"] <= 0.8)]
    out += [
        ("absn_b1_eps_imag_min_x_0.4_0.8", float(sel["exact_b1_eps_imag"].min())),
        ("absn_b1_eps_imag_max_x_0.4_0.8", float(sel["exact_b1_eps_imag"].max())),
        (
            "absn_a1_max_rel_dev_from_2x3_x_le_0.2",
            float(
                ((absn["exact_a1_eps_imag"] - absn["eq12_2x3"]) / absn["eq12_2x3"])[
                    absn["x"] <= 0.2
                ]
                .abs()
                .max()
            ),
        ),
    ]
    return pd.DataFrame(out, columns=["quantity", "value"])


def main() -> None:
    """Compute all curves, write the CSVs and the figures."""
    from wavecheck.analysis import plots

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    elec = pd.DataFrame(compute_curves(np.linspace(0.01, 1.0, 100)))
    mag = compute_magnetic_errors(np.linspace(0.05, 1.0, 96))  # step 0.01
    absn = compute_max_absorption(np.linspace(0.02, 1.0, 99))  # step 0.01
    paths = {
        "elec": RESULTS / "repro_prb2016.csv",
        "mag": RESULTS / "repro_prb2016_fig1.csv",
        "absn": RESULTS / "repro_prb2016_fig2.csv",
        "summary": RESULTS / "repro_prb2016_summary.csv",
    }
    elec.to_csv(paths["elec"], index=False, float_format="%.10g")
    mag.to_csv(paths["mag"], index=False, float_format="%.10g")
    absn.to_csv(paths["absn"], index=False, float_format="%.10g")
    summary = summarise(elec, mag, absn)
    summary.to_csv(paths["summary"], index=False, float_format="%.6g")
    plots.fig_repro(paths["elec"], FIGURES / "fig1_repro.png")
    plots.fig_repro_magnetic_error(paths["mag"], FIGURES / "fig1b_prb2016_fig1.png")
    plots.fig_repro_max_absorption(paths["absn"], FIGURES / "fig1c_prb2016_fig2.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
