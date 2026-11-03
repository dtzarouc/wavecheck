"""Reproduce a figure from Tzarouchis, Ylä-Oijala & Sihvola, Phys. Rev. B 94, 140301(R) (2016).

Suggested target: the electric-dipole resonance condition eps_res(x) of a small sphere, comparing
  - the exact Mie root (physics.mie.dipole_resonance_eps),
  - the static Fröhlich value eps = -2,
  - the Padé-Mie prediction (physics.quasistatic.pade_dipole_resonance),
plus the published data points if available (put them in data/published/, cite the figure number).

Output: figures/fig1_repro.png and results/repro_prb2016.csv (the plotted numbers).

TODO(Dimi): choose the figure, implement compute_curves() and main().
"""

from __future__ import annotations

import numpy as np


def compute_curves(x_grid: np.ndarray) -> dict[str, np.ndarray]:
    """Return {"x": ..., "exact": ..., "static": ..., "pade": ...} on the given grid."""
    raise NotImplementedError


def main() -> None:
    """Compute the curves, write the CSV and the figure."""
    raise NotImplementedError
