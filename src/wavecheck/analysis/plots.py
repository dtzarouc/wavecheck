"""The four figures of the write-up. Each reads from results/ and writes to figures/.

Keep them plain: one message per figure, labelled axes with units, the physical limit drawn as a
line wherever it applies, and 95% confidence intervals (Wilson) on every proportion.

TODO: implement with Claude Code once the data exist.
"""

from __future__ import annotations

from pathlib import Path


def fig_repro(csv_path: Path, out: Path) -> None:
    """Fig. 1: eps_res(x) from exact Mie, static -2 and Padé-Mie (plus published points)."""
    raise NotImplementedError


def fig_calibration(summary_csv: Path, out: Path) -> None:
    """Fig. 2: reliability diagram per model (resonance_shift), with ECE in the legend."""
    raise NotImplementedError


def fig_authority(summary_csv: Path, out: Path) -> None:
    """Fig. 3: accuracy by framing (neutral / journal / expert) per model, with Wilson CIs."""
    raise NotImplementedError


def fig_hacking(gradient_csv: Path, episodes_csv: Path, out: Path) -> None:
    """Fig. 4: left, Q_sca vs optimisation step (naive vs corrected) with the 6/x^2 line;
    right, exploitation and disclosure rates per condition and model."""
    raise NotImplementedError
