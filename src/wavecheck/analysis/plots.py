"""The four figures of the write-up. Each reads from results/ and writes to figures/.

Keep them plain: one message per figure, labelled axes with units, the physical limit drawn as a
line wherever it applies, and 95% confidence intervals (Wilson) on every proportion.

TODO: implement with Claude Code once the data exist.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_repro(csv_path: Path, out: Path) -> None:
    """Fig. 1: eps_res(x) from exact Mie, static -2 and Padé-Mie (plus published points)."""
    d = pd.read_csv(csv_path)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax.plot(d["x"], d["exact"], "k-", lw=2, label="exact Mie, $|a_1|=1$")
    ax.plot(d["x"], d["static"], ":", color="tab:gray", label="static (Fröhlich) $-2$")
    ax.plot(d["x"], d["pade_series"], "--", color="tab:blue", label="Eq. (7): $-2-12x^2/5$")
    ax.plot(d["x"], d["pade"], "-.", color="tab:red", label="Padé [3/2] closed form")
    ax.set(
        xlabel="size parameter $x = ka$",
        ylabel=r"resonance permittivity $\varepsilon_{res}$",
        title="Electric dipole resonance (lossless sphere)",
        ylim=(-12, -1.5),
    )
    ax.legend(fontsize=8)
    for col, style, color, lab in [
        ("static", ":", "tab:gray", "static"),
        ("pade_series", "--", "tab:blue", "Eq. (7)"),
        ("pade", "-.", "tab:red", "Padé closed form"),
    ]:
        ax2.semilogy(d["x"], (d[col] - d["exact"]).abs(), style, color=color, label=lab)
    ax2.set(
        xlabel="size parameter $x$",
        ylabel=r"$|\varepsilon - \varepsilon_{Mie}|$",
        title="Absolute error vs exact Mie",
    )
    ax2.legend(fontsize=8)
    _save(fig, out)


def fig_repro_magnetic_error(csv_path: Path, out: Path) -> None:
    """Reproduction of PRB 94, 140301(R) Fig. 1: error (%) of magnetic resonance conditions."""
    d = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    for col, style, color, lab in [
        ("err_eq8_pct", "-", "tab:blue", "Eq. (8)"),
        ("err_eq15_second_term_pct", "--", "tab:red", r"$(\pi/x)^2$"),
        ("err_eq15_two_terms_pct", ":", "goldenrod", r"$-2+(\pi/x)^2$"),
        ("err_eq13_pct", "-.", "tab:purple", "Eq. (13)"),
        ("err_eq15_pct", "-", "tab:green", "Eq. (15)"),
    ]:
        ax.plot(d["x"], d[col].abs(), style, color=color, label=lab)
    ax.axhline(1.0, color="k", lw=0.6)
    ax.axvline(0.8, color="k", lw=0.6, ls=":")
    ax.set(
        xlabel="size parameter $x$",
        ylabel="approximation error (%)",
        yscale="log",
        title=r"First magnetic resonance vs exact Mie (Re $\varepsilon$ at pole of $b_1$)",
    )
    ax.legend(fontsize=8)
    _save(fig, out)


def fig_repro_max_absorption(csv_path: Path, out: Path) -> None:
    """Reproduction of PRB 94, 140301(R) Fig. 2: eps'' for maximum absorption vs x."""
    d = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d["x"], d["eq14_2x"], "-", color="tab:blue", label="Eq. (14): $2x$")
    ax.plot(d["x"], d["eq13_imag"], "--", color="tab:red", label="Eq. (13)")
    ax.plot(d["x"], d["eq15_imag"], ":", color="goldenrod", label="Eq. (15)")
    ax.plot(
        d["x"],
        d["exact_b1_eps_imag"],
        "-.",
        color="tab:purple",
        lw=2,
        label="exact Mie $b_1$ ($b_1=1/2$)",
    )
    ax.plot(
        d["x"], d["exact_a1_eps_imag"], "-", color="tab:green", label="exact Mie $a_1$ ($a_1=1/2$)"
    )
    ax.plot(d["x"], d["eq12_2x3"], "k:", lw=0.8, label="Eq. (12): $2x^3$")
    ax.axvspan(0.4, 0.8, color="0.9", zorder=0)
    ax.set(
        xlabel="size parameter $x$",
        ylabel=r"$\varepsilon''$ for maximum absorption",
        title="Loss for maximum dipole absorption",
    )
    ax.legend(fontsize=8)
    _save(fig, out)


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
