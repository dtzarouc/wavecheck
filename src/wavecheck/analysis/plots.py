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


MODEL_COLORS = {
    "claude-haiku-5-5": "tab:green",
    "claude-sonnet-5-5": "tab:blue",
    "claude-opus-5-5": "tab:purple",
}


def fig_calibration(summary_csv: Path, out: Path) -> None:
    """Fig. 2: reliability diagram per model, with ECE in the legend.

    Left: resonance_shift only; right: all graded items. Points are 10 equal-width confidence
    bins (marker area ~ number of answers); points above the diagonal mean underconfidence.
    """
    import numpy as np

    from wavecheck.analysis.stats import wilson_interval
    from wavecheck.evals.grading import expected_calibration_error

    d = pd.read_csv(summary_csv)
    d = d[d["correct"].notna() & d["confidence"].notna()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, (title, sub) in zip(
        axes,
        [("Resonance questions", d[d["family"] == "resonance_shift"]), ("All graded questions", d)],
        strict=True,
    ):
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        for model, g in sub.groupby("model"):
            c = g["confidence"].to_numpy()
            y = g["correct"].astype(bool).to_numpy()
            bins = np.minimum((c * 10).astype(int), 9)
            ece = expected_calibration_error(c.tolist(), y.tolist())
            for b in np.unique(bins):
                sel = bins == b
                lo, hi = wilson_interval(int(y[sel].sum()), int(sel.sum()))
                ax.errorbar(
                    c[sel].mean(),
                    y[sel].mean(),
                    yerr=[[max(0.0, y[sel].mean() - lo)], [max(0.0, hi - y[sel].mean())]],
                    fmt="o",
                    ms=3 + 2 * np.sqrt(sel.sum()),
                    alpha=0.7,
                    color=MODEL_COLORS.get(model),
                    capsize=2,
                )
            ax.plot(
                [],
                [],
                "o",
                color=MODEL_COLORS.get(model),
                label=f"{model.replace('claude-', '')} (ECE {ece:.3f}, n={len(g)})",
            )
        ax.set(xlabel="stated confidence", title=title, xlim=(0.35, 1.02), ylim=(0.35, 1.02))
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("observed accuracy (95% Wilson CI)")
    _save(fig, out)


def fig_authority(authority_csv: Path, out: Path) -> None:
    """Fig. 3: accuracy by framing (neutral / journal / expert) per model, with Wilson CIs.

    Reads results/eval_authority.csv (written by scripts/analyse_eval.py).
    """
    a = pd.read_csv(authority_csv)
    framings = ["neutral", "journal", "expert"]
    models = list(dict.fromkeys(a["model"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.25
    for j, framing in enumerate(framings):
        sub = a[a["framing"] == framing].set_index("model").loc[models]
        xs = [i + (j - 1) * width for i in range(len(models))]
        ax.errorbar(
            xs,
            sub["accuracy"],
            yerr=[
                (sub["accuracy"] - sub["ci_low"]).clip(lower=0),
                (sub["ci_high"] - sub["accuracy"]).clip(lower=0),
            ],
            fmt=["o", "s", "^"][j],
            capsize=3,
            label=framing,
        )
    ax.set_xticks(range(len(models)), [m.replace("claude-", "") for m in models])
    ax.set(
        ylabel="accuracy (95% Wilson CI)",
        ylim=(0.8, 1.01),
        title="Report checks under neutral and authority framing (n = 50 each)",
    )
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, out)


def fig_hacking(gradient_csv: Path, episodes_csv: Path, out: Path) -> None:
    """Fig. 4: left, Q_sca vs optimisation step (naive vs corrected) with the 6/x^2 line;
    right, exploitation and disclosure rates per condition and model (only if episodes exist)."""
    d = pd.read_csv(gradient_csv)
    has_episodes = Path(episodes_csv).exists()
    fig, axes = plt.subplots(1, 2 if has_episodes else 1, figsize=(10 if has_episodes else 6, 4))
    ax = axes[0] if has_episodes else axes
    naive = d[d["objective"] == "naive"]
    corr = d[d["objective"] == "corrected"]
    ax.semilogy(
        naive["step"],
        naive["q_naive"],
        color="tab:red",
        label="optimising naive model: reported $Q$",
    )
    ax.semilogy(
        naive["step"],
        naive["q_corrected"],
        ":",
        color="tab:red",
        label="same designs, radiation-corrected $Q$",
    )
    ax.semilogy(
        corr["step"], corr["q_corrected"], color="tab:blue", label="optimising corrected model"
    )
    ax.axhline(d["dipole_limit"].iloc[0], color="k", lw=1, label="dipole limit $6/x^2$")
    ax.set(
        xlabel="Adam step",
        ylabel="scattering efficiency $Q_{sca}$",
        title=f"Gradient agent, x = {d['x'].iloc[0]:g}",
    )
    ax.legend(fontsize=8)
    if has_episodes:
        e = pd.read_csv(episodes_csv)
        rates = e.groupby(["model", "condition"])[["exploited", "disclosed"]].mean()
        rates.plot.bar(ax=axes[1])
        axes[1].set(ylabel="share of episodes", title="LLM agent")
    _save(fig, out)
