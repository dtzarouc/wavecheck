"""Regenerate every figure from results/ (never from numbers typed by hand)."""

from wavecheck.analysis import plots
from wavecheck.config import ROOT


def main() -> None:
    res, fig = ROOT / "results", ROOT / "figures"
    plots.fig_repro(res / "repro_prb2016.csv", fig / "fig1_repro.png")
    plots.fig_calibration(res / "summary_eval.csv", fig / "fig2_calibration.png")
    plots.fig_authority(res / "eval_authority.csv", fig / "fig3_authority.png")
    plots.fig_hacking(
        res / "hacking_gradient.csv", res / "hacking_episodes.csv", fig / "fig4_hacking.png"
    )


if __name__ == "__main__":
    main()
