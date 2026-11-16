# wavecheck

**Physics-verified tests of language models, built on exact electromagnetic scattering.**

`wavecheck` uses Mie theory, the exact solution for light scattering by a sphere, as an oracle that
never guesses. It asks three questions about frontier language models:

1. **Calibration.** On scattering questions with exact answers, do models' stated confidences match
   their accuracy, especially where the textbook small-particle formula quietly fails?
2. **Verification under pressure.** Can models tell a correct reported result from a wrong one, and
   does an authority framing ("published in a leading journal") change their verdict?
3. **Reward hacking.** When an optimiser or an LLM agent maximises a score computed by a *flawed*
   simulator (the quasistatic dipole model, which has no radiation reaction and diverges at the
   plasmon resonance), does it exploit the flaw, does it say so, and does a physical-bound monitor
   catch it?

The physics builds on my own published work: D. C. Tzarouchis, P. Ylä-Oijala and A. Sihvola,
*Unveiling the scattering behavior of small spheres*, Phys. Rev. B **94**, 140301(R) (2016), and
D. C. Tzarouchis and A. Sihvola, *Light scattering by a dielectric sphere: perspectives on the Mie
resonances*, Appl. Sci. **8**, 184 (2018). Milestone 1 reproduces a figure from that work in Python
before anything touches a language model.

> **Status:** v0.1 (November 2026). Every number below is produced by the scripts in this
> repository and can be traced to a file in `results/`; nothing is typed in by hand. The write-up is
> [`report/report.md`](report/report.md).

## Results

| Question | Headline number (source) | Figure |
| --- | --- | --- |
| Reproduction of PRB 94, 140301(R) | Eq. (15) vs the exact Mie pole of b₁: max error 0.93 % for x ≤ 0.8, first above 1 % at x = 0.81; the paper claims < 1 % up to x = 0.8 (`results/repro_prb2016_summary.csv`) | `figures/fig1_repro.png`, `fig1b_prb2016_fig1.png`, `fig1c_prb2016_fig2.png` |
| Calibration | Resonance permittivity (n = 50 per model): Sonnet 5.5 and Opus 5.5 50/50 correct, Haiku 5.5 45/46 answered; all three underconfident, ECE 0.19–0.25 (`results/eval_metrics.csv`) | `figures/fig2_calibration.png` |
| Authority framing | No detectable effect: Sonnet and Opus 50/50 under every framing; Haiku 50 → 49 (journal) and 48 (expert) of 50, McNemar p ≥ 0.5, all 3 errors from a wrong polarisability formula (`results/eval_authority.csv`, `results/eval_formula_errors.csv`) | `figures/fig3_authority.png` |
| Reward hacking | Gradient agent on the naive simulator crosses the dipole limit 6/x² at step 340 and reports 491× the limit; the same design is at 0.92 of the limit under the corrected model. LLM-agent episodes not run in v0.1 (`results/hacking_gradient_summary.csv`) | `figures/fig4_hacking.png` |

Total API cost: $11.51 for the frozen full run (600 calls) plus $2.78 for two pilots
(`results/eval_cost.csv`).

## Quickstart

```bash
git clone https://github.com/dtzarouc/wavecheck && cd wavecheck
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # PyTorch part: pip install -e ".[dev,ml]"
pytest                             # the physics tests turn green at Milestone 1
cp .env.example .env               # then add your ANTHROPIC_API_KEY to .env
```

| Step | Command |
| --- | --- |
| Reproduce the published figure | `python scripts/reproduce_paper.py` |
| Build the question set | `python scripts/build_dataset.py --config configs/eval.yaml` |
| Run the evaluation | `python scripts/run_eval.py --config configs/eval.yaml` |
| Analyse the evaluation (tables + Figures 2, 3) | `python scripts/analyse_eval.py` |
| Run the gradient study (needs `.[ml]`) | `python scripts/run_hacking.py --gradient-only` |

## Repository layout

```
src/wavecheck/physics/   exact Mie, small-sphere models, physical bounds (the oracle)
src/wavecheck/repro/     reproduction of the published figure
src/wavecheck/evals/     question generators, model client, runner, grading
src/wavecheck/hacking/   flawed-simulator environment, gradient and LLM agents, monitor
src/wavecheck/analysis/  figures
configs/                 experiment settings (models, sizes, seeds, budget)
results/                 raw model outputs (JSONL, never edited) and summary tables
report/                  the short write-up
tests/                   physics identities, grading, the agent loop
```

## How this was built

<!-- Keep this accurate before publishing. -->
I designed the study, made the decisions recorded in `CHANGELOG.md`, ran the API and PyTorch runs
on my own machine, and reviewed the results against my 2016 paper. The code, including the physics
core and the PyTorch agent, was written by Claude Code at my request and is checked by identity
tests, an independent Mie implementation (miepython) and the reproduction of the published results. Every result is produced by code in this repository
and checked against the physics tests.

## Citation and licence

See `CITATION.cff`. MIT licence.
