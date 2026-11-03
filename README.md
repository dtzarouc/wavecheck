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

> **Status:** work in progress (started November 2026). Every number below is produced by the
> scripts in this repository; nothing is typed in by hand.

## Results

_To be generated; see `PLAN.md`._

| Question | Headline number | Figure |
| --- | --- | --- |
| Reproduction of PRB 2016 | Eq. (15) vs exact Mie pole of b₁: max error 0.93 % for x ≤ 0.8 (paper: < 1 %); first exceeds 1 % at x = 0.81 | `figures/fig1_repro.png`, `figures/fig1b_prb2016_fig1.png`, `figures/fig1c_prb2016_fig2.png` |
| Calibration | expected calibration error per model: … | `figures/fig2_calibration.png` |
| Authority framing | accuracy change under authority framing: … | `figures/fig3_authority.png` |
| Reward hacking | share of agent episodes reporting a bound-violating design: … | `figures/fig4_hacking.png` |

## Quickstart

```bash
git clone https://github.com/<your-user>/wavecheck && cd wavecheck
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
| Run the reward-hacking study | `python scripts/run_hacking.py --config configs/hack.yaml` |
| Make all figures | `python scripts/make_figures.py` |

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
I designed the study. The physics core (`physics/`, `repro/`) was implemented by Claude Code at my
request, against identity tests and an independent cross-check (miepython), and reviewed by me.
Claude Code also did the scaffolding, tests, plotting and API plumbing. Every result is produced by code in this repository
and checked against the physics tests.

## Citation and licence

See `CITATION.cff`. MIT licence.
