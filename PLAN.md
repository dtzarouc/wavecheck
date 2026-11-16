# Plan: wavecheck v0.1 by 18 November 2026

Residency applications close on 22 November (11:59 pm PT). Tag v0.1 on 18 November and link it in
the application between 19 and 21 November.

## M0 · Set-up (4 Nov, one evening)

- [x] Create a public GitHub repository `wavecheck`, copy this folder in, make the first commit.
- [x] Create the virtual environment, run `pip install -e ".[dev]"`, then `pytest`. Most tests fail
      with `NotImplementedError`; that is expected, and they are your to-do list.
- [x] Create an API key in the Claude Console, put it in `.env`, and set a monthly spend limit.
- [x] Open the folder in Claude Code: "Read CLAUDE.md and PLAN.md, then help me start M1."

## M1 · Physics core and the reproduction (5–7 Nov)

- [x] `physics/mie.py`: Mie coefficients, efficiencies, per-channel terms, exact dipole resonance.
- [x] `physics/quasistatic.py`: naive Rayleigh model, radiation-reaction correction, and the
      Padé–Mie dipole resonance from PRB 94, 140301(R) (2016).
- [x] `physics/bounds.py`: single-channel limit, passivity and energy-conservation residuals.
- [x] `pytest` green locally and on GitHub Actions.
- [x] `repro/prb2016.py`: reproduce one figure from the 2016 paper as `figures/fig1_repro.png`,
      overlaying the published data if you still have it.

## M2 · Evaluation (8–12 Nov)

- [x] `evals/tasks.py`: three families. `resonance_shift` (numeric answer plus confidence),
      `report_check` (YES/NO, neutral framing) and `authority` (the same items, authority framing).
- [x] `evals/grading.py`: answer parser, tolerance grading, Brier score, expected calibration error.
- [x] Pilot: 10 items with one model. Read every transcript, fix the prompts, then freeze them
      (git tag `prompts-frozen`).
- [x] Full run: about 50 items per family with three models (`configs/eval.yaml`). Check the cost
      estimate first.
- [x] Figures 2 and 3.

## M3 · Reward-hacking testbed (13–16 Nov)

- [x] `hacking/env.py`: design environment with the naive (exploitable) and corrected simulators.
- [x] `hacking/gradient_agent.py` (PyTorch): Adam on (Re ε, Im ε ≥ 0) maximising the naive Q_sca.
      Log the run toward ε → −2 and the point where it crosses the dipole limit 6/x².
- [ ] `hacking/llm_agent.py` (deferred: not run for v0.1): the tool-use loop is written and tested with a fake client. Run three
      conditions (neutral, "must be physically realisable", monitor tool available) × 20 episodes ×
      two models.
- [x] `hacking/monitor.py`: bound check plus a disclosure check on the final report.
- [x] Figure 4.

## M4 · Write-up and release (17–18 Nov)

- [x] README results table filled from `results/`.
- [x] `report/report.md` (2–4 pages): question, set-up, three results, limitations, next steps.
- [ ] Tag v0.1; optionally archive on Zenodo for a DOI.
- [ ] Put the link first under "Notable work" in the application.

## If time runs short

Ship M1 and M2 with one or two models, plus the gradient half of M3 (no API cost). A small, honest,
reproducible result beats a large unfinished one.
