# CLAUDE.md

Instructions for Claude Code in this repository. Read this file and `PLAN.md` before any task.

## Project

wavecheck is a two-week public mini-study (November 2026) by Dimitrios (Dimi) C. Tzarouchis, an
electromagnetics researcher, prepared for an AI-safety research residency application. It uses exact
Mie scattering as ground truth to test language models (calibration; verification under authority
framing) and builds a small reward-hacking testbed: a flawed quasistatic simulator plus a
physical-bound monitor. Target: a presentable v0.1 tagged by **18 November 2026**.

## How to work with Dimi

- Dimi is an expert in scattering theory and is building fluency in modern Python, PyTorch and LLM
  APIs. Briefly explain non-obvious Python, numpy or PyTorch choices; skip physics explanations.
- The physics core (`src/wavecheck/physics/`, `src/wavecheck/repro/`) was implemented by Claude at
  Dimi's request (3 Nov 2026); Dimi reviews it. The PyTorch agent
  (`src/wavecheck/hacking/gradient_agent.py`) was also implemented by Claude at his request
  (16 Nov 2026).
- Before a change that touches more than one file, propose a 3–5 bullet plan and wait for a go-ahead.
- Keep functions small, pure and typed; vectorise with numpy; docstrings state the equation, units
  and conventions.
- Every function gets a test. A physics function is done only when it passes an identity test
  (energy conservation, passivity, a known limit) or matches an independent implementation.
- Commit in small steps, one idea per commit, with clear messages. Reviewers will read the history.

## Research integrity (non-negotiable)

1. Never invent numbers, results, figures or citations. Every number in `README.md` or `report/`
   comes from a file in `results/` written by a script in this repository.
2. Ground truth comes only from `src/wavecheck/physics/`, never from a model's output or from memory.
3. Log every API call (model ID, timestamp, parameters, full prompt, full response, usage) to
   `results/raw/*.jsonl`. Never edit or delete raw files; reruns write new files.
4. Freeze prompts, question sets and grading rules before the full run (git tag `prompts-frozen`).
   If anything changes afterwards, rerun everything and record why in `CHANGELOG.md`.
5. Report null results and failures as plainly as positive ones, and list limitations.
6. Secrets stay out of the repository: the API key lives in `.env` (gitignored). Never print it.
7. Estimate the cost before any API run, and stop to ask if it exceeds the `budget_usd` in the config.

## Conventions

- Time dependence exp(-iωt); passive media have Im(ε) ≥ 0. Non-magnetic sphere (μ = 1) in vacuum.
- Size parameter x = k·a = 2πa/λ. Efficiencies Q = σ / (πa²).
- Mie coefficients in the Bohren & Huffman convention (a_n electric, b_n magnetic).
- Randomness only through `numpy.random.default_rng(seed)`, with seeds from `configs/`.
- Model IDs come from `configs/*.yaml`; never hard-code them in code.

## Commands

- Install: `pip install -e ".[dev]"` (PyTorch part: `pip install -e ".[dev,ml]"`)
- Tests: `pytest`; one file: `pytest tests/test_mie.py`
- Lint and format: `ruff check . --fix` then `ruff format .`
- Scripts: see the table in `README.md`.

## Done means

Tests pass, ruff is clean, README and `report/report.md` numbers are regenerated from `results/`,
and the change is committed.
