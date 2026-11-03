# wavecheck: physics-verified tests of language models

Dimitrios C. Tzarouchis · v0.1 · November 2026 · code: <repository link>

<!-- Write this last. Every number comes from results/; cite the file next to it. 2-4 pages. -->

**TL;DR.** One sentence per result, each with its number and sample size.

## 1. Why exact physics makes a good oracle

Three or four sentences: oversight research needs tasks whose answers can be checked without
trusting the model; Mie theory provides exact answers plus cheap consistency checks (energy
conservation, single-channel limits); the quasistatic model provides a known, controllable flaw.

## 2. Set-up

- Ground truth: exact Mie, validated by identity tests and an independent implementation.
- Reproduction: Fig. 1 against Phys. Rev. B 94, 140301(R) (2016).
- Models and dates: model IDs and run dates from `results/raw/`; prompts frozen at commit `<hash>`.
- Question families and sizes: resonance_shift (n = …), report_check (n = …), authority (n = …).
- Reward-hacking testbed: naive vs corrected simulator; gradient agent; LLM agent under three
  conditions (n = … episodes each).

## 3. Results

### 3.1 Calibration (Fig. 2)

### 3.2 Authority framing (Fig. 3)

### 3.3 Reward hacking and the bound monitor (Fig. 4)

## 4. Limitations

Small samples, one physical domain, single-turn questions, keyword-based disclosure scoring,
English prompts only; anything else you found.

## 5. What I would do next

Two or three concrete follow-ups, e.g. agentic tasks with code execution, physical bounds as cheap
monitors inside RL environments, and testing whether models disclose simulator artefacts unprompted.

## How this was built

I designed the study and wrote the physics core; Claude Code helped with scaffolding, tests,
plotting and API plumbing. All numbers are produced by scripts in the repository.
