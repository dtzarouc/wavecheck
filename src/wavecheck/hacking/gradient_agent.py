"""Gradient-based optimiser that games the naive simulator (PyTorch: pip install -e ".[ml]").

This is the no-API-cost half of the reward-hacking study, and Dimi's PyTorch exercise.

Plan
----
1. Parameters: eps_real (free) and s, with eps_imag = softplus(s) >= 0 so the design stays
   passive; keep x fixed from the config (or map it into x_range with a sigmoid).
2. Objective: maximise log Q_sca of the chosen model (the log keeps gradients well scaled).
   Write the models with real arithmetic or torch complex tensors; both support autograd.
3. Adam for cfg["steps"] iterations; record each step's eps_real, eps_imag, x, Q_naive,
   Q_corrected and the dipole limit 6/x^2.
4. Run once against the naive model and once against the corrected one.

Expected outcome: against the naive model, eps_imag -> 0 and eps_real -> -2 while Q_naive grows
without bound and crosses the dipole limit. Against the corrected model the run settles at
eps = -2 with Q = 6/x^2. Plot both; the bound line is the monitor.

TODO(Dimi): implement.
"""

from __future__ import annotations

from typing import Any


def naive_qsca_torch(eps_real, eps_imag, x):
    """Torch version of physics.quasistatic.naive_qsca (differentiable)."""
    raise NotImplementedError


def corrected_qsca_torch(eps_real, eps_imag, x):
    """Torch version of physics.quasistatic.radiative_qsca (differentiable)."""
    raise NotImplementedError


def optimise(objective: str, steps: int, lr: float, init: dict[str, float]) -> list[dict[str, Any]]:
    """Run Adam against objective "naive" or "corrected"; return one record per step."""
    raise NotImplementedError
