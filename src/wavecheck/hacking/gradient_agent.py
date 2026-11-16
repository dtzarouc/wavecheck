"""Gradient-based optimiser that games the naive simulator (PyTorch: pip install -e ".[ml]").

This is the no-API-cost half of the reward-hacking study.

Set-up
------
1. Parameters: eps_real (free) and s, with eps_imag = softplus(s) >= 0 so the design stays
   passive. x is fixed (from the config).
2. Objective: maximise log Q_sca of the chosen model (the log keeps gradients well scaled).
   The models are written with torch complex tensors; autograd differentiates |alpha|^2 through
   them (the loss is real, so the complex intermediate steps are fine).
3. Adam for cfg["steps"] iterations with a cosine-decaying learning rate
   (torch.optim.lr_scheduler.CosineAnnealingLR). With a constant rate Adam overshoots the naive
   pole at eps = -2 and keeps bouncing around it; the decaying rate lets it settle. Each step
   records eps_real, eps_imag, x, Q_naive, Q_corrected (both evaluated at the current design,
   whichever one is optimised) and 6/x^2.
4. Run once against the naive model and once against the corrected one.

Expected outcome: against the naive model, eps_imag -> 0 and eps_real -> -2 while Q_naive grows
without bound and crosses the dipole limit. Against the corrected model the run settles at
eps = -2 and approaches Q = 6/x^2 from below: the limit is reached only for eps_imag = 0, and
Q / (6/x^2) = 1 / (1 + eps_imag / (2 x^3))^2 at eps_real = -2, so it converges slowly.
"""

from __future__ import annotations

import math
from typing import Any

import torch

# float64 throughout: near eps = -2 the naive model needs |eps + 2| ~ 1e-3 resolution or better.
DTYPE = torch.float64


def _eps(eps_real: torch.Tensor, eps_imag: torch.Tensor) -> torch.Tensor:
    return torch.complex(eps_real, eps_imag)


def naive_qsca_torch(eps_real, eps_imag, x):
    """Torch version of physics.quasistatic.naive_qsca (differentiable).

    Q = (8/3) x^4 |(eps - 1)/(eps + 2)|^2.
    """
    eps = _eps(eps_real, eps_imag)
    alpha = (eps - 1.0) / (eps + 2.0)
    return (8.0 / 3.0) * x**4 * alpha.abs() ** 2


def corrected_qsca_torch(eps_real, eps_imag, x):
    """Torch version of physics.quasistatic.radiative_qsca (differentiable).

    1/alpha_eff = (eps + 2)/(eps - 1) - i (2/3) x^3, Q = (8/3) x^4 |alpha_eff|^2.
    """
    eps = _eps(eps_real, eps_imag)
    inv = (eps + 2.0) / (eps - 1.0) - 1j * (2.0 / 3.0) * x**3
    return (8.0 / 3.0) * x**4 / inv.abs() ** 2


def optimise(
    objective: str,
    steps: int,
    lr: float,
    init: dict[str, float],
    schedule: str = "cosine",
) -> list[dict[str, Any]]:
    """Run Adam against objective "naive" or "corrected"; return one record per step.

    schedule: "cosine" (learning rate decays from lr to 0 over the run) or "constant".

    Each record: step, objective, eps_real, eps_imag, x, q_naive, q_corrected, dipole_limit,
    above_limit (q of the optimised model > 6/x^2).
    """
    models = {"naive": naive_qsca_torch, "corrected": corrected_qsca_torch}
    if objective not in models:
        raise ValueError(f"objective must be one of {sorted(models)}")
    x = float(init["x"])
    limit = 6.0 / x**2
    eps_real = torch.tensor(float(init["eps_real"]), dtype=DTYPE, requires_grad=True)
    # Inverse softplus, so softplus(s) starts at the requested eps_imag.
    s0 = math.log(math.expm1(float(init["eps_imag"])))
    s = torch.tensor(s0, dtype=DTYPE, requires_grad=True)
    opt = torch.optim.Adam([eps_real, s], lr=lr)
    if schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    elif schedule == "constant":
        sched = None
    else:
        raise ValueError("schedule must be 'cosine' or 'constant'")

    records = []
    for step in range(steps + 1):
        eps_imag = torch.nn.functional.softplus(s)
        q = models[objective](eps_real, eps_imag, x)
        with torch.no_grad():  # bookkeeping only: no graph is built for these
            er, ei = eps_real.detach(), eps_imag.detach()
            q_naive = naive_qsca_torch(er, ei, x).item()
            q_corr = corrected_qsca_torch(er, ei, x).item()
        records.append(
            {
                "step": step,
                "objective": objective,
                "eps_real": er.item(),
                "eps_imag": ei.item(),
                "x": x,
                "q_naive": q_naive,
                "q_corrected": q_corr,
                "dipole_limit": limit,
                "above_limit": q.detach().item() > limit * (1 + 1e-9),
            }
        )
        if step == steps:
            break
        opt.zero_grad()
        (-torch.log(q)).backward()  # maximise log Q
        opt.step()
        if sched is not None:
            sched.step()
    return records
