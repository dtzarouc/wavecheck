"""The PyTorch gradient agent (skipped when torch is not installed, e.g. on CI)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wavecheck.hacking import gradient_agent as ga  # noqa: E402
from wavecheck.physics import quasistatic as qs  # noqa: E402


@pytest.mark.parametrize("eps", [-3 + 0.5j, -2.01 + 1e-3j, 2.25 + 0j, -1.5 + 0.2j])
def test_torch_models_match_numpy(eps):
    x = 0.1
    er = torch.tensor(eps.real, dtype=torch.float64)
    ei = torch.tensor(eps.imag, dtype=torch.float64)
    assert float(ga.naive_qsca_torch(er, ei, x)) == pytest.approx(float(qs.naive_qsca(eps, x)))
    assert float(ga.corrected_qsca_torch(er, ei, x)) == pytest.approx(
        float(qs.radiative_qsca(eps, x))
    )


def test_gradients_flow():
    er = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
    ei = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    ga.naive_qsca_torch(er, ei, 0.1).backward()
    assert er.grad is not None and np.isfinite(float(er.grad))


INIT = {"eps_real": -3.0, "eps_imag": 0.5, "x": 0.1}


def test_naive_run_exploits_the_flaw():
    rec = ga.optimise("naive", steps=2000, lr=0.01, init=INIT)
    assert rec[0]["eps_imag"] == pytest.approx(0.5)
    assert any(r["above_limit"] for r in rec)
    assert rec[-1]["q_naive"] > rec[-1]["dipole_limit"]
    assert rec[-1]["eps_imag"] >= 0


def test_corrected_run_saturates_at_the_limit():
    rec = ga.optimise("corrected", steps=2000, lr=0.01, init=INIT)
    assert not any(r["above_limit"] for r in rec)
    assert rec[-1]["q_corrected"] == pytest.approx(rec[-1]["dipole_limit"], rel=1e-3)
