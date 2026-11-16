"""The evaluation runner with a fake client: no network, no cost."""

from types import SimpleNamespace as NS

import pytest

from wavecheck.config import load_config
from wavecheck.evals import runner


class FakeUsage:
    def model_dump(self):
        return {"input_tokens": 10, "output_tokens": 20}


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        block = NS(type="text", text="ANSWER: YES\nCONFIDENCE: 0.9")
        return NS(
            model=kwargs["model"],
            id=f"msg_{self.calls}",
            stop_reason="end_turn",
            content=[block],
            usage=FakeUsage(),
            model_dump=lambda: {"fake": True},
        )


@pytest.fixture
def cfg(tmp_path):
    c = load_config("configs/eval.yaml")
    c["output_dir"] = str(tmp_path)  # absolute, so ROOT / output_dir points into tmp_path
    c["models"] = ["model-a", "model-b"]
    c["price_per_mtok"] = {"model-a": [1.0, 1.0], "model-b": [1.0, 1.0]}
    return c


def test_cost_estimate():
    c = {"models": ["m"], "price_per_mtok": {"m": [2.0, 10.0]}}
    assert runner.estimate_cost_usd(100, c, tokens_in=1000, tokens_out=1000) == pytest.approx(1.2)


def test_run_logs_every_pair_and_resumes(cfg, tmp_path):
    client = FakeClient()
    raw = runner.run(cfg, limit=2, client=client, workers=3)
    assert client.calls == 2 * 3 * 2  # 2 per family x 3 families x 2 models
    assert raw.name.startswith("pilot_")
    assert len(runner.done_pairs(raw.parent, "pilot")) == client.calls
    runner.run(cfg, limit=2, client=client)
    assert client.calls == 12  # nothing re-run


def test_budget_guard_stops_before_any_call(cfg):
    cfg["budget_usd"] = 0.0
    client = FakeClient()
    with pytest.raises(SystemExit):
        runner.run(cfg, limit=2, client=client)
    assert client.calls == 0


def test_hard_stop_on_actual_spend(cfg, monkeypatch):
    monkeypatch.setattr(runner, "estimate_cost_usd", lambda *a, **k: 0.0)  # pass the pre-check
    cfg["budget_usd"] = 1e-4  # each fake call costs 30 tokens x $1/Mtok = 3e-5
    client = FakeClient()
    with pytest.raises(SystemExit, match="budget"):
        runner.run(cfg, limit=10, client=client, workers=1)
    assert client.calls == 4  # 4 x 3e-5 >= 1e-4, then no new calls (60 were queued)
