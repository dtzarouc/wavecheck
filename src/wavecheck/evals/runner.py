"""Run the question set against the configured models, with resume and a cost guard.

TODO: implement with Claude Code once tasks.py and grading.py pass their tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def estimate_cost_usd(
    n_calls_per_model: int, cfg: dict[str, Any], tokens_in: int = 600, tokens_out: int = 1500
) -> float:
    """Rough upper estimate from cfg["price_per_mtok"]; thinking tokens are billed as output."""
    raise NotImplementedError


def run(cfg: dict[str, Any], *, dry_run: bool = False, limit: int | None = None) -> Path:
    """Run every task x model pair that is not already in the raw log.

    1. Build the dataset (tasks.build_dataset) and save it to data/generated/.
    2. Estimate the cost; stop with a clear message if it exceeds cfg["budget_usd"].
    3. For each remaining pair call models.complete(..., log_path=results/raw/eval_<YYYYMMDD>.jsonl,
       meta={"task_id": ..., "family": ...}).
    dry_run prints the plan and the estimate only; limit runs the first N tasks (the pilot).
    Returns the raw log path.
    """
    raise NotImplementedError


def summarise(raw_path: str | Path):
    """One row per task x model: parsed answer, confidence, correct (True/False/None), usage.

    Returns a pandas DataFrame and writes results/summary_eval.csv.
    """
    raise NotImplementedError
