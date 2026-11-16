"""Run the question set against the configured models, with resume and a cost guard.

Calls run in a small thread pool (the API is I/O-bound, so threads are enough; the GIL is released
while waiting on the network). Every response is appended to results/raw/eval_<YYYYMMDD>.jsonl
under a lock, so concurrent writes never interleave. A rerun skips every (task, model) pair that
already has a logged response in any results/raw/eval_*.jsonl, which makes the run resumable.
Failed calls go to results/raw/eval_errors_<YYYYMMDD>.jsonl and are retried on the next run.
"""

from __future__ import annotations

import datetime as dt
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from wavecheck.config import ROOT
from wavecheck.evals import grading, models
from wavecheck.evals.tasks import Task, build_dataset
from wavecheck.io import append_jsonl, read_jsonl, write_jsonl


def estimate_cost_usd(
    n_calls_per_model: int, cfg: dict[str, Any], tokens_in: int = 600, tokens_out: int = 1500
) -> float:
    """Rough upper estimate from cfg["price_per_mtok"]; thinking tokens are billed as output."""
    total = 0.0
    for model in cfg["models"]:
        p_in, p_out = cfg["price_per_mtok"][model]
        total += n_calls_per_model * (tokens_in * p_in + tokens_out * p_out) / 1e6
    return total


def _raw_dir(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg.get("output_dir", "results") / "raw"


def done_pairs(raw_dir: Path, prefix: str = "eval") -> set[tuple[str, str]]:
    """(task_id, requested model) pairs that already have a logged response."""
    pairs: set[tuple[str, str]] = set()
    for path in sorted(raw_dir.glob(f"{prefix}_2*.jsonl")):
        for rec in read_jsonl(path):
            pairs.add((rec["meta"]["task_id"], rec["request"]["model"]))
    return pairs


def run(
    cfg: dict[str, Any],
    *,
    dry_run: bool = False,
    limit: int | None = None,
    client: Any = None,
    workers: int = 6,
) -> Path:
    """Run every task x model pair that is not already in the raw log.

    1. Build the dataset (tasks.build_dataset) and save it to data/generated/.
    2. Estimate the cost; stop with a clear message if it exceeds cfg["budget_usd"].
    3. For each remaining pair call models.complete(..., log_path=results/raw/eval_<YYYYMMDD>.jsonl,
       meta={"task_id": ..., "family": ...}).
    dry_run prints the plan and the estimate only; limit runs the first N tasks of each family
    (the pilot). Returns the raw log path.
    """
    tasks = build_dataset(cfg)
    write_jsonl(ROOT / "data" / "generated" / "eval_items.jsonl", (t.to_dict() for t in tasks))
    if limit is not None:
        per_family: dict[str, int] = {}
        kept = []
        for t in tasks:
            if per_family.get(t.family, 0) < limit:
                kept.append(t)
                per_family[t.family] = per_family.get(t.family, 0) + 1
        tasks = kept

    raw_dir = _raw_dir(cfg)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    # Pilot runs (limit set) log to pilot_*.jsonl so they never count towards the frozen full run.
    prefix = "pilot" if limit is not None else "eval"
    raw_path = raw_dir / f"{prefix}_{stamp}.jsonl"
    err_path = raw_dir / f"{prefix}_errors_{stamp}.jsonl"
    done = done_pairs(raw_dir, prefix)
    todo = [(t, m) for m in cfg["models"] for t in tasks if (t.id, m) not in done]

    n_per_model = max((sum(1 for _, m in todo if m == mm) for mm in cfg["models"]), default=0)
    cost = estimate_cost_usd(n_per_model, cfg, tokens_out=cfg.get("max_tokens", 4096))
    print(
        f"{len(tasks)} tasks x {len(cfg['models'])} models; {len(todo)} calls to make; "
        f"upper cost estimate ${cost:.2f} (every call at max_tokens) vs budget "
        f"${cfg['budget_usd']:.2f}"
    )
    if dry_run or not todo:
        return raw_path
    if cost > cfg["budget_usd"]:
        raise SystemExit(f"Estimated cost ${cost:.2f} exceeds budget_usd; ask before running.")

    client = client or models.get_client()
    lock = threading.Lock()
    stop = threading.Event()  # set on an authentication error: no point making more calls

    def one(task: Task, model: str) -> None:
        if stop.is_set():
            return
        meta = {"task_id": task.id, "family": task.family}
        try:
            record = models.complete(
                client, model, task.prompt, max_tokens=cfg.get("max_tokens", 4096), meta=meta
            )
        except Exception as exc:  # logged and retried on the next run
            if type(exc).__name__ in {"AuthenticationError", "PermissionDeniedError"}:
                stop.set()
            with lock:
                append_jsonl(
                    err_path,
                    {
                        "model": model,
                        "meta": meta,
                        "error": repr(exc),
                        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                    },
                )
            return
        with lock:
            append_jsonl(raw_path, record)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, t, m) for t, m in todo]
        for i, _ in enumerate(as_completed(futures), 1):
            if i % 25 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} calls finished", flush=True)
    if stop.is_set():
        raise SystemExit("API key rejected (authentication error); fix .env and rerun.")
    return raw_path


def summarise(raw_path: str | Path | None = None, prefix: str = "eval"):
    """One row per task x model: parsed answer, confidence, correct (True/False/None), usage.

    Reads every results/raw/<prefix>_2*.jsonl (raw_path only selects the folder), keeps the first
    logged response per (task, model), grades it against data/generated/eval_items.jsonl, and
    writes results/summary_<prefix>.csv. Returns the DataFrame.
    """
    raw_dir = Path(raw_path).parent if raw_path else ROOT / "results" / "raw"
    items = {
        d["id"]: Task(**d) for d in read_jsonl(ROOT / "data" / "generated" / "eval_items.jsonl")
    }
    rows, seen = [], set()
    for path in sorted(raw_dir.glob(f"{prefix}_2*.jsonl")):
        for rec in read_jsonl(path):
            key = (rec["meta"]["task_id"], rec["request"]["model"])
            if key in seen or key[0] not in items:
                continue
            seen.add(key)
            task = items[key[0]]
            answer, confidence = grading.parse_answer(rec["text"])
            rows.append(
                {
                    "task_id": task.id,
                    "family": task.family,
                    "model": key[1],
                    "framing": task.meta.get("framing", ""),
                    "item": task.meta.get("item", task.id),
                    "x": task.meta.get("x"),
                    "truth": task.answer,
                    "answer": answer,
                    "confidence": confidence,
                    "correct": grading.grade(task, answer),
                    "stop_reason": rec["stop_reason"],
                    "input_tokens": rec["usage"].get("input_tokens"),
                    "output_tokens": rec["usage"].get("output_tokens"),
                    "raw_file": path.name,
                }
            )
    df = pd.DataFrame(rows)
    out = ROOT / "results" / f"summary_{prefix}.csv"
    df.to_csv(out, index=False)
    return df
