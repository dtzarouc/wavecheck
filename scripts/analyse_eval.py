"""Turn results/summary_eval.csv into the M2 tables (and Figures 2 and 3).

Writes results/eval_metrics.csv (per model: accuracy, calibration), results/eval_authority.csv
(per model and framing: accuracy with Wilson 95% CI, paired flips vs neutral, exact McNemar p)
and results/eval_formula_errors.csv (report items whose reasoning uses the factor
(m^2 - 1)/(m^2 + 1) instead of (eps - 1)/(eps + 2); flagged by a regular expression, the
flagged errors were also read by hand).
"""

import re

import pandas as pd

from wavecheck.analysis import plots
from wavecheck.analysis.stats import mcnemar_exact, wilson_interval
from wavecheck.config import ROOT
from wavecheck.evals import grading
from wavecheck.io import read_jsonl

RES = ROOT / "results"
WRONG_FACTOR = re.compile(r"(m²|m\^2|ε|eps|epsilon)\s*\+\s*1\s*\)")
# Below this x the static answer -2 is already within the 2% tolerance (exact eps_res ~ -2-2.4x^2).
STATIC_OK = "static_within_tol"


def metrics(d: pd.DataFrame, items: dict) -> pd.DataFrame:
    rows = []
    for model, m in d.groupby("model"):
        rs = m[m["family"] == "resonance_shift"]
        rs_g = rs[rs["correct"].notna()]
        hard = rs_g[[not items[t]["meta"].get(STATIC_OK, False) for t in rs_g["task_id"]]]
        g = m[m["correct"].notna() & m["confidence"].notna()]
        c, y = g["confidence"].tolist(), g["correct"].astype(bool).tolist()
        rg = rs_g[rs_g["confidence"].notna()]
        rows.append(
            {
                "model": model,
                "resonance_n": len(rs),
                "resonance_no_answer": int(rs["correct"].isna().sum()),
                "resonance_accuracy": rs_g["correct"].mean(),
                "resonance_accuracy_static_wrong": hard["correct"].mean(),
                "resonance_n_static_wrong": len(hard),
                "resonance_mean_confidence": rg["confidence"].mean(),
                "resonance_ece": grading.expected_calibration_error(
                    rg["confidence"].tolist(), rg["correct"].astype(bool).tolist()
                ),
                "all_n_graded": len(g),
                "all_accuracy": g["correct"].mean(),
                "all_mean_confidence": g["confidence"].mean(),
                "all_ece": grading.expected_calibration_error(c, y),
                "all_brier": grading.brier(c, y),
            }
        )
    return pd.DataFrame(rows)


def authority(d: pd.DataFrame) -> pd.DataFrame:
    a = d[d["family"].isin(["report_check", "authority"])]
    wide = a.pivot_table(
        index=["model", "item"], columns="framing", values="correct", aggfunc="first"
    )
    rows = []
    for model, w in wide.groupby(level="model"):
        for framing in ["neutral", "journal", "expert"]:
            col = w[framing].dropna().astype(bool)
            lo, hi = wilson_interval(int(col.sum()), len(col))
            row = {
                "model": model,
                "framing": framing,
                "n": len(col),
                "accuracy": col.mean(),
                "ci_low": lo,
                "ci_high": hi,
            }
            if framing != "neutral":
                pair = w[["neutral", framing]].dropna().astype(bool)
                b = int((pair["neutral"] & ~pair[framing]).sum())
                c_ = int((~pair["neutral"] & pair[framing]).sum())
                row.update(
                    {
                        "lost_vs_neutral": b,
                        "gained_vs_neutral": c_,
                        "mcnemar_p": mcnemar_exact(b, c_),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def formula_errors(d: pd.DataFrame) -> pd.DataFrame:
    texts = {
        (r["meta"]["task_id"], r["request"]["model"]): r["text"]
        for r in read_jsonl(RES / "raw" / "eval_20261116.jsonl")
    }
    a = d[d["family"] != "resonance_shift"].copy()
    a["wrong_factor"] = [
        bool(WRONG_FACTOR.search(texts[(t, m)]))
        for t, m in zip(a["task_id"], a["model"], strict=True)
    ]
    return (
        a.groupby("model")
        .apply(
            lambda g: pd.Series(
                {
                    "report_answers": len(g),
                    "wrong_factor_flagged": int(g["wrong_factor"].sum()),
                    "errors": int(g["correct"].eq(False).sum()),
                    "errors_with_wrong_factor": int(
                        (g["correct"].eq(False) & g["wrong_factor"]).sum()
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )


def costs(cfg: dict) -> pd.DataFrame:
    """API usage and cost per run (pilot, pilot2, full) and model, from the logged token counts."""
    rows = []
    for prefix in ("pilot", "pilot2", "eval"):
        for path in sorted((RES / "raw").glob(f"{prefix}_2*.jsonl")):
            for rec in read_jsonl(path):
                model = rec["request"]["model"]
                p_in, p_out = cfg["price_per_mtok"][model]
                u = rec["usage"]
                rows.append(
                    {
                        "run": prefix,
                        "model": model,
                        "date": rec["timestamp"][:10],
                        "input_tokens": u["input_tokens"],
                        "output_tokens": u["output_tokens"],
                        "cost_usd": (u["input_tokens"] * p_in + u["output_tokens"] * p_out) / 1e6,
                    }
                )
    df = pd.DataFrame(rows)
    out = (
        df.groupby(["run", "model", "date"])
        .agg(
            calls=("cost_usd", "size"),
            input_tokens=("input_tokens", "sum"),
            output_tokens=("output_tokens", "sum"),
            cost_usd=("cost_usd", "sum"),
        )
        .reset_index()
    )
    total = out.groupby("run", as_index=False)[
        ["calls", "input_tokens", "output_tokens", "cost_usd"]
    ].sum()
    total["model"], total["date"] = "all", ""
    return pd.concat([out, total], ignore_index=True)


def errors(d: pd.DataFrame) -> pd.DataFrame:
    """Every wrong or unanswered item of the full run, for the write-up."""
    e = d[d["correct"] != True].copy()  # noqa: E712  (True/False/NaN column)
    return e[
        [
            "model",
            "task_id",
            "family",
            "framing",
            "x",
            "truth",
            "answer",
            "confidence",
            "correct",
            "stop_reason",
            "output_tokens",
        ]
    ]


def main() -> None:
    d = pd.read_csv(RES / "summary_eval.csv")
    items = {i["id"]: i for i in read_jsonl(ROOT / "data" / "generated" / "eval_items.jsonl")}
    for i in items.values():  # mark resonance items where answering -2 would count as correct
        if i["family"] == "resonance_shift":
            i["meta"][STATIC_OK] = abs(-2.0 / i["answer"] - 1) <= i["tolerance_rel"]
    from wavecheck.config import load_config

    m, a, f = metrics(d, items), authority(d), formula_errors(d)
    costs(load_config("configs/eval.yaml")).to_csv(
        RES / "eval_cost.csv", index=False, float_format="%.4f"
    )
    errors(d).to_csv(RES / "eval_errors.csv", index=False)
    m.to_csv(RES / "eval_metrics.csv", index=False, float_format="%.4g")
    a.to_csv(RES / "eval_authority.csv", index=False, float_format="%.4g")
    f.to_csv(RES / "eval_formula_errors.csv", index=False)
    plots.fig_calibration(RES / "summary_eval.csv", ROOT / "figures" / "fig2_calibration.png")
    plots.fig_authority(RES / "eval_authority.csv", ROOT / "figures" / "fig3_authority.png")
    with pd.option_context("display.width", 200):
        print(
            m.round(3).to_string(index=False),
            a.round(3).to_string(index=False),
            f.to_string(index=False),
            sep="\n\n",
        )


if __name__ == "__main__":
    main()
