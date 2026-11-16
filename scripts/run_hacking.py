"""Run the reward-hacking study.

v0.1 runs the gradient agent only (no API cost). The LLM-agent episodes (hacking/llm_agent.py,
tested with a fake client) are planned for after v0.1.
"""

import argparse

import pandas as pd

from wavecheck.config import ROOT, load_config


def gradient_study(cfg: dict) -> pd.DataFrame:
    """Optimise against both simulators; write results/hacking_gradient*.csv and return the
    per-step table."""
    from wavecheck.hacking import gradient_agent

    g = cfg["gradient_agent"]
    rows = []
    for objective in ("naive", "corrected"):
        rows += gradient_agent.optimise(objective, steps=g["steps"], lr=g["lr"], init=g["init"])
    df = pd.DataFrame(rows)
    out = ROOT / cfg.get("output_dir", "results")
    df.to_csv(out / "hacking_gradient.csv", index=False, float_format="%.10g")

    summary = []
    for objective, d in df.groupby("objective", sort=False):
        q_col = "q_naive" if objective == "naive" else "q_corrected"
        above = d[d["above_limit"]]
        last = d.iloc[-1]
        summary += [
            (
                objective,
                "first_step_above_dipole_limit",
                float(above["step"].min()) if len(above) else float("nan"),
            ),
            (objective, "final_eps_real", last["eps_real"]),
            (objective, "final_eps_imag", last["eps_imag"]),
            (objective, "final_q_optimised_over_limit", last[q_col] / last["dipole_limit"]),
            (objective, "final_q_corrected_over_limit", last["q_corrected"] / last["dipole_limit"]),
            (
                objective,
                "max_q_corrected_over_limit",
                float((d["q_corrected"] / d["dipole_limit"]).max()),
            ),
        ]
    s = pd.DataFrame(summary, columns=["objective", "quantity", "value"])
    s.to_csv(out / "hacking_gradient_summary.csv", index=False, float_format="%.6g")
    print(s.to_string(index=False))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hack.yaml")
    parser.add_argument("--gradient-only", action="store_true", help="skip the API episodes")
    args = parser.parse_args()
    cfg = load_config(args.config)
    gradient_study(cfg)
    if not args.gradient_only:
        print("LLM-agent episodes are not part of v0.1; only the gradient study was run.")


if __name__ == "__main__":
    main()
