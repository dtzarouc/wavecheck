"""Build the question set from configs/eval.yaml and save it to data/generated/."""

import argparse

from wavecheck.config import ROOT, load_config
from wavecheck.evals.tasks import build_dataset
from wavecheck.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tasks = build_dataset(cfg)
    out = ROOT / "data" / "generated" / "eval_items.jsonl"
    write_jsonl(out, (t.to_dict() for t in tasks))
    print(f"wrote {len(tasks)} items to {out}")


if __name__ == "__main__":
    main()
