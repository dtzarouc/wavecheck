"""Run the evaluation. Start with --dry-run, then a pilot with --limit 10."""

import argparse

from wavecheck.config import load_config
from wavecheck.evals.runner import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and cost only")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N tasks")
    args = parser.parse_args()

    raw_path = run(load_config(args.config), dry_run=args.dry_run, limit=args.limit)
    print(f"raw log: {raw_path}")


if __name__ == "__main__":
    main()
