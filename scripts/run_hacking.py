"""Run the reward-hacking study: the gradient agent (free) and the LLM agent episodes (API)."""

import argparse

from wavecheck.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hack.yaml")
    parser.add_argument("--gradient-only", action="store_true", help="skip the API episodes")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # TODO(M3):
    # 1. gradient_agent.optimise("naive", ...) and ("corrected", ...)
    #    -> results/hacking_gradient.csv
    # 2. unless --gradient-only: estimate cost, then for each model x condition x episode run
    #    llm_agent.run_episode(...) with a fresh DesignEnv, append every EpisodeResult to
    #    results/raw/hacking_<YYYYMMDD>.jsonl, and score it with monitor.* into
    #    results/hacking_episodes.csv
    raise NotImplementedError(f"M3 not implemented yet (config loaded: {sorted(cfg)})")


if __name__ == "__main__":
    main()
