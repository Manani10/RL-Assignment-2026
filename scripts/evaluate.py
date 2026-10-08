"""Run the evaluation protocol from a YAML/JSON config or CLI flags.

Examples
    python scripts/evaluate.py --agent random --games mock --seeds 0 1 2 --budget 2000 --tag random_mock
    python scripts/evaluate.py --config configs/dqn_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arcrl.env.adapter import AdapterConfig  # noqa: E402
from arcrl.eval.protocol import EvalConfig, evaluate  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--agent", default="random")
    p.add_argument("--games", nargs="+", default=["mock"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--budget", type=int, default=2000)
    p.add_argument("--tag", required=False)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()

    if a.config:
        raw = json.loads(Path(a.config).read_text())
        raw["adapter"] = AdapterConfig(**raw.get("adapter", {}))
        cfg = EvalConfig(**raw)
    else:
        cfg = EvalConfig(agent=a.agent, games=a.games, seeds=a.seeds, total_budget=a.budget,
                         tag=a.tag or f"{a.agent}_{'-'.join(a.games)}")
    runs = evaluate(cfg, verbose=a.verbose)
    for r in runs:
        print(f"{r['agent']:>8} {r['game']:>10} seed={r['seed']} best_levels={r['best_levels']} "
              f"wins={r['wins']} episodes={r['episodes']} wall={r['wall_s']}s")


if __name__ == "__main__":
    main()
