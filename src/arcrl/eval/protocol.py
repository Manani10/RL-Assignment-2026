"""Reproducible evaluation protocol — OWNER: Member 1.

One "run" = (agent, game, seed). The agent plays the game with a fixed
interaction budget, learning online if it is a learning agent. We log every
episode (successful or not) and the action count at which each level was
completed, so we can draw learning curves vs interaction budget and compute
the local RHAE.

Outputs JSONL, one record per episode, to results/<tag>/episodes.jsonl and a
run-level summary to results/<tag>/runs.jsonl. Never overwrite: tags are unique.
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from arcrl.agents import build_agent
from arcrl.env.adapter import AdapterConfig, make_env


@dataclass
class EvalConfig:
    agent: str = "random"
    games: list[str] = field(default_factory=lambda: ["mock"])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    total_budget: int = 5_000            # env actions per (game, seed) run, incl. resets
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    tag: str = "dev"
    out_dir: str = "results"


def run_one(cfg: EvalConfig, game: str, seed: int, ep_log, verbose=False) -> dict:
    env = make_env(game, cfg.adapter, seed=seed)
    agent = build_agent(cfg.agent, env.n_actions, 16 * cfg.adapter.frame_stack, seed)
    agent.begin_game(game)

    used, episode, best_levels, wins = 0, 0, 0, 0
    curve = []  # (actions_used_total, levels_completed_this_episode)
    t0 = time.time()
    while used < cfg.total_budget:
        obs, info = env.reset(seed=seed)
        level_actions, last_mark, done = [], 0, False
        while not done and used < cfg.total_budget:
            a = agent.act(obs, info)
            nxt, r, term, trunc, info = env.step(a)
            used += 1
            done = term or trunc
            agent.observe(obs, a, r, nxt, term, info)
            if r > 0:  # level(s) completed
                level_actions.append(env.actions_used - last_mark)
                last_mark = env.actions_used
                curve.append((used, info["levels_completed"]))
            obs = nxt
        best_levels = max(best_levels, info["levels_completed"])
        wins += int(info["state"] == "WIN")
        ep_log.write(json.dumps({
            "agent": cfg.agent, "game": game, "seed": seed, "episode": episode,
            "levels_completed": info["levels_completed"], "win_levels": info["win_levels"],
            "actions": env.actions_used, "actions_per_level": level_actions,
            "state": info["state"], "budget_used": used,
        }) + "\n")
        episode += 1
        if verbose:
            print(f"[{cfg.agent}] {game} seed={seed} ep={episode} levels={info['levels_completed']} used={used}")
    return {
        "agent": cfg.agent, "game": game, "seed": seed, "episodes": episode,
        "best_levels": best_levels, "wins": wins, "actions": used,
        "wall_s": round(time.time() - t0, 2), "curve": curve,
    }


def evaluate(cfg: EvalConfig, verbose=False) -> list[dict]:
    out = Path(cfg.out_dir) / cfg.tag
    out.mkdir(parents=True, exist_ok=False)
    meta = {"config": asdict(cfg), "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__}
    try:
        import arc_agi  # noqa: F401
        from importlib.metadata import version
        meta["arc_agi"] = version("arc-agi")
    except Exception:
        meta["arc_agi"] = None
    try:
        import torch
        meta["torch"] = torch.__version__
        meta["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        meta["torch"] = None
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    runs = []
    with open(out / "episodes.jsonl", "w") as ep_log, open(out / "runs.jsonl", "w") as run_log:
        for game in cfg.games:
            for seed in cfg.seeds:
                res = run_one(cfg, game, seed, ep_log, verbose)
                run_log.write(json.dumps(res) + "\n")
                runs.append(res)
    return runs
