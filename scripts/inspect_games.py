"""Day-1 environment inspection — OWNER: Member 1.

The brief (§2) asks us to inspect the observation format, available actions,
reset behaviour, level progression and scoring BEFORE designing agents. This
script does that for every public game and writes:

    results/inspection/games.json     per-game facts (+ package versions)
    results/inspection/games.md       table to paste into the report / appendix

Per game it records: number of levels, frame shape(s) returned per step,
colours used, which actions are offered, and — from a short random probe —
how often each action actually changes the frame ("action effect rate"),
what RESET does, and whether GAME_OVER happens.

Usage
    python scripts/inspect_games.py                  # all games
    python scripts/inspect_games.py --games ls20 --probe 300
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arcrl.env.encoding import grid_hash, last_grid  # noqa: E402


def pkg(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def probe(arc, game_id: str, steps: int, seed: int = 0) -> dict:
    from arcengine import GameAction

    env = arc.make(game_id, seed=seed)
    if env is None:
        return {"game_id": game_id, "error": "could not create"}
    obs = env.reset() or env.step(GameAction.RESET)
    rng = random.Random(seed)
    first = np.asarray(obs.frame)
    info = {
        "game_id": game_id,
        "win_levels": int(getattr(obs, "win_levels", 0) or 0),
        "reset_frame_layers": int(first.shape[0]) if first.ndim == 3 else 1,
        "initial_state": obs.state.name,
    }
    offered, tried, changed = Counter(), Counter(), Counter()
    layers_per_step, colours = Counter(), set(np.unique(last_grid(obs.frame)).tolist())
    states, max_levels, game_overs, seen = Counter(), 0, 0, {grid_hash(last_grid(obs.frame))}
    prev = last_grid(obs.frame)
    for _ in range(steps):
        acts = [a for a in env.action_space if a.name != "RESET"]
        for a in acts:
            offered[a.name] += 1
        if not acts:
            break
        a = rng.choice(acts)
        data = {"x": rng.randint(0, 63), "y": rng.randint(0, 63)} if a.is_complex() else {}
        obs = env.step(a, data=data)
        arr = np.asarray(obs.frame)
        layers_per_step[int(arr.shape[0]) if arr.ndim == 3 else 1] += 1
        g = last_grid(obs.frame)
        colours |= set(np.unique(g).tolist())
        tried[a.name] += 1
        changed[a.name] += int(not np.array_equal(g, prev))
        seen.add(grid_hash(g))
        states[obs.state.name] += 1
        max_levels = max(max_levels, int(obs.levels_completed))
        prev = g
        if obs.state.name == "GAME_OVER":
            game_overs += 1
            obs = env.step(GameAction.RESET)
            prev = last_grid(obs.frame)
        elif obs.state.name == "WIN":
            break
    info.update({
        "actions_offered": sorted(offered),
        "action_effect_rate": {k: round(changed[k] / tried[k], 2) for k in sorted(tried)},
        "colours_seen": sorted(colours),
        "frames_per_step": dict(layers_per_step),
        "distinct_frames_seen": len(seen),
        "levels_completed_by_random": max_levels,
        "game_overs": game_overs,
        "states": dict(states),
        "probe_steps": steps,
    })
    sc = arc.get_scorecard()
    info["scorecard_score"] = getattr(sc, "score", None) if sc else None
    return info


def to_markdown(rows) -> str:
    head = "| game | levels | actions offered | effect rate (ACTION: frac frames changed) | random levels | game-overs |\n"
    head += "|---|---|---|---|---|---|\n"
    for r in rows:
        if "error" in r:
            head += f"| {r['game_id']} | error | | | | |\n"
            continue
        eff = ", ".join(f"{k[-1]}:{v}" for k, v in r["action_effect_rate"].items())
        acts = ",".join(a.replace("ACTION", "") for a in r["actions_offered"])
        head += (f"| {r['game_id']} | {r['win_levels']} | {acts} | {eff} | "
                 f"{r['levels_completed_by_random']} | {r['game_overs']} |\n")
    return head


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", nargs="*")
    p.add_argument("--probe", type=int, default=500, help="random steps per game")
    p.add_argument("--out", default="results/inspection")
    a = p.parse_args()

    import arc_agi

    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    catalogue = [{"game_id": e.game_id, "title": getattr(e, "title", None),
                  "tags": list(getattr(e, "tags", []) or [])} for e in envs]
    games = a.games or [c["game_id"] for c in catalogue]
    rows = []
    for gid in games:
        print(f"probing {gid} ...", flush=True)
        try:
            rows.append(probe(arc_agi.Arcade(), gid, a.probe))
        except Exception as e:  # keep going, record the failure
            rows.append({"game_id": gid, "error": repr(e)})

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"python": platform.python_version(), "arc-agi": pkg("arc-agi"), "arcengine": pkg("arcengine"),
            "numpy": np.__version__, "n_games": len(catalogue)}
    (out / "games.json").write_text(json.dumps({"meta": meta, "catalogue": catalogue, "probe": rows}, indent=2))
    (out / "games.md").write_text(to_markdown(rows))
    print(to_markdown(rows))
    print(f"\nversions: {meta}\nwrote {out}/games.json and games.md")


if __name__ == "__main__":
    main()
