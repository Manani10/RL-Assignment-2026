"""Turn evaluation runs into report tables and figures — OWNER: Member 1.

Reads results/<tag>/runs.jsonl for each tag and writes to results/analysis/<name>/:
    summary.md / summary.csv   per (tag, game): seeds, levels, official score, wins,
                               actions to first level, wall time  (mean ± sd over seeds)
    overall.md                 per tag: mean official score over games, mean ± sd over seeds
    curves.pdf / curves.png    learning curves: best levels completed vs interaction budget,
                               one panel per game, thin line per seed + thick mean

Every agent version (baseline, imp1, imp2, ablations) goes through this same
script so all figures in the report share one style and one protocol.

Usage
    python scripts/analyze.py --tags random_baseline_v1 dqn_baseline_v1 ppo_baseline_v1 --name baselines
Pass tags in the SAME order every time: colour follows the position in this list.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

# Categorical slots in fixed order (validated palette); random/sanity check is neutral grey.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GREY = "#8a8a85"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"


def load(results: Path, tag: str) -> list[dict]:
    path = results / tag / "runs.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ms(xs) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:.3g}"
    return f"{st.mean(xs):.3g} ± {st.stdev(xs):.2g}"


def first_level_actions(run):
    return run["curve"][0][0] if run.get("curve") else None


def best_so_far(run, grid):
    """Best levels completed in any episode so far, evaluated on a budget grid."""
    pts = sorted(run.get("curve") or [])
    out, best, j = np.zeros(len(grid)), 0, 0
    for i, g in enumerate(grid):
        while j < len(pts) and pts[j][0] <= g:
            best = max(best, pts[j][1])
            j += 1
        out[i] = best
    return out


def tables(runs_by_tag: dict[str, list[dict]]):
    rows, overall = [], []
    for tag, runs in runs_by_tag.items():
        by_game = defaultdict(list)
        for r in runs:
            by_game[r["game"]].append(r)
        for game, rs in sorted(by_game.items()):
            rows.append({
                "tag": tag, "game": game, "seeds": len(rs),
                "best_levels": ms([r["best_levels"] for r in rs]),
                "official_score": ms([r.get("official_score") for r in rs]),
                "wins": sum(r["wins"] for r in rs),
                "actions_to_first_level": ms([first_level_actions(r) for r in rs]),
                "solved_seeds": sum(1 for r in rs if r["best_levels"] > 0),
                "budget": rs[0]["actions"],
                "wall_s": ms([r["wall_s"] for r in rs]),
            })
        per_seed = defaultdict(list)  # seed -> official scores across games
        for r in runs:
            if r.get("official_score") is not None:
                per_seed[r["seed"]].append(r["official_score"])
        seed_means = [st.mean(v) for v in per_seed.values()]
        overall.append({"tag": tag, "games": len(by_game), "seeds": len({r["seed"] for r in runs}),
                        "mean_official_score": ms(seed_means),
                        "total_levels": sum(r["best_levels"] for r in runs)})
    return rows, overall


def to_md(rows) -> str:
    if not rows:
        return ""
    keys = list(rows[0])
    out = "| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n"
    for r in rows:
        out += "| " + " | ".join(str(r[k]) for k in keys) + " |\n"
    return out


def plot(runs_by_tag, path_stem: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping curves")
        return
    games = sorted({r["game"] for rs in runs_by_tag.values() for r in rs})
    budget = max(r["actions"] for rs in runs_by_tag.values() for r in rs)
    grid = np.linspace(0, budget, 200)
    ncol = min(3, len(games))
    nrow = int(np.ceil(len(games) / ncol))
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.0 * nrow), squeeze=False, sharex=True)
    colour, slot = {}, 0
    for tag in runs_by_tag:
        if tag.startswith("random"):
            colour[tag] = GREY
        else:
            colour[tag] = SLOTS[slot % len(SLOTS)]
            slot += 1
    for ax, game in zip(axes.flat, games):
        for tag, runs in runs_by_tag.items():
            rs = [r for r in runs if r["game"] == game]
            if not rs:
                continue
            curves = np.stack([best_so_far(r, grid) for r in rs])
            for c in curves:
                ax.plot(grid, c, color=colour[tag], lw=0.8, alpha=0.35, drawstyle="steps-post")
            ax.plot(grid, curves.mean(0), color=colour[tag], lw=2, drawstyle="steps-post", label=tag)
        ax.set_title(game, fontsize=8, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_ylim(bottom=0)
    for ax in axes.flat[len(games):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("interaction budget (actions)")
    for ax in axes[:, 0]:
        ax.set_ylabel("levels completed")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(path_stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--name", default="latest")
    p.add_argument("--results", default="results")
    a = p.parse_args()
    results = Path(a.results)
    runs_by_tag = {t: load(results, t) for t in a.tags}
    out = results / "analysis" / a.name
    out.mkdir(parents=True, exist_ok=True)
    rows, overall = tables(runs_by_tag)
    (out / "summary.md").write_text(to_md(rows))
    (out / "overall.md").write_text(to_md(overall))
    import csv
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(runs_by_tag, out / "curves")
    print(to_md(overall))
    print(to_md(rows))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
