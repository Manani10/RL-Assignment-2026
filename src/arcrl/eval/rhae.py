"""Local re-implementation of the ARC-AGI-3 RHAE score, for analysis only.

Official definition (docs.arcprize.org/methodology):
  level_score = min((human_actions / ai_actions)^2, 1.15)  for completed levels, else 0
  game_score  = sum_l l * level_score_l / sum_l l           (l = 1-indexed level)
  total       = mean of game scores

ALWAYS report the official scorecard / Kaggle score as THE score.
This module is for plots, ablations and sanity checks, because human
baselines are only known for public games.
"""
from __future__ import annotations

from collections.abc import Sequence

CAP = 1.15


def level_score(human_actions: float, ai_actions: float | None) -> float:
    if ai_actions is None or ai_actions <= 0:
        return 0.0
    return min((human_actions / ai_actions) ** 2, CAP)


def game_score(human_actions: Sequence[float], ai_actions: Sequence[float | None]) -> float:
    """ai_actions[i] = actions used on level i+1, or None if not completed."""
    n = len(human_actions)
    ai = list(ai_actions) + [None] * (n - len(ai_actions))
    weights = range(1, n + 1)
    num = sum(w * level_score(h, a) for w, h, a in zip(weights, human_actions, ai))
    return num / sum(weights)


def total_score(games: Sequence[tuple[Sequence[float], Sequence[float | None]]]) -> float:
    if not games:
        return 0.0
    return sum(game_score(h, a) for h, a in games) / len(games)
