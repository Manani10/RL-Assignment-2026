"""Common agent interface so the evaluation protocol treats every agent the same."""
from __future__ import annotations

import numpy as np


class Agent:
    """
    Lifecycle used by eval/protocol.py:
        agent.begin_game(game_id)          # new game: decide what persists
        a = agent.act(obs, info)           # choose an action index
        agent.observe(obs, a, r, next_obs, done, info)   # learning hook
    Document in the report which parameters persist between actions,
    levels, games and evaluation runs (assignment §4.4).
    """

    name = "base"

    def begin_game(self, game_id: str) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict) -> int:
        raise NotImplementedError

    def observe(self, obs, action, reward, next_obs, done, info) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def diagnostics(self):
        """Optional JSON-serialisable training statistics, logged by eval/protocol.py."""
        return None


def masked_choice(rng: np.random.Generator, mask: np.ndarray) -> int:
    idx = np.flatnonzero(mask)
    return int(rng.choice(idx))
