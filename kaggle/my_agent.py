"""Bridge: plug our agents into the official Kaggle starter — OWNER: Member 1.

Usage
    1. git clone https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter (outside this repo)
    2. Copy this file over  ARC-AGI-3-Kaggle-Starter/agent/my_agent.py
       and copy our  src/arcrl  package next to it (or pip install -e this repo).
    3. make play-local  ->  make submit  ->  "Submit to Competition" on kaggle.com

The starter calls `choose_action(frames, latest_frame)` once per step and
`is_done(...)` to stop. We translate that into our Agent.act/observe API so the
SAME agent code runs in local experiments and on Kaggle (online learning only,
internet is disabled during Kaggle evaluation).

TODO(Member 1): confirm the two framework imports below against the sample agent
(`make pull-sample`) — names can change between toolkit versions.
"""
from __future__ import annotations

import os

import numpy as np
from arcengine import FrameData, GameAction, GameState  # noqa: F401

from agents.agent import Agent  # framework base class (from the starter's vendor/)

from arcrl.agents import build_agent
from arcrl.env.adapter import SIMPLE_ACTIONS
from arcrl.env.encoding import FrameStack, last_grid

AGENT_NAME = os.environ.get("ARCRL_AGENT", "random")   # e.g. dqn, ppo, dqn_imp2
CLICK_CELL = 8
FRAME_STACK = 1
MAX_ACTIONS = int(os.environ.get("ARCRL_MAX_ACTIONS", "5000"))


def _names(actions) -> set[str]:
    out = set()
    for a in actions or []:
        if isinstance(a, int):
            out.add(f"ACTION{a}")
        else:
            out.add(getattr(a, "name", str(a)))
    return out


class MyAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_cells = 64 // CLICK_CELL
        self.clicks = [(y * CLICK_CELL + CLICK_CELL // 2, x * CLICK_CELL + CLICK_CELL // 2)
                       for y in range(n_cells) for x in range(n_cells)]
        self.n_actions = len(SIMPLE_ACTIONS) + len(self.clicks)
        self.inner = build_agent(AGENT_NAME, self.n_actions, 16 * FRAME_STACK, seed=0)
        self.inner.begin_game(getattr(self, "game_id", "kaggle"))
        self.stack = FrameStack(FRAME_STACK)
        self.prev = None          # (obs, action_idx)
        self.prev_levels = 0
        self.n = 0

    def _mask(self, latest) -> np.ndarray:
        allowed = _names(getattr(latest, "available_actions", None))
        m = np.zeros(self.n_actions, bool)
        for i, n in enumerate(SIMPLE_ACTIONS):
            m[i] = n in allowed
        m[len(SIMPLE_ACTIONS):] = "ACTION6" in allowed
        return m if m.any() else np.ones_like(m)

    def is_done(self, frames, latest_frame) -> bool:
        return latest_frame.state is GameState.WIN or self.n >= MAX_ACTIONS

    def choose_action(self, frames, latest_frame):
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.prev = None
            self.n += 1
            return GameAction.RESET

        grid = last_grid(latest_frame.frame)
        obs = self.stack.push(grid) if self.prev else self.stack.reset(grid)
        info = {"action_mask": self._mask(latest_frame)}
        lv = int(latest_frame.levels_completed)
        if self.prev is not None:
            r = float(max(0, lv - self.prev_levels))
            self.inner.observe(self.prev[0], self.prev[1], r, obs, False, info)
        self.prev_levels = lv

        idx = self.inner.act(obs, info)
        self.prev = (obs, idx)
        self.n += 1
        if idx < len(SIMPLE_ACTIONS):
            return getattr(GameAction, SIMPLE_ACTIONS[idx])
        y, x = self.clicks[idx - len(SIMPLE_ACTIONS)]
        action = GameAction.ACTION6
        action.set_data({"x": int(x), "y": int(y)})
        return action
