"""Bridge: plug our agents into the official Kaggle starter — OWNER: Member 1.

Do NOT copy this file directly: the starter only packages agent/my_agent.py,
so `arcrl` would be missing on Kaggle. Build a single self-contained file:

    python scripts/bundle_agent.py --agent dqn --out build/my_agent.py
    cp build/my_agent.py ../ARC-AGI-3-Kaggle-Starter/agent/my_agent.py
    cd ../ARC-AGI-3-Kaggle-Starter && make play-local && make submit
    then "Submit to Competition" on kaggle.com

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
from arcrl.env.actions import ActionConfig, ActionSpace
from arcrl.env.encoding import FrameStack, last_grid

AGENT_NAME = os.environ.get("ARCRL_AGENT", "random")   # e.g. dqn, ppo, dqn_imp1
CLICK_MODE = os.environ.get("ARCRL_CLICK_MODE", "grid")            # "grid" | "objects"
PRUNE_NOOPS = os.environ.get("ARCRL_PRUNE_NOOPS", "0") == "1"
CLICK_CELL = 8
FRAME_STACK = 1
MAX_ACTIONS = int(os.environ.get("ARCRL_MAX_ACTIONS", "5000"))


def _state(frame) -> str:
    return getattr(frame.state, "name", str(frame.state))


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
        # same action space as local experiments (env/actions.py)
        self.actions = ActionSpace(ActionConfig(click_cell=CLICK_CELL, click_mode=CLICK_MODE,
                                                prune_noops=PRUNE_NOOPS))
        self.inner = build_agent(AGENT_NAME, self.actions.n, 16 * FRAME_STACK, seed=0)
        self.inner.begin_game(getattr(self, "game_id", "kaggle"))
        self.stack = FrameStack(FRAME_STACK)
        self.prev = None          # (obs, action_idx, grid)
        self.prev_levels = 0
        self.n = 0

    def is_done(self, frames, latest_frame) -> bool:
        return _state(latest_frame) == "WIN" or self.n >= MAX_ACTIONS

    def choose_action(self, frames, latest_frame):
        if _state(latest_frame) in ("NOT_PLAYED", "GAME_OVER"):
            self.prev = None
            self.n += 1
            return GameAction.RESET

        grid = last_grid(latest_frame.frame)
        if self.prev is not None:  # must run BEFORE update(): uses the previous frame's hash
            self.actions.record(self.prev[1], changed=not np.array_equal(grid, self.prev[2]))
        self.actions.update(grid)
        obs = self.stack.push(grid) if self.prev else self.stack.reset(grid)
        info = {"action_mask": self.actions.mask(_names(getattr(latest_frame, "available_actions", None)))}
        lv = int(latest_frame.levels_completed)
        if self.prev is not None:
            r = float(max(0, lv - self.prev_levels))
            self.inner.observe(self.prev[0], self.prev[1], r, obs, False, info)
        self.prev_levels = lv

        idx = self.inner.act(obs, info)
        self.prev = (obs, idx, grid)
        self.n += 1
        name, data = self.actions.decode(idx)
        action = getattr(GameAction, name)
        if data:
            action.set_data(data)
        return action
