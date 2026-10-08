"""Gymnasium-style adapter around the official ARC-AGI-3 environment.

The official toolkit (`pip install arc-agi`, Python 3.12) is NOT a Gymnasium env,
so the assignment asks us to write and document an adapter. Semantics:

Observation : (16*k, 64, 64) float32 one-hot of the last k grids (see encoding.py)
Action      : Discrete(N). Index -> (GameAction, data):
                0..5  -> ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION7
                6..   -> ACTION6 click at the centre of a coarse cell
                         (64/click_cell)^2 targets, default 8x8 cells -> 64 targets
              `info["action_mask"]` marks which indices the game currently allows.
Reward      : +1 each time `levels_completed` increases (environment-derived).
              Optional shaping terms are added by WRAPPERS, never here, and are
              logged separately — shaped return is NOT the ARC score.
Termination : terminated=True on GameState.WIN.
              On GameState.GAME_OVER we either issue RESET (default, counted as
              an action) or terminate, depending on `on_game_over`.
Truncation  : truncated=True once `action_budget` actions are used.

Everything the agent sees comes from the public observation. No hidden state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .encoding import FrameStack, last_grid

try:  # Gymnasium is optional so the code also runs on bare Kaggle images
    import gymnasium as gym
    from gymnasium import spaces

    _Base = gym.Env
except ImportError:  # pragma: no cover
    gym = None
    spaces = None
    _Base = object

SIMPLE_ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7"]


@dataclass
class AdapterConfig:
    frame_stack: int = 1
    click_cell: int = 8          # 8 -> 8x8 px cells -> 64 click targets
    action_budget: int = 2000    # actions per episode (game attempt)
    on_game_over: str = "reset"  # "reset" | "terminate"


def _name(x: Any) -> str:
    return getattr(x, "name", str(x))


class ArcGymEnv(_Base):
    metadata = {"render_modes": []}

    def __init__(self, raw_env, action_enum, cfg: AdapterConfig | None = None):
        """
        raw_env     : object from `arc_agi.Arcade().make(game_id)` (or MockGame)
        action_enum : `arcengine.GameAction` (or MockAction) — looked up by name
        """
        self.raw = raw_env
        self.A = action_enum
        self.cfg = cfg or AdapterConfig()
        n_cells = 64 // self.cfg.click_cell
        self.click_targets = [
            (y * self.cfg.click_cell + self.cfg.click_cell // 2,
             x * self.cfg.click_cell + self.cfg.click_cell // 2)
            for y in range(n_cells) for x in range(n_cells)
        ]
        self.n_actions = len(SIMPLE_ACTIONS) + len(self.click_targets)
        self.stack = FrameStack(self.cfg.frame_stack)
        if spaces is not None:
            self.observation_space = spaces.Box(0.0, 1.0, (16 * self.cfg.frame_stack, 64, 64), np.float32)
            self.action_space = spaces.Discrete(self.n_actions)
        self._levels = 0
        self.actions_used = 0
        self.last_obs = None

    # ---------------------------------------------------------------- helpers
    def decode(self, idx: int):
        """Discrete index -> (GameAction, data dict)."""
        if idx < len(SIMPLE_ACTIONS):
            return getattr(self.A, SIMPLE_ACTIONS[idx]), {}
        y, x = self.click_targets[idx - len(SIMPLE_ACTIONS)]
        return getattr(self.A, "ACTION6"), {"x": int(x), "y": int(y)}

    def action_mask(self) -> np.ndarray:
        allowed = {_name(a) for a in self.raw.action_space}
        mask = np.zeros(self.n_actions, dtype=bool)
        for i, n in enumerate(SIMPLE_ACTIONS):
            mask[i] = n in allowed
        if "ACTION6" in allowed:
            mask[len(SIMPLE_ACTIONS):] = True
        if not mask.any():  # never hand the agent an empty mask
            mask[:] = True
        return mask

    def _info(self, obs) -> dict:
        return {
            "action_mask": self.action_mask(),
            "levels_completed": int(obs.levels_completed),
            "win_levels": int(getattr(obs, "win_levels", 0) or 0),
            "state": _name(obs.state),
            "actions_used": self.actions_used,
            "grid": last_grid(obs.frame),
        }

    # -------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if gym is not None:
            super().reset(seed=seed)
        obs = self.raw.reset()
        if obs is None:
            obs = self.raw.step(self.A.RESET)
        self.last_obs = obs
        self._levels = int(obs.levels_completed)
        self.actions_used = 0
        return self.stack.reset(last_grid(obs.frame)), self._info(obs)

    def step(self, idx: int):
        action, data = self.decode(int(idx))
        obs = self.raw.step(action, data=data)
        self.actions_used += 1
        reward, terminated, truncated = 0.0, False, False

        lv = int(obs.levels_completed)
        if lv > self._levels:
            reward += float(lv - self._levels)
        self._levels = lv

        state = _name(obs.state)
        if state == "WIN":
            terminated = True
        elif state == "GAME_OVER":
            if self.cfg.on_game_over == "terminate":
                terminated = True
            else:
                obs = self.raw.step(self.A.RESET)
                self.actions_used += 1  # resets cost an action in our accounting
                self._levels = int(obs.levels_completed)
        if self.actions_used >= self.cfg.action_budget and not terminated:
            truncated = True

        self.last_obs = obs
        return self.stack.push(last_grid(obs.frame)), reward, terminated, truncated, self._info(obs)


def make_env(game_id: str, cfg: AdapterConfig | None = None, seed: int = 0):
    """Build an adapter around a real game. Requires `pip install arc-agi`."""
    if game_id.startswith("mock"):
        from .mock import MockAction, MockGame
        return ArcGymEnv(MockGame(seed), MockAction, cfg)
    import arc_agi  # noqa: WPS433 (lazy import: only needed for real games)
    from arcengine import GameAction

    arc = arc_agi.Arcade()
    raw = arc.make(game_id, seed=seed)
    if raw is None:
        raise RuntimeError(f"Could not create environment {game_id!r}")
    return ArcGymEnv(raw, GameAction, cfg)
