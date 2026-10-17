"""Gymnasium-style adapter around the official ARC-AGI-3 environment.

The official toolkit (`pip install arc-agi`, Python 3.12) is NOT a Gymnasium env,
so the assignment asks us to write and document an adapter. Semantics:

Observation : (16*k, 64, 64) float32 one-hot of the last k grids (see encoding.py)
Action      : Discrete(70) by default (see actions.py). Index -> (GameAction, data):
                0..5  -> ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION7
                6..69 -> ACTION6 clicks: grid-cell centres (baseline) or detected
                         objects (click_mode="objects", shared improvement)
              `info["action_mask"]` marks which indices are allowed now
              (optionally with known no-op actions removed: prune_noops=True).
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

from .actions import SIMPLE_ACTIONS, ActionConfig, ActionSpace  # noqa: F401
from .encoding import FrameStack, last_grid

try:  # Gymnasium is optional so the code also runs on bare Kaggle images
    import gymnasium as gym
    from gymnasium import spaces

    _Base = gym.Env
except ImportError:  # pragma: no cover
    gym = None
    spaces = None
    _Base = object

@dataclass
class AdapterConfig:
    frame_stack: int = 1
    click_cell: int = 8          # 8 -> 8x8 px cells -> 64 click slots
    action_budget: int = 2000    # actions per episode (game attempt)
    on_game_over: str = "reset"  # "reset" | "terminate"
    # --- shared improvement (Member 1); defaults = baseline behaviour ---
    click_mode: str = "grid"     # "grid" | "objects"  (see actions.py)
    prune_noops: bool = False    # mask actions known not to change the frame


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
        self.actions = ActionSpace(ActionConfig(click_cell=self.cfg.click_cell, click_mode=self.cfg.click_mode,
                                                prune_noops=self.cfg.prune_noops))
        self.n_actions = self.actions.n
        self.stack = FrameStack(self.cfg.frame_stack)
        if spaces is not None:
            self.observation_space = spaces.Box(0.0, 1.0, (16 * self.cfg.frame_stack, 64, 64), np.float32)
            self.action_space = spaces.Discrete(self.n_actions)
        self._levels = 0
        self.actions_used = 0
        self.last_obs = None
        self._grid = None

    # ---------------------------------------------------------------- helpers
    def decode(self, idx: int):
        """Discrete index -> (GameAction, data dict)."""
        name, data = self.actions.decode(int(idx))
        return getattr(self.A, name), data

    def action_mask(self) -> np.ndarray:
        return self.actions.mask({_name(a) for a in self.raw.action_space})

    def _observe(self, obs) -> dict:
        self._grid = last_grid(obs.frame)
        self.actions.update(self._grid)
        return {
            "action_mask": self.action_mask(),
            "levels_completed": int(obs.levels_completed),
            "win_levels": int(getattr(obs, "win_levels", 0) or 0),
            "state": _name(obs.state),
            "actions_used": self.actions_used,
            "grid": self._grid,
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
        info = self._observe(obs)
        return self.stack.reset(self._grid), info

    def step(self, idx: int):
        prev_grid = self._grid
        action, data = self.decode(int(idx))
        obs = self.raw.step(action, data=data)
        self.actions_used += 1
        self.actions.record(int(idx), changed=not np.array_equal(last_grid(obs.frame), prev_grid))
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
        info = self._observe(obs)
        return self.stack.push(self._grid), reward, terminated, truncated, info


def make_env(game_id: str, cfg: AdapterConfig | None = None, seed: int = 0, save_recording: bool = False):
    """Build an adapter around a real game. Requires `pip install arc-agi`.

    save_recording=True writes a JSONL replay to ./recordings/ — use it for the
    success/failure case figures the report needs.
    """
    if game_id.startswith("mock"):
        from .mock import MockAction, MockGame
        return ArcGymEnv(MockGame(seed), MockAction, cfg)
    import arc_agi  # noqa: WPS433 (lazy import: only needed for real games)
    from arcengine import GameAction

    arc = arc_agi.Arcade()
    raw = arc.make(game_id, seed=seed, save_recording=save_recording)
    if raw is None:
        raise RuntimeError(f"Could not create environment {game_id!r}")
    env = ArcGymEnv(raw, GameAction, cfg)
    env.arcade = arc  # keep the client so the protocol can read the OFFICIAL scorecard
    return env


def official_score(env) -> dict | None:
    """Read the official scorecard for this env's Arcade client (None for mock games)."""
    arc = getattr(env, "arcade", None)
    if arc is None:
        return None
    sc = arc.get_scorecard()
    if sc is None:
        return None
    if hasattr(sc, "model_dump"):
        d = sc.model_dump()
    elif hasattr(sc, "__dict__"):
        d = dict(vars(sc))
    else:
        d = {"raw": str(sc)}
    d.setdefault("score", getattr(sc, "score", None))
    return d
