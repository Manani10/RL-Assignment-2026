"""A tiny fake ARC-style game so everything can be tested offline / in CI.

It imitates the parts of the official interface our adapter relies on:
`reset()`, `step(action, data=...)`, `action_space`, and an observation with
`frame`, `state`, `levels_completed`, `win_levels`.

Game: a player pixel (colour 3) must reach a goal pixel (colour 4).
ACTION1..4 move up/down/left/right. Two levels. ACTION6 (click) does nothing
except on level 2, where clicking the goal teleports you there.
It is NOT a substitute for real games — only for plumbing tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class MockState(Enum):
    NOT_FINISHED = "NOT_FINISHED"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"


class MockAction(Enum):
    RESET = 0
    ACTION1 = 1
    ACTION2 = 2
    ACTION3 = 3
    ACTION4 = 4
    ACTION6 = 6

    def is_complex(self) -> bool:
        return self is MockAction.ACTION6


@dataclass
class MockObs:
    frame: list
    state: MockState
    levels_completed: int
    win_levels: int
    available_actions: list = field(default_factory=list)


class MockGame:
    LEVELS = [((60, 2), (60, 10)), ((10, 10), (50, 50))]  # (player, goal) as (y, x)
    MAX_STEPS = 300

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    @property
    def action_space(self):
        acts = [MockAction.ACTION1, MockAction.ACTION2, MockAction.ACTION3, MockAction.ACTION4]
        if self.level == 1:
            acts.append(MockAction.ACTION6)
        return acts

    def _obs(self) -> MockObs:
        g = np.zeros((64, 64), dtype=np.int8)
        g[0, :] = 5  # a "UI bar" so frames are not trivially empty
        gy, gx = self.goal
        g[gy, gx] = 4
        py, px = self.player
        g[py, px] = 3
        return MockObs([g], self.state, self.level, len(self.LEVELS), [a.value for a in self.action_space])

    def _load(self, lvl: int):
        self.level = lvl
        self.player, self.goal = (list(p) for p in self.LEVELS[lvl])

    def reset(self) -> MockObs:
        self.steps = 0
        self.state = MockState.NOT_FINISHED
        self._load(0)
        return self._obs()

    def step(self, action, data: dict | None = None) -> MockObs:
        if action is MockAction.RESET:
            return self.reset()
        self.steps += 1
        dy, dx = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action.value, (0, 0))
        if action is MockAction.ACTION6 and data and self.level == 1:
            if (data.get("y"), data.get("x")) == tuple(self.goal):
                self.player = list(self.goal)
        self.player[0] = int(np.clip(self.player[0] + dy, 1, 63))
        self.player[1] = int(np.clip(self.player[1] + dx, 0, 63))
        if self.player == self.goal:
            if self.level + 1 >= len(self.LEVELS):
                self.level += 1
                self.state = MockState.WIN
                self.level = len(self.LEVELS)
                g = np.zeros((64, 64), dtype=np.int8)
                return MockObs([g], self.state, self.level, len(self.LEVELS), [])
            self._load(self.level + 1)
        if self.steps >= self.MAX_STEPS:
            self.state = MockState.GAME_OVER
        return self._obs()
