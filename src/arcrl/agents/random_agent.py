"""Sanity-check reference policy (required by the assignment, §3.2).

Uniform over currently-allowed actions. Not a learning baseline.
"""
from __future__ import annotations

import numpy as np

from .base import Agent, masked_choice


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int = 0, **_):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, info):
        return masked_choice(self.rng, info["action_mask"])
