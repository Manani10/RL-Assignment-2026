"""Agent registry. Torch agents are imported lazily so the random baseline and
the evaluation code work on machines without PyTorch."""
from __future__ import annotations


def build_agent(name: str, n_actions: int, in_ch: int, seed: int):
    if name == "random":
        from .random_agent import RandomAgent
        return RandomAgent(seed=seed)
    if name == "dqn":
        from .dqn import DQNAgent
        return DQNAgent(n_actions, in_ch, seed)
    if name == "ppo":
        from .ppo import PPOAgent
        return PPOAgent(n_actions, in_ch, seed)
    # Register improved variants here, e.g. "dqn_imp1": DQNAgent(..., cfg=DQNConfig(novelty=True))
    raise KeyError(f"Unknown agent {name!r}")
