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
    if name == "dqn_imp1":
        from .dqn import DQNAgent, DQNConfig
        return DQNAgent(n_actions, in_ch, seed, cfg=DQNConfig(novelty=True, novelty_beta=0.01))
    if name == "ppo":
        from .ppo import PPOAgent
        return PPOAgent(n_actions, in_ch, seed)
    if name.startswith("hdqn"):   # hierarchical DQN family (see agents/hdqn.py VARIANTS)
        from .hdqn import HDQNAgent, hdqn_config
        return HDQNAgent(n_actions, in_ch, seed, cfg=hdqn_config(name))
    raise KeyError(f"Unknown agent {name!r}")