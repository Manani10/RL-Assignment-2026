"""Smoke tests for DQN/PPO. Skipped automatically if torch is not installed."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arcrl.agents.dqn import DQNAgent, DQNConfig  # noqa: E402
from arcrl.agents.ppo import PPOAgent, PPOConfig  # noqa: E402
from arcrl.env.adapter import AdapterConfig, make_env  # noqa: E402


def _drive(agent, steps=300):
    env = make_env("mock", AdapterConfig(action_budget=steps))
    obs, info = env.reset()
    for _ in range(steps):
        a = agent.act(obs, info)
        assert info["action_mask"][a]
        nxt, r, term, trunc, info = env.step(a)
        agent.observe(obs, a, r, nxt, term, info)
        obs = nxt
        if term or trunc:
            obs, info = env.reset()


def test_dqn_learns_without_crashing():
    env = make_env("mock")
    ag = DQNAgent(env.n_actions, 16, 0, DQNConfig(learn_start=64, batch_size=16, buffer_size=500, device="cpu"))
    _drive(ag)
    assert hasattr(ag, "last_loss")


def test_ppo_updates_without_crashing():
    env = make_env("mock")
    ag = PPOAgent(env.n_actions, 16, 0, PPOConfig(rollout_len=64, minibatch=16, epochs=1, device="cpu"))
    _drive(ag)
    assert hasattr(ag, "last_loss")
