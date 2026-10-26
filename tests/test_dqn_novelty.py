"""Unit tests for DQN Improvement 1: count-based novelty."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arcrl.agents.dqn import DQNAgent, DQNConfig  # noqa: E402
from arcrl.env.encoding import one_hot  # noqa: E402


BETA = 0.01
N_ACTIONS = 70


def _agent(*, novelty=True, beta=BETA):
    # Keep test agents small so these tests do not allocate the full default replay buffer.
    cfg = DQNConfig(
        buffer_size=8,
        batch_size=2,
        learn_start=100,
        train_every=4,
        target_update=100,
        novelty=novelty,
        novelty_beta=beta,
        device="cpu",
    )
    return DQNAgent(N_ACTIONS, in_ch=16, seed=0, cfg=cfg)


def _grid(value=0):
    grid = np.zeros((64, 64), dtype=np.int8)
    grid[0, 0] = value
    return grid


def _observe(agent, grid, reward=0.0):
    obs = one_hot(_grid(0))
    next_obs = one_hot(grid)
    info = {"grid": grid, "action_mask": np.ones(N_ACTIONS, dtype=bool)}
    agent.observe(obs, 0, reward, next_obs, False, info)


def test_novelty_bonus_is_largest_on_first_visit_and_decays():
    agent = _agent()
    grid = _grid(1)

    first = agent._novelty_bonus(grid)
    second = agent._novelty_bonus(grid)
    third = agent._novelty_bonus(grid)

    assert first == pytest.approx(BETA)
    assert second == pytest.approx(BETA / np.sqrt(2))
    assert third == pytest.approx(BETA / np.sqrt(3))
    assert len(agent._novelty_counts) == 1
    assert next(iter(agent._novelty_counts.values())) == 3


def test_observe_stores_environment_reward_plus_novelty_bonus():
    agent = _agent()
    original_reward = 1.0
    _observe(agent, _grid(1), reward=original_reward)

    assert agent.buf.r[0] == pytest.approx(original_reward + BETA)
    # The external reward value is an input; shaping happens only inside DQN.
    assert original_reward == 1.0


def test_novelty_disabled_keeps_original_replay_reward_and_no_counts():
    agent = _agent(novelty=False)
    _observe(agent, _grid(1), reward=1.0)

    assert agent.buf.r[0] == pytest.approx(1.0)
    assert agent._novelty_counts == {}


def test_novelty_counts_persist_until_begin_game():
    agent = _agent()
    grid = _grid(1)

    _observe(agent, grid)
    _observe(agent, grid)
    assert next(iter(agent._novelty_counts.values())) == 2

    # Avoid rebuilding the neural networks; this test is only about the reset boundary.
    agent._build = lambda: None
    agent.begin_game("next-game-seed-run")
    assert agent._novelty_counts == {}


def test_dqn_imp1_registry_enables_novelty(monkeypatch):
    import arcrl.agents.dqn as dqn_module
    from arcrl.agents import build_agent

    class CapturingAgent:
        def __init__(self, n_actions, in_ch, seed, cfg=None):
            self.cfg = cfg or DQNConfig()

    monkeypatch.setattr(dqn_module, "DQNAgent", CapturingAgent)
    agent = build_agent("dqn_imp1", N_ACTIONS, 16, 0)

    assert agent.cfg.novelty is True
    assert agent.cfg.novelty_beta == pytest.approx(BETA)


def test_dqn_baseline_registry_keeps_novelty_disabled(monkeypatch):
    import arcrl.agents.dqn as dqn_module
    from arcrl.agents import build_agent

    class CapturingAgent:
        def __init__(self, n_actions, in_ch, seed, cfg=None):
            self.cfg = cfg or DQNConfig()

    monkeypatch.setattr(dqn_module, "DQNAgent", CapturingAgent)
    agent = build_agent("dqn", N_ACTIONS, 16, 0)

    assert agent.cfg.novelty is False
