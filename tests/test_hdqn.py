"""Tests for the hierarchical DQN agent. Skipped automatically if torch is not installed."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arcrl.agents import build_agent  # noqa: E402
from arcrl.agents.hdqn import VARIANTS, HDQNAgent, MetaNet, hdqn_config  # noqa: E402
from arcrl.env.adapter import AdapterConfig, make_env  # noqa: E402


def _small(name):
    return replace(hdqn_config(name), learn_start=64, batch_size=16, buffer_size=500, meta_learn_start=8,
                   meta_batch=4, seq_len=4, burn_in=1, device="cpu")


def _drive(agent, steps=400, budget=60):
    """Several short episodes, so the episode-boundary logic is exercised too."""
    env = make_env("mock", AdapterConfig(action_budget=budget))
    agent.begin_game("mock")
    obs, info = env.reset()
    for _ in range(steps):
        a = agent.act(obs, info)
        assert info["action_mask"][a]
        nxt, r, term, trunc, info = env.step(a)
        agent.observe(obs, a, r, nxt, term, info)
        obs = nxt
        if term or trunc:
            obs, info = env.reset()


@pytest.mark.parametrize("name", list(VARIANTS))
def test_variant_runs_and_learns(name):
    env = make_env("mock")
    ag = HDQNAgent(env.n_actions, 16, 0, _small(name))
    _drive(ag)
    assert ag.last_closs is not None            # controller updated
    assert ag.mt > 0 and ag.log                 # segments finished and logged
    if name != "hdqn_randmeta":
        assert ag.last_mloss is not None        # meta-controller updated
    assert all(0 <= r["g"] < ag.n_goals for r in ag.mrec)


def test_registry_builds_every_variant():
    for name in VARIANTS:
        assert build_agent(name, 70, 16, 0).cfg is not None
    with pytest.raises(KeyError):
        build_agent("hdqn_nope", 70, 16, 0)


def test_goal_masks_cover_one_cell_each():
    ag = HDQNAgent(70, 16, 0, _small("hdqn"))
    cell = ag.cfg.goal_cell
    assert ag.goal_masks.shape == (ag.n_goals, 64, 64)
    assert all(int(m.sum()) == cell * cell for m in ag.goal_masks)
    assert int(ag.goal_masks.sum(0).max()) == 1   # cells do not overlap


@pytest.mark.parametrize("gru", [False, True])
def test_metanet_shapes(gru):
    net = MetaNet(16, 16, gru, 32)
    q = net.seq(torch.zeros(2, 5, 16, 64, 64), torch.zeros(2, 5, 18))
    assert q.shape == (2, 5, 16)
    q1, h = net.step(torch.zeros(1, 16, 64, 64), torch.zeros(1, 18), None)
    assert q1.shape == (1, 16)
    assert (h is not None) == gru


def test_gru_step_matches_sequence_unroll():
    """Acting one step at a time must equal unrolling the same inputs as a sequence."""
    torch.manual_seed(0)
    net = MetaNet(16, 16, True, 32).eval()
    x, aux = torch.rand(1, 4, 16, 64, 64), torch.rand(1, 4, 18)
    with torch.no_grad():
        full = net.seq(x, aux)
        h, outs = None, []
        for t in range(4):
            q, h = net.step(x[:, t], aux[:, t], h)
            outs.append(q)
    assert torch.allclose(full[0], torch.cat(outs), atol=1e-5)
