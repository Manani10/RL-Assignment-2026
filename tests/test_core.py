"""Fast tests that run without arc-agi or torch (CI runs these on every push)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arcrl.env.adapter import AdapterConfig, make_env  # noqa: E402
from arcrl.env.encoding import FrameStack, grid_hash, last_grid, one_hot  # noqa: E402
from arcrl.eval import rhae  # noqa: E402
from arcrl.eval.protocol import EvalConfig, evaluate  # noqa: E402


def test_one_hot_roundtrip():
    g = np.random.default_rng(0).integers(0, 16, (64, 64)).astype(np.int8)
    oh = one_hot(g)
    assert oh.shape == (16, 64, 64)
    assert np.array_equal(oh.argmax(0), g)
    assert oh.sum() == 64 * 64


def test_last_grid_handles_animation_layers():
    layers = [np.zeros((64, 64)), np.ones((64, 64))]
    assert last_grid(layers).max() == 1


def test_frame_stack_shape():
    fs = FrameStack(3)
    g = np.zeros((64, 64), np.int8)
    assert fs.reset(g).shape == (48, 64, 64)
    assert fs.push(g).shape == (48, 64, 64)
    assert grid_hash(g) == grid_hash(g.copy())


def test_adapter_mask_and_reward():
    env = make_env("mock", AdapterConfig(action_budget=500))
    obs, info = env.reset()
    assert obs.shape == (16, 64, 64)
    assert info["action_mask"][:4].all() and not info["action_mask"][4]
    assert not info["action_mask"][6:].any()  # no clicks on level 1
    total = 0.0
    for _ in range(8):  # ACTION4 = move right; goal is 8 cells right
        obs, r, term, trunc, info = env.step(3)
        total += r
    assert total == 1.0 and info["levels_completed"] == 1
    assert info["action_mask"][6:].all()  # clicks enabled on level 2


def test_click_decode():
    env = make_env("mock")
    action, data = env.decode(6)
    assert action.name == "ACTION6" and data == {"x": 4, "y": 4}


def test_truncation():
    env = make_env("mock", AdapterConfig(action_budget=5))
    env.reset()
    for _ in range(5):
        *_, trunc, info = env.step(0)
    assert trunc


def test_rhae_matches_docs_example():
    assert rhae.level_score(10, 10) == 1.0
    assert rhae.level_score(10, 20) == 0.25
    assert rhae.level_score(10, 1) == 1.15
    # 5 levels, first 4 perfect -> 10/15
    assert abs(rhae.game_score([10] * 5, [10, 10, 10, 10, None]) - 10 / 15) < 1e-9


def test_protocol_random_agent(tmp_path):
    cfg = EvalConfig(agent="random", games=["mock"], seeds=[0, 1], total_budget=600,
                     tag="t", out_dir=str(tmp_path))
    runs = evaluate(cfg)
    assert len(runs) == 2 and all(r["actions"] == 600 for r in runs)
    lines = (tmp_path / "t" / "episodes.jsonl").read_text().splitlines()
    assert len(lines) >= 2 and "levels_completed" in json.loads(lines[0])
