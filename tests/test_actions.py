"""Tests for the shared action-space improvement (object clicks + no-op pruning)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arcrl.env.actions import N_SIMPLE, ActionConfig, ActionSpace, find_objects  # noqa: E402
from arcrl.env.adapter import AdapterConfig, make_env  # noqa: E402


def test_find_objects_excludes_background_and_huge_regions():
    g = np.zeros((64, 64), np.int8)
    g[5, 5] = 3                 # 1-pixel object
    g[20:23, 30:33] = 4         # 3x3 object
    g[40:, :] = 7               # 24 rows -> 1536 px = 37% of screen -> excluded
    objs = find_objects(g, max_frac=0.25)
    assert [(s, c) for s, _, _, c in objs] == [(1, 3), (9, 4)]
    assert objs[1][1:3] == (21, 31)  # pixel nearest the centroid


def test_object_click_completes_mock_level_2():
    env = make_env("mock", AdapterConfig(click_mode="objects"))
    obs, info = env.reset()
    for _ in range(8):               # level 1: move right 8 times
        obs, r, term, trunc, info = env.step(3)
    assert info["levels_completed"] == 1
    # level 2: objects sorted by (size, y, x): player (10,10), goal (50,50), top bar
    assert info["action_mask"][N_SIMPLE:N_SIMPLE + 3].all()
    assert not info["action_mask"][N_SIMPLE + 3:].any()    # unused click slots masked
    action, data = env.decode(N_SIMPLE + 1)
    assert data == {"x": 50, "y": 50}
    obs, r, term, trunc, info = env.step(N_SIMPLE + 1)     # click the goal
    assert term and info["state"] == "WIN"


def test_noop_pruning_masks_actions_that_did_nothing():
    env = make_env("mock", AdapterConfig(prune_noops=True))
    env.reset()
    for _ in range(2):               # player starts at x=2: move left to the wall
        *_, info = env.step(2)
    assert info["action_mask"][2]    # not yet known to be a no-op
    *_, info = env.step(2)           # bump the wall: frame unchanged
    assert not info["action_mask"][2]
    assert info["action_mask"].any()


def test_mask_never_empty_even_if_everything_pruned():
    sp = ActionSpace(ActionConfig(prune_noops=True))
    g = np.zeros((64, 64), np.int8)
    sp.update(g)
    for i in range(N_SIMPLE):
        sp.record(i, changed=False)
    m = sp.mask({"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7"})
    assert m.any()


def test_flags_off_is_baseline():
    sp = ActionSpace()
    sp.update(np.zeros((64, 64), np.int8))
    assert sp.n == 70
    assert sp.decode(N_SIMPLE) == ("ACTION6", {"x": 4, "y": 4})
    assert sp.mask({"ACTION6"})[N_SIMPLE:].all()
