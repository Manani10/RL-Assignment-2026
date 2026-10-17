"""Discrete action space shared by the local adapter AND the Kaggle bridge.

Index layout (fixed size, so DQN/PPO networks never change shape):
    0..5            ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION7
    6..6+N-1        ACTION6 clicks, N = (64 / click_cell)^2  (default 64)

Baseline   click_mode="grid":    click slot i = centre of cell i of a fixed grid.
IMPROVEMENT (shared, Member 1) — two independent flags:
    click_mode="objects": click slot i = a pixel inside the i-th detected object
        (4-connected region of one colour; background = most common colour and
        very large regions are excluded). Unused slots are masked out.
    prune_noops=True: remember, per (frame hash, action index), whether the
        action changed the frame; mask actions known to do nothing in the
        current frame. Never masks every action. Memory lasts for one game.

Both use only the public frame — no hidden game state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .encoding import grid_hash

SIMPLE_ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7"]
N_SIMPLE = len(SIMPLE_ACTIONS)


@dataclass
class ActionConfig:
    click_cell: int = 8                 # grid mode: 8x8 px cells -> 64 slots (also sets N)
    click_mode: str = "grid"            # "grid" | "objects"
    prune_noops: bool = False
    max_object_frac: float = 0.25       # objects larger than this fraction of the screen are ignored


def find_objects(grid: np.ndarray, max_frac: float = 0.25) -> list[tuple[int, int, int, int]]:
    """Connected single-colour regions (4-connectivity), background excluded.

    Returns [(size, y, x, colour)] where (y, x) is the member pixel closest to the
    region's centroid, sorted by (size, y, x) so the order is deterministic.
    """
    h, w = grid.shape
    colours, counts = np.unique(grid, return_counts=True)
    background = colours[counts.argmax()]
    seen = np.zeros_like(grid, dtype=bool)
    limit = max_frac * h * w
    objs = []
    for sy in range(h):
        for sx in range(w):
            if seen[sy, sx] or grid[sy, sx] == background:
                continue
            c = grid[sy, sx]
            stack, pix = [(sy, sx)], []
            seen[sy, sx] = True
            while stack:
                y, x = stack.pop()
                pix.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and grid[ny, nx] == c:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(pix) > limit:
                continue
            p = np.asarray(pix)
            cy, cx = p.mean(0)
            k = int(((p[:, 0] - cy) ** 2 + (p[:, 1] - cx) ** 2).argmin())
            objs.append((len(pix), int(p[k, 0]), int(p[k, 1]), int(c)))
    objs.sort()
    return objs


class ActionSpace:
    def __init__(self, cfg: ActionConfig | None = None):
        self.cfg = cfg or ActionConfig()
        cell = self.cfg.click_cell
        n = 64 // cell
        self.grid_targets = [(y * cell + cell // 2, x * cell + cell // 2) for y in range(n) for x in range(n)]
        self.n_clicks = len(self.grid_targets)
        self.n = N_SIMPLE + self.n_clicks
        self.targets = list(self.grid_targets)
        self.noop: set[tuple[str, int]] = set()
        self._hash = None

    # -- per-game memory -------------------------------------------------
    def reset_memory(self):
        self.noop.clear()

    # -- per-step update -------------------------------------------------
    def update(self, grid: np.ndarray):
        """Call with every new frame before mask()/decode()."""
        self._hash = grid_hash(grid) if self.cfg.prune_noops else None
        if self.cfg.click_mode == "objects":
            objs = find_objects(grid, self.cfg.max_object_frac)[: self.n_clicks]
            self.targets = [(y, x) for _, y, x, _ in objs]
        else:
            self.targets = self.grid_targets

    def mask(self, allowed_names: set[str]) -> np.ndarray:
        m = np.zeros(self.n, dtype=bool)
        for i, name in enumerate(SIMPLE_ACTIONS):
            m[i] = name in allowed_names
        if "ACTION6" in allowed_names:
            m[N_SIMPLE:N_SIMPLE + len(self.targets)] = True
        if not m.any():
            m[:] = True  # never hand the agent an empty mask
        if self.cfg.prune_noops and self._hash is not None:
            pruned = m.copy()
            for i in np.flatnonzero(m):
                if (self._hash, int(i)) in self.noop:
                    pruned[i] = False
            if pruned.any():
                m = pruned
        return m

    def decode(self, idx: int) -> tuple[str, dict]:
        """Index -> (action name, data)."""
        if idx < N_SIMPLE:
            return SIMPLE_ACTIONS[idx], {}
        j = idx - N_SIMPLE
        y, x = self.targets[j] if j < len(self.targets) else self.grid_targets[j]
        return "ACTION6", {"x": int(x), "y": int(y)}

    def record(self, idx: int, changed: bool):
        """After a step: note whether action `idx` changed the frame it was taken in."""
        if self.cfg.prune_noops and self._hash is not None and not changed:
            self.noop.add((self._hash, int(idx)))
