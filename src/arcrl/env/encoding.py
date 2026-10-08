"""Observation encoding shared by every agent.

ARC-AGI-3 frames are 64x64 grids with integer colours 0..15. A single env step
can return several frames (an animation); we keep the LAST one as the state and
optionally stack the previous K observations as a short history.

Keep this file the single source of truth so DQN and PPO see identical inputs
(required for a fair comparison).
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np

GRID = 64
N_COLOURS = 16


def last_grid(frame) -> np.ndarray:
    """Return the final 64x64 grid from whatever the SDK gives us.

    The SDK's `frame` field may be a 2-D grid or a list of 2-D grids
    (animation layers). We always return an int8 array of shape (64, 64).
    """
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected frame shape {arr.shape}")
    return arr.astype(np.int8)


def one_hot(grid: np.ndarray) -> np.ndarray:
    """(64,64) ints -> (16,64,64) float32 one-hot planes."""
    planes = np.zeros((N_COLOURS, *grid.shape), dtype=np.float32)
    idx = np.clip(grid, 0, N_COLOURS - 1).astype(np.int64)
    np.put_along_axis(planes, idx[None], 1.0, axis=0)
    return planes


def grid_hash(grid: np.ndarray) -> str:
    """Stable hash of a grid, used for novelty counts / state graphs."""
    return hashlib.blake2b(np.ascontiguousarray(grid).tobytes(), digest_size=8).hexdigest()


class FrameStack:
    """Keeps the last `k` one-hot grids -> tensor of shape (16*k, 64, 64)."""

    def __init__(self, k: int = 1):
        self.k = k
        self.buf: deque[np.ndarray] = deque(maxlen=k)

    def reset(self, grid: np.ndarray) -> np.ndarray:
        self.buf.clear()
        oh = one_hot(grid)
        for _ in range(self.k):
            self.buf.append(oh)
        return self.get()

    def push(self, grid: np.ndarray) -> np.ndarray:
        self.buf.append(one_hot(grid))
        return self.get()

    def get(self) -> np.ndarray:
        return np.concatenate(list(self.buf), axis=0)
