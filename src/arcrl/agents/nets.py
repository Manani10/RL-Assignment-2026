"""Shared CNN encoder. Use the SAME encoder for DQN and PPO so that differences
in results come from the learning algorithm, not the architecture."""
from __future__ import annotations

import torch
from torch import nn


class GridEncoder(nn.Module):
    """(B, C, 64, 64) one-hot -> (B, feat_dim)."""

    def __init__(self, in_ch: int, feat_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.ReLU(),   # 32x32
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),      # 16x16
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),      # 8x8
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, feat_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, -1e9)
