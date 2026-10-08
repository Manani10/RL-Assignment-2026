"""DQN baseline (algorithm covered in the course) — OWNER: Member 2.

Baseline = Double DQN with target network, uniform replay, epsilon-greedy over
the *allowed* actions only (action mask from the adapter).

What is learned : Q(s, a) for every discrete action index.
When            : every `train_every` env steps once the buffer has `learn_start` items.
Effect          : act = argmax_a Q(s, a) over allowed a (with prob 1-eps).

Improvement slots (fill in, one file/flag per improvement so ablations are easy):
    IMP1  e.g. count-based novelty bonus on grid hashes (exploration)
    IMP2  e.g. object-centric click targets / prioritised replay / episodic memory
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import Agent, masked_choice
from .nets import GridEncoder


@dataclass
class DQNConfig:
    lr: float = 2.5e-4
    gamma: float = 0.99
    buffer_size: int = 50_000
    batch_size: int = 32
    learn_start: int = 500
    train_every: int = 4
    target_update: int = 1_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000
    persist_across_games: bool = False   # False = re-initialise per game (pure online)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class QNet(nn.Module):
    def __init__(self, in_ch: int, n_actions: int):
        super().__init__()
        self.enc = GridEncoder(in_ch)
        self.head = nn.Linear(256, n_actions)

    def forward(self, x):
        return self.head(self.enc(x))


class Replay:
    """Stores colour-index grids (int8, k x 64 x 64) instead of one-hot to save RAM."""

    def __init__(self, cap: int, k: int, n_actions: int):
        self.cap, self.i, self.full = cap, 0, False
        self.s = np.zeros((cap, k, 64, 64), np.int8)
        self.s2 = np.zeros_like(self.s)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)
        self.m2 = np.zeros((cap, n_actions), bool)

    def add(self, s, a, r, s2, d, m2):
        j = self.i
        self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j], self.m2[j] = s, a, r, s2, d, m2
        self.i = (j + 1) % self.cap
        self.full = self.full or self.i == 0

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, n, rng):
        idx = rng.integers(0, len(self), n)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx], self.m2[idx]


def to_index(obs: np.ndarray) -> np.ndarray:
    """(16k,64,64) one-hot -> (k,64,64) int8 colour indices."""
    k = obs.shape[0] // 16
    return obs.reshape(k, 16, 64, 64).argmax(1).astype(np.int8)


def to_onehot_batch(idx: np.ndarray, device) -> torch.Tensor:
    """(B,k,64,64) int8 -> (B,16k,64,64) float tensor."""
    t = torch.as_tensor(idx, dtype=torch.long, device=device)
    oh = F.one_hot(t, 16).permute(0, 1, 4, 2, 3).float()  # B,k,16,64,64
    return oh.reshape(t.shape[0], -1, 64, 64)


class DQNAgent(Agent):
    name = "dqn"

    def __init__(self, n_actions: int, in_ch: int = 16, seed: int = 0, cfg: DQNConfig | None = None):
        self.cfg = cfg or DQNConfig()
        self.n_actions, self.in_ch = n_actions, in_ch
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self._build()

    def _build(self):
        c = self.cfg
        self.q = QNet(self.in_ch, self.n_actions).to(c.device)
        self.q_t = QNet(self.in_ch, self.n_actions).to(c.device)
        self.q_t.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=c.lr)
        self.buf = Replay(c.buffer_size, self.in_ch // 16, self.n_actions)
        self.t = 0

    def begin_game(self, game_id):
        if not self.cfg.persist_across_games:
            self._build()

    def epsilon(self):
        c = self.cfg
        frac = min(1.0, self.t / c.eps_decay_steps)
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def act(self, obs, info):
        mask = info["action_mask"]
        if self.rng.random() < self.epsilon():
            return masked_choice(self.rng, mask)
        with torch.no_grad():
            x = torch.as_tensor(obs[None], device=self.cfg.device)
            q = self.q(x)[0].cpu().numpy()
        q[~mask] = -np.inf
        return int(q.argmax())

    def observe(self, obs, action, reward, next_obs, done, info):
        self.buf.add(to_index(obs), action, reward, to_index(next_obs), float(done), info["action_mask"])
        self.t += 1
        c = self.cfg
        if len(self.buf) >= c.learn_start and self.t % c.train_every == 0:
            self._learn()
        if self.t % c.target_update == 0:
            self.q_t.load_state_dict(self.q.state_dict())

    def _learn(self):
        c = self.cfg
        s, a, r, s2, d, m2 = self.buf.sample(c.batch_size, self.rng)
        s, s2 = to_onehot_batch(s, c.device), to_onehot_batch(s2, c.device)
        a = torch.as_tensor(a, device=c.device)
        r = torch.as_tensor(r, device=c.device)
        d = torch.as_tensor(d, device=c.device)
        m2 = torch.as_tensor(m2, device=c.device)
        q = self.q(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():  # Double DQN target with masked argmax
            a2 = self.q(s2).masked_fill(~m2, -1e9).argmax(1, keepdim=True)
            target = r + c.gamma * (1 - d) * self.q_t(s2).gather(1, a2).squeeze(1)
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.last_loss = float(loss)
