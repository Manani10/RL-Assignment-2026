"""PPO baseline (algorithm NOT covered in the course) — OWNER: Member 3.

Clipped-surrogate PPO (Schulman et al., 2017) with GAE(lambda), shared CNN
encoder, masked categorical policy over allowed actions.

Learning objective (maximised):
    L = E[ min(rho*A, clip(rho, 1-eps, 1+eps)*A) ] - c_v * (V - R)^2 + c_e * H(pi)
    rho = pi_theta(a|s) / pi_old(a|s),  A = GAE advantage.
What is learned : policy pi(a|s) and value V(s).
When            : every `rollout_len` env steps -> `epochs` passes of minibatch SGD.
Effect          : actions are sampled from the masked policy.

Improvement slots:
    IMP1  (Member 3) RND intrinsic reward (exploration under sparse level rewards)
    IMP2  (shared, built by Member 1) structured action space: object click
          targets + no-op pruning, switched on via AdapterConfig flags
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .base import Agent
from .nets import GridEncoder, masked_logits


@dataclass
class PPOConfig:
    lr: float = 2.5e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    rollout_len: int = 256
    epochs: int = 4
    minibatch: int = 64
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    persist_across_games: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ActorCritic(nn.Module):
    def __init__(self, in_ch: int, n_actions: int):
        super().__init__()
        self.enc = GridEncoder(in_ch)
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)

    def forward(self, x, mask):
        h = self.enc(x)
        return Categorical(logits=masked_logits(self.pi(h), mask)), self.v(h).squeeze(-1)


class PPOAgent(Agent):
    name = "ppo"

    def __init__(self, n_actions: int, in_ch: int = 16, seed: int = 0, cfg: PPOConfig | None = None):
        self.cfg = cfg or PPOConfig()
        self.n_actions, self.in_ch = n_actions, in_ch
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self._build()

    def _build(self):
        self.net = ActorCritic(self.in_ch, self.n_actions).to(self.cfg.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)
        self._clear()

    def _clear(self):
        self.obs, self.masks, self.acts, self.logps, self.vals, self.rews, self.dones = ([] for _ in range(7))

    def begin_game(self, game_id):
        if not self.cfg.persist_across_games:
            self._build()

    def _t(self, x, dtype=torch.float32):
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.cfg.device)

    def act(self, obs, info):
        with torch.no_grad():
            dist, v = self.net(self._t(obs[None]), self._t(info["action_mask"][None], torch.bool))
            a = dist.sample()
        self._pending = (obs, info["action_mask"], int(a), float(dist.log_prob(a)), float(v))
        return int(a)

    def observe(self, obs, action, reward, next_obs, done, info):
        o, m, a, lp, v = self._pending
        self.obs.append(o); self.masks.append(m); self.acts.append(a)
        self.logps.append(lp); self.vals.append(v); self.rews.append(reward); self.dones.append(float(done))
        if len(self.rews) >= self.cfg.rollout_len:
            self._update(next_obs, info["action_mask"], done)

    def _update(self, last_obs, last_mask, last_done):
        c = self.cfg
        with torch.no_grad():
            _, last_v = self.net(self._t(last_obs[None]), self._t(last_mask[None], torch.bool))
        last_v = 0.0 if last_done else float(last_v)
        # --- GAE
        T = len(self.rews)
        adv = np.zeros(T, np.float32)
        gae, next_v = 0.0, last_v
        for t in reversed(range(T)):
            nonterm = 1.0 - self.dones[t]
            delta = self.rews[t] + c.gamma * next_v * nonterm - self.vals[t]
            gae = delta + c.gamma * c.lam * nonterm * gae
            adv[t], next_v = gae, self.vals[t]
        ret = adv + np.asarray(self.vals, np.float32)

        obs = self._t(np.stack(self.obs)); masks = self._t(np.stack(self.masks), torch.bool)
        acts = self._t(self.acts, torch.long); old_lp = self._t(self.logps)
        adv_t = self._t(adv); ret_t = self._t(ret)
        for _ in range(c.epochs):
            for idx in np.array_split(self.rng.permutation(T), max(1, T // c.minibatch)):
                dist, v = self.net(obs[idx], masks[idx])
                lp = dist.log_prob(acts[idx])
                a_mb = adv_t[idx]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                ratio = torch.exp(lp - old_lp[idx])
                pg = -torch.min(ratio * a_mb, torch.clamp(ratio, 1 - c.clip, 1 + c.clip) * a_mb).mean()
                vf = (v - ret_t[idx]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pg + c.vf_coef * vf - c.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.opt.step()
        self.last_loss = float(loss)
        self._clear()
