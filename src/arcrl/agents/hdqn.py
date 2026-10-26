"""Hierarchical DQN (h-DQN) for ARC-AGI-3 -- OWNER: Member 3 (replaces the PPO agent).

Based on Kulkarni et al. (2016), "Hierarchical Deep Reinforcement Learning:
Integrating Temporal Abstraction and Intrinsic Motivation".

Two levels, both value-based (DQN family, Double-DQN targets, target networks):

  META-CONTROLLER  Q2(s, g)      picks a SUBGOAL g every few steps.
  CONTROLLER       Q1(s, g, a)   picks primitive actions until the subgoal is
                                 reached or the horizon H runs out.

Subgoals (public frame only, no hidden state, no game-specific scripts):
    the 64x64 frame is cut into (64/goal_cell)^2 cells (default 16 px -> 4x4 = 16 cells;
    8 px -> 64 cells is a hyperparameter).
    Subgoal g = "make at least `min_change_px` pixels change inside cell g".
    In a game where actions move an avatar this means "go to cell g"; where
    clicks toggle objects it means "act on whatever is in cell g".

Rewards
    controller : +goal_reward when the subgoal cell changed (intrinsic, dense).
    meta       : sum of the environment's level rewards over the segment
                 (+ an optional novelty bonus, IMP1).
    Nothing intrinsic is ever reported as a score.

What is learned / when
    Q1 : every `train_every` env steps once `learn_start` transitions exist.
    Q2 : every `meta_train_every` segments once `meta_learn_start` exist.
    Everything is re-initialised per game (persist_across_games=False) and
    persists across levels and resets inside a game, including the GRU state
    and the novelty counts.

Improvement slots (one flag each, so ablations are single switches):
    IMP1  novelty=True   count-based novelty bonus on the meta reward: the bonus
          nov_coef/sqrt(N(frame hash at the end of the segment)) rewards subgoals
          that lead to frames not seen before in this game.
    IMP2  gru=True       the meta-controller becomes recurrent (GRU over the
          sequence of meta decisions, fed with the previous subgoal and its
          outcome), trained on stored sequences with a short burn-in
          (DRQN, Hausknecht & Stone 2015; burn-in as in R2D2, Kapturowski et al. 2019).
    ABLATION random_meta=True   subgoals chosen uniformly at random: isolates what
          the *learned* meta-controller adds on top of the controller.

Simplifications to list in the report: per-goal epsilon annealing of the paper is
replaced by one decaying epsilon per level; timeouts of a subgoal are not terminal
for the controller; GAME_OVER auto-resets are ordinary transitions; GRU sequences
start from a zero state with a 2-step burn-in (no stored recurrent state).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..env.encoding import grid_hash
from .base import Agent, masked_choice
from .dqn import to_index, to_onehot_batch
from .nets import GridEncoder


@dataclass
class HDQNConfig:
    # --- controller (low level) ---
    lr: float = 2.5e-4
    gamma: float = 0.99
    buffer_size: int = 20_000
    batch_size: int = 32
    learn_start: int = 500
    train_every: int = 4
    target_update: int = 1_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 3_000
    # --- subgoals ---
    goal_cell: int = 16           # 16 -> 16x16 px cells -> 16 subgoals (8 -> 64)
    horizon: int = 10             # max primitive steps per subgoal
    goal_reward: float = 1.0
    min_change_px: int = 1
    # --- meta-controller (high level) ---
    meta_lr: float = 2.5e-4
    meta_gamma: float = 0.95      # per segment
    meta_batch: int = 16          # windows per update
    seq_len: int = 8              # window = seq_len + 1 consecutive segments
    burn_in: int = 2              # GRU only: steps unrolled without loss
    meta_learn_start: int = 32
    meta_train_every: int = 2
    meta_target_update: int = 50
    meta_buffer: int = 5_000
    meta_eps_start: float = 1.0
    meta_eps_end: float = 0.1
    meta_eps_decay: int = 300     # in segments
    gru_hidden: int = 128
    # --- variants (all off = baseline h-DQN) ---
    random_meta: bool = False
    novelty: bool = False         # IMP1
    nov_coef: float = 0.5
    gru: bool = False             # IMP2
    persist_across_games: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


VARIANTS = {
    "hdqn": {},
    "hdqn_randmeta": {"random_meta": True},
    "hdqn_nov": {"novelty": True},
    "hdqn_gru": {"gru": True},
    "hdqn_nov_gru": {"novelty": True, "gru": True},
}


def hdqn_config(name: str) -> HDQNConfig:
    if name not in VARIANTS:
        raise KeyError(f"Unknown h-DQN variant {name!r}")
    return replace(HDQNConfig(), **VARIANTS[name])


class GoalQNet(nn.Module):
    """Controller Q1(s, g, .): the subgoal enters as one extra input plane (1 inside cell g)."""

    def __init__(self, in_ch: int, n_actions: int):
        super().__init__()
        self.enc = GridEncoder(in_ch + 1)
        self.head = nn.Linear(256, n_actions)

    def forward(self, x: torch.Tensor, goal_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(torch.cat([x, goal_mask[:, None]], 1)))


class MetaNet(nn.Module):
    """Meta-controller Q2(s, .) over subgoals; optionally recurrent over meta steps.

    aux = [one-hot previous subgoal, previous segment hit flag, previous segment length / H].
    """

    def __init__(self, in_ch: int, n_goals: int, gru: bool, hidden: int = 128):
        super().__init__()
        self.gru_on = gru
        self.enc = GridEncoder(in_ch)
        if gru:
            self.rnn = nn.GRU(256 + n_goals + 2, hidden, batch_first=True)
            self.head = nn.Linear(hidden, n_goals)
        else:
            self.head = nn.Linear(256, n_goals)

    def seq(self, x: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """x (B,T,C,64,64), aux (B,T,G+2) -> Q (B,T,G); the GRU starts from a zero state."""
        B, T = x.shape[:2]
        f = self.enc(x.reshape(B * T, *x.shape[2:])).reshape(B, T, -1)
        if self.gru_on:
            f, _ = self.rnn(torch.cat([f, aux], -1))
        return self.head(f)

    def step(self, x: torch.Tensor, aux: torch.Tensor, h):
        """One decision: x (1,C,64,64), aux (1,G+2), h (1,1,H) or None -> (Q (1,G), new h)."""
        f = self.enc(x)
        if self.gru_on:
            out, h = self.rnn(torch.cat([f, aux], -1).unsqueeze(1), h)
            f = out[:, 0]
        return self.head(f), h


class GoalReplay:
    """Controller replay: colour-index grids (int8) plus the subgoal of each transition."""

    def __init__(self, cap: int, k: int, n_actions: int):
        self.cap, self.i, self.full = cap, 0, False
        self.s = np.zeros((cap, k, 64, 64), np.int8)
        self.s2 = np.zeros_like(self.s)
        self.g = np.zeros(cap, np.int64)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)
        self.m2 = np.zeros((cap, n_actions), bool)

    def add(self, s, g, a, r, s2, d, m2):
        j = self.i
        self.s[j], self.g[j], self.a[j], self.r[j], self.s2[j], self.d[j], self.m2[j] = s, g, a, r, s2, d, m2
        self.i = (j + 1) % self.cap
        self.full = self.full or self.i == 0

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, n, rng):
        idx = rng.integers(0, len(self), n)
        return self.s[idx], self.g[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx], self.m2[idx]


class HDQNAgent(Agent):
    name = "hdqn"

    def __init__(self, n_actions: int, in_ch: int = 16, seed: int = 0, cfg: HDQNConfig | None = None):
        self.cfg = cfg or HDQNConfig()
        self.n_actions, self.in_ch = n_actions, in_ch
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self._build()

    # ------------------------------------------------------------------ setup
    def _build(self):
        c = self.cfg
        n = 64 // c.goal_cell
        self.n_goals = n * n
        self.ctrl = GoalQNet(self.in_ch, self.n_actions).to(c.device)
        self.ctrl_t = GoalQNet(self.in_ch, self.n_actions).to(c.device)
        self.ctrl_t.load_state_dict(self.ctrl.state_dict())
        self.copt = torch.optim.Adam(self.ctrl.parameters(), lr=c.lr)
        self.cbuf = GoalReplay(c.buffer_size, self.in_ch // 16, self.n_actions)
        self.meta = MetaNet(self.in_ch, self.n_goals, c.gru, c.gru_hidden).to(c.device)
        self.meta_t = MetaNet(self.in_ch, self.n_goals, c.gru, c.gru_hidden).to(c.device)
        self.meta_t.load_state_dict(self.meta.state_dict())
        self.mopt = torch.optim.Adam(self.meta.parameters(), lr=c.meta_lr)
        self.goal_masks = torch.zeros(self.n_goals, 64, 64, device=c.device)
        for g in range(self.n_goals):
            sy, sx = self._cell(g)
            self.goal_masks[g, sy, sx] = 1.0
        self.mrec: list[dict] = []          # one record per finished segment, in time order
        self.t = 0                          # primitive steps
        self.mt = 0                         # finished segments
        self.goal = None                    # current subgoal (None = pick a new one)
        self.seg_steps, self.seg_reward, self.seg_hit = 0, 0.0, False
        self.seg_start, self.seg_prev = None, (-1, (0.0, 0.0))
        self.prev_goal, self.prev_out = -1, (0.0, 0.0)
        self.h = None                       # GRU state, persists across episodes within a game
        self.counts: dict[str, int] = {}    # novelty counts (per game)
        self.goal_counts = np.zeros(self.n_goals)
        self.recent = deque(maxlen=25)
        self.log: list[dict] = []
        self.last_closs, self.last_mloss = None, None
        self._cur_grid, self._last_used = None, -1

    def begin_game(self, game_id):
        if not self.cfg.persist_across_games:
            self._build()

    def _cell(self, g: int):
        cell = self.cfg.goal_cell
        gy, gx = divmod(g, 64 // cell)
        return slice(gy * cell, (gy + 1) * cell), slice(gx * cell, (gx + 1) * cell)

    def _eps_c(self):
        c = self.cfg
        return c.eps_start + min(1.0, self.t / c.eps_decay_steps) * (c.eps_end - c.eps_start)

    def _eps_m(self):
        c = self.cfg
        return c.meta_eps_start + min(1.0, self.mt / c.meta_eps_decay) * (c.meta_eps_end - c.meta_eps_start)

    def _aux(self, pg, po):
        a = np.zeros((1, self.n_goals + 2), np.float32)
        if pg >= 0:
            a[0, pg] = 1.0
        a[0, -2:] = po
        return torch.as_tensor(a, device=self.cfg.device)

    # -------------------------------------------------------------------- act
    def act(self, obs, info):
        used = int(info.get("actions_used", 0))
        if self.goal is not None and used < self._last_used:
            self._end_segment(done=False, end_grid=None)   # new episode began: close the dangling segment
        self._last_used = used
        self._cur_grid = info["grid"]
        if self.goal is None:
            self._start_segment(obs)
        mask = info["action_mask"]
        if self.rng.random() < self._eps_c():
            return masked_choice(self.rng, mask)
        with torch.no_grad():
            x = torch.as_tensor(obs[None], device=self.cfg.device)
            q = self.ctrl(x, self.goal_masks[self.goal][None])[0].cpu().numpy()
        q[~mask] = -np.inf
        return int(q.argmax())

    def _start_segment(self, obs):
        c = self.cfg
        if c.random_meta:
            g = int(self.rng.integers(self.n_goals))
        else:
            x = torch.as_tensor(obs[None], device=c.device)
            with torch.no_grad():   # always advance the GRU state, even when exploring
                q, self.h = self.meta.step(x, self._aux(self.prev_goal, self.prev_out), self.h)
            if self.rng.random() < self._eps_m():
                g = int(self.rng.integers(self.n_goals))
            else:
                g = int(q[0].argmax())
        self.goal_counts[g] += 1
        self.goal, self.seg_steps, self.seg_reward, self.seg_hit = g, 0, 0.0, False
        self.seg_start = to_index(obs)
        self.seg_prev = (self.prev_goal, self.prev_out)

    # ---------------------------------------------------------------- observe
    def observe(self, obs, action, reward, next_obs, done, info):
        c = self.cfg
        g = self.goal
        sy, sx = self._cell(g)
        nxt = info["grid"]
        hit = int(np.count_nonzero(self._cur_grid[sy, sx] != nxt[sy, sx])) >= c.min_change_px
        r_int = c.goal_reward if hit else 0.0
        # controller transition is terminal when the subgoal is reached (or the game ends);
        # a time-out is NOT terminal.
        self.cbuf.add(to_index(obs), g, action, r_int, to_index(next_obs), float(hit or done), info["action_mask"])
        self.t += 1
        self.seg_steps += 1
        self.seg_reward += float(reward)
        self.seg_hit = self.seg_hit or hit
        self._cur_grid = nxt
        self._last_used = int(info.get("actions_used", self._last_used))
        if len(self.cbuf) >= c.learn_start and self.t % c.train_every == 0:
            self._learn_ctrl()
        if self.t % c.target_update == 0:
            self.ctrl_t.load_state_dict(self.ctrl.state_dict())
        if hit or reward > 0 or done or self.seg_steps >= c.horizon:
            self._end_segment(done=done, end_grid=nxt)

    def _end_segment(self, done: bool, end_grid):
        c = self.cfg
        bonus = 0.0
        if c.novelty and end_grid is not None:     # IMP1: count-based novelty of the resulting frame
            key = grid_hash(end_grid)
            self.counts[key] = self.counts.get(key, 0) + 1
            bonus = c.nov_coef / float(np.sqrt(self.counts[key]))
        out = (float(self.seg_hit), self.seg_steps / c.horizon)
        self.mrec.append({"s": self.seg_start, "g": self.goal, "r": self.seg_reward + bonus, "d": float(done),
                          "pg": self.seg_prev[0], "po": self.seg_prev[1]})
        if len(self.mrec) > c.meta_buffer:
            del self.mrec[0]
        self.prev_goal, self.prev_out = self.goal, out
        self.recent.append((float(self.seg_hit), self.seg_steps, self.seg_reward, bonus))
        self.goal = None
        self.mt += 1
        if not c.random_meta:
            if len(self.mrec) >= max(c.meta_learn_start, c.seq_len + 1) and self.mt % c.meta_train_every == 0:
                self._learn_meta()
            if self.mt % c.meta_target_update == 0:
                self.meta_t.load_state_dict(self.meta.state_dict())
        if self.mt % 25 == 0:
            self._log()

    # ---------------------------------------------------------------- learning
    def _learn_ctrl(self):
        c = self.cfg
        s, g, a, r, s2, d, m2 = self.cbuf.sample(c.batch_size, self.rng)
        s, s2 = to_onehot_batch(s, c.device), to_onehot_batch(s2, c.device)
        g = torch.as_tensor(g, device=c.device)
        gm = self.goal_masks[g]
        a, r = torch.as_tensor(a, device=c.device), torch.as_tensor(r, device=c.device)
        d, m2 = torch.as_tensor(d, device=c.device), torch.as_tensor(m2, device=c.device)
        q = self.ctrl(s, gm).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():   # Double DQN target, masked argmax, same subgoal for s and s'
            a2 = self.ctrl(s2, gm).masked_fill(~m2, -1e9).argmax(1, keepdim=True)
            target = r + c.gamma * (1 - d) * self.ctrl_t(s2, gm).gather(1, a2).squeeze(1)
        loss = F.smooth_l1_loss(q, target)
        self.copt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.ctrl.parameters(), 10.0)
        self.copt.step()
        self.last_closs = float(loss)

    def _learn_meta(self):
        c = self.cfg
        L = c.seq_len + 1
        n = len(self.mrec)
        starts = self.rng.integers(0, n - L + 1, c.meta_batch)
        win = [self.mrec[i:i + L] for i in starts]
        B = len(win)
        s = np.stack([np.stack([r["s"] for r in w]) for w in win])              # (B,L,k,64,64) int8
        x = to_onehot_batch(s.reshape(B * L, *s.shape[2:]), c.device).reshape(B, L, -1, 64, 64)
        g = torch.as_tensor(np.array([[r["g"] for r in w] for w in win]), device=c.device)
        rew = torch.as_tensor(np.array([[r["r"] for r in w] for w in win], np.float32), device=c.device)
        d = torch.as_tensor(np.array([[r["d"] for r in w] for w in win], np.float32), device=c.device)
        aux = np.zeros((B, L, self.n_goals + 2), np.float32)
        for bi, w in enumerate(win):
            for ti, r in enumerate(w):
                if r["pg"] >= 0:
                    aux[bi, ti, r["pg"]] = 1.0
                aux[bi, ti, -2:] = r["po"]
        aux = torch.as_tensor(aux, device=c.device)
        q_on = self.meta.seq(x, aux)                                           # (B,L,G)
        with torch.no_grad():
            q_tg = self.meta_t.seq(x, aux)
            a_star = q_on[:, 1:].argmax(2, keepdim=True)                       # Double DQN
            nxt = q_tg[:, 1:].gather(2, a_star).squeeze(2)
            target = rew[:, :-1] + c.meta_gamma * (1 - d[:, :-1]) * nxt
        q = q_on[:, :-1].gather(2, g[:, :-1, None]).squeeze(2)                 # (B,L-1)
        lo = c.burn_in if c.gru else 0
        loss = F.smooth_l1_loss(q[:, lo:], target[:, lo:])
        self.mopt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.meta.parameters(), 10.0)
        self.mopt.step()
        self.last_mloss = float(loss)

    # ------------------------------------------------------------ diagnostics
    def _log(self):
        rec = list(self.recent)
        p = self.goal_counts / max(1.0, self.goal_counts.sum())
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        self.log.append({
            "t": self.t, "segments": self.mt,
            "goal_success": float(np.mean([r[0] for r in rec])),   # fraction of recent subgoals reached
            "seg_len": float(np.mean([r[1] for r in rec])),
            "ext_reward": float(sum(r[2] for r in rec)),
            "nov_bonus": float(sum(r[3] for r in rec)),
            "goal_entropy": ent,                                    # nats; ln(n_goals) = uniform choice
            "eps_ctrl": self._eps_c(), "eps_meta": self._eps_m(),
            "ctrl_loss": self.last_closs, "meta_loss": self.last_mloss,
        })

    def diagnostics(self):
        return {"meta_log": self.log}
