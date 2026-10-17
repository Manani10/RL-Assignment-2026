# Project plan — ARC-AGI-3 RL assignment (COMS4061A/COMS7071A)

**Moodle deadline: Tue 27 Oct 2026, 23:59 SAST.** Kaggle: submissions 2 Nov, papers 8 Nov (verify on Kaggle). Team-merge cut-off on Kaggle is earlier — join as a team **this week**.

---

## 1. What this assignment is about

ARC-AGI-3 is a set of small video-game-like puzzles. Each game is a **64×64 grid with 16 colours**. The agent gets **no instructions**: it doesn't know what the buttons do, what the goal is, or what "winning" looks like. It must find out by playing.

- **Actions:** `RESET`, `ACTION1–5` (simple, e.g. up/down/left/right/interact — meaning differs per game), `ACTION6` (click at an (x, y) cell), `ACTION7` (undo, where supported). Each game exposes only some of these, and the set can change per step.
- **Feedback:** there is no reward signal. All we see is the new frame, the game state (`NOT_FINISHED`, `WIN`, `GAME_OVER`) and `levels_completed`. So our RL reward is essentially "+1 when a level is completed" — extremely sparse.
- **Scoring (RHAE):** per level, `(human_actions / ai_actions)^2`, capped at 1.15; per game, a level-weighted average (later levels count more); total = mean over games. Completing levels matters most; wasting actions is punished quadratically. A score of 0 is normal for naive agents.
- **Kaggle:** a code competition with **no internet** during evaluation. Our notebook plays **unseen games**, so whatever we submit has to **learn online during the game** (or carry over general knowledge from offline training on public games).

**What we must deliver (marks):** two learning agents — **DQN** (from the course) and **PPO** (not in the course) — each taken through **baseline → improvement 1 → improvement 2**, each step motivated by the previous step's evidence; a random-policy sanity check; fair comparison + ablations + ≥3 seeds; an 8-page RLC-template report; a Kaggle code submission (10 marks from class rank) and a Paper Track write-up.

> The marks are mostly for the *investigation* (problem formulation 10, baselines 20, improvements 25, rigour 25, report 10, leaderboard 10). A well-analysed improvement that *doesn't* help still earns credit.

---

## 2. Who does what

| | **Member 1 — Lead: environment, shared action space, evaluation, comparison** | **Member 2 — DQN** | **Member 3 — PPO** |
|---|---|---|---|
| Owns code | `src/arcrl/env/` (incl. the **shared action-space improvement**), `src/arcrl/eval/`, `scripts/`, `kaggle/` | `agents/dqn.py` + DQN exploration improvement | `agents/ppo.py` + PPO exploration improvement |
| Runs | Random sanity check, **all ablations, all held-out evaluations**, Kaggle submissions | DQN baseline, imp1, imp2 on dev games (3 seeds) | PPO baseline, imp1, imp2 on dev games (3 seeds) |
| Writes | Intro, problem formulation, setup, shared-improvement method, **DQN vs PPO comparison + ablations + generalisation**, abstract/conclusion, Paper Track, integration | DQN background + DQN iterative results | PPO background (full PPO explanation) + PPO iterative results |

Each algorithm still goes baseline → improvement 1 → improvement 2:
- **Improvement 1 (own):** exploration, built by the algorithm owner (DQN: count-based novelty bonus; PPO: RND).
- **Improvement 2 (shared):** the structured action space built by Member 1. Each algorithm owner integrates it, runs it, and analyses its effect on *their* algorithm (the brief allows a shared idea if its effect is evaluated separately for each algorithm).

### Member 1 — Lead: environment, shared action space, evaluation, comparison
Detailed, dated checklist: **[MEMBER1_CHECKLIST.md](MEMBER1_CHECKLIST.md)**.
1. **Admin:** Kaggle team on **both** tracks (same members, same team name); GitHub (collaborators, protect `main`); Overleaf with the RLC 2026 template, cover page disabled.
2. **Environment:** verify the adapter on real games with `scripts/inspect_games.py`; record package versions.
3. **Dev / held-out split** with `scripts/make_split.py`; freeze it; never tune on held-out games.
4. **Evaluation protocol:** freeze budgets and seeds, run the random sanity check, keep `scripts/analyze.py` as the single source of tables and figures.
5. **Shared improvement — structured action space** (in `src/arcrl/env/`, behind flags, ready by **Mon 19 Oct**):
   - *Object click targets:* replace the fixed 8×8 click grid with clicks on detected objects (connected regions of one colour, background excluded), keeping the action dimension fixed so networks don't change.
   - *No-op pruning:* remember which actions did not change the frame in a given state and mask them out (always leave at least one action).
   - Tests on the mock game; a short method write-up for the report.
6. **Ablations and held-out:** run every ablation (full agent / minus imp1 / minus imp2, for both algorithms) and the one-time held-out evaluation, on the same games, budgets and seeds.
7. **DQN vs PPO comparison:** final table, learning curves, interaction efficiency, compute, held-out generalisation, ablation attribution, failure-case replays.
8. **Kaggle:** first (random) submission by Mon 12 Oct; submit the best agent; nominate the leaderboard entry and save evidence; Paper Track write-up (< 1500 words).
9. **Report:** write own sections, integrate all sections, enforce the 8-page limit, build the Moodle ZIP.

### Member 2 — DQN (algorithm covered in the course)
1. Get the DQN baseline running on the mock game, then on real dev games via `scripts/evaluate.py` (3 seeds, shared budgets).
2. From baseline results, identify the main limitation (likely: almost never sees a reward).
3. **Improvement 1 — exploration:** count-based novelty bonus on frame hashes (`grid_hash`). Limitation → hypothesis → implement behind a flag → evaluate.
4. **Improvement 2 — shared action space:** turn on Member 1's object clicks / no-op pruning for DQN, evaluate, and explain what it changes for a value-based agent.
5. Write the DQN background, method and iterative-results sections (limitation → hypothesis → change → result for each step), including failure cases.

### Member 3 — PPO (algorithm not covered in the course)
1. Get the PPO baseline running with the identical encoder, action space and budgets.
2. **Explain PPO properly in the report** (required): clipped surrogate objective, GAE, value and entropy terms, when updates happen, what is learned.
3. **Improvement 1 — exploration:** Random Network Distillation (RND) intrinsic reward.
4. **Improvement 2 — shared action space:** turn on Member 1's object clicks / no-op pruning for PPO, evaluate, and explain what it changes for a policy-gradient agent.
5. Write the PPO background, method and iterative-results sections, including failure cases.

### Everyone
- Log every experiment in `docs/EXPERIMENT_LOG.md` (including failures — they earn marks).
- Use the shared encoder, adapter and evaluation protocol unchanged so comparisons are fair. Disclose any unequal budgets.
- Hand Member 1 your final result tags and configs by **Thu 22 Oct** for the comparison.
- Each member writes a one-paragraph **contribution statement**, including any generative-AI use (Wits policy).
- Cite every paper, repo and pretrained model used.

---

## 3. Timeline (≈19 days)

| Dates | Member 1 | Member 2 (DQN) | Member 3 (PPO) |
|---|---|---|---|
| **Thu 8 – Sun 11 Oct** | Kaggle team, GitHub, Overleaf; inspect games; verify adapter | DQN runs on mock + 1 real game | PPO runs on mock + 1 real game |
| **Mon 12 Oct** | Freeze split + protocol; random baseline; first Kaggle submission | — | — |
| **Tue 13 – Thu 15 Oct** | Test shared action space on real games (code is in `env/actions.py`); write intro, problem, setup | DQN baseline evaluated (3 seeds) | PPO baseline evaluated (3 seeds) |
| **Fri 16 – Mon 19 Oct** | Shared action space ready for M2/M3 (**Mon 19**); Overleaf report skeleton | Imp1: novelty bonus | Imp1: RND |
| **Tue 20 – Thu 22 Oct** | Ablation runs; held-out evaluation; best agent to Kaggle | Imp2: shared action space for DQN; hand over results | Imp2: shared action space for PPO; hand over results |
| **Fri 23 – Sun 25 Oct** | Comparison section; integrate report; nominate leaderboard entry | DQN sections | PPO sections |
| **Mon 26 Oct** | Proofread, page limit, Paper Track, code cleanup | Review | Review |
| **Tue 27 Oct, before 23:59** | Submit `Kaggle-team-name.zip` on Moodle | — | — |

---

## 4. Ground rules that cost marks if broken
- Report the **official** ARC score (scorecard / Kaggle), never shaped training return. Say which set and scoring version; separate local vs Kaggle scores.
- No hidden game state, no per-game scripted solutions, no tuning on held-out games.
- State what persists between actions, levels, games and runs, and what is offline training vs online learning.
- Same games, same interaction and compute budgets for DQN vs PPO; report training cost separately from evaluation cost.
- Show unsuccessful runs and seed variability, not just the best trajectory.
- Merely changing hyperparameters is **not** an improvement.
