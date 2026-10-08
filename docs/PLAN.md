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

| | **Member 1 — Lead / Infrastructure & Evaluation** | **Member 2 — DQN track** | **Member 3 — PPO track** |
|---|---|---|---|
| Owns code | `src/arcrl/env/`, `src/arcrl/eval/`, `scripts/`, `kaggle/`, random agent, plotting | `src/arcrl/agents/dqn.py` + DQN improvements | `src/arcrl/agents/ppo.py` + PPO improvements |
| Rubric focus | Problem formulation (10), rigour/fair comparison (25), reproducibility (10), leaderboard (10) | Baseline (20) + improvements (25) for DQN | Baseline (20) + improvements (25) for PPO |

### Member 1 — Lead, environment, evaluation, Kaggle, report integration
1. **Admin (day 1):** create the Kaggle team, join **both** tracks (ARC-AGI-3 + Paper Track) with the same members and team name; set up this repo and invite members; Overleaf with the **RLC 2026 template, cover page disabled**.
2. **Environment:** install `arc-agi` (Python 3.12), list public games, verify adapter semantics on real games (observation, action mask, reset cost, level progression). Record package versions.
3. **Dev / held-out split:** pick the development games and a disjoint held-out set; write it into `configs/splits.json` and *never* tune on held-out games. Disclose every game used.
4. **Evaluation protocol:** freeze budgets, seeds (≥3), stopping rules, checkpoint policy; run the **random sanity check**; build the plotting script (learning curve vs. interaction budget, per-game table, seed spread).
5. **Kaggle pipeline:** get `kaggle/my_agent.py` working inside the official starter; first (random) submission by **Mon 12 Oct**; later submit the best agent, choose the **nominated** leaderboard entry, save submission ID/score/timestamp evidence.
6. **Final comparison & ablations section** (DQN vs PPO: score, completion, actions, compute, held-out generalisation), plus the report's intro, MDP formulation, environment/adapter and evaluation sections. Integrate everyone's sections, enforce 8-page limit, write the Paper Track condensed version (<1500 words), build the ZIP.

### Member 2 — DQN (algorithm covered in the course)
1. Get the DQN baseline training on the mock game, then on real dev games via `scripts/evaluate.py` (3 seeds, Member 1's budgets).
2. Identify the main limitation from the baseline results (likely: never sees a reward → nothing to learn from; huge click action space).
3. **Improvement 1 (suggested):** exploration — count-based novelty bonus on frame hashes (`grid_hash`) or "action-effect" pruning (mask actions that never change the frame). Hypothesis → implement behind a flag → evaluate.
4. **Improvement 2 (suggested), motivated by Imp-1 results:** object-centric click targets (click connected-component centroids instead of an 8×8 grid), prioritised replay, or an episodic state-graph memory.
5. Ablations: each improvement on/off. Write the DQN background, method and iterative-results sections (limitation → hypothesis → change → result for each step).

### Member 3 — PPO (algorithm not covered in the course)
1. Get PPO baseline running with identical encoder, action space and budgets.
2. **Explain PPO properly in the report** (required): clipped surrogate objective, GAE, value/entropy terms, when updates happen, what is learned.
3. **Improvement 1 (suggested):** intrinsic motivation — Random Network Distillation (RND) bonus for sparse level rewards.
4. **Improvement 2 (suggested):** memory — recurrent (GRU) policy, or a curriculum over games/levels; motivate it from Imp-1 failure cases.
5. Ablations, PPO background/method/results sections. If you reuse an idea Member 2 also uses, evaluate it separately for PPO.

### Everyone
- Log every experiment in `docs/EXPERIMENT_LOG.md` (including failures — they earn marks).
- Use the shared encoder, adapter and evaluation protocol unchanged so comparisons are fair. Disclose any unequal budgets.
- Each member writes their **contribution statement** (one paragraph) for the submission.
- Cite every paper, repo and pretrained model used; declare any generative-AI use per Wits policy.

---

## 3. Timeline (≈19 days)

| Dates | Milestone | Owner |
|---|---|---|
| **Thu 8 – Sun 11 Oct** | Kaggle team on both tracks; repo + Overleaf; adapter verified on real games; DQN & PPO run end-to-end on mock + 1 real game | M1 / M2 / M3 |
| **Mon 12 Oct** | Evaluation protocol + dev/held-out split **frozen**; random baseline results; first Kaggle (random) submission | M1 |
| **Tue 13 – Thu 15 Oct** | DQN & PPO baselines evaluated (3 seeds, dev games). Team meeting: agree on limitations | M2, M3 |
| **Fri 16 – Mon 19 Oct** | Improvement 1 for each algorithm, evaluated | M2, M3 |
| **Tue 20 – Thu 22 Oct** | Improvement 2 + ablations; held-out evaluation; best agent submitted to Kaggle | all |
| **Fri 23 – Sun 25 Oct** | Report draft complete; figures final; nominate leaderboard entry | all, M1 integrates |
| **Mon 26 Oct** | Proofread, page limit, Paper Track write-up, code cleanup (no API keys) | all |
| **Tue 27 Oct (before 23:59)** | Submit `Kaggle-team-name.zip` on Moodle | M1 |

---

## 4. Ground rules that cost marks if broken
- Report the **official** ARC score (scorecard / Kaggle), never shaped training return. Say which set and scoring version; separate local vs Kaggle scores.
- No hidden game state, no per-game scripted solutions, no tuning on held-out games.
- State what persists between actions, levels, games and runs, and what is offline training vs online learning.
- Same games, same interaction and compute budgets for DQN vs PPO; report training cost separately from evaluation cost.
- Show unsuccessful runs and seed variability, not just the best trajectory.
- Merely changing hyperparameters is **not** an improvement.
