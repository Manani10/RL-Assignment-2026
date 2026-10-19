# Experiment log

One entry per experiment — failures included. This becomes the "iterative results" section of the report.

## Template

### [ID] short name — owner — date
- **Version:** baseline / imp1 / imp2 / ablation
- **Observed limitation (from which run?):**
- **Hypothesis:**
- **Intervention (what changed, file + flag):**
- **Setup:** games (dev/held-out), seeds, interaction budget, hardware, wall-clock, commit hash
- **Result tag:** `results/<tag>/`
- **Outcome:** levels completed, official score (local / Kaggle), actions used, seed spread
- **What it shows / what is still uncertain:**
- **Next step:**

---

### R0 random sanity check — Member 1 — 2026-10-12
- (fill in)

### R-1 wrapper check on real game — Member 1 — 2026-10-17
- Toolkit: arc-agi 0.9.9, arcengine 0.9.3, Python 3.12.2. 25 public games, 6–10 levels each.
- Random agent, ls20, seed 0, 500 actions: ran without errors (0.83 s).
- Official scorecard: actions = 500, resets = 3 → RESET counts as an action officially; matches our adapter's accounting.
- levels_completed (official) = ours = 0; level_count = 7 matches inspection.
- Human baseline actions per level for ls20: 22, 123, 73, 84, 96, 192, 186.
- Inspection (500 random actions/game): only sp80 and r11l reached level 1. Random click effect rate very low in lp85 (0.01), ft09 (0.07), sb26 (0.05), sc25 (0.06) → motivates object click targets. Many games change every frame (timers), so no-op pruning will rarely trigger there.

### R0 random sanity check — Member 1 — 2026-10-17
- Config: configs/random_baseline.json (15 dev games, seeds 0–2, 5000 actions/run, 2000/episode, 8x8 grid clicks). Tag: random_baseline_v1.
- Official score (mean over seeds of per-game mean, %): 0.0104 ± 0.011. Total best levels over 45 runs: 9.
- Level 1 reached on all seeds: lp85 (~992 actions), r11l (~187). 1/3 seeds: sp80, tu93 (tu93 seed 1 reached level 2). 11/15 games: nothing.
- Wall-clock: 2–28 s per run on laptop CPU.
- Takeaway: reward is extremely sparse; random play almost never completes a level.

### M1-1 shared action space with random agent — Member 1 — 2026-10-17
- Limitation (from inspection): random clicks rarely change the frame (lp85 0.01, sb26 0.05, sc25 0.06, ft09 0.07), so most of the grid-click action space is wasted.
- Hypothesis: clicking detected objects and pruning observed no-op actions makes exploration more efficient, even without learning.
- Change: adapter click_mode="objects", prune_noops=true. Config configs/random_objects.json, tag random_objects_v1. Same games/seeds/budget as R0.
- Result: official score 0.0104 ± 0.011 → 0.429 ± 0.18 (%); total levels 9 → 14. r11l first level in ~14 actions (was 187; beats human baseline, score 4.76 on all seeds); sp80 3/3 seeds (was 1/3; ~552 vs ~4020 actions); tn36 newly solved 3/3 (~687 actions). lp85, tu93 unchanged. 10/15 games still 0 levels.
- Cost: runs ~5–10x slower (object detection each step), e.g. lp85 68 s vs 3 s.
- Uncertain: which of the two changes causes the gain → ablation M1-2.
- Next: give the setting to Members 2 and 3 as Improvement 2.

### M1-2 action-space ablation with random agent — Member 1 — 2026-10-17
- Question: which of the two M1-1 changes (object clicks, no-op pruning) causes the gain?
- Configs: random_pruneonly.json (grid + pruning), random_objonly.json (object clicks, no pruning). Same games/seeds/budget as R0. Tags random_pruneonly_v1, random_objonly_v1.
- Results (official score %, total levels over 45 runs): baseline 0.0104 ± 0.011 / 9; prune only 0.0115 ± 0.013 / 9; objects only 0.430 ± 0.18 / 15; both 0.429 ± 0.18 / 14.
- Pruning never fires in practice: r11l, sp80, tu93 trajectories identical to baseline (same actions to level 1: 187, 4020, 3570) → frames almost never recur exactly (timers / animations change every frame).
- Object clicks alone give the whole gain (objects-only also reached level 1 on sb26 in 1/3 seeds).
- Decision: Improvement 2 for DQN and PPO = object clicks only ("click_mode": "objects", "prune_noops": false). Pruning reported as a negative result.

### M2-0 DQN baseline — Member 2 — 2026-10-18
- **Version:** baseline
- **Observed limitation (from which run?):** Random sanity check R0 showed sparse rewards, while random exploration reached level 1 in only a small subset of games. The DQN baseline was evaluated to establish a learning-agent reference.
- **Hypothesis:** Double DQN with replay, a target network, and epsilon-greedy exploration over valid actions may learn useful action sequences better than random selection.
- **Intervention (what changed, file + flag):** Baseline Double DQN implementation in `src/arcrl/agents/dqn.py`; agent `dqn`. No novelty bonus or action-space changes.
- **Setup:** 15 development games, seeds 0–2, 5,000 actions per run, 2,000 actions per episode, Windows 11, Python 3.12.13, PyTorch CPU. Commit: `b85e2edb7993fd871dfab8603b12a524325851d3`.
- **Result tag:** `results/dqn_baseline_v1/`; config: `configs/dqn_baseline.json`.
- **Outcome:** 45 runs completed. Mean official score across seed-level means: 0.00646 ± 0.00614. Sum of per-run best levels: 7. Non-zero official score in 7/45 runs; progress in 3/15 games.
- **What it shows / what is still uncertain:** The DQN baseline did not demonstrate an overall advantage over random. Successful runs were sparse, and performance varied between games and seeds.
- **Next step:** Test count-based novelty as Improvement 1.

### M2-1 count-based novelty bonus — Member 2 — 2026-10-19
- **Version:** imp1
- **Observed limitation (from which run?):** DQN baseline `dqn_baseline_v1` had non-zero official scores in 7/45 runs and made progress in 3/15 games.
- **Hypothesis:** Rewarding less-visited grid states may encourage more useful exploration and increase level progress.
- **Intervention (what changed, file + flag):** Added the count-based bonus `b(s') = beta / sqrt(N(s'))`, with `beta = 0.01`, in `src/arcrl/agents/dqn.py`. Registered `dqn_imp1` in `src/arcrl/agents/__init__.py`. Novelty is enabled for this variant only. The bonus affects training rewards stored in replay memory, not the environment reward or official ARC score. Added `tests/test_dqn_novelty.py`.
- - **Setup:** Same 15 development games, seeds 0–2, 5,000 actions per run, 2,000 actions per episode, Windows 11, Python 3.12.13, PyTorch CPU. Baseline commit: `b85e2edb7993fd871dfab8603b12a524325851d3`. The Improvement 1 code and config were uncommitted when the evaluation ran; they were subsequently committed unchanged in `a34b023b4e013f2ff962e99eb0a3026d2a6e986f`.
- **Result tag:** `results/dqn_imp1_v1/`; config: `configs/dqn_imp1.json`.
- **Outcome:** 45 runs completed. Mean official score across seed-level means: 0.00696 ± 0.00561. Sum of per-run best levels: 8. Non-zero official score in 8/45 runs; progress in 3/15 games. All 25 automated tests passed, including six novelty tests.
- **What it shows / what is still uncertain:** Compared with baseline DQN, the novelty variant produced one additional successful run and a small increase in mean official score, but did not increase the number of games with progress or outperform random. The evidence does not establish a reliable general improvement.
- **Next step:** Preserve this result and evaluate the planned action-space improvement separately.

### M2-1a novelty pipeline smoke test — Member 2 — 2026-10-19
- **Version:** imp1 validation
- **Observed limitation (from which run?):** Before committing to the full development evaluation, the new agent needed a basic integration check.
- **Hypothesis:** The novelty-enabled agent can execute the evaluation loop without errors in the mock environment.
- **Intervention (what changed, file + flag):** Ran `dqn_imp1` using the mock environment with three seeds.
- - **Setup:** Mock environment, seeds 0–2, 2,000-action budget, Windows 11, Python 3.12.13, PyTorch CPU. Baseline commit: `b85e2edb7993fd871dfab8603b12a524325851d3`. The Improvement 1 implementation was uncommitted when the smoke test ran; it was subsequently committed unchanged in `a34b023b4e013f2ff962e99eb0a3026d2a6e986f`.
- **Result tag:** `results/dqn_imp1_mock_full_20261019/`.
- **Outcome:** All three runs completed, each reporting `best_levels=1`, `wins=0`, and one episode. Runtime was approximately 21 seconds per run.
- **What it shows / what is still uncertain:** The agent and evaluation pipeline ran, but this mock check does not demonstrate successful learning on real ARC games.
- **Next step:** Run the full 15-game Improvement 1 evaluation.