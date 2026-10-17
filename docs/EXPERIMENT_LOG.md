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