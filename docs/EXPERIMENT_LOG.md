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
