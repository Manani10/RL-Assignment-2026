# h-DQN experiment log — Member 3

Kept separate from the shared `docs/EXPERIMENT_LOG.md` (same entry format). Results folders are not committed (gitignored); `hdqn_results.zip` is on the team drive.

### H0 h-DQN baseline, hierarchy ablation and two improvements — Member 3 — 2026-10-22
- Code: `src/arcrl/agents/hdqn.py` (own implementation of Kulkarni et al. 2016), commit `fc31373`. Configs `configs/hdqn_*.json` are identical to `configs/dqn_baseline.json` except `agent` and `tag`: 15 dev games, seeds 0–2, 5000 actions/run, 2000/episode, 8x8 grid clicks. Run in a Kaggle notebook (`hdqn_runner.ipynb`), arc-agi 0.9.9, torch 2.11.0+cu130.
- Design: 16 subgoals (4x4 cells of the frame), subgoal reached = at least one pixel changed in that cell, horizon 10, controller intrinsic reward +1 on reaching; meta-controller gets the level reward only.
- Results (official score %, mean over seeds of per-game mean ± sd over seeds; total best levels over 45 runs; wall-clock s/run):

| Version (tag) | Score % | Levels | Wall s |
|---|---|---|---|
| random meta (`hdqn_randmeta_v1`) | 0.0025 ± 0.0035 | 7 | 14.1 |
| h-DQN baseline (`hdqn_baseline_v1`) | 0.0049 ± 0.0052 | 6 | 19.4 |
| + Imp 1 novelty (`hdqn_nov_v1`) | 0.0051 ± 0.0050 | 7 | 21.2 |
| + Imp 2 GRU (`hdqn_gru_v1`) | 0.0055 ± 0.0049 | 7 | 19.8 |
| + Imp 1 + 2 (`hdqn_nov_gru_v1`) | 0.0059 ± 0.0046 | 8 | 20.8 |

- What it shows: no version is distinguishable from another or from random subgoals at 5000 actions (H0, "the learned meta-controller adds nothing", not rejected; novelty and GRU hypotheses not supported). Six levels come from lp85 and r11l, solved by every version on every seed. Extra levels: sp80 (random meta 1, novelty 1, novelty+GRU 2) and ls20 (GRU 1). 11/15 games: no level in any version. Controller subgoal-success share does not rise within a run (baseline 0.276 → 0.272; random meta 0.263 → 0.274).
- Caveats: versions with the same seed share an identical random exploration prefix (lp85 seeds 1, 2 reach level 1 at 415 and 122 actions in all four learned versions), random meta is not seed-matched with the others; r11l level 1 is faster for all learned seeds (143–479 actions) than for random meta (664–1008) but this rests on 3 seeds in one game.
- Comparison with flat DQN (numbers from entry M2-0 of `docs/EXPERIMENT_LOG.md`, not recomputed from raw runs): flat DQN `dqn_baseline_v1` 0.00646 ± 0.00614, 7 levels, vs h-DQN baseline 0.0049 ± 0.0052, 6 levels, vs random 0.0104 ± 0.011, 9 levels (entry R0). All within seed-to-seed variation, same grid-click action space. With object clicks flat DQN reached 0.3624 ± 0.3406 / 17 levels (M2-2); h-DQN has not been run with object clicks.
- Next step: object-click variants of the same five configs (Improvement 2 of the shared action space; random agent gained 0.0104 → 0.430 from it), flat-DQN comparison from raw runs, held-out games (Member 1).
