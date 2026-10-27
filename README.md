# ARC-AGI-3 RL agents — DQN vs hierarchical DQN (h-DQN)

Group project for **Reinforcement Learning (COMS4061A/COMS7071A), Wits 2026**, and our entry to
**ARC Prize 2026 – ARC-AGI-3** and **ARC Prize 2026 – Paper Track** on Kaggle.

> Plan, work split and timeline: **[docs/PLAN.md](docs/PLAN.md)**  ·  How the code fits together: [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md)  ·  Experiment log: [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)  ·  Team workflow: [CONTRIBUTING.md](CONTRIBUTING.md)

## Team
| Role | Name |
|---|---|
| Member 1 | Sinethemba Nani |
| Member 2 | Cassia Rene Rieckhoff |
| Member 3 | Margaret Keitshokile Molefe |

Kaggle team name: **`just2gurls`** (ARC Prize 2026 – ARC-AGI-3 and ARC Prize 2026 – Paper Track)  
Moodle submission file: `just2gurls.zip`

## Layout
```
src/arcrl/
  env/encoding.py     64x64 grid -> one-hot (16 x 64 x 64), frame stacking, grid hashing
  env/actions.py      action space + shared improvement (object clicks, no-op pruning)
  env/adapter.py      Gymnasium-style adapter over the official env (action mask, reward, resets)
  env/mock.py         tiny fake game for offline tests / CI
  agents/random_agent.py   sanity-check policy
  agents/dqn.py       Double DQN baseline + count-based novelty bonus   (Member 2)
  agents/hdqn.py      hierarchical DQN: meta-controller + controller    (Member 3)
  agents/ppo.py       PPO starter code (not used; replaced by h-DQN)
  agents/nets.py      shared CNN encoder (same for both algorithms)
  eval/protocol.py    reproducible evaluation: games x seeds x interaction budget -> JSONL logs
  eval/rhae.py        local copy of the official RHAE formula (analysis only)
scripts/evaluate.py   CLI entry point for every experiment
scripts/inspect_games.py  day-1 inspection of all public games
scripts/make_split.py     frozen dev / held-out split
scripts/analyze.py        tables + learning curves for the report
scripts/bundle_agent.py   single-file agent for Kaggle
scripts/build_kaggle_notebook.py  builds notebooks/submission.ipynb for Kaggle
kaggle/my_agent.py    bridge into the official ARC-AGI-3 Kaggle starter
configs/              matched baseline configs + dev/held-out split
report/               LaTeX (RLC 2026 template)
tests/                pytest; runs offline on the mock game
```

## Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                   # should pass on the mock game

# real games (downloads game files on first run, then works offline)
export ARC_API_KEY=...                      # optional, from three.arcprize.org — never commit it
python -c "import arc_agi; print(arc_agi.Arcade().get_environments())"
python scripts/evaluate.py --agent random --games ls20 --seeds 0 1 2 --budget 2000 --tag random_ls20
```

## Adapter semantics (summary — full text in `src/arcrl/env/adapter.py`)
- **Observation:** one-hot of the last *k* frames, shape `(16k, 64, 64)`.
- **Actions:** `Discrete(6 + 64)` — ACTION1–5, ACTION7, plus 64 ACTION6 click slots: centres of an 8×8 grid (baseline) or detected objects (`"click_mode": "objects"`, Improvement 2); `info["action_mask"]` = actions the game currently allows.
- **Reward:** +1 per newly completed level (exploration bonuses and h-DQN intrinsic reward are used for learning only, never reported as the ARC score).
- **Episode end:** WIN → terminated; GAME_OVER → auto-RESET (counted as an action); action budget → truncated.

## Kaggle submission
The official starter only uploads `agent/my_agent.py`, so we bundle our whole library into that one file:
```bash
python scripts/bundle_agent.py --agent dqn --click-mode objects --out build/my_agent.py
python scripts/build_kaggle_notebook.py      # writes notebooks/submission.ipynb
```
Upload `notebooks/submission.ipynb` to Kaggle (competition data attached, Internet off), **Save & Run All**, then **Submit to Competition** (5 submissions/day). The bundle header records the git commit.

## Status
- Game split frozen: 15 dev / 10 held-out games (`configs/splits.json`).
- Random reference done: 0.010% official score; with object clicks 0.43% (see `docs/EXPERIMENT_LOG.md`, R0, M1-1, M1-2).
- DQN  and h-DQN: baselines and improvements in progress.
