# ARC-AGI-3 RL agents — DQN vs PPO

Group project for **Reinforcement Learning (COMS4061A/COMS7071A), Wits 2026**, and our entry to
**ARC Prize 2026 – ARC-AGI-3** and **ARC Prize 2026 – Paper Track** on Kaggle.

> Plan, work split and timeline: **[docs/PLAN.md](docs/PLAN.md)**  ·  How the code fits together: [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md)  ·  Experiment log: [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)  ·  Team workflow: [CONTRIBUTING.md](CONTRIBUTING.md)

## Team
| Role | Name | Student no. |

|---|---|---|---|
| Member 1 |1607717 | Sinethemba Nani|
| Member 2 |1396207 | Cassia Rene Rieckhoff
| Member 3 |456052 | Margaret Keitshokile Molefe

Kaggle team name: `just2gurls`

## Layout
```
src/arcrl/
  env/encoding.py     64x64 grid -> one-hot (16 x 64 x 64), frame stacking, grid hashing
  env/actions.py      action space + shared improvement (object clicks, no-op pruning)
  env/adapter.py      Gymnasium-style adapter over the official env (action mask, reward, resets)
  env/mock.py         tiny fake game for offline tests / CI
  agents/random_agent.py   sanity-check policy
  agents/dqn.py       Double DQN baseline           (Member 2)
  agents/ppo.py       PPO + GAE baseline            (Member 3)
  agents/nets.py      shared CNN encoder (same for both algorithms)
  eval/protocol.py    reproducible evaluation: games x seeds x interaction budget -> JSONL logs
  eval/rhae.py        local copy of the official RHAE formula (analysis only)
scripts/evaluate.py   CLI entry point for every experiment
scripts/inspect_games.py  day-1 inspection of all public games
scripts/make_split.py     frozen dev / held-out split
scripts/analyze.py        tables + learning curves for the report
scripts/bundle_agent.py   single-file agent for Kaggle
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
- **Actions:** `Discrete(6 + 64)` — ACTION1–5, ACTION7, plus ACTION6 clicks at the centres of an 8×8 grid of cells; `info["action_mask"]` = actions the game currently allows.
- **Reward:** +1 per newly completed level (shaping only via separate wrappers; never reported as the ARC score).
- **Episode end:** WIN → terminated; GAME_OVER → auto-RESET (counted as an action); action budget → truncated.

## Kaggle submission
The official starter only uploads `agent/my_agent.py`, so we bundle our whole library into that one file:
```bash
git clone https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter ../ARC-AGI-3-Kaggle-Starter
python scripts/bundle_agent.py --agent dqn --out build/my_agent.py
cp build/my_agent.py ../ARC-AGI-3-Kaggle-Starter/agent/my_agent.py
cd ../ARC-AGI-3-Kaggle-Starter && make play-local && make submit
```
Then **Submit to Competition** on kaggle.com (5 submissions/day). The bundle header records the git commit.

## Status
Torch agents (`dqn.py`, `ppo.py`) are baseline implementations that compile but have **not yet been run on real games** — first job for Members 2 and 3 is `pytest tests/test_torch_agents.py` and a 1-game run.
