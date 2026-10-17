# How the code is organised

```
arc-rl/
├── src/arcrl/        the library: environment, agents, evaluation
├── scripts/          command-line tools you actually run
├── configs/          experiment settings (JSON) + the frozen game split
├── results/          outputs of runs (NOT committed to git)
├── kaggle/           bridge from our agents to the Kaggle competition
├── report/           LaTeX sections + bibliography
├── tests/            offline tests on a fake game
└── docs/             plan, checklists, experiment log, this guide
```

Data flows one way:

```
configs/*.json ──> scripts/evaluate.py ──> src/arcrl (env + agent + protocol) ──> results/<tag>/
                                                                                     │
                       report figures/tables <── scripts/analyze.py <────────────────┘
src/arcrl + kaggle/my_agent.py ──> scripts/bundle_agent.py ──> build/my_agent.py ──> Kaggle
```

## `src/arcrl/` — the library

| File | What it does | Owner |
|---|---|---|
| `env/encoding.py` | Turns a 64×64 grid of colours 0–15 into a one-hot tensor (16×64×64); stacks the last *k* frames; hashes frames (for novelty counts / no-op pruning). | M1 |
| `env/actions.py` | The action space: 6 simple actions + 64 click slots, the action mask, and the **shared improvement** (`click_mode="objects"` clicks on detected objects; `prune_noops` hides actions that did nothing). Used by both the adapter and the Kaggle bridge. | M1 |
| `env/adapter.py` | Wraps the official ARC-AGI-3 game in a Gym-style `reset()` / `step()`. Maps an action number (0–69) to a real game action (ACTION1–5, 7, or a click), builds the **action mask** of allowed actions, gives reward +1 per completed level, auto-RESETs after GAME_OVER, stops at the action budget. `make_env(game_id)` builds it; `official_score(env)` reads the official scorecard. The shared action-space improvement goes here. | M1 |
| `env/mock.py` | A tiny fake game (move a pixel to a goal) so code can be tested without the real games. | M1 |
| `agents/base.py` | The interface every agent follows: `begin_game`, `act(obs, info) → action`, `observe(...)` (learning happens here). | all |
| `agents/random_agent.py` | Sanity-check policy: random allowed action. | M1 |
| `agents/nets.py` | The CNN encoder **shared** by DQN and PPO (so the comparison is fair). | shared |
| `agents/dqn.py` | Double DQN: replay buffer, target network, ε-greedy over allowed actions. | M2 |
| `agents/ppo.py` | PPO: rollouts, GAE, clipped objective, masked policy. | M3 |
| `agents/__init__.py` | `build_agent(name, ...)` — the registry. **Every new variant gets a name here** (e.g. `dqn_imp1`). | all |
| `eval/protocol.py` | The evaluation loop: for each game × seed, play with a fixed action budget while the agent learns online; log every episode; record the official score. | M1 |
| `eval/rhae.py` | Local copy of the official scoring formula, for analysis only. | M1 |

## `scripts/` — what you run

| Command | Purpose |
|---|---|
| `python scripts/inspect_games.py` | Probe every public game → `results/inspection/` |
| `python scripts/make_split.py --from-inspection results/inspection/games.json` | Create the frozen dev / held-out split → `configs/splits.json` |
| `python scripts/evaluate.py --config configs/dqn_baseline.json` | Run an experiment → `results/<tag>/` |
| `python scripts/analyze.py --tags tagA tagB --name baselines` | Tables + learning curves → `results/analysis/baselines/` |
| `python scripts/bundle_agent.py --agent dqn` | Build the single-file Kaggle agent → `build/my_agent.py` |

## `configs/` — experiment settings

One JSON file per experiment. Changing a run means changing (or copying) a config, never editing code.

```json
{
  "agent": "dqn",                       // name in agents/__init__.py
  "games": ["ls20", "..."],             // dev games from splits.json
  "seeds": [0, 1, 2],                   // ≥ 3 seeds
  "total_budget": 5000,                 // actions per (game, seed) run, incl. resets
  "adapter": {
    "frame_stack": 1,                   // how many past frames the agent sees
    "click_cell": 8,                    // 8×8-pixel click grid → 64 click targets
    "action_budget": 2000,              // max actions per episode
    "on_game_over": "reset"             // auto-RESET or end the episode
  },
  "tag": "dqn_baseline_v1",             // unique name → results/dqn_baseline_v1/
  "record": false                       // true = save replays for failure-case figures
}
```

Rules: same `games`, `seeds`, `total_budget` and `adapter` for DQN and PPO; a new `tag` for every run (the script refuses to overwrite). `splits.json` is created once and then frozen.

## `results/` — outputs (gitignored)

```
results/
├── inspection/            games.json, games.md       (inspect_games.py)
├── <tag>/                 one folder per experiment  (evaluate.py)
│   ├── meta.json          config + Python / arc-agi / torch / GPU versions
│   ├── runs.jsonl         one line per (game, seed): best levels, wins, actions,
│   │                      official score, wall time, learning curve
│   └── episodes.jsonl     one line per episode, failures included
└── analysis/<name>/       summary.md / .csv, overall.md, curves.pdf / .png  (analyze.py)
```

Results are too big and too many for git — share result folders via a shared drive, and copy the summary tables into `docs/EXPERIMENT_LOG.md` with the commit hash.

## `kaggle/` — competition submission

`kaggle/my_agent.py` is a **bridge**: the Kaggle framework calls `choose_action(frames, latest_frame)` once per step; the bridge converts the frame with our encoding, builds the action mask, passes level rewards to our agent's `observe`, asks `act` for an action and converts it back to a `GameAction`. So the same agent code runs locally and on Kaggle.

The Kaggle starter only uploads one file, so never copy the bridge directly:

```bash
python scripts/bundle_agent.py --agent dqn --out build/my_agent.py   # packs src/arcrl inside
cp build/my_agent.py ../ARC-AGI-3-Kaggle-Starter/agent/my_agent.py
cd ../ARC-AGI-3-Kaggle-Starter && make play-local && make submit
```

The bundle's first line records the agent name and git commit, which is your evidence that the submission matches the code.

## `tests/`
`pytest -q` runs everything offline on the mock game (DQN/PPO tests are skipped if torch isn't installed). GitHub runs them on every push.
