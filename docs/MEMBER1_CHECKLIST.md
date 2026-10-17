# Member 1 — step-by-step guide (Windows / VS Code)

Run every command in the **VS Code terminal** (Terminal → New Terminal), inside the
`RL-Assignment-2026` folder. Tick each box when its "Done when" is true.

**Member 1 owns:** setup, Kaggle, the shared action-space improvement, all ablation
and held-out runs, the DQN vs PPO comparison, and putting the report together.

**Deadline:** Moodle, Tue 27 Oct, 23:59.

---

## Part A — Today: GitHub and team (Thu 8 Oct)

### A1. Get the latest code into GitHub
1. Unzip `member1-update.zip`.
2. Copy everything **inside** the unzipped `arc-rl` folder into `RL-Assignment-2026`, and choose **Replace** when Windows asks.
3. Save it to GitHub:
```powershell
git add -A
git commit -m "Member 1 tools, shared action space, guides"
git push
```
- [ ] **Done when** `docs/MEMBER1_CHECKLIST.md` and `src/arcrl/env/actions.py` are visible on github.com.

### A2. Add teammates
- [ ] GitHub repo → **Settings → Collaborators → Add people** → Members 2 and 3.

### A3. Kaggle team (both competitions)
- [ ] Open "ARC Prize 2026 – ARC-AGI-3" on Kaggle → **Join Competition** → accept rules → **Team** tab → create the team and invite Members 2 and 3.
- [ ] Do the same for "ARC Prize 2026 – Paper Track", with the **same team name**.
- [ ] Write the team name in `README.md`, with all names and student numbers.

### A4. Tell the team
- [ ] Send Members 2 and 3 the link to `docs/PLAN.md`. Tell them:
  - Improvement 2 for both of them is your shared action space.
  - Their results are due to you on Thu 22 Oct.

---

## Part B — Set up Python (Fri 9 Oct)

### B1. Install Python 3.12
The ARC package needs exactly Python 3.12.
```powershell
py -3.12 --version
```
If that fails, install it from python.org (tick "Add to PATH"), or run `winget install Python.Python.3.12`. Then reopen VS Code.

### B2. Create the environment and install packages
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If `Activate.ps1` is blocked, run this once and try again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
You'll see `(.venv)` at the start of the prompt when it's active. **Activate it every time you open a new terminal.**

### B3. Run the tests
```powershell
pytest -q
```
- [ ] **Done when** all tests pass (about 18).

---

## Part C — Look at the real games (Fri 9 – Sat 10 Oct)

### C1. Inspect every public game
```powershell
python scripts/inspect_games.py --probe 500
```
The first run downloads the games, so it needs internet.
- [ ] Open `results\inspection\games.md` and note:
  - how many games there are;
  - levels per game;
  - which actions each game uses;
  - how often clicks change the screen.

### C2. Run one game and check the wrapper
```powershell
python scripts/evaluate.py --agent random --games ls20 --seeds 0 --budget 500 --tag check_ls20 -v
```
- [ ] It runs without errors.
- [ ] Open `results\check_ls20\runs.jsonl`. Is `official_score` filled in?
- [ ] Does RESET count as an action? Compare `"scorecard"` (the official action count) with `"actions"` (ours). If they differ, tell me and we'll fix the counting.
- [ ] Write the answers in `docs/EXPERIMENT_LOG.md`, as entry **R-1: wrapper check**.

### C3. Freeze the game split
```powershell
python scripts/make_split.py --from-inspection results/inspection/games.json --held-out-frac 0.4
```
- [ ] Open `configs/splits.json` and copy the `"dev"` list into these 4 files, replacing `"REPLACE_WITH_DEV_GAME_IDS"`:
  - `configs/random_baseline.json`
  - `configs/dqn_baseline.json`
  - `configs/ppo_baseline.json`
  - `configs/random_objects.json`
- [ ] Commit and push:
```powershell
git add -A
git commit -m "Freeze dev/held-out split"
git push
```
**Never run `make_split.py` again** after this.

---

## Part D — Kaggle submission setup (Sat 10 – Sun 11 Oct)

The Kaggle starter uses `make`, which Windows doesn't have, so use **WSL (Ubuntu)** for this part only.

### D1. Install WSL (once)
In PowerShell **as Administrator**:
```powershell
wsl --install
```
Restart the PC, open **Ubuntu** from the Start menu, and create a username and password.

### D2. Set up the starter inside Ubuntu
```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv make git
cd ~
git clone https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter
```
Ubuntu can see your Windows project folder directly at `/mnt/c/Users/nanis/RL-Assignment-2026`, so you don't need a second copy of our repo.

### D3. Kaggle token
1. On kaggle.com, go to **Settings → API → Create New Token** and copy the token.
2. In Ubuntu:
```bash
cd ~/ARC-AGI-3-Kaggle-Starter
mkdir -p .kaggle && echo "PASTE_TOKEN_HERE" > .kaggle/access_token
chmod 600 .kaggle/access_token
nano notebooks/kernel-metadata.json
```
In `nano`, replace `REPLACE_WITH_YOUR_USERNAME` with your Kaggle username. Save with Ctrl+O, Enter, then exit with Ctrl+X.

### D4. Build our agent and test it locally
```bash
cd /mnt/c/Users/nanis/RL-Assignment-2026
python3.12 scripts/bundle_agent.py --agent random --out build/my_agent.py
cp build/my_agent.py ~/ARC-AGI-3-Kaggle-Starter/agent/my_agent.py
cd ~/ARC-AGI-3-Kaggle-Starter
make setup
make play-local
```
- [ ] **Done when** `make play-local` plays every game without errors. If you see an import error, paste it to me.
- [ ] Check the Kaggle competition's **Overview / Code Requirements** page: what is the runtime limit? Note it in the log.

---

## Part E — Freeze the rules and first submission (Mon 12 Oct)

### E1. Random baseline (sanity check)
Back in VS Code:
```powershell
python scripts/evaluate.py --config configs/random_baseline.json
python scripts/analyze.py --tags random_baseline_v1 --name random
```
- [ ] Results go in `docs/EXPERIMENT_LOG.md`, entry **R0**.
- [ ] Tell the team the frozen rules:
  - same dev games;
  - seeds 0, 1, 2;
  - 5000 actions per run, 2000 per episode;
  - always use `scripts/evaluate.py`.

### E2. First Kaggle submission
In Ubuntu:
```bash
cd ~/ARC-AGI-3-Kaggle-Starter
make submit
make status
```
1. When `make status` shows `complete`, open the notebook on kaggle.com.
2. Click **Submit to Competition**, then choose `submission.parquet`.

- [ ] Screenshot the submission ID, score and time, and save it in `docs/kaggle_evidence/`.

---

## Part F — Shared action-space improvement (Tue 13 – Mon 19 Oct)

**The code is already written** in `src/arcrl/env/actions.py`, switched on with two config settings:
```json
"adapter": { ..., "click_mode": "objects", "prune_noops": true }
```
- `click_mode: "objects"` replaces the 8×8 click grid with one click per detected object on the screen.
- `prune_noops: true` hides actions that already did nothing in the same screen state.

Your job is to **understand it, test it on real games, and explain it**.

- [ ] Read `src/arcrl/env/actions.py` until you can explain every function. You'll present it as your work, so this matters (declare the AI help in your contribution statement).
- [ ] Run the tests: `pytest tests/test_actions.py -q`.
- [ ] Compare the random agent with and without it on the dev games:
```powershell
python scripts/evaluate.py --config configs/random_objects.json
python scripts/analyze.py --tags random_baseline_v1 random_objects_v1 --name actionspace_check
```
- [ ] Look at `results\analysis\actionspace_check\summary.md`. Does it complete levels sooner? Write the result in the log as entry **M1-1**.
- [ ] Tell Members 2 and 3 how to switch it on: add `"click_mode": "objects", "prune_noops": true` to their config's `adapter`, with a new `tag`.
- [ ] Write a short method paragraph for the report: what it does, why (huge click space, no reward signal), and what persists (no-op memory lasts one game).

**Same week, writing:** fill the `\todo{}`s you can already answer in:
- [ ] `report/sections/01_introduction.tex`
- [ ] `report/sections/02_problem.tex` (use your C1/C2 findings)
- [ ] `report/sections/04_setup.tex`

### Overleaf (any time this week)
- [ ] Download the **RLC 2026** LaTeX template, upload it to Overleaf, and turn off the cover page.
- [ ] Upload `report/sections/*.tex` and `report/references.bib`.
- [ ] Add `\usepackage{xcolor}` and `\newcommand{\todo}[1]{\textcolor{red}{[TODO: #1]}}` to the preamble.
- [ ] Share the project with Members 2 and 3.

---

## Part G — Ablations and held-out runs (Tue 20 – Thu 22 Oct)

- [ ] Get from Members 2 and 3 their final configs and result tags.
- [ ] For each algorithm, make 4 configs: same games, seeds and budget; only the improvement settings and the `tag` change. Run each one:

| Variant | DQN tag | PPO tag |
|---|---|---|
| baseline | `dqn_baseline_v1` | `ppo_baseline_v1` |
| + imp1 only (exploration) | `dqn_imp1_v1` | `ppo_imp1_v1` |
| + imp2 only (your action space) | `dqn_imp2only_v1` | `ppo_imp2only_v1` |
| + both (full) | `dqn_full_v1` | `ppo_full_v1` |

```powershell
python scripts/evaluate.py --config configs/dqn_imp2only.json
```
Repeat for each config.

- [ ] **Held-out games, run once only:** copy the configs for random, both baselines and both full agents. Set `"games"` to the `held_out` list, add `"record": true`, and use new tags ending in `_heldout`.
- [ ] Best agent to Kaggle (in Ubuntu):
```bash
cd /mnt/c/Users/nanis/RL-Assignment-2026
python3.12 scripts/bundle_agent.py --agent dqn --click-mode objects --prune-noops --out build/my_agent.py
cp build/my_agent.py ~/ARC-AGI-3-Kaggle-Starter/agent/my_agent.py
cd ~/ARC-AGI-3-Kaggle-Starter && make play-local && make submit
```
Use `--agent ppo` if PPO is better.

---

## Part H — DQN vs PPO comparison and report (Fri 23 – Sun 25 Oct)

```powershell
python scripts/analyze.py --tags random_baseline_v1 dqn_baseline_v1 dqn_full_v1 ppo_baseline_v1 ppo_full_v1 --name final
python scripts/analyze.py --tags dqn_baseline_v1 dqn_imp1_v1 dqn_imp2only_v1 dqn_full_v1 --name ablation_dqn
python scripts/analyze.py --tags ppo_baseline_v1 ppo_imp1_v1 ppo_imp2only_v1 ppo_full_v1 --name ablation_ppo
python scripts/analyze.py --tags random_heldout dqn_full_heldout ppo_full_heldout --name heldout
```
Write `report/sections/07_comparison.tex` using these outputs:
- [ ] Final table (copy numbers from `results\analysis\final\summary.md`).
- [ ] Learning-curve figure: upload `results\analysis\final\curves.pdf` to Overleaf.
- [ ] Ablations: what each improvement adds, for DQN and for PPO separately.
- [ ] Efficiency and compute: actions to first level, time per run, GPU.
- [ ] Generalisation: dev vs held-out.
- [ ] One success and one failure replay (in the `recordings\` folder) and what each shows.
- [ ] Limitations.
- [ ] Put Members 2 and 3's sections into Overleaf, check the report is ≤ 8 pages before references, and write the abstract and conclusion.
- [ ] Pick the **nominated** Kaggle submission (made before 27 Oct) and save its ID, score and time.

---

## Part I — Finish (Mon 26 – Tue 27 Oct)
- [ ] Paper Track: a shortened report under 1500 words, submitted on Kaggle and linked to the code submission. Save a copy as `docs/paper_track.md`.
- [ ] Contribution statements from all three members, including generative-AI use.
- [ ] Check no secrets are in the code: `git grep -i -e api_key -e KGAT` should show nothing.
- [ ] Download the nominated submission's notebook from Kaggle.
- [ ] Make `TEAMNAME.zip` (named after your Kaggle team) containing:
  - the report PDF;
  - the notebook;
  - the Kaggle evidence: submission ID, score and time screenshots, the nominated entry, and the Paper Track link plus its saved text;
  - the contribution statements.
- [ ] Upload to **Moodle before 23:59 on Tue 27 Oct**.
