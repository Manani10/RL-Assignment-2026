# How we work in this repo

**Branches**
- `main` — always runs; protected (pull request + 1 review).
- `m1/infra-*` (Member 1), `m2/dqn-*` (Member 2), `m3/ppo-*` (Member 3).
- One branch per improvement, e.g. `m2/dqn-imp1-novelty`, so ablations map to code.

**Pull requests**
- `pytest -q` passes (CI runs it automatically).
- New behaviour goes behind a config flag; the baseline must stay reproducible.
- Link the experiment-log entry and results tag in the PR description.

**Experiments**
- Run through `scripts/evaluate.py` only — never a private eval loop.
- Unique `tag` per run; don't commit `results/` (gitignored). Share result folders via the team drive and put the summary table in `docs/EXPERIMENT_LOG.md`.
- Note the commit hash for every reported number.

**Secrets**
- Never commit `.env`, `.kaggle/`, `ARC_API_KEY` or Kaggle tokens.
