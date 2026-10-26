"""Build a readable, ready-to-upload Kaggle submission notebook.

    python scripts/build_kaggle_notebook.py --agent dqn --click-mode objects
    -> notebooks/submission_dqn.ipynb

The notebook is the "source code" the course marks, so it shows our library as
ordinary source files: one `%%writefile` cell per module of `src/arcrl` that
the agent needs, then the bridge `kaggle/my_agent.py`, a self-test, and the
official run cell. Nothing is minified or embedded as strings.

Kaggle steps: competition page -> Code -> New Notebook -> File -> Import
Notebook (this .ipynb), Internet OFF, Run All, Save Version (Save & Run All),
then "Submit to competition".

Cell layout mirrors the official ARC-AGI-3 Kaggle starter: install the offline
wheels; write the agent; during the competition rerun only, copy the attached
ARC-AGI-3-Agents framework, register our agent and play every game against the
gateway (which writes submission.parquet). On a normal "Save & Run All" a
placeholder submission.parquet is written so the version commits.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "arcrl"
COMP = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3"
LIB = "/tmp/arcrl_src"  # where the notebook writes our package

# Modules the Kaggle agent needs. Evaluation code (eval/, env/adapter.py, env/mock.py)
# and the unused PPO starter are left out: they never run on Kaggle.
SKIP = {"agents/ppo.py", "env/adapter.py", "env/mock.py"}
SKIP_DIRS = {"eval"}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def library_files() -> list[Path]:
    files = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel in SKIP or rel.split("/")[0] in SKIP_DIRS:
            continue
        files.append(path)
    return files


def bridge_source(agent: str, click_mode: str, prune_noops: bool, max_actions: int) -> str:
    src = (ROOT / "kaggle" / "my_agent.py").read_text()
    required = ['os.environ.get("ARCRL_CLICK_MODE"', "MAX_ACTIONS = MAX_ACTIONS", "from arcrl.agents import build_agent"]
    missing = [r for r in required if r not in src]
    if missing:
        raise SystemExit(f"kaggle/my_agent.py is an old version (missing {missing}). "
                         "Replace it with the latest bridge before building.")
    # bake this submission's settings in as the defaults
    for var, val in [("ARCRL_AGENT", agent), ("ARCRL_CLICK_MODE", click_mode),
                     ("ARCRL_PRUNE_NOOPS", str(int(prune_noops))), ("ARCRL_MAX_ACTIONS", str(max_actions))]:
        start = src.index(f'os.environ.get("{var}", "') + len(f'os.environ.get("{var}", "')
        end = src.index('"', start)
        src = src[:start] + val + src[end:]
    # make our package importable from where the notebook wrote it
    src = src.replace("from arcrl.agents import build_agent",
                      f'import sys\nsys.path.insert(0, "{LIB}")  # our library, written by the cells above\n'
                      "from arcrl.agents import build_agent", 1)
    return src


def cell(kind: str, src: str) -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def build(agent: str, click_mode: str, prune_noops: bool, max_actions: int) -> dict:
    commit = git_commit()
    files = library_files()
    dirs = sorted({str(Path(LIB, "arcrl", p.relative_to(SRC).parent).as_posix()) for p in files})

    intro = f"""# ARC Prize 2026 — ARC-AGI-3 submission (team just2gurls)

**Agent:** `{agent}` · **click targets:** `{click_mode}` · **no-op pruning:** `{prune_noops}` · **max actions per game:** {max_actions}
**Code:** [github.com/Manani10/RL-Assignment-2026](https://github.com/Manani10/RL-Assignment-2026), commit `{commit}`; this notebook is generated from it by `scripts/build_kaggle_notebook.py`.

The agent learns **online, inside each game**: no pretrained model, no internet, no game-specific rules.
At every step it sees the 64×64 frame, chooses one of 70 actions (6 simple actions + 64 click targets;
with `objects`, each click target is the centre of a detected single-colour object), and receives
+1 when a level is completed.

| Cell | What it does |
|---|---|
| 1 | Install the ARC-AGI-3 toolkit from the competition's offline wheels |
| 2 | Write our library `arcrl` to `{LIB}` (one cell per source file, unchanged from the repository) |
| 3 | Write `my_agent.py`, the bridge between the official framework and our agent |
| 4 | Self-test on a synthetic frame (normal runs only) |
| 4b | Placeholder `submission.parquet` (must exist in every run, including scoring) |
| 5 | Competition rerun only: register the agent and play every hidden game (log: `framework.log`) |
"""
    install = f"""# 1) Install the competition's offline wheels (internet is disabled during scoring)
!pip install --no-index --find-links {COMP}/arc_agi_3_wheels arc-agi python-dotenv -q
"""
    mkdirs = ("# 2) Our library `arcrl`: create the package folders, then one cell per module below\n"
              "import os\n"
              f"for d in {dirs!r}:\n"
              "    os.makedirs(d, exist_ok=True)\n")
    lib_cells = []
    for p in files:
        rel = p.relative_to(SRC).as_posix()
        body = p.read_text() or '"""(package marker)"""\n'
        lib_cells.append(cell("code", f"%%writefile {LIB}/arcrl/{rel}\n" + body))

    bridge = "%%writefile /tmp/my_agent.py\n" + bridge_source(agent, click_mode, prune_noops, max_actions)

    selftest = f'''# 4) Self-test (normal runs only): the library imports, the action space finds targets,
#    and the agent returns a legal action. Catches packaging errors before Kaggle scoring does.
import os, sys
import numpy as np

if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    sys.path.insert(0, "{LIB}")
    from arcrl.agents import build_agent
    from arcrl.env.actions import ActionConfig, ActionSpace
    from arcrl.env.encoding import FrameStack

    grid = np.zeros((64, 64), dtype=np.int64)
    grid[10:14, 20:24] = 3            # two small "objects" on a background
    grid[40:42, 50:56] = 7
    space = ActionSpace(ActionConfig(click_mode="{click_mode}", prune_noops={prune_noops}))
    space.update(grid)
    mask = space.mask({{"ACTION1", "ACTION2", "ACTION6"}})
    agent = build_agent("{agent}", space.n, 16, seed=0)
    agent.begin_game("self-test")
    a = agent.act(FrameStack(1).reset(grid), {{"action_mask": mask}})
    assert mask[a], "agent chose a masked-out action"
    print(f"self-test OK: agent={{agent.__class__.__name__}}, {{int(mask.sum())}} legal actions, chose {{space.decode(a)}}")
'''
    run = f'''# 5) Competition rerun only: plug our agent into the official framework and play every game.
#    The scoring gateway records every action; submission.parquet (cell 4b) must exist when the notebook ends.
import os, shutil, subprocess, pathlib

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    subprocess.run(["curl", "-s", "--retry", "60", "--retry-delay", "5", "--retry-connrefused",
                    "http://gateway:8001/api/games"], check=False)
    src = "{COMP}/ARC-AGI-3-Agents"
    dst = "/kaggle/working/ARC-AGI-3-Agents"
    shutil.copytree(src, dst, dirs_exist_ok=True)
    shutil.copy("/tmp/my_agent.py", f"{{dst}}/agents/templates/my_agent.py")
    pathlib.Path(f"{{dst}}/agents/__init__.py").write_text(
        "from typing import Type\\n"
        "from dotenv import load_dotenv\\n"
        "from .agent import Agent, Playback\\n"
        "from .swarm import Swarm\\n"
        "from .templates.random_agent import Random\\n"
        "from .templates.my_agent import MyAgent\\n"
        "load_dotenv()\\n"
        "AVAILABLE_AGENTS: dict[str, Type[Agent]] = {{'random': Random, 'myagent': MyAgent}}\\n"
        "__all__ = ['Agent', 'Playback', 'Swarm', 'Random', 'MyAgent', 'AVAILABLE_AGENTS']\\n"
    )
    pathlib.Path(f"{{dst}}/.env").write_text(
        "HOST=gateway\\nPORT=8001\\nOPERATION_MODE=online\\n"
        "RECORDINGS_DIR=/kaggle/working/server_recording\\n"
    )
    env = {{**os.environ, "MPLBACKEND": "agg"}}
    with open("/kaggle/working/framework.log", "w") as log:   # kept as notebook output for debugging
        r = subprocess.run(["python", "main.py", "--agent", "myagent"], cwd=dst, env=env,
                           stdout=log, stderr=subprocess.STDOUT)
    print("framework exit code:", r.returncode)  # never fail the notebook here: the gateway has the scores
'''
    dummy = '''# 4b) Placeholder submission.parquet, written BEFORE the agent plays and in every mode
#     (the official starter does the same: without this file the competition rerun ends in "Kaggle Error").
#     During scoring the gateway records the real results; this file only has to exist.
import pandas as pd
pd.DataFrame({"row_id": pd.Series(dtype=str), "game_id": pd.Series(dtype=str),
              "end_of_game": pd.Series(dtype=bool), "score": pd.Series(dtype=float)}) \\
    .to_parquet("/kaggle/working/submission.parquet", index=False)
print("Placeholder submission.parquet written (zero rows).")
'''
    cells = [cell("markdown", intro), cell("code", install), cell("code", mkdirs), *lib_cells,
             cell("markdown", "## 3) Bridge: the official framework calls `choose_action`; we translate it to our agent's `act` / `observe`"),
             cell("code", bridge), cell("code", selftest), cell("code", dummy), cell("code", run)]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "none", "isInternetEnabled": False,
                       "language": "python", "sourceType": "notebook", "isGpuEnabled": False},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="random")
    p.add_argument("--click-mode", default="grid", choices=["grid", "objects"])
    p.add_argument("--prune-noops", action="store_true")
    p.add_argument("--max-actions", type=int, default=1000)
    p.add_argument("--out", default="notebooks/submission.ipynb")
    a = p.parse_args()
    nb = build(a.agent, a.click_mode, a.prune_noops, a.max_actions)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1))
    n_files = sum(1 for c in nb["cells"] if "".join(c["source"]).startswith(f"%%writefile {LIB}"))
    print(f"wrote {out} (agent={a.agent}, click_mode={a.click_mode}, max_actions={a.max_actions}, "
          f"{n_files} library files, commit={git_commit()})")


if __name__ == "__main__":
    main()
