"""Make held-out copies of the final agents' configs (same config, held-out games, new tag).
Run each held-out config ONCE, for the final report only.
Usage: python scripts/make_heldout_configs.py [config names...]
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ["random_baseline", "random_objonly", "dqn_imp2only", "dqn_full"]

names = sys.argv[1:] or DEFAULT
held_out = json.loads((ROOT / "configs" / "splits.json").read_text())["held_out"]
assert len(held_out) == 10, f"expected 10 held-out games, found {len(held_out)}"
for name in names:
    src = ROOT / "configs" / f"{name}.json"
    if not src.exists():
        print(f"skip {name}: not found"); continue
    cfg = json.loads(src.read_text())
    cfg["games"] = held_out
    cfg["tag"] = f"{name}_heldout_v1"
    out = ROOT / "configs" / f"{name}_heldout.json"
    out.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"wrote {out.name}  tag={cfg['tag']}")
