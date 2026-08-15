"""Seed a replacement resume root for the two-run control, including the data
source state file this time (position-exact resume)."""
import json
import shutil
from pathlib import Path

import modal

app = modal.App("seed-resume-root2")
vol = modal.Volume.from_name("chess-rl-miles-checkpoints", create_if_missing=False)
ROOT = Path("/rl/chess-rl-miles-interleave")
SRC = "e3p2band-lr1e4-rl1500"
DST = "e3p2band-lr1e4-rl3000r2"


@app.function(volumes={"/rl": vol}, timeout=3600, cpu=8.0, memory=32 * 1024)
def seed() -> str:
    vol.reload()
    src = ROOT / SRC / "iter_0001500"
    for marker in ("model/.metadata", "rng.pt", "meta.json"):
        if not (src / marker).exists():
            raise RuntimeError(f"{src} missing {marker}")
    state = ROOT / SRC / "rollout" / "global_dataset_state_dict_1499.pt"
    if not state.exists():
        raise RuntimeError(f"missing data state {state}")
    dst_root = ROOT / DST
    if (dst_root / "latest_checkpointed_iteration.txt").exists():
        return json.dumps({DST: "already seeded"})
    if dst_root.exists():
        raise RuntimeError(f"{dst_root} exists but is not a seeded root")
    dst_root.mkdir(parents=True)
    shutil.copytree(src, dst_root / "iter_0001500")
    (dst_root / "rollout").mkdir()
    shutil.copy2(state, dst_root / "rollout" / "global_dataset_state_dict_1499.pt")
    (dst_root / "latest_checkpointed_iteration.txt").write_text("1500")
    n = sum(1 for _ in dst_root.rglob("*") if _.is_file())
    vol.commit()
    return json.dumps({DST: f"seeded ({n} files, incl. data state)"})


@app.local_entrypoint()
def main() -> None:
    print("RESULT::" + seed.remote())
