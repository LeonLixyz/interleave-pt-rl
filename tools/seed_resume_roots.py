"""Seed new run roots with the 1,500-step checkpoints of the two baseline RL
runs so the launcher's auto-resume continues them to 3,000 updates."""
import json
import shutil
from pathlib import Path

import modal

app = modal.App("seed-resume-roots")
vol = modal.Volume.from_name("chess-rl-miles-checkpoints", create_if_missing=False)
ROOT = Path("/rl/chess-rl-miles-interleave")

PAIRS = [
    ("e2w1band-lr1e4-rl1500", "e2w1band-lr1e4-rl3000r"),
    ("e3p2band-lr1e4-rl1500", "e3p2band-lr1e4-rl3000r"),
]


@app.function(volumes={"/rl": vol}, timeout=3600, cpu=8.0, memory=32 * 1024)
def seed() -> str:
    vol.reload()
    out = {}
    for src_run, dst_run in PAIRS:
        src = ROOT / src_run / "iter_0001500"
        for marker in ("model/.metadata", "rng.pt", "meta.json"):
            if not (src / marker).exists():
                raise RuntimeError(f"{src} missing {marker}")
        dst_root = ROOT / dst_run
        if (dst_root / "latest_checkpointed_iteration.txt").exists():
            out[dst_run] = "already seeded"
            continue
        if dst_root.exists():
            raise RuntimeError(f"{dst_root} exists but is not a seeded root")
        dst_root.mkdir(parents=True)
        shutil.copytree(src, dst_root / "iter_0001500")
        (dst_root / "latest_checkpointed_iteration.txt").write_text("1500")
        n_files = sum(1 for _ in (dst_root / "iter_0001500").rglob("*") if _.is_file())
        out[dst_run] = f"seeded ({n_files} files)"
    vol.commit()
    return json.dumps(out)


@app.local_entrypoint()
def main() -> None:
    print("RESULT::" + seed.remote())
