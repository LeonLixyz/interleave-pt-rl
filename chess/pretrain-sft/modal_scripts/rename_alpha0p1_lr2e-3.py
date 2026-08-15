"""
One-shot helper: rename 6p5e18/{200m,410m,680m}_C_6p5e18_alpha0.100 dirs
to *_lr2e-3 so we can relaunch at the new convention (lr=1e-3, eta_min=1e-4)
under the original names.

Usage:
  modal run modal_scripts/rename_alpha0p1_lr2e-3.py
"""
import shutil
from pathlib import Path

import modal

ckpt_volume = modal.Volume.from_name("rl-reasoning-checkpoints")

app = modal.App("rename-helper")


@app.function(volumes={"/checkpoints": ckpt_volume}, timeout=600)
def rename():
    targets = [
        "200m_C_6p5e18_alpha0.100",
        "410m_C_6p5e18_alpha0.100",
        "680m_C_6p5e18_alpha0.100",
    ]
    base = Path("/checkpoints/6p5e18")
    for name in targets:
        src = base / name
        dst = base / f"{name}_lr2e-3"
        if not src.is_dir():
            print(f"  SKIP (no src): {src}")
            continue
        if dst.exists():
            print(f"  SKIP (dst exists): {dst}")
            continue
        print(f"  RENAME: {src} -> {dst}")
        shutil.move(str(src), str(dst))
    ckpt_volume.commit()
    print("done")


@app.local_entrypoint()
def main():
    rename.remote()
