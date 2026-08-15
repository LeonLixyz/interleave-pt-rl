"""
One-shot helper: delete contaminated 6p5e18/{200m,410m,680m}_C_6p5e18_alpha0.100
dirs (lr=2e-3 resume artifacts that should not exist; clean *_lr2e-3 copies are
preserved separately).

Usage:
  modal run modal_scripts/delete_alpha0p1_contaminated.py
"""
import shutil
from pathlib import Path

import modal

ckpt_volume = modal.Volume.from_name("rl-reasoning-checkpoints")

app = modal.App("delete-helper")


@app.function(volumes={"/checkpoints": ckpt_volume}, timeout=600)
def delete():
    ckpt_volume.reload()
    targets = [
        "200m_C_6p5e18_alpha0.100",
        "410m_C_6p5e18_alpha0.100",
        "680m_C_6p5e18_alpha0.100",
    ]
    base = Path("/checkpoints/6p5e18")
    for name in targets:
        path = base / name
        if not path.is_dir():
            print(f"  SKIP (not dir): {path}")
            continue
        print(f"  DELETE: {path}")
        shutil.rmtree(path)
    ckpt_volume.commit()
    print("done")


@app.local_entrypoint()
def main():
    delete.remote()
