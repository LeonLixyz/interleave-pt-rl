"""
Verify peak lr in completed runs by reading metrics.jsonl from the volume.
"""
import json
from pathlib import Path

import modal

ckpt_volume = modal.Volume.from_name("rl-reasoning-checkpoints")

app = modal.App("check-lr")


@app.function(volumes={"/checkpoints": ckpt_volume}, timeout=300)
def check():
    runs = [
        "6p5e18/100m_C_6p5e18_alpha0.100",
        "6p5e18/200m_C_6p5e18_alpha0.100",
        "6p5e18/410m_C_6p5e18_alpha0.100",
        "6p5e18/680m_C_6p5e18_alpha0.100",
        "6p5e18/32m_C_6p5e18_alpha0.750",
    ]
    for r in runs:
        m = Path("/checkpoints") / r / "metrics.jsonl"
        if not m.exists():
            print(f"{r}: NO metrics.jsonl")
            continue
        max_lr = 0.0
        last_lr = 0.0
        n = 0
        last_step = 0
        with open(m) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                lr = d.get("lr") or d.get("train/lr")
                if lr is None:
                    continue
                max_lr = max(max_lr, lr)
                last_lr = lr
                last_step = d.get("step", last_step)
                n += 1
        print(f"{r}: peak_lr={max_lr:.2e}  last_lr={last_lr:.2e}  last_step={last_step}  n_logs={n}")


@app.local_entrypoint()
def main():
    check.remote()
