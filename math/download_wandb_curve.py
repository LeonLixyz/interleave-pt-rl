"""Download training loss curve from wandb -> CSV + PNG.

Usage:
    python download_wandb_curve.py [--run-path leon-modal-modal/math-pretraining/nkhcxisa]

Outputs:
    training_curve.csv  — per-step columns: _step, train/CE loss, lr, MFU
    training_curve.png  — loss vs step
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-path",
        default="leon-modal-modal/math-pretraining/nkhcxisa",
        help="entity/project/run_id",
    )
    parser.add_argument("--out-csv", default="training_curve.csv")
    parser.add_argument("--out-png", default="training_curve.png")
    parser.add_argument(
        "--keys",
        default="train/CE loss,optim/LR,throughput/device/MFU,train/Z loss",
        help="comma-separated wandb metric keys to include",
    )
    parser.add_argument("--samples", type=int, default=None,
                        help="downsample to N points; None = all")
    args = parser.parse_args()

    import wandb
    import pandas as pd
    import matplotlib.pyplot as plt

    api = wandb.Api()
    run = api.run(args.run_path)
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    print(f"[wandb] downloading run {run.path}: {run.name}")
    print(f"[wandb] keys={keys}")

    hist = run.history(keys=keys, samples=args.samples, pandas=True)
    print(f"[wandb] got {len(hist)} rows, columns={list(hist.columns)}")

    out_csv = Path(args.out_csv).resolve()
    hist.to_csv(out_csv, index=False)
    print(f"[csv] -> {out_csv}")

    # Plot CE loss
    loss_col = next((c for c in hist.columns if "CE loss" in c), None)
    if loss_col is None:
        print("[png] no CE loss column found; available:", list(hist.columns))
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(10, 5))
    df = hist.dropna(subset=[loss_col])
    ax.plot(df["_step"], df[loss_col], lw=0.7, color="steelblue", label="CE loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("train/CE loss")
    ax.set_title(f"{run.name} — training loss (entity/project: {run.entity}/{run.project})")
    ax.grid(alpha=0.3)
    # secondary axis: tokens
    gbs = 512 * 4096  # ~2.1M tokens / step
    sec = ax.secondary_xaxis(
        "top",
        functions=(lambda x: x * gbs / 1e9, lambda x: x * 1e9 / gbs),
    )
    sec.set_xlabel("training tokens (B)")
    ax.legend()
    out_png = Path(args.out_png).resolve()
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[png] -> {out_png}")


if __name__ == "__main__":
    main()
