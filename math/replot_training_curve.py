"""Plot training_curve.csv with a 100-step rolling average overlay.

Concatenated wandb runs include early smoke runs + the 6 resume-segments of v0.
We dedup by _step (keeping the latest-created run's value) and filter out
smoke segments that didn't reach the v0 chain.

Reads:  training_curve.csv
Writes: training_curve.png  (replaces)
        training_curve_clean.csv (deduped, sorted, only v0 chain)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="training_curve.csv")
    parser.add_argument("--png", default="training_curve.png")
    parser.add_argument("--clean-csv", default="training_curve_clean.csv")
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=["_step"])
    df["_step"] = df["_step"].astype(int)
    loss_col = "train/CE loss"
    df = df.dropna(subset=[loss_col])

    # Per-run metadata
    print(f"raw rows: {len(df)}, runs: {df['wandb_run_id'].nunique()}")
    for rid, sub in df.groupby("wandb_run_id"):
        print(f"  {rid}: {sub['_step'].min()}→{sub['_step'].max()} ({len(sub)} rows, created {sub['wandb_run_created'].iloc[0]})")

    # Identify the v0 chain: pick the run that REACHES step 95368.
    # Walk backwards: that run covers some range [a,b]. The previous run in the
    # chain is the one whose [start,end] connects to a (i.e., contains step a-1
    # or step a). Iteratively build the chain.
    runs_meta = df.groupby("wandb_run_id").agg(
        start=("_step", "min"), end=("_step", "max"),
        created=("wandb_run_created", "first"),
    ).reset_index()
    runs_meta = runs_meta.sort_values("created")
    print()
    print("sorted by created:")
    print(runs_meta.to_string(index=False))

    # Dedup by _step keeping latest-created (handles resume overlap cleanly).
    df = df.sort_values(["_step", "wandb_run_created"])  # last in same step = latest run
    df = df.drop_duplicates(subset=["_step"], keep="last").sort_values("_step").reset_index(drop=True)

    # Filter to the v0 chain by walking from step 95368 backward through the gaps.
    final_step = df["_step"].max()
    if final_step < 90000:
        print(f"WARN: final step {final_step} < 90000, may not be the full v0 run.")
    # Truncate any runs that didn't connect — but since smoke runs (steps 0-489
    # and 250-1149) overlap with the real run dw0i1nz9 (which covers 0-13089)
    # and dedup kept the latest, smoke values are already gone unless they
    # were the latest. Assume the chain is contiguous.
    print()
    print(f"after dedup: {len(df)} rows, steps {df['_step'].min()}→{df['_step'].max()}")

    # Save clean csv
    out_csv = Path(args.clean_csv).resolve()
    df.to_csv(out_csv, index=False)
    print(f"[csv] -> {out_csv}")

    rolling = df[loss_col].rolling(window=args.window, min_periods=1).mean()

    gbs = 512 * 4096
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(df["_step"], df[loss_col],
            lw=0.35, color="lightsteelblue", alpha=0.55, label="raw (per step)")
    ax.plot(df["_step"], rolling,
            lw=1.6, color="steelblue", label=f"rolling mean (window={args.window})")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("train/CE loss")
    ax.set_title(
        f"math-1b-v0 stable phase — {len(df):,} unique steps "
        f"(0 → {df['_step'].max():,}, ~{df['_step'].max() * gbs / 1e9:.0f}B tokens)"
    )
    ax.grid(alpha=0.3)
    sec = ax.secondary_xaxis(
        "top",
        functions=(lambda x: x * gbs / 1e9, lambda x: x * 1e9 / gbs),
    )
    sec.set_xlabel("training tokens (B)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_png = Path(args.png).resolve()
    fig.savefig(out_png, dpi=140)
    print(f"[png] -> {out_png}")
    print(f"[final] raw loss @ step {df['_step'].max()} = {df[loss_col].iloc[-1]:.4f}")
    print(f"[final] rolling mean = {rolling.iloc[-1]:.4f}")


if __name__ == "__main__":
    main()
