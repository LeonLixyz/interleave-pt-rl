"""Fan-out launcher for fork-and-anneal runs across a list of stable-phase checkpoints.

After the stable-phase training writes checkpoints at the configured milestones
(10/25/50/65/80/100/130B tokens), this script spawns one anneal run per
checkpoint, each running independently on its own GPU node.

Usage:
    # Anneal every milestone with a 5B-token linear-to-zero decay:
    python launch_anneals.py \\
        --stable-folder /checkpoints/math-1b-stable-v0 \\
        --milestones 10B,25B,50B,65B,80B,100B,130B \\
        --anneal-tokens 5_000_000_000

This relies on the `train.py` Modal app already being deployed (or being
launchable with `modal run`). Each anneal is launched detached via `.spawn()`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import modal

from common import CHECKPOINT_MOUNT, TOKENIZED_MOUNT


def _parse_tokens(s: str) -> int:
    s = s.strip().replace("_", "").upper()
    suffixes = {"B": 1_000_000_000, "M": 1_000_000, "K": 1_000}
    if s and s[-1] in suffixes:
        return int(float(s[:-1]) * suffixes[s[-1]])
    return int(s)


def _expected_step(token_count: int, global_batch_size: int) -> int:
    """Convert tokens to optimizer steps (matches train_inner.GLOBAL_BATCH_SIZE)."""
    return token_count // global_batch_size


def _find_checkpoint(stable_folder: Path, expected_step: int) -> Path | None:
    """Find a checkpoint dir whose step number is closest to expected_step."""
    if not stable_folder.is_dir():
        return None
    candidates = []
    for child in stable_folder.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith("step-"):
            try:
                step = int(name.removeprefix("step-"))
                candidates.append((step, child))
            except ValueError:
                continue
        elif name.startswith("step"):
            try:
                step = int(name.removeprefix("step"))
                candidates.append((step, child))
            except ValueError:
                continue
    if not candidates:
        return None
    # Pick the checkpoint closest to expected_step
    candidates.sort(key=lambda x: abs(x[0] - expected_step))
    return candidates[0][1]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-folder", required=True, help="Volume path containing step-* checkpoints.")
    parser.add_argument("--milestones", required=True, help="Comma-separated token milestones, e.g. 10B,25B,50B")
    parser.add_argument("--anneal-tokens", type=_parse_tokens, default=5_000_000_000)
    parser.add_argument("--data-glob", default=f"{TOKENIZED_MOUNT}/3/part_*.npy")
    parser.add_argument("--gpu-type", default="H200")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--save-interval", type=int, default=0,
                        help="Save interval for anneal runs (0 = save only the final checkpoint).")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without launching anything.")
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=512 * 4096,
        help="Must match the GLOBAL_BATCH_SIZE used during stable training (default: 512 * 4096).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    milestones = [_parse_tokens(m) for m in args.milestones.split(",")]
    stable_folder = Path(args.stable_folder)

    plan = []
    for milestone in milestones:
        expected_step = _expected_step(milestone, args.global_batch_size)
        ckpt = _find_checkpoint(stable_folder, expected_step)
        plan.append({
            "milestone": milestone,
            "expected_step": expected_step,
            "ckpt": str(ckpt) if ckpt else None,
        })

    print(f"Stable folder: {stable_folder}")
    print(f"Milestones (tokens): {milestones}")
    print(f"Anneal length: {args.anneal_tokens:,} tokens each")
    print()
    for p in plan:
        mark = "OK" if p["ckpt"] else "MISSING"
        print(
            f"  [{mark}] {p['milestone'] / 1e9:.0f}B  step~{p['expected_step']}  -> {p['ckpt']}"
        )

    if any(p["ckpt"] is None for p in plan):
        print("\nERROR: one or more checkpoints are missing — refusing to launch.")
        return 1

    if args.dry_run:
        print("\n[dry-run] would launch", len(plan), "anneal runs")
        return 0

    # Resolve the Modal function from the deployed train.py app.
    fn_name = "run_h200" if (args.gpu_type.upper() == "H200" and args.nodes == 1) else (
        "run_h100" if (args.gpu_type.upper() == "H100" and args.nodes == 1) else "run_h200_4node"
    )
    try:
        fn = modal.Function.from_name("math-pretraining-train", fn_name)
    except modal.exception.NotFoundError:
        print(
            "\nERROR: app 'math-pretraining-train' is not deployed. Run "
            "`modal deploy train.py` first, or launch each anneal directly with "
            "`modal run train.py --mode anneal ...`."
        )
        return 1

    print()
    for p in plan:
        run_name = f"math-1b-anneal-{p['milestone'] // 1_000_000_000}B"
        save_folder = f"{CHECKPOINT_MOUNT}/{run_name}"
        kwargs = {
            "mode": "anneal",
            "data_glob": args.data_glob,
            "tokens": args.anneal_tokens,
            "warmup_tokens": 0,  # ignored in anneal mode
            "lr": args.lr,
            "load_path": p["ckpt"],
            "load_optim": False,
            "save_folder": save_folder,
            "run_name": run_name,
            "rank_microbatch_size_tokens": 8 * 4096,
            "compile_model": True,
            "num_workers": 4,
            "save_interval": args.save_interval,
            "ephemeral_save_interval": 0,
            "save_async": False,
            "benchmark_steps": None,
            "shard_degree": None,
        }
        call = fn.spawn(**kwargs)
        print(f"  [launched] {run_name}  call_id={call.object_id}  ckpt={p['ckpt']}")
    print(f"\nLaunched {len(plan)} anneal runs. Track with `modal app list`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
