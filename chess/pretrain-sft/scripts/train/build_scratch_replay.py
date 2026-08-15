"""Build Exp 4's immutable scratch P2 + positive-replay manifest."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.scratch_replay import build_scratch_replay_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one deterministic sample-level shuffle containing every "
            "P2 PT/SFT record and every extracted positive replay record"
        )
    )
    parser.add_argument("--p2-manifest", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--replay-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shuffle-seed", type=int, default=44)
    parser.add_argument("--model-init-seed", type=int, default=42)
    parser.add_argument(
        "--trust-extractor-validation",
        action="store_true",
        help=(
            "Skip the full row-by-row validation pass. Checksums and replay "
            "manifest row counts are still verified."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_scratch_replay_manifest(
        p2_manifest_path=args.p2_manifest,
        replay_path=args.replay,
        replay_manifest_path=args.replay_manifest,
        output_dir=args.output_dir,
        shuffle_seed=args.shuffle_seed,
        model_init_seed=args.model_init_seed,
        validate_all_replay_rows=not args.trust_extractor_validation,
    )
    print(
        json.dumps(
            {
                "metadata_path": str(manifest.metadata_path),
                "metadata_hash": manifest.metadata_hash,
                "base_records": (
                    manifest.pretrain_records + manifest.sft_records
                ),
                "replay_records": manifest.replay_records,
                "padding_records": manifest.padding_records,
                "baseline_cosine_steps": manifest.baseline_cosine_steps,
                "floor_tail_steps": manifest.floor_tail_steps,
                "total_steps": manifest.total_steps,
                "model_init_seed": manifest.model_init_seed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
