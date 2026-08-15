from __future__ import annotations

import shutil
from pathlib import Path

import modal

from chess_rl_miles.data import COT_TYPE, model_id_from_spec
from chess_rl_miles.scripts.run_chess_miles import SPECS

CKPT_DIR = "/checkpoints"

image = (
    modal.Image.from_registry("python:3.11-slim")
    .add_local_dir(
        "/Users/leonli66/Desktop/Research/RL/Chess RL/chess-rl-miles",
        remote_path="/root/chess-rl-miles",
    )
)

ckpt_vol = modal.Volume.from_name("chess-rl-miles-checkpoints", create_if_missing=False)
app = modal.App("rl training", image=image)


def _choose_specs(specs: str) -> list[str]:
    chosen = [s.strip() for s in specs.split(",") if s.strip()]
    if not chosen:
        raise ValueError("--specs is required")
    unknown = sorted(set(chosen) - set(SPECS))
    if unknown:
        raise ValueError(f"Unknown spec(s): {', '.join(unknown)}")
    return chosen


@app.function(cpu=2.0, memory=8 * 1024, timeout=60 * 60, volumes={CKPT_DIR: ckpt_vol})
def cleanup_debug_artifact(spec: str, hparam_tag: str, artifact_kind: str) -> str:
    if artifact_kind not in {"policy_loss_debug", "train_data", "rollout_data"}:
        raise ValueError(f"Unsupported artifact_kind={artifact_kind!r}")
    model_id = model_id_from_spec(spec)
    artifact_root = Path(CKPT_DIR) / "chess-rl-miles" / COT_TYPE / hparam_tag / model_id
    debug_dir = artifact_root / "dump_details" / artifact_kind
    if not debug_dir.exists():
        return f"{model_id}: no {artifact_kind} dir"
    count = sum(1 for path in debug_dir.rglob("*") if path.is_file())
    shutil.rmtree(debug_dir)
    ckpt_vol.commit()
    return f"{model_id}: removed {count} {artifact_kind} files"


@app.local_entrypoint()
def main(specs: str, hparam_tag: str, artifact_kind: str = "policy_loss_debug", dry_run: bool = False) -> None:
    chosen = _choose_specs(specs)
    print(f"Cleanup {artifact_kind} for {len(chosen)} run(s)")
    for spec in chosen:
        model_id = model_id_from_spec(spec)
        path = Path(CKPT_DIR) / "chess-rl-miles" / COT_TYPE / hparam_tag / model_id / "dump_details" / artifact_kind
        print(f"  {model_id}: {path}")
    if dry_run:
        return
    handles = [(spec, cleanup_debug_artifact.spawn(spec, hparam_tag, artifact_kind)) for spec in chosen]
    for spec, handle in handles:
        print(handle.get())
