from __future__ import annotations

import shutil
from pathlib import Path

import modal

from chess_rl_miles.data import COT_TYPE

CKPT_DIR = "/checkpoints"
ROOT = Path(CKPT_DIR) / "chess-rl-miles"
BASE = ROOT / COT_TYPE


image = (
    modal.Image.from_registry("python:3.11-slim")
    .add_local_dir(
        "/Users/leonli66/Desktop/Research/RL/Chess RL/chess-rl-miles",
        remote_path="/root/chess-rl-miles",
    )
)

ckpt_vol = modal.Volume.from_name("chess-rl-miles-checkpoints", create_if_missing=False)
app = modal.App("rl training", image=image)


def _split_paths(paths: str) -> list[str]:
    return [part.strip().strip("/") for part in paths.replace("\n", ",").split(",") if part.strip()]


def _resolve_safe(rel_path: str) -> Path:
    rel = rel_path.strip().strip("/")
    if rel.startswith("chess-rl-miles/"):
        rel = rel.removeprefix("chess-rl-miles/")
        target = (ROOT / rel).resolve()
    elif rel.startswith(f"{COT_TYPE}/"):
        target = (ROOT / rel).resolve()
    else:
        target = (BASE / rel).resolve()
    root = ROOT.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Refusing to prune outside {root}: {target}")
    return target


@app.function(cpu=2.0, memory=16 * 1024, timeout=60 * 60 * 4, volumes={CKPT_DIR: ckpt_vol})
def prune_paths(paths: str, dry_run: bool = True, count_limit: int = 2000) -> list[str]:
    ckpt_vol.reload()
    results: list[str] = []
    for rel_path in _split_paths(paths):
        target = _resolve_safe(rel_path)
        if not target.exists():
            results.append(f"missing {rel_path}")
            continue
        file_count = 1 if target.is_file() else 0
        dir_count = 0
        truncated = False
        if target.is_dir():
            for path in target.rglob("*"):
                if path.is_file():
                    file_count += 1
                elif path.is_dir():
                    dir_count += 1
                if file_count + dir_count >= count_limit:
                    truncated = True
                    break
        count_text = f"files={file_count} dirs={dir_count}" + ("+" if truncated else "")
        results.append(f"{'would delete' if dry_run else 'delete'} {rel_path} {count_text}")
        if not dry_run:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
    if not dry_run:
        ckpt_vol.commit()
    return results


@app.local_entrypoint()
def main(paths: str, dry_run: bool = True, count_limit: int = 2000) -> None:
    selected = _split_paths(paths)
    print(f"Prune uploaded artifacts: {len(selected)} path(s)")
    for path in selected:
        print(f"  {path}")
    handle = prune_paths.spawn(paths, dry_run, count_limit)
    for line in handle.get():
        print(line)
