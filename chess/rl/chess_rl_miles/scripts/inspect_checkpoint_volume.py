from __future__ import annotations

import os
from pathlib import Path

import modal

CKPT_DIR = "/checkpoints"
ROOT = Path(CKPT_DIR) / "chess-rl-miles"

image = (
    modal.Image.from_registry("python:3.11-slim")
    .add_local_dir(
        "/Users/leonli66/Desktop/Research/RL/Chess RL/chess-rl-miles",
        remote_path="/root/chess-rl-miles",
    )
)

ckpt_vol = modal.Volume.from_name("chess-rl-miles-checkpoints", create_if_missing=False)
app = modal.App("rl training", image=image)


def _count_tree(path: Path) -> tuple[int, int]:
    files = 0
    dirs = 0
    for item in path.rglob("*"):
        if item.is_file():
            files += 1
        elif item.is_dir():
            dirs += 1
    return files, dirs


@app.function(cpu=2.0, memory=16 * 1024, timeout=60 * 60 * 4, volumes={CKPT_DIR: ckpt_vol})
def inspect() -> list[str]:
    ckpt_vol.reload()
    st = os.statvfs(CKPT_DIR)
    used = st.f_files - st.f_ffree
    pct = (used / st.f_files * 100) if st.f_files else 0.0
    lines = [f"statvfs inodes: used={used} total={st.f_files} pct={pct:.1f}%"]
    if not ROOT.exists():
        lines.append(f"missing {ROOT}")
        return lines

    rows: list[tuple[int, int, str]] = []
    for child in ROOT.iterdir():
        if child.is_dir():
            files, dirs = _count_tree(child)
            rows.append((files, dirs, child.name))
        elif child.is_file():
            rows.append((1, 0, child.name))
    for files, dirs, name in sorted(rows, reverse=True):
        lines.append(f"{name}: files={files} dirs={dirs} entries={files + dirs}")

    traj = ROOT / "trajectory_sep_no_labels"
    if traj.exists():
        lines.append("trajectory_sep_no_labels children:")
        rows = []
        for child in traj.iterdir():
            if child.is_dir():
                files, dirs = _count_tree(child)
                rows.append((files, dirs, child.name))
            elif child.is_file():
                rows.append((1, 0, child.name))
        for files, dirs, name in sorted(rows, reverse=True):
            lines.append(f"  {name}: files={files} dirs={dirs} entries={files + dirs}")
    return lines


@app.local_entrypoint()
def main() -> None:
    for line in inspect.remote():
        print(line)
