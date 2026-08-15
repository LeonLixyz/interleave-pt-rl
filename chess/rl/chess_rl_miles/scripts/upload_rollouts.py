from __future__ import annotations

import os
import re
import sys
import tarfile
from pathlib import Path

import modal

from chess_rl_miles.data import COT_TYPE, model_id_from_spec
from chess_rl_miles.scripts.run_chess_miles import SPECS

CKPT_DIR = "/checkpoints"
DEFAULT_REPO_PREFIX = "chess-pre-to-post/rl_"
DEFAULT_PATH_PREFIX = "miles_cispo_minimax"

DEFAULT_RUN_SUFFIX_BY_SPEC = {
    "6p5e18|680m|1.000|0.296": "miles_cispo_minimax_jingyan_from_verl_step60",
    "6p5e19|680m|0.750|0.030": "miles_cispo_minimax_jingyan_from_verl_step280",
}


def _find_project_dir() -> Path:
    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        if (parent / "chess_rl_miles").is_dir() and (parent / "pyproject.toml").exists():
            return parent
    return current.parent


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if key != "HF_TOKEN":
            continue
        value = value.strip().strip('"').strip("'")
        if value and not os.environ.get(key):
            os.environ[key] = value


PROJECT_LOCAL = _find_project_dir()
WORKSPACE_LOCAL = PROJECT_LOCAL.parent
_load_env_file(WORKSPACE_LOCAL / "chess_reasoning" / ".env")

runtime_secrets = [modal.Secret.from_name("huggingface-secret")]
if hf_token := os.environ.get("HF_TOKEN"):
    runtime_secrets.append(modal.Secret.from_dict({"HF_TOKEN": hf_token}))

image = modal.Image.from_registry("python:3.11-slim").pip_install("huggingface-hub>=0.28.0")

ckpt_vol = modal.Volume.from_name("chess-rl-miles-checkpoints", create_if_missing=False)

app = modal.App("rl training", image=image, secrets=runtime_secrets)


def _choose_specs(specs: str) -> list[str]:
    chosen = [s.strip() for s in specs.split(",") if s.strip()] if specs else list(DEFAULT_RUN_SUFFIX_BY_SPEC)
    unknown = sorted(set(chosen) - set(SPECS))
    if unknown:
        raise ValueError(f"Unknown spec(s): {', '.join(unknown)}")
    return chosen


def _run_name(spec: str, run_suffix: str = "") -> str:
    model_id = model_id_from_spec(spec)
    suffix = run_suffix or DEFAULT_RUN_SUFFIX_BY_SPEC.get(spec)
    if not suffix:
        raise ValueError(f"No default Miles rollout run suffix for spec {spec}; pass --run-suffix.")
    return f"{model_id}_{suffix}"


def _artifact_root(spec: str, layout: str, run_suffix: str, hparam_tag: str) -> Path:
    model_id = model_id_from_spec(spec)
    if layout == "flat":
        return Path(CKPT_DIR) / "chess-rl-miles" / _run_name(spec, run_suffix)
    if not hparam_tag:
        raise ValueError("--hparam-tag is required for --layout chess-rl")
    return Path(CKPT_DIR) / "chess-rl-miles" / COT_TYPE / hparam_tag / model_id


ARTIFACT_DIR_CANDIDATES = [
    ("rollouts", ("rollouts",)),
    ("dump_details/rollout_data", ("dump_details", "rollout_data")),
    ("dump_details/train_data", ("dump_details", "train_data")),
    ("dump_details/policy_loss_debug", ("dump_details", "policy_loss_debug")),
    ("rollout", ("rollout",)),
    ("checkpoints/rollout", ("checkpoints", "rollout")),
]


def _collect_artifact_dirs(root: Path, artifact_kinds: str = "all") -> list[tuple[str, Path]]:
    candidates = [(name, root.joinpath(*parts)) for name, parts in ARTIFACT_DIR_CANDIDATES]
    if artifact_kinds and artifact_kinds != "all":
        wanted = {kind.strip() for kind in artifact_kinds.split(",") if kind.strip()}
        known = {name for name, _ in candidates}
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError(f"Unknown artifact kind(s): {', '.join(unknown)}. Known: {', '.join(sorted(known))}")
        candidates = [(name, path) for name, path in candidates if name in wanted]
    return [(name, path) for name, path in candidates if path.exists()]


def _file_sort_key(path: Path) -> tuple[tuple[int, ...], str]:
    numbers = tuple(int(value) for value in re.findall(r"\d+", path.stem))
    return numbers, path.name


def _collect_files(artifact_dirs: list[tuple[str, Path]], latest_n: int = 0) -> list[Path]:
    files: list[Path] = []
    for _, artifact_dir in artifact_dirs:
        artifact_files = sorted((p for p in artifact_dir.rglob("*") if p.is_file()), key=_file_sort_key)
        if latest_n > 0:
            artifact_files = artifact_files[-latest_n:]
        files.extend(artifact_files)
    return sorted(files)


@app.function(
    cpu=4.0,
    memory=32 * 1024,
    timeout=60 * 60 * 4,
    volumes={CKPT_DIR: ckpt_vol},
)
def upload_rollout(
    spec: str,
    repo_prefix: str = DEFAULT_REPO_PREFIX,
    path_prefix: str = DEFAULT_PATH_PREFIX,
    run_suffix: str = "",
    layout: str = "flat",
    hparam_tag: str = "",
    artifact_kinds: str = "all",
    compresslevel: int = 1,
    latest_n: int = 0,
) -> str:
    from huggingface_hub import HfApi

    model_id = model_id_from_spec(spec)
    run_name = _run_name(spec, run_suffix) if layout == "flat" else f"{model_id}_{hparam_tag}"
    root = _artifact_root(spec, layout, run_suffix, hparam_tag)
    artifact_dirs = _collect_artifact_dirs(root, artifact_kinds)
    if not artifact_dirs:
        raise FileNotFoundError(f"No Miles rollout/debug artifact dirs under {root}")

    files = _collect_files(artifact_dirs, latest_n)
    if not files:
        raise ValueError(f"No files found under artifact dirs for {root}")

    manifest = Path("/tmp") / f"{run_name}_manifest.txt"
    latest_file = root / "latest_checkpointed_iteration.txt"
    if not latest_file.exists():
        latest_file = root / "checkpoints" / "latest_checkpointed_iteration.txt"
    latest = latest_file.read_text().strip() if latest_file.exists() else ""
    manifest.write_text(
        "\n".join(
            [
                f"spec={spec}",
                f"model_id={model_id}",
                f"run_name={run_name}",
                f"source_root={root}",
                f"latest_checkpointed_iteration={latest}",
                f"artifact_kinds={artifact_kinds}",
                f"compresslevel={compresslevel}",
                f"latest_n={latest_n}",
                "artifact_dirs=",
                *[f"{name}={path}" for name, path in artifact_dirs],
                f"num_files={len(files)}",
                "files=",
                *[str(p.relative_to(root)) for p in files],
                "",
            ]
        )
    )

    tarball = Path("/tmp") / f"{run_name}_rollout.tar.gz"
    print(f"[{run_name}] tarring {len(files)} file(s) -> {tarball}", flush=True)
    with tarfile.open(tarball, "w:gz", compresslevel=compresslevel) as tf:
        for file in files:
            tf.add(str(file), arcname=str(file.relative_to(root)))
        tf.add(str(manifest), arcname="manifest.txt")
    size_mb = tarball.stat().st_size / (1024 * 1024)
    print(f"[{run_name}] tarball size: {size_mb:.1f} MB", flush=True)

    repo_id = f"{repo_prefix}{model_id}"
    path_in_repo = f"{path_prefix}/{run_name}_rollout.tar.gz"
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(tarball),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
    )
    url = f"https://huggingface.co/{repo_id}/blob/main/{path_in_repo}"
    print(f"[done] {url}", flush=True)
    return url


@app.local_entrypoint()
def main(
    specs: str = "",
    repo_prefix: str = DEFAULT_REPO_PREFIX,
    path_prefix: str = DEFAULT_PATH_PREFIX,
    run_suffix: str = "",
    layout: str = "flat",
    hparam_tag: str = "",
    artifact_kinds: str = "all",
    compresslevel: int = 1,
    latest_n: int = 0,
    dry_run: bool = False,
    wait: bool = True,
) -> None:
    chosen = _choose_specs(specs)
    print(f"Upload Miles rollout/debug artifacts: {len(chosen)} job(s)")
    print(f"repo_prefix={repo_prefix}  path_prefix={path_prefix}")
    print(
        f"layout={layout}  hparam_tag={hparam_tag or '(default/none)'}  "
        f"artifact_kinds={artifact_kinds}  compresslevel={compresslevel}  latest_n={latest_n}"
    )
    for spec in chosen:
        print(f"  {model_id_from_spec(spec)} -> {_artifact_root(spec, layout, run_suffix, hparam_tag)}")
    if dry_run:
        print("(dry-run)")
        return

    handles = [
        (
            spec,
            upload_rollout.spawn(
                spec,
                repo_prefix,
                path_prefix,
                run_suffix,
                layout,
                hparam_tag,
                artifact_kinds,
                compresslevel,
                latest_n,
            ),
        )
        for spec in chosen
    ]
    print(f"Spawned {len(handles)} upload job(s).")
    if not wait:
        return

    failed = []
    for spec, handle in handles:
        try:
            url = handle.get()
            print(f"  OK  {model_id_from_spec(spec)} -> {url}")
        except Exception as exc:
            print(f"  FAIL {model_id_from_spec(spec)}: {exc}")
            failed.append(spec)
    if failed:
        raise SystemExit(1)
