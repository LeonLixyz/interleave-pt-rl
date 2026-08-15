from __future__ import annotations

import os
from pathlib import Path

import modal

from chess_rl_miles.data import COT_TYPE, model_id_from_spec
from chess_rl_miles.scripts.run_chess_miles import SPECS

MILES_CKPT_DIR = "/miles-checkpoints"
VERL_CKPT_DIR = "/verl-checkpoints"
DEFAULT_REPO_PREFIX = "chess-pre-to-post/rl_"


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

image = (
    modal.Image.from_registry("python:3.11-slim")
    .pip_install("huggingface-hub>=0.28.0")
    .add_local_dir(str(PROJECT_LOCAL), remote_path="/root/chess-rl-miles")
)

miles_ckpt_vol = modal.Volume.from_name("chess-rl-miles-checkpoints", create_if_missing=False)
verl_ckpt_vol = modal.Volume.from_name("chess-rl-checkpoints", create_if_missing=False)

app = modal.App("rl training", image=image, secrets=runtime_secrets)


def _choose_specs(specs: str) -> list[str]:
    chosen = [s.strip() for s in specs.split(",") if s.strip()]
    if not chosen:
        raise ValueError("--specs is required")
    unknown = sorted(set(chosen) - set(SPECS))
    if unknown:
        raise ValueError(f"Unknown spec(s): {', '.join(unknown)}")
    return chosen


def _step_num(step_dir: Path) -> int:
    if step_dir.name.startswith("iter_"):
        return int(step_dir.name.split("_")[-1])
    if step_dir.name.startswith("global_step_"):
        return int(step_dir.name.split("_")[-1])
    raise ValueError(f"Cannot parse checkpoint step from {step_dir}")


def _checkpoint_root(backend: str, hparam_tag: str, model_id: str) -> Path:
    if not hparam_tag:
        raise ValueError("--hparam-tag is required")
    if backend == "miles":
        return Path(MILES_CKPT_DIR) / "chess-rl-miles" / COT_TYPE / hparam_tag / model_id / "checkpoints"
    if backend == "verl":
        return Path(VERL_CKPT_DIR) / COT_TYPE / hparam_tag / model_id / "checkpoints"
    raise ValueError("--backend must be either miles or verl")


def _selected_step_dirs(root: Path, steps: str, latest_n: int) -> list[Path]:
    all_steps = sorted(
        [d for d in root.iterdir() if d.is_dir() and (d.name.startswith("iter_") or d.name.startswith("global_step_"))],
        key=_step_num,
    )
    if not all_steps:
        raise FileNotFoundError(f"No checkpoint step dirs under {root}")

    if steps:
        if steps.strip().lower() == "all":
            latest_file = root / "latest_checkpointed_iteration.txt"
            if latest_file.exists():
                latest = int(latest_file.read_text().strip())
                return [d for d in all_steps if _step_num(d) <= latest]
            return all_steps
        wanted = {int(step.strip()) for step in steps.split(",") if step.strip()}
        selected = [d for d in all_steps if _step_num(d) in wanted]
        missing = sorted(wanted - {_step_num(d) for d in selected})
        if missing:
            raise FileNotFoundError(f"Missing requested checkpoint steps under {root}: {missing}")
        return selected

    latest_file = root / "latest_checkpointed_iteration.txt"
    if latest_file.exists():
        latest = int(latest_file.read_text().strip())
        all_steps = [d for d in all_steps if _step_num(d) <= latest]

    if latest_n <= 0:
        latest_n = 1
    return all_steps[-latest_n:]


def _target_repo_and_path(
    *,
    model_id: str,
    repo_prefix: str,
    path_prefix: str,
    run_label: str,
    single_repo_id: str,
    model_path_prefix: str,
) -> tuple[str, str]:
    if not single_repo_id:
        return f"{repo_prefix}{model_id}", f"{path_prefix.strip('/')}/raw_checkpoints/{run_label}"

    parts = [
        model_path_prefix.strip("/"),
        model_id,
        path_prefix.strip("/"),
        "raw_checkpoints",
        run_label,
    ]
    return single_repo_id, "/".join(part for part in parts if part)


@app.function(
    cpu=4.0,
    memory=32 * 1024,
    timeout=60 * 60 * 8,
    volumes={
        MILES_CKPT_DIR: miles_ckpt_vol,
        VERL_CKPT_DIR: verl_ckpt_vol,
    },
)
def upload_raw_checkpoint_archive(
    spec: str,
    backend: str,
    hparam_tag: str,
    repo_prefix: str = DEFAULT_REPO_PREFIX,
    path_prefix: str = "raw_checkpoints",
    steps: str = "",
    latest_n: int = 3,
    single_repo_id: str = "",
    model_path_prefix: str = "models",
    target_repo_type: str = "model",
) -> str:
    from huggingface_hub import HfApi

    model_id = model_id_from_spec(spec)
    root = _checkpoint_root(backend, hparam_tag, model_id)
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root not on volume: {root}")

    step_dirs = _selected_step_dirs(root, steps, latest_n)
    step_numbers = [_step_num(step_dir) for step_dir in step_dirs]
    if len(step_numbers) > 6:
        step_label = f"all_{len(step_numbers)}_steps_{step_numbers[0]}_to_{step_numbers[-1]}"
    else:
        step_label = "-".join(str(step) for step in step_numbers)
    run_label = f"{backend}_{hparam_tag}_steps_{step_label}".replace("/", "_")
    manifest = Path("/tmp") / f"{model_id}_{run_label}_manifest.txt"
    latest_file = root / "latest_checkpointed_iteration.txt"
    latest = latest_file.read_text().strip() if latest_file.exists() else ""

    manifest.write_text(
        "\n".join(
            [
                f"spec={spec}",
                f"model_id={model_id}",
                f"backend={backend}",
                f"hparam_tag={hparam_tag}",
                f"source_root={root}",
                f"latest_checkpointed_iteration={latest}",
                f"selected_steps={','.join(str(step) for step in step_numbers)}",
                "selected_dirs=",
                *[step_dir.name for step_dir in step_dirs],
                "",
            ]
        )
    )

    repo_id, path_in_repo = _target_repo_and_path(
        model_id=model_id,
        repo_prefix=repo_prefix,
        path_prefix=path_prefix,
        run_label=run_label,
        single_repo_id=single_repo_id,
        model_path_prefix=model_path_prefix,
    )
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    create_kwargs = {"space_sdk": "static"} if target_repo_type == "space" else {}
    api.create_repo(repo_id=repo_id, repo_type=target_repo_type, exist_ok=True, **create_kwargs)
    allow_patterns = [f"{step_dir.name}/**" for step_dir in step_dirs]
    if latest_file.exists():
        allow_patterns.append("latest_checkpointed_iteration.txt")
    print(f"[raw] {model_id}: uploading {[d.name for d in step_dirs]} -> {repo_id}/{path_in_repo}", flush=True)
    api.upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        repo_type=target_repo_type,
        path_in_repo=path_in_repo,
        allow_patterns=allow_patterns,
    )
    api.upload_file(
        path_or_fileobj=str(manifest),
        repo_id=repo_id,
        repo_type=target_repo_type,
        path_in_repo=f"{path_in_repo}/manifest.txt",
    )
    url = f"https://huggingface.co/{repo_id}/tree/main/{path_in_repo}"
    print(f"[done] {url}", flush=True)
    return url


@app.local_entrypoint()
def main(
    specs: str,
    backend: str,
    hparam_tag: str,
    repo_prefix: str = DEFAULT_REPO_PREFIX,
    path_prefix: str = "preserve",
    steps: str = "",
    latest_n: int = 3,
    single_repo_id: str = "",
    model_path_prefix: str = "models",
    target_repo_type: str = "model",
    dry_run: bool = False,
    wait: bool = True,
) -> None:
    chosen = _choose_specs(specs)
    print(f"Upload raw checkpoints: {len(chosen)} job(s)")
    print(
        f"backend={backend} hparam_tag={hparam_tag} path_prefix={path_prefix} "
        f"single_repo_id={single_repo_id or '(per-model repos)'} "
        f"target_repo_type={target_repo_type} "
        f"steps={steps or f'latest {latest_n}'}"
    )
    for spec in chosen:
        model_id = model_id_from_spec(spec)
        print(f"  {model_id} -> {_checkpoint_root(backend, hparam_tag, model_id)}")
    if dry_run:
        print("(dry-run)")
        return

    handles = [
        (
            spec,
            upload_raw_checkpoint_archive.spawn(
                spec,
                backend,
                hparam_tag,
                repo_prefix,
                path_prefix,
                steps,
                latest_n,
                single_repo_id,
                model_path_prefix,
                target_repo_type,
            ),
        )
        for spec in chosen
    ]
    print(f"Spawned {len(handles)} raw checkpoint upload job(s).")
    if not wait:
        return

    failed: list[str] = []
    for spec, handle in handles:
        try:
            print(f"  OK  {model_id_from_spec(spec)} -> {handle.get()}")
        except Exception as exc:
            print(f"  FAIL {model_id_from_spec(spec)}: {exc}")
            failed.append(spec)
    if failed:
        raise SystemExit(1)
