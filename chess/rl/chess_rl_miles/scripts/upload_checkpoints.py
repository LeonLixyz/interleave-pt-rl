from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

PROJECT_DIR = "/root/chess-rl-miles"
MILES_DIR = "/root/miles"
for mounted_source_dir in (PROJECT_DIR, MILES_DIR):
    if mounted_source_dir not in sys.path:
        sys.path.insert(0, mounted_source_dir)

from chess_rl_miles.data import COT_TYPE, ensure_sft_model, model_id_from_spec
from chess_rl_miles.scripts.run_chess_miles import SPECS

SFT_DIR = "/sft"
CKPT_DIR = "/checkpoints"
HF_CACHE_DIR = "/root/.cache/huggingface"
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
MILES_LOCAL = WORKSPACE_LOCAL / "miles"
_load_env_file(WORKSPACE_LOCAL / "chess_reasoning" / ".env")

runtime_secrets = [modal.Secret.from_name("huggingface-secret")]
if hf_token := os.environ.get("HF_TOKEN"):
    runtime_secrets.append(modal.Secret.from_dict({"HF_TOKEN": hf_token}))

image = (
    modal.Image.from_registry(os.environ.get("CHESS_RL_MILES_IMAGE", "radixark/miles:dev"))
    .apt_install("git", "curl")
    .pip_install("huggingface-hub>=0.28.0", "safetensors>=0.5.0")
    .add_local_dir(str(MILES_LOCAL), remote_path=MILES_DIR)
    .add_local_dir(str(PROJECT_LOCAL), remote_path=PROJECT_DIR)
)

sft_vol = modal.Volume.from_name("chess-rl-miles-sft", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("chess-rl-miles-checkpoints", create_if_missing=False)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

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
        raise ValueError(f"No default Miles checkpoint run suffix for spec {spec}; pass --run-suffix.")
    return f"{model_id}_{suffix}"


def _step_num(step_dir: Path) -> int:
    if step_dir.name.startswith("iter_"):
        return int(step_dir.name.split("_")[-1])
    if step_dir.name.startswith("global_step_"):
        return int(step_dir.name.split("_")[-1])
    raise ValueError(f"Cannot parse step from {step_dir}")


def _selected_step_dirs(run_root: Path, steps: str) -> list[Path]:
    all_steps = sorted(
        [d for d in run_root.glob("iter_*") if d.is_dir()],
        key=_step_num,
    )
    if not all_steps:
        all_steps = sorted(
            [d for d in run_root.glob("global_step_*") if d.is_dir()],
            key=_step_num,
        )
    if not all_steps:
        raise ValueError(f"No iter_* or global_step_* checkpoint dirs under {run_root}")

    if steps.strip().lower() == "all":
        latest_file = run_root / "latest_checkpointed_iteration.txt"
        if latest_file.exists():
            latest = int(latest_file.read_text().strip())
            return [d for d in all_steps if _step_num(d) <= latest]
        return all_steps

    if steps:
        wanted = {int(s.strip()) for s in steps.split(",") if s.strip()}
        selected = [d for d in all_steps if _step_num(d) in wanted]
        if not selected:
            raise ValueError(f"No requested step in {run_root}. wanted={sorted(wanted)}")
        return selected

    latest_file = run_root / "latest_checkpointed_iteration.txt"
    if latest_file.exists():
        latest = int(latest_file.read_text().strip())
        for step_dir in all_steps:
            if _step_num(step_dir) == latest:
                return [step_dir]
    return [all_steps[-1]]


def _run_root(
    *,
    spec: str,
    layout: str,
    run_suffix: str,
    hparam_tag: str,
) -> Path:
    model_id = model_id_from_spec(spec)
    if layout == "flat":
        return Path(CKPT_DIR) / "chess-rl-miles" / _run_name(spec, run_suffix)
    if not hparam_tag:
        raise ValueError("--hparam-tag is required for --layout chess-rl")
    return Path(CKPT_DIR) / "chess-rl-miles" / COT_TYPE / hparam_tag / model_id / "checkpoints"


def _target_repo_and_prefix(
    *,
    model_id: str,
    repo_prefix: str,
    path_prefix: str,
    single_repo_id: str,
    model_path_prefix: str,
) -> tuple[str, str]:
    if not single_repo_id:
        return f"{repo_prefix}{model_id}", path_prefix.strip("/")

    parts = [model_path_prefix.strip("/"), model_id, path_prefix.strip("/")]
    return single_repo_id, "/".join(part for part in parts if part)


def _is_rate_limited(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "Too Many Requests" in text or "rate limit" in text.lower()


def _rate_limit_sleep_seconds(message: str, default_seconds: int, path_in_repo: str) -> int:
    match = re.search(r"Retry after\s+(\d+)\s+seconds", message)
    seconds = int(match.group(1)) if match else default_seconds
    if "repository commits" in message and "320 per hour" in message:
        seconds = max(seconds, default_seconds)
    # Spread concurrent Modal workers a little after the hourly reset.
    seconds += sum(ord(ch) for ch in path_in_repo) % 180
    return seconds


def _upload_folder_with_rate_limit_retry(
    *,
    api,
    folder_path: str,
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
    allow_patterns: list[str] | None,
    attempts: int,
    rate_limit_sleep_seconds: int,
) -> None:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            api.upload_folder(
                folder_path=folder_path,
                repo_id=repo_id,
                repo_type=repo_type,
                path_in_repo=path_in_repo,
                allow_patterns=allow_patterns,
            )
            return
        except Exception as exc:
            last_exc = exc
            if not _is_rate_limited(exc) or attempt == attempts:
                raise
            message = str(exc)
            sleep_seconds = _rate_limit_sleep_seconds(message, rate_limit_sleep_seconds, path_in_repo)
            print(
                f"[hf] rate limited on {repo_id}/{path_in_repo}; "
                f"sleeping {sleep_seconds}s before retry {attempt + 1}/{attempts}",
                flush=True,
            )
            time.sleep(sleep_seconds)
    if last_exc is not None:
        raise last_exc


@app.function(
    cpu=16.0,
    memory=128 * 1024,
    timeout=60 * 60 * 8,
    volumes={
        SFT_DIR: sft_vol,
        CKPT_DIR: ckpt_vol,
        HF_CACHE_DIR: hf_cache,
    },
)
def convert_and_upload(
    spec: str,
    repo_prefix: str = DEFAULT_REPO_PREFIX,
    path_prefix: str = DEFAULT_PATH_PREFIX,
    steps: str = "",
    layout: str = "flat",
    run_suffix: str = "",
    hparam_tag: str = "",
    delete_existing_prefix: bool = False,
    skip_existing: bool = False,
    single_repo_id: str = "",
    model_path_prefix: str = "models",
    target_repo_type: str = "model",
    commit_batch_size: int = 1,
    upload_retry_attempts: int = 4,
    rate_limit_sleep_seconds: int = 3700,
) -> list[str]:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    from huggingface_hub import HfApi

    model_id = model_id_from_spec(spec)
    run_root = _run_root(spec=spec, layout=layout, run_suffix=run_suffix, hparam_tag=hparam_tag)
    if not run_root.is_dir():
        raise FileNotFoundError(f"Miles checkpoint root not on volume: {run_root}")

    origin_hf = ensure_sft_model(model_id, SFT_DIR, cot_type=COT_TYPE)
    step_dirs = _selected_step_dirs(run_root, steps)
    print(f"[convert] {model_id}: {[d.name for d in step_dirs]} from {run_root}", flush=True)

    target_root = Path("/tmp/hf_miles") / model_id
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_DIR}:{MILES_DIR}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env.setdefault("TQDM_DISABLE", "1")
    env.setdefault("HF_HOME", HF_CACHE_DIR)

    urls: list[str] = []
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    repo_id, target_prefix = _target_repo_and_prefix(
        model_id=model_id,
        repo_prefix=repo_prefix,
        path_prefix=path_prefix,
        single_repo_id=single_repo_id,
        model_path_prefix=model_path_prefix,
    )
    create_kwargs = {"space_sdk": "static"} if target_repo_type == "space" else {}
    api.create_repo(repo_id=repo_id, repo_type=target_repo_type, exist_ok=True, **create_kwargs)
    existing_steps: set[int] = set()
    if delete_existing_prefix:
        try:
            existing = list(
                api.list_repo_tree(
                    repo_id=repo_id,
                    repo_type=target_repo_type,
                    path_in_repo=target_prefix,
                    recursive=False,
                    expand=False,
                )
            )
        except Exception:
            existing = []
        if existing:
            print(f"[hf] deleting existing {repo_id}/{target_prefix}", flush=True)
            api.delete_folder(repo_id=repo_id, repo_type=target_repo_type, path_in_repo=target_prefix)
    elif skip_existing:
        try:
            existing = list(
                api.list_repo_tree(
                    repo_id=repo_id,
                    repo_type=target_repo_type,
                    path_in_repo=target_prefix,
                    recursive=False,
                    expand=False,
                )
            )
        except Exception:
            existing = []
        for item in existing:
            name = Path(getattr(item, "path", "")).name
            if name.startswith("global_step_"):
                try:
                    existing_steps.add(int(name.split("_")[-1]))
                except ValueError:
                    pass
        if existing_steps:
            print(f"[hf] {repo_id}/{target_prefix}: {len(existing_steps)} existing step(s)", flush=True)

    commit_batch_size = max(1, commit_batch_size)
    pending_steps: list[int] = []

    def flush_pending_steps() -> None:
        nonlocal pending_steps
        if not pending_steps:
            return
        upload_steps = list(pending_steps)
        allow_patterns = [f"global_step_{step}/**" for step in upload_steps]
        print(
            f"[hf] uploading batch {model_id} steps {upload_steps} -> {repo_id}/{target_prefix}",
            flush=True,
        )
        _upload_folder_with_rate_limit_retry(
            api=api,
            folder_path=str(target_root),
            repo_id=repo_id,
            repo_type=target_repo_type,
            path_in_repo=target_prefix,
            allow_patterns=allow_patterns,
            attempts=upload_retry_attempts,
            rate_limit_sleep_seconds=rate_limit_sleep_seconds,
        )
        for step in upload_steps:
            path_in_repo = f"{target_prefix}/global_step_{step}"
            url = f"https://huggingface.co/{repo_id}/tree/main/{path_in_repo}"
            print(f"[done] {url}", flush=True)
            urls.append(url)
            shutil.rmtree(target_root / f"global_step_{step}", ignore_errors=True)
        pending_steps = []

    for step_dir in step_dirs:
        step = _step_num(step_dir)
        if skip_existing and step in existing_steps:
            print(f"[skip] {model_id} step {step}: already on HF", flush=True)
            continue
        target_step = target_root / f"global_step_{step}"
        cmd = [
            sys.executable,
            str(Path(MILES_DIR) / "tools" / "convert_fsdp_to_hf.py"),
            "--input-dir",
            str(step_dir),
            "--origin-hf-dir",
            str(origin_hf),
            "--output-dir",
            str(target_step),
            "--force",
        ]
        print(f"[merge] step {step}: {step_dir} -> {target_step}", flush=True)
        result = subprocess.run(
            cmd,
            cwd=MILES_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            print(result.stdout[-8000:], flush=True)
            raise RuntimeError(f"convert_fsdp_to_hf failed for {model_id} step {step}: exit {result.returncode}")
        print(f"[merge done] step {step}", flush=True)

        manifest = target_step / "miles_checkpoint_manifest.json"
        manifest.write_text(
            "{\n"
            f'  "spec": "{spec}",\n'
            f'  "model_id": "{model_id}",\n'
            f'  "layout": "{layout}",\n'
            f'  "source": "{step_dir}",\n'
            f'  "origin_hf": "{origin_hf}",\n'
            f'  "step": {step}\n'
            "}\n"
        )

        pending_steps.append(step)
        if len(pending_steps) >= commit_batch_size:
            flush_pending_steps()

    flush_pending_steps()

    sft_vol.commit()
    return urls


@app.function(
    cpu=2.0,
    memory=8 * 1024,
    timeout=60 * 30,
    volumes={
        CKPT_DIR: ckpt_vol,
    },
)
def audit_hf_steps(
    spec: str,
    repo_prefix: str = DEFAULT_REPO_PREFIX,
    path_prefix: str = DEFAULT_PATH_PREFIX,
    steps: str = "all",
    layout: str = "flat",
    run_suffix: str = "",
    hparam_tag: str = "",
    single_repo_id: str = "",
    model_path_prefix: str = "models",
    target_repo_type: str = "model",
) -> dict[str, object]:
    from huggingface_hub import HfApi

    model_id = model_id_from_spec(spec)
    run_root = _run_root(spec=spec, layout=layout, run_suffix=run_suffix, hparam_tag=hparam_tag)
    if not run_root.is_dir():
        raise FileNotFoundError(f"Miles checkpoint root not on volume: {run_root}")

    step_dirs = _selected_step_dirs(run_root, steps)
    expected_steps = [_step_num(step_dir) for step_dir in step_dirs]

    repo_id, target_prefix = _target_repo_and_prefix(
        model_id=model_id,
        repo_prefix=repo_prefix,
        path_prefix=path_prefix,
        single_repo_id=single_repo_id,
        model_path_prefix=model_path_prefix,
    )
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    existing_steps: set[int] = set()
    try:
        existing = list(
                api.list_repo_tree(
                    repo_id=repo_id,
                    repo_type=target_repo_type,
                    path_in_repo=target_prefix,
                    recursive=False,
                    expand=False,
                )
            )
    except Exception:
        existing = []
    for item in existing:
        name = Path(getattr(item, "path", "")).name
        if name.startswith("global_step_"):
            try:
                existing_steps.add(int(name.split("_")[-1]))
            except ValueError:
                pass

    expected = set(expected_steps)
    existing_in_scope = sorted(expected & existing_steps)
    missing = sorted(expected - existing_steps)
    extra = sorted(existing_steps - expected)
    latest_file = run_root / "latest_checkpointed_iteration.txt"
    latest = int(latest_file.read_text().strip()) if latest_file.exists() else None
    return {
        "model_id": model_id,
        "repo_id": repo_id,
        "path_prefix": target_prefix,
        "run_root": str(run_root),
        "latest": latest,
        "expected_count": len(expected_steps),
        "existing_count": len(existing_in_scope),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "expected_steps": expected_steps,
        "existing_steps": existing_in_scope,
        "missing_steps": missing,
        "extra_steps": extra,
    }


@app.local_entrypoint()
def main(
    specs: str = "",
    repo_prefix: str = DEFAULT_REPO_PREFIX,
    path_prefix: str = DEFAULT_PATH_PREFIX,
    steps: str = "",
    layout: str = "flat",
    run_suffix: str = "",
    hparam_tag: str = "",
    delete_existing_prefix: bool = False,
    skip_existing: bool = False,
    single_repo_id: str = "",
    model_path_prefix: str = "models",
    target_repo_type: str = "model",
    commit_batch_size: int = 1,
    upload_retry_attempts: int = 4,
    rate_limit_sleep_seconds: int = 3700,
    dry_run: bool = False,
    wait: bool = True,
) -> None:
    chosen = _choose_specs(specs)
    print(f"Convert/upload Miles checkpoints: {len(chosen)} job(s)")
    print(
        f"repo_prefix={repo_prefix}  path_prefix={path_prefix}  "
        f"single_repo_id={single_repo_id or '(per-model repos)'}  "
        f"target_repo_type={target_repo_type}  "
        f"steps={steps or 'latest'} delete_existing_prefix={delete_existing_prefix} "
        f"skip_existing={skip_existing} commit_batch_size={commit_batch_size}"
    )
    print(
        f"upload_retry_attempts={upload_retry_attempts} "
        f"rate_limit_sleep_seconds={rate_limit_sleep_seconds}"
    )
    print(f"layout={layout}  hparam_tag={hparam_tag or '(default/none)'}")
    for spec in chosen:
        root = _run_root(spec=spec, layout=layout, run_suffix=run_suffix, hparam_tag=hparam_tag)
        print(f"  {model_id_from_spec(spec)} -> {root}")
    if dry_run:
        print("(dry-run)")
        return

    handles = [
        (
            spec,
            convert_and_upload.spawn(
                spec,
                repo_prefix,
                path_prefix,
                steps,
                layout,
                run_suffix,
                hparam_tag,
                delete_existing_prefix,
                skip_existing,
                single_repo_id,
                model_path_prefix,
                target_repo_type,
                commit_batch_size,
                upload_retry_attempts,
                rate_limit_sleep_seconds,
            ),
        )
        for spec in chosen
    ]
    print(f"Spawned {len(handles)} conversion job(s).")
    if not wait:
        return

    failed = []
    for spec, handle in handles:
        try:
            urls = handle.get()
            for url in urls:
                print(f"  OK  {model_id_from_spec(spec)} -> {url}")
        except Exception as exc:
            print(f"  FAIL {model_id_from_spec(spec)}: {exc}")
            failed.append(spec)
    if failed:
        raise SystemExit(1)
