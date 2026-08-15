from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from chess_rl_miles.data import model_id_from_spec


DEFAULT_HPARAM = "multi_turn_lr1e-5_bs2048_kl0.001_res2560_adamw_grpo_miles_sglang_grpo_adamw_sgl64_cvd_mrouter_ctx16fix"
DEFAULT_PATH_PREFIX = "miles_sglang_grpo_adamw_bs2048_sgl64_cvd_mrouter_ctx16fix"
DEFAULT_REPO_PREFIX = "Pre-to-Post-2/rl_"


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def run_capture(cmd: list[str], cwd: Path) -> str:
    return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True)


def latest_step(*, project: Path, ckpt_root: str) -> int | None:
    try:
        output = run_capture(["modal", "volume", "ls", "chess-rl-miles-checkpoints", ckpt_root], project)
    except subprocess.CalledProcessError as exc:
        log(f"WARN latest_step failed rc={exc.returncode}: {exc.output[-500:]}")
        return None
    steps = [int(match.group(1)) for match in re.finditer(r"iter_0*(\d+)", output)]
    return max(steps) if steps else None


def tracker_step(*, project: Path, ckpt_root: str) -> int | None:
    marker = f"{ckpt_root}/latest_checkpointed_iteration.txt"
    try:
        output = run_capture(
            ["modal", "volume", "get", "chess-rl-miles-checkpoints", marker, "-"],
            project,
        )
    except subprocess.CalledProcessError as exc:
        log(f"WARN tracker_step failed rc={exc.returncode}: {exc.output[-500:]}")
        return None
    match = re.search(r"(?m)^\s*(\d+)", output)
    return int(match.group(1)) if match else None


def checkpoint_complete(*, project: Path, ckpt_root: str, step: int) -> tuple[bool, str]:
    step_root = f"{ckpt_root}/iter_{step:07d}"
    try:
        output = run_capture(
            ["modal", "volume", "ls", "chess-rl-miles-checkpoints", step_root],
            project,
        )
    except subprocess.CalledProcessError as exc:
        return False, f"ls failed rc={exc.returncode}: {exc.output[-300:]}"

    present = {Path(line.strip()).name for line in output.splitlines() if line.strip()}
    required = {"model", "optimizer", "lr_scheduler", "rng.pt", "meta.json"}
    missing = sorted(required - present)
    if missing:
        return False, f"missing={','.join(missing)}"

    try:
        meta_output = run_capture(
            [
                "modal",
                "volume",
                "get",
                "chess-rl-miles-checkpoints",
                f"{step_root}/meta.json",
                "-",
            ],
            project,
        )
    except subprocess.CalledProcessError as exc:
        return False, f"meta read failed rc={exc.returncode}: {exc.output[-300:]}"

    iteration_match = re.search(r'"iteration"\s*:\s*(\d+)', meta_output)
    next_rollout_match = re.search(r'"next_rollout_id"\s*:\s*(\d+)', meta_output)
    if not iteration_match or int(iteration_match.group(1)) != step:
        return False, "meta iteration mismatch"
    if not next_rollout_match or int(next_rollout_match.group(1)) < step:
        return False, "meta next_rollout_id mismatch"
    return True, "ok"


def app_active(*, project: Path, app_id: str) -> bool | None:
    if not app_id:
        return False
    try:
        output = run_capture(["modal", "container", "list", "--json"], project)
    except subprocess.CalledProcessError as exc:
        log(f"WARN container list failed rc={exc.returncode}: {exc.output[-500:]}")
        return None
    try:
        containers = json.loads(output)
    except json.JSONDecodeError as exc:
        log(f"WARN container list returned invalid JSON: {exc}")
        return None
    return any(container.get("App ID") == app_id for container in containers)


def run_logged(cmd: list[str], *, project: Path, miles: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project}:{miles}:{env.get('PYTHONPATH', '')}"
    log("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(project),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"command failed rc={rc}: {cmd}")


def upload_all(args: argparse.Namespace, *, project: Path, miles: Path, model_id: str) -> None:
    checkpoint_steps = ",".join(str(step) for step in range(args.save_interval, args.target + 1, args.save_interval))
    log(f"target reached for {model_id}; starting converted checkpoint upload")
    run_logged(
        [
            "modal",
            "run",
            "-m",
            "chess_rl_miles.scripts.upload_checkpoints",
            "--specs",
            args.spec,
            "--repo-prefix",
            args.repo_prefix,
            "--path-prefix",
            args.path_prefix,
            "--steps",
            checkpoint_steps,
            "--layout",
            "chess-rl",
            "--hparam-tag",
            args.hparam,
            "--skip-existing",
            "--commit-batch-size",
            str(args.commit_batch_size),
            "--upload-retry-attempts",
            str(args.upload_retry_attempts),
        ],
        project=project,
        miles=miles,
    )

    log(f"converted checkpoints uploaded for {model_id}; starting final raw checkpoint upload")
    run_logged(
        [
            "modal",
            "run",
            "-m",
            "chess_rl_miles.scripts.upload_raw_checkpoints",
            "--specs",
            args.spec,
            "--backend",
            "miles",
            "--hparam-tag",
            args.hparam,
            "--repo-prefix",
            args.repo_prefix,
            "--path-prefix",
            args.path_prefix,
            "--steps",
            str(args.target),
        ],
        project=project,
        miles=miles,
    )

    log(f"raw final checkpoint uploaded for {model_id}; starting rollout/debug upload")
    run_logged(
        [
            "modal",
            "run",
            "-m",
            "chess_rl_miles.scripts.upload_rollouts",
            "--specs",
            args.spec,
            "--repo-prefix",
            args.repo_prefix,
            "--path-prefix",
            args.path_prefix,
            "--layout",
            "chess-rl",
            "--hparam-tag",
            args.hparam,
            "--compresslevel",
            str(args.compresslevel),
        ],
        project=project,
        miles=miles,
    )
    log(f"all uploads complete for {model_id}")


def audit_hf(*, model_id: str, repo_prefix: str, path_prefix: str) -> None:
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        repo = f"{repo_prefix}{model_id}"
        files = api.list_repo_files(repo, repo_type="model")
        steps = sorted(
            {
                int(match.group(1))
                for path in files
                for match in [re.search(rf"^{re.escape(path_prefix)}/global_step_(\d+)/", path)]
                if match
            }
        )
        raw = sorted(
            {
                int(match.group(1))
                for path in files
                for match in [re.search(rf"^{re.escape(path_prefix)}/raw_checkpoints/.*steps_(\d+)(?:/|$)", path)]
                if match
            }
        )
        rollout_count = sum(1 for path in files if path.startswith(f"{path_prefix}/") and "rollout" in path.lower())
        log(
            "HF audit "
            f"converted_count={len(steps)} converted_max={steps[-1] if steps else None} "
            f"raw_max={raw[-1] if raw else None} rollout_files={rollout_count}"
        )
    except Exception as exc:
        log(f"WARN HF audit failed: {type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="/Users/leonli66/Desktop/Research/RL/Chess RL")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--app-id", default="")
    parser.add_argument("--hparam", default=DEFAULT_HPARAM)
    parser.add_argument("--path-prefix", default=DEFAULT_PATH_PREFIX)
    parser.add_argument("--repo-prefix", default=DEFAULT_REPO_PREFIX)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--save-interval", type=int, default=20)
    parser.add_argument("--commit-batch-size", type=int, default=4)
    parser.add_argument("--upload-retry-attempts", type=int, default=4)
    parser.add_argument("--compresslevel", type=int, default=1)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    project = workspace / "chess-rl-miles"
    miles = workspace / "miles"
    model_id = model_id_from_spec(args.spec)
    ckpt_root = f"/chess-rl-miles/trajectory_sep_no_labels/{args.hparam}/{model_id}/checkpoints"

    log(
        f"monitor started model={model_id} target={args.target} "
        f"interval={args.interval_seconds}s ckpt_root={ckpt_root}"
    )
    while True:
        step = latest_step(project=project, ckpt_root=ckpt_root)
        tracker = tracker_step(project=project, ckpt_root=ckpt_root)
        complete, completeness_reason = checkpoint_complete(
            project=project,
            ckpt_root=ckpt_root,
            step=args.target,
        )
        active = app_active(project=project, app_id=args.app_id)
        log(
            f"status max_step={step}/{args.target} tracker={tracker} "
            f"target_complete={complete}({completeness_reason}) app_active={active}"
        )
        ready = (
            step is not None
            and step >= args.target
            and tracker is not None
            and tracker >= args.target
            and complete
            and active is False
        )
        if ready:
            upload_all(args, project=project, miles=miles, model_id=model_id)
            audit_hf(model_id=model_id, repo_prefix=args.repo_prefix, path_prefix=args.path_prefix)
            log("monitor exiting success")
            return 0
        if step is not None and step < args.target and args.app_id and active is False:
            log("WARN training app is not active before target; monitor will wait but will not auto-relaunch")
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
