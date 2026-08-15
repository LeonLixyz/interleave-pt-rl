from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

WORKSPACE = Path("/Users/leonli66/Desktop/Research/RL/Chess RL")
PROJECT = WORKSPACE / "chess-rl-miles"
MILES = WORKSPACE / "miles"

HPARAM = "multi_turn_lr1e-5_bs2048_kl0.001_res2560_adamw_grpo_miles_sglang_grpo_adamw_sgl64_cvd_mrouter_ctx16fix"
PATH_PREFIX = "miles_sglang_grpo_adamw_bs2048_sgl64_cvd_mrouter_ctx16fix"

STATE_PATH = Path("/tmp/chess_rl_miles_auto_upload_state.txt")

SPECS = [
    ("6p5e18|20m|1.000|0.008", "C6p5e18_20m_alpha1.000_beta0.008", 5000),
    ("6p5e19|50m|0.180|0.002", "C6p5e19_50m_alpha0.180_beta0.002", 2000),
    ("6p5e18|680m|0.750|0.296", "C6p5e18_680m_alpha0.750_beta0.296", 2000),
    ("6p5e18|680m|1.000|0.296", "C6p5e18_680m_alpha1.000_beta0.296", 2000),
    ("6p5e19|200m|0.750|0.007", "C6p5e19_200m_alpha0.750_beta0.007", 2000),
    ("6p5e19|680m|0.750|0.030", "C6p5e19_680m_alpha0.750_beta0.030", 2000),
    ("6p5e19|680m|1.000|0.030", "C6p5e19_680m_alpha1.000_beta0.030", 2000),
    ("6p5e19|200m|0.400|0.007", "C6p5e19_200m_alpha0.400_beta0.007", 2000),
    ("6p5e19|200m|1.000|0.007", "C6p5e19_200m_alpha1.000_beta0.007", 2000),
    ("6p5e19|680m|1.500|0.030", "C6p5e19_680m_alpha1.500_beta0.030", 2000),
    ("6p5e19|680m|2.000|0.030", "C6p5e19_680m_alpha2.000_beta0.030", 2000),
    ("6p5e19|200m|0.200|0.007", "C6p5e19_200m_alpha0.200_beta0.007", 2000),
    ("6p5e19|680m|3.000|0.030", "C6p5e19_680m_alpha3.000_beta0.030", 2000),
]

KNOWN_UPLOADED = {
    "6p5e19|50m|0.180|0.002",
    "6p5e18|680m|1.000|0.296",
    "6p5e19|200m|0.750|0.007",
    "6p5e19|200m|0.400|0.007",
}


def log(message: str) -> None:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def load_uploaded() -> set[str]:
    uploaded = set(KNOWN_UPLOADED)
    if STATE_PATH.exists():
        uploaded.update(line.strip() for line in STATE_PATH.read_text().splitlines() if line.strip())
    return uploaded


def save_uploaded(uploaded: set[str]) -> None:
    STATE_PATH.write_text("\n".join(sorted(uploaded)) + "\n")


def latest_step(model: str) -> int | None:
    dest = Path("/tmp") / f"latest_{model}_{uuid.uuid4().hex}.txt"
    ckpt_root = f"chess-rl-miles/trajectory_sep_no_labels/{HPARAM}/{model}/checkpoints"
    marker_src = f"{ckpt_root}/latest_checkpointed_iteration.txt"
    marker_step: int | None = None
    max_iter_step: int | None = None
    try:
        subprocess.run(
            ["modal", "volume", "get", "chess-rl-miles-checkpoints", marker_src, str(dest)],
            cwd=str(WORKSPACE),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        marker_step = int(dest.read_text().strip())
        listing = subprocess.check_output(
            ["modal", "volume", "ls", "chess-rl-miles-checkpoints", ckpt_root],
            cwd=str(WORKSPACE),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        steps = [int(match.group(1)) for match in re.finditer(r"iter_(\d+)", listing)]
        if steps:
            max_iter_step = max(steps)
        if marker_step is not None and max_iter_step is not None and max_iter_step > marker_step:
            log(f"WARN {model}: marker={marker_step} max_iter={max_iter_step}; using max_iter")
        return max(step for step in (marker_step, max_iter_step) if step is not None)
    except Exception as exc:
        log(f"WARN marker unavailable for {model}: {type(exc).__name__}")
        return None
    finally:
        dest.unlink(missing_ok=True)


def upload_spec(spec: str, model: str) -> bool:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT}:{MILES}:{env.get('PYTHONPATH', '')}"
    cmd = [
        "modal",
        "run",
        "-m",
        "chess_rl_miles.scripts.upload_checkpoints",
        "--specs",
        spec,
        "--repo-prefix",
        "Pre-to-Post-2/rl_",
        "--path-prefix",
        PATH_PREFIX,
        "--target-repo-type",
        "model",
        "--layout",
        "chess-rl",
        "--hparam-tag",
        HPARAM,
        "--steps",
        "all",
        "--skip-existing",
        "--commit-batch-size",
        "20",
        "--upload-retry-attempts",
        "4",
        "--rate-limit-sleep-seconds",
        "3700",
    ]
    log(f"UPLOAD start {model} ({spec})")
    proc = subprocess.Popen(
        cmd,
        cwd=str(WORKSPACE),
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
    if rc == 0:
        log(f"UPLOAD done {model}")
        return True
    log(f"UPLOAD failed {model}: exit {rc}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=int, default=300)
    args = parser.parse_args()

    uploaded = load_uploaded()
    save_uploaded(uploaded)
    log("monitor started")

    while True:
        all_uploaded = True
        summary: list[str] = []
        for spec, model, target in SPECS:
            step = latest_step(model)
            if step is None:
                all_uploaded = False
                continue
            done = step >= target
            if not done:
                all_uploaded = False
            summary.append(f"{model}:{step}/{target}{'*' if done else ''}")
            if done and spec not in uploaded and upload_spec(spec, model):
                uploaded.add(spec)
                save_uploaded(uploaded)
            if spec not in uploaded:
                all_uploaded = False

        log("status " + " ".join(summary))
        if all_uploaded:
            log("all target checkpoints uploaded; monitor exiting")
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
