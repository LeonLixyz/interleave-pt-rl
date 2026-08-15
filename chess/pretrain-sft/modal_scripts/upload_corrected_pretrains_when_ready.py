"""Wait for the corrected r4 pretrains and synchronously upload every checkpoint.

Usage:
  modal run --detach modal_scripts/upload_corrected_pretrains_when_ready.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

FRESH_TAG = "corrected_lr1e-3_20260726_r4"
OUTPUT_ROOT = Path(f"/checkpoints/6p5e18_{FRESH_TAG}")
POLL_SECONDS = 60
RETRY_SECONDS = 5 * 60

RUNS = {
    "50m_a0p400": {
        "experiment": f"50m_C_6p5e18_alpha0.400_{FRESH_TAG}",
        "repo": (
            "Pre-to-Post-2/"
            f"pretrain_C6p5e18_50m_alpha0.400_{FRESH_TAG}"
        ),
    },
    "200m_a0p200": {
        "experiment": f"200m_C_6p5e18_alpha0.200_{FRESH_TAG}",
        "repo": (
            "Pre-to-Post-2/"
            f"pretrain_C6p5e18_200m_alpha0.200_{FRESH_TAG}"
        ),
    },
    "200m_a0p750": {
        "experiment": f"200m_C_6p5e18_alpha0.750_{FRESH_TAG}",
        "repo": (
            "Pre-to-Post-2/"
            f"pretrain_C6p5e18_200m_alpha0.750_{FRESH_TAG}"
        ),
    },
}

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface-hub==0.35.3",
)
checkpoint_volume = modal.Volume.from_name(
    "rl-reasoning-checkpoints",
    create_if_missing=False,
)
app = modal.App(
    "chess-pretrain-corrected-r4-uploader",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/checkpoints": checkpoint_volume},
)


def _final_is_complete(run_dir: Path) -> bool:
    final_dir = run_dir / "final"
    if not (final_dir / "config.json").is_file():
        return False
    return any(final_dir.glob("*.safetensors"))


@app.function(
    cpu=4.0,
    memory=16 * 1024,
    timeout=47 * 60 * 60,
    retries=modal.Retries(initial_delay=5.0, max_retries=2),
)
def monitor_and_upload() -> dict[str, str]:
    from huggingface_hub import HfApi

    api = HfApi()
    pending = dict(RUNS)
    uploaded: dict[str, str] = {}
    retry_after = {target: 0.0 for target in pending}

    # Create and permission-check the destinations immediately.
    for target, spec in pending.items():
        api.create_repo(
            repo_id=spec["repo"],
            repo_type="model",
            exist_ok=True,
        )
        print(
            f"[uploader] destination ready: {target} -> {spec['repo']}",
            flush=True,
        )

    while pending:
        checkpoint_volume.reload()
        now = time.monotonic()
        made_progress = False

        for target, spec in list(pending.items()):
            if now < retry_after[target]:
                continue
            run_dir = OUTPUT_ROOT / spec["experiment"]
            if not _final_is_complete(run_dir):
                continue

            print(
                f"[uploader] final detected; syncing full run: "
                f"{run_dir} -> {spec['repo']}",
                flush=True,
            )
            try:
                commit = api.upload_folder(
                    folder_path=str(run_dir),
                    repo_id=spec["repo"],
                    repo_type="model",
                    path_in_repo="",
                    commit_message=(
                        f"Complete corrected pretraining run {FRESH_TAG}"
                    ),
                )
            except Exception as exc:
                retry_after[target] = time.monotonic() + RETRY_SECONDS
                print(
                    f"[uploader] upload failed for {target}; retrying in "
                    f"{RETRY_SECONDS}s: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

            commit_url = str(commit)
            marker = run_dir / "hf_upload_complete.json"
            marker.write_text(
                json.dumps(
                    {
                        "target": target,
                        "repo": spec["repo"],
                        "fresh_tag": FRESH_TAG,
                        "commit": commit_url,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            checkpoint_volume.commit()
            uploaded[target] = commit_url
            del pending[target]
            made_progress = True
            print(
                f"[uploader] COMPLETE {target}: {commit_url}",
                flush=True,
            )

        if pending:
            names = ",".join(sorted(pending))
            print(f"[uploader] waiting for: {names}", flush=True)
            time.sleep(5 if made_progress else POLL_SECONDS)

    return uploaded


@app.local_entrypoint()
def main() -> None:
    result = monitor_and_upload.remote()
    print(json.dumps(result, indent=2), flush=True)
