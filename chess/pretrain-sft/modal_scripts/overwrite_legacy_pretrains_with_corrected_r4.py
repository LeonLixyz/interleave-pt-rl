"""Replace legacy final-only pretrain repos with the complete corrected r4 runs.

The upload is guarded by the legacy repository HEAD observed immediately before
this migration. Matching files are overwritten, the missing training metadata
and scheduled checkpoints are added, and the tagged corrected repositories are
left untouched as immutable backups.

Usage:
  modal run --detach modal_scripts/overwrite_legacy_pretrains_with_corrected_r4.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

FRESH_TAG = "corrected_lr1e-3_20260726_r4"
OUTPUT_ROOT = Path(f"/checkpoints/6p5e18_{FRESH_TAG}")

RUNS = {
    "50m_a0p400": {
        "experiment": f"50m_C_6p5e18_alpha0.400_{FRESH_TAG}",
        "repo": "Pre-to-Post-2/50m_C_6p5e18_alpha0.400",
        "legacy_head_prefix": "3e2931a",
        "expected_step": 17_512,
    },
    "200m_a0p200": {
        "experiment": f"200m_C_6p5e18_alpha0.200_{FRESH_TAG}",
        "repo": "Pre-to-Post-2/200m_C_6p5e18_alpha0.200",
        "legacy_head_prefix": "0dfaa80",
        "expected_step": 2_038,
    },
    "200m_a0p750": {
        "experiment": f"200m_C_6p5e18_alpha0.750_{FRESH_TAG}",
        "repo": "Pre-to-Post-2/200m_C_6p5e18_alpha0.750",
        "legacy_head_prefix": "2cd8aa7",
        "expected_step": 7_646,
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
    "chess-pretrain-corrected-r4-legacy-overwrite",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/checkpoints": checkpoint_volume},
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _final_model(run_dir: Path) -> Path:
    models = sorted((run_dir / "final").glob("*.safetensors"))
    if len(models) != 1:
        raise RuntimeError(
            f"Expected exactly one final safetensors file in {run_dir / 'final'}; "
            f"found {len(models)}"
        )
    return models[0]


def _remote_lfs_sha(api, repo_id: str, path: str) -> str | None:
    info = api.model_info(repo_id=repo_id, files_metadata=True)
    for sibling in info.siblings:
        if sibling.rfilename == path and sibling.lfs:
            return sibling.lfs.get("sha256")
    return None


@app.function(
    cpu=4.0,
    memory=16 * 1024,
    timeout=47 * 60 * 60,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
)
def overwrite_legacy_repos() -> dict[str, dict[str, object]]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    results: dict[str, dict[str, object]] = {}
    checkpoint_volume.reload()

    for target, spec in RUNS.items():
        run_dir = OUTPUT_ROOT / str(spec["experiment"])
        state_path = run_dir / "latest" / "training_state.json"
        config_path = run_dir / "final" / "config.json"
        if not state_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"Incomplete corrected source: {run_dir}")

        state = json.loads(state_path.read_text(encoding="utf-8"))
        step = int(state["step"])
        if step != int(spec["expected_step"]):
            raise RuntimeError(
                f"Unexpected source step for {target}: {step} != {spec['expected_step']}"
            )

        local_model = _final_model(run_dir)
        local_sha = _sha256(local_model)
        repo_id = str(spec["repo"])
        before = api.model_info(repo_id=repo_id, files_metadata=True)

        # A retry after a successful commit is safe: verify the corrected model
        # and state and skip instead of rejecting the now-changed repository HEAD.
        remote_sha = _remote_lfs_sha(api, repo_id, "final/model.safetensors")
        if remote_sha == local_sha:
            downloaded_state = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename="latest/training_state.json",
                    repo_type="model",
                )
            )
            remote_state = json.loads(downloaded_state.read_text(encoding="utf-8"))
            if int(remote_state["step"]) == step:
                print(f"[skip] {repo_id} already contains corrected step {step}", flush=True)
                results[target] = {
                    "repo": repo_id,
                    "commit": before.sha,
                    "step": step,
                    "final_sha256": local_sha,
                    "status": "already_corrected",
                }
                continue

        expected_prefix = str(spec["legacy_head_prefix"])
        if not before.sha.startswith(expected_prefix):
            raise RuntimeError(
                f"Refusing to overwrite changed repo {repo_id}: "
                f"HEAD {before.sha} does not start with audited {expected_prefix}"
            )

        print(
            f"[upload] {run_dir} -> {repo_id} at parent {before.sha}",
            flush=True,
        )
        commit = api.upload_folder(
            folder_path=str(run_dir),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="",
            ignore_patterns=["hf_upload_complete.json"],
            commit_message=f"Replace legacy pretrain with complete corrected r4 ({FRESH_TAG})",
            parent_commit=before.sha,
        )

        after = api.model_info(repo_id=repo_id, files_metadata=True)
        files = {sibling.rfilename for sibling in after.siblings}
        required = {
            "final/config.json",
            "final/model.safetensors",
            "latest/training_state.json",
        }
        missing = sorted(required - files)
        if missing:
            raise RuntimeError(f"Post-upload verification failed for {repo_id}: missing {missing}")

        remote_sha = _remote_lfs_sha(api, repo_id, "final/model.safetensors")
        if remote_sha != local_sha:
            raise RuntimeError(
                f"Post-upload SHA mismatch for {repo_id}: {remote_sha} != {local_sha}"
            )
        downloaded_state = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename="latest/training_state.json",
                repo_type="model",
                revision=after.sha,
            )
        )
        remote_state = json.loads(downloaded_state.read_text(encoding="utf-8"))
        if int(remote_state["step"]) != step:
            raise RuntimeError(
                f"Post-upload step mismatch for {repo_id}: "
                f"{remote_state.get('step')} != {step}"
            )

        step_models = {
            path
            for path in files
            if path.startswith("step_") and path.endswith("/model.safetensors")
        }
        if len(step_models) != 10:
            raise RuntimeError(
                f"Post-upload checkpoint count mismatch for {repo_id}: "
                f"{len(step_models)} != 10"
            )

        results[target] = {
            "repo": repo_id,
            "commit": str(commit),
            "head": after.sha,
            "file_count": len(files),
            "checkpoint_models": len(step_models),
            "step": step,
            "final_sha256": local_sha,
            "status": "uploaded_and_verified",
        }
        print(f"[complete] {repo_id}: {json.dumps(results[target])}", flush=True)

    return results


@app.local_entrypoint()
def main() -> None:
    result = overwrite_legacy_repos.remote()
    print(json.dumps(result, indent=2), flush=True)
