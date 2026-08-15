import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi


VOLUME_NAME = "rl-reasoning-checkpoints"
TMP_ROOT = Path("/tmp/hf_final_uploads")

UPLOADS = [
    ("800m_C_total_03_9p344e18_alpha0.740_beta0.03", "C9e18_800m_alpha0.740_beta0.03"),
    ("400m_C_total_03_9p344e18_alpha0.970_beta0.03", "C9e18_400m_alpha0.970_beta0.03"),
    ("200m_C_total_03_9p344e18_alpha0.970_beta0.03", "C9e18_200m_alpha0.970_beta0.03"),
    ("200m_C_total_03_9p344e18_alpha0.740_beta0.03", "C9e18_200m_alpha0.740_beta0.03"),
    ("100m_C_total_03_9p344e18_alpha0.740_beta0.03", "C9e18_100m_alpha0.740_beta0.03"),
]


def ensure_download(old_name: str, new_name: str) -> Path:
    local_dir = TMP_ROOT / new_name
    final_dir = local_dir / "final"
    model_file = local_dir / "model.safetensors"
    final_model_file = final_dir / "model.safetensors"
    if final_model_file.is_file():
        return final_dir
    if model_file.is_file():
        return local_dir

    remote_path = f"/C9e18_beta0.03/{old_name}/final"
    print(f"DOWNLOADING {remote_path}", flush=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["modal", "volume", "get", VOLUME_NAME, remote_path, str(local_dir), "--force"],
        check=True,
    )
    if final_model_file.is_file():
        return final_dir
    if not model_file.is_file():
        raise RuntimeError(f"Downloaded path missing model.safetensors under: {local_dir}")
    return local_dir


def upload_final(final_dir: Path, repo_suffix: str, hf_token: str) -> None:
    repo_id = f"chess-pre-to-post/{repo_suffix}"
    print(f"UPLOADING {repo_id}", flush=True)
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(final_dir),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo="final",
        commit_message="Manual upload of final checkpoint",
    )
    print(f"UPLOADED {repo_id}", flush=True)


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is not set")

    TMP_ROOT.mkdir(parents=True, exist_ok=True)

    for old_name, new_name in UPLOADS:
        final_dir = ensure_download(old_name, new_name)
        upload_final(final_dir, new_name, hf_token)
        shutil.rmtree(TMP_ROOT / new_name, ignore_errors=True)


if __name__ == "__main__":
    main()
