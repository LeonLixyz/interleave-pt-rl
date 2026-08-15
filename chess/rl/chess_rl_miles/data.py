from __future__ import annotations

from pathlib import Path

DATA_HF_REPO = "chess-pre-to-post/chess-rl-data"
SFT_HF_REPO = "chess-pre-to-post/sft_trajectory_no_labels"
COT_TYPE = "trajectory_sep_no_labels"

DEFAULT_TRAIN_FILE = "train_v4_dataset_balanced_multi_turn.parquet"
DEFAULT_TRAIN_FILE_SHA256 = (
    "bcf131d88cd9916f6ec12a12f8e85901a05e0dabc5d5a1fdebd8ea0b0ae86c30"
)
DEFAULT_EVAL_FILE = "puzzles/test_multi_turn_final.parquet"


def model_id_from_spec(spec: str) -> str:
    compute, size, alpha, beta = spec.split("|")
    return f"C{compute}_{size}_alpha{alpha}_beta{beta}"


def ensure_hf_snapshot(repo_id: str, local_dir: str | Path, *, repo_type: str, allow_patterns=None) -> Path:
    from huggingface_hub import snapshot_download

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
    )
    return local_dir


def ensure_chess_data(data_dir: str | Path) -> Path:
    return ensure_hf_snapshot(DATA_HF_REPO, data_dir, repo_type="dataset")


def ensure_sft_model(model_id: str, sft_root: str | Path, *, cot_type: str = COT_TYPE) -> Path:
    target_root = Path(sft_root) / cot_type
    target = target_root / model_id
    if (target / "config.json").exists():
        return target
    ensure_hf_snapshot(
        SFT_HF_REPO,
        target_root,
        repo_type="model",
        allow_patterns=[f"{model_id}/*", f"{model_id}/**"],
    )
    return target
