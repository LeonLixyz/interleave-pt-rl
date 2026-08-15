"""Launch the revised 20B interleaved pretraining experiment on Modal.

This is deliberately separate from ``launch_50m_interleaved.py``.  The older
launcher is bound to immutable 10B/5B artifacts and production-gate markers;
reusing it would either return old outputs or mix experiment identities.

The initial runnable DAG is:

* P1: 10B pretraining targets + the first deterministic SFT half;
* E2: the exact P1 || P2 stream (20B + all SFT), one cosine arc.

After P1 completes, the same launcher can start a fresh P2 arc from either the
P1 endpoint (E3) or an E1 RL1 HF export.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import modal


# ---------------------------------------------------------------------------
# Frozen experiment identity and accounting
# ---------------------------------------------------------------------------

EXPERIMENT_VERSION = "mix20b_sft77k_once_3072_v1_20260730"
DATA_ARTIFACT_VERSION = "mix20b_sft77k_once_clean_v1_20260730"
APP_NAME = "chess-50m-interleaved-pretrain-20b"

SOURCE_REPO = "chess-pre-to-post/pretrain_v1_20b"
SOURCE_REVISION = "07dd1b7090ca5f0fb05ef624c26b20bff19483c8"
SOURCE_DIR = Path("/data/pretrain_v1_20b")
SOURCE_TOKENS = 53_970_293_905
SOURCE_SHARDS = 47_090
SOURCE_MANIFEST_TEMPLATE = Path(
    "/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate"
    "/source_manifest.json"
)
SOURCE_MANIFEST_FILE_SHA256 = (
    "7f144d2329628759f2529540bfb9b10692e374d0c8b1933ec43c7c634b979253"
)
SOURCE_MANIFEST_HASH = (
    "5e2bd529811066c0c9c264eaf39a820f139ad4a4b1e9c9395fca42118e95a275"
)

SFT_REPO = "Pre-to-Post-2/200M_SFT_dataset"
SFT_REVISION = "fd343bd28f6a40fc3dab4dcfb6e74c11b7a20b90"
SFT_ROWS = 77_717
SFT_CACHE_DIR = Path(
    "/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/sft_cache"
)
SFT_CACHE_HASH = (
    "d82378522d43d5db3e8333588c24b1f864bb9e8ecd46303e1d2cd2e31d31df98"
)
SFT_PROMPT_FIELD = "pgn"
SFT_COT_FIELD = "cot_by_method.trajectory_sep.cot_format_no_labels"
SFT_TARGETS_P1 = 26_289_598
SFT_TARGETS_P2 = 26_193_155
SFT_TARGETS_TOTAL = 52_482_753

SEQUENCE_LENGTH = 3_072
PRETRAIN_TOTAL_TOKENS = 20_000_000_000
PRETRAIN_LEG_TOKENS = 10_000_000_000
DATA_SEED = 42
P1_SHUFFLE_SEED = 42
P2_SHUFFLE_SEED = 43

GPU_TYPE = "H200"
WORLD_SIZE = 8
LOCAL_BATCH_SIZE = 21
GLOBAL_BATCH_SIZE = WORLD_SIZE * LOCAL_BATCH_SIZE
GRADIENT_ACCUMULATION = 1
LEG_PT_RECORDS = math.ceil(PRETRAIN_LEG_TOKENS / SEQUENCE_LENGTH)
LEG_STEPS = 19_608
MONOLITHIC_STEPS = 39_216
P1_PADDING_RECORDS = 77
P2_PADDING_RECORDS = 76

# Preserve the clean-v2 equal-integrated PT:SFT objective after doubling the
# PT budget while exposing the same SFT rows exactly once.
P1_SFT_LOSS_WEIGHT = PRETRAIN_LEG_TOKENS / SFT_TARGETS_P1
P2_SFT_LOSS_WEIGHT = PRETRAIN_LEG_TOKENS / SFT_TARGETS_P2
MONOLITHIC_SFT_LOSS_WEIGHT = PRETRAIN_TOTAL_TOKENS / SFT_TARGETS_TOTAL

ARTIFACT_ROOT = Path(f"/data/50m_interleaved_{DATA_ARTIFACT_VERSION}")
SOURCE_MANIFEST_PATH = ARTIFACT_ROOT / "source_manifest.json"
PRETRAIN_SELECTION_PATH = ARTIFACT_ROOT / "pretrain_selection.json"
LEGS_ROOT = ARTIFACT_ROOT / "legs"
P1_METADATA_PATH = LEGS_ROOT / "p1" / "metadata.json"
P1_ORDER_PATH = LEGS_ROOT / "p1" / "order.npy"
P2_METADATA_PATH = LEGS_ROOT / "p2" / "metadata.json"
P2_ORDER_PATH = LEGS_ROOT / "p2" / "order.npy"
EXP2_METADATA_PATH = LEGS_ROOT / "exp2" / "metadata.json"
CANARY_METADATA_PATH = LEGS_ROOT / "canary" / "metadata.json"
MANIFEST_SET_PATH = ARTIFACT_ROOT / "manifest_set.json"
CANARY_GATE_PATH = ARTIFACT_ROOT / "mixed_canary_gate_v2.json"

CHECKPOINT_ROOT = Path(
    f"/checkpoints/interleave_50m/pretrain/{EXPERIMENT_VERSION}"
)
BASE_CONFIG = "config/configs/interleaved_50m/base_3072.yaml"
BASE_CONFIG_SHA256 = (
    "3ec2303cca8ada094124be8d36c380640b0a2cb8fa6001dc3b1d08d20d46a518"
)
TRAIN_CLI = "scripts/train/train_interleaved_hf.py"
WANDB_ENTITY = "jingyanshen-new-york-university"
WANDB_PROJECT = "chess-50m-interleaved-20b"

EXPECTED_SELECTION_HASH = (
    "fb8292387ee2f9b4d410500881010dca54dfc1073ef9b5244139616958c4a340"
)
EXPECTED_SELECTION_FILE_SHA256 = (
    "26bcf9b4f31743dfd9511e6e939b44712d2ef06a945cbd3336450ade8fd058ba"
)
EXPECTED_P1_ORDER_SHA256 = (
    "b9422d53781f6c2faaeaf1de90c4d021ef9e67f6aa265af13084caea80e5b77c"
)
EXPECTED_P2_ORDER_SHA256 = (
    "0e876e032aecf0e723dce53fbf7bd632c5a685a9283f42d4fb8a0bdd10ad3fae"
)
EXPECTED_P1_METADATA_HASH = (
    "96ae34da2bb51f8c64b44696953bb30b85cfb91e63ca65552315f5169b84ea01"
)
EXPECTED_P2_METADATA_HASH = (
    "6ff6b18bf1b005f725228838bad82d93653426a04898a3a7f87ef747c39ec67b"
)
EXPECTED_EXP2_METADATA_HASH = (
    "13ba5d17aa5658434dce582970867f50cc26f3a31a0c5f150114a5548475e786"
)

_RUN_ID_RE = re.compile(r"[a-z0-9][a-z0-9.-]{0,79}")
_CALL_ID_RE = re.compile(r"fc-[A-Za-z0-9]+")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


CONTRACT: dict[str, Any] = {
    "schema": "interleaved-20b-launch-contract-v1",
    "experiment_version": EXPERIMENT_VERSION,
    "data_artifact_version": DATA_ARTIFACT_VERSION,
    "model": {
        "parameters": 47_245_312,
        "context_length": SEQUENCE_LENGTH,
        "architecture": "qwen3-qk-norm",
        "seed": 42,
    },
    "data": {
        "pretrain_tokens": PRETRAIN_TOTAL_TOKENS,
        "p1_tokens": PRETRAIN_LEG_TOKENS,
        "p2_tokens": PRETRAIN_LEG_TOKENS,
        "sft_rows": SFT_ROWS,
        "p1_sft_rows": 38_858,
        "p2_sft_rows": 38_859,
        "sft_exposures": 1,
        "stream_order": "p1-shuffle-then-p2-shuffle",
        "selection_seed": DATA_SEED,
        "p1_shuffle_seed": P1_SHUFFLE_SEED,
        "p2_shuffle_seed": P2_SHUFFLE_SEED,
    },
    "topology": {
        "gpu": f"{GPU_TYPE}:{WORLD_SIZE}",
        "local_batch_size": LOCAL_BATCH_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
    },
    "optimizer": {
        "name": "adamw",
        "peak_lr": 1e-3,
        "weight_decay": 0.1,
        "betas": [0.9, 0.95],
        "warmup_ratio": 0.05,
        "schedule": "cosine",
        "floor_lr": 1e-5,
    },
    "steps": {
        "leg": LEG_STEPS,
        "monolithic": MONOLITHIC_STEPS,
        "leg_warmup": int(LEG_STEPS * 0.05),
        "monolithic_warmup": int(MONOLITHIC_STEPS * 0.05),
    },
    "sft_loss_weights": {
        "p1": P1_SFT_LOSS_WEIGHT,
        "p2": P2_SFT_LOSS_WEIGHT,
        "monolithic": MONOLITHIC_SFT_LOSS_WEIGHT,
    },
    "rl": {
        "e1_steps": [1_500, 1_500],
        "e2_steps": 3_000,
        "e3_steps": 3_000,
        "filters": ["unfiltered", "miles-dynamic"],
        "save_interval": 40,
    },
}
CONTRACT_SHA256 = hashlib.sha256(_canonical_json(CONTRACT)).hexdigest()


def _validate_static_contract() -> None:
    if PRETRAIN_TOTAL_TOKENS != 2 * PRETRAIN_LEG_TOKENS:
        raise RuntimeError("20B stream must be exactly two 10B legs")
    if (WORLD_SIZE, LOCAL_BATCH_SIZE, GRADIENT_ACCUMULATION) != (8, 21, 1):
        raise RuntimeError("Production topology drifted")
    if LEG_PT_RECORDS != 3_255_209:
        raise RuntimeError("10B packed-record accounting drifted")
    if (
        LEG_PT_RECORDS + 38_858 + P1_PADDING_RECORDS
        != LEG_STEPS * GLOBAL_BATCH_SIZE
    ):
        raise RuntimeError("P1 record accounting drifted")
    if (
        LEG_PT_RECORDS + 38_859 + P2_PADDING_RECORDS
        != LEG_STEPS * GLOBAL_BATCH_SIZE
    ):
        raise RuntimeError("P2 record accounting drifted")


_validate_static_contract()


# ---------------------------------------------------------------------------
# Local source identity and Modal resources
# ---------------------------------------------------------------------------

REPO_DIR = Path(__file__).resolve().parent.parent


def _source_tree_sha256() -> str:
    paths: list[Path] = []
    for relative in ("config", "llm_tokens", "scripts", "training"):
        root = REPO_DIR / relative
        if root.is_dir():
            paths.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}
            )
    paths.append(Path(__file__).resolve())
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda item: str(item.relative_to(REPO_DIR))):
        relative = str(path.relative_to(REPO_DIR))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


_COMPUTED_SOURCE_TREE_SHA256 = _source_tree_sha256()
SOURCE_TREE_SHA256 = os.environ.get(
    "CHESS_INTERLEAVE_20B_SOURCE_TREE_SHA256",
    _COMPUTED_SOURCE_TREE_SHA256,
).strip()
if not re.fullmatch(r"[0-9a-f]{64}", SOURCE_TREE_SHA256):
    raise RuntimeError("Invalid CHESS_INTERLEAVE_20B_SOURCE_TREE_SHA256")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("curl", "git")
    .pip_install(
        "torch==2.9.0",
        "accelerate==1.10.1",
        "transformers==4.57.0",
        "datasets==4.2.0",
        "huggingface-hub==0.35.3",
        "numpy==2.2.6",
        "safetensors==0.6.2",
        "pyarrow>=17.0.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "wandb>=0.19.0",
        "einops>=0.7.0",
        "tokenizers==0.22.1",
        "tqdm>=4.66.0",
        "chess>=1.11.0",
        "sentencepiece>=0.2.0",
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/root/chess",
            "WANDB_ENTITY": WANDB_ENTITY,
            "CHESS_INTERLEAVE_20B_SOURCE_TREE_SHA256": SOURCE_TREE_SHA256,
        }
    )
    .add_local_dir(str(REPO_DIR / "scripts"), remote_path="/root/chess/scripts")
    .add_local_dir(str(REPO_DIR / "training"), remote_path="/root/chess/training")
    .add_local_dir(str(REPO_DIR / "config"), remote_path="/root/chess/config")
    .add_local_dir(
        str(REPO_DIR / "llm_tokens"), remote_path="/root/chess/llm_tokens"
    )
)

data_volume = modal.Volume.from_name(
    "rl-reasoning-training-data", create_if_missing=False
)
checkpoint_volume = modal.Volume.from_name(
    "rl-reasoning-checkpoints", create_if_missing=False
)

app = modal.App(
    APP_NAME,
    image=image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
    },
)


# ---------------------------------------------------------------------------
# Immutable data preparation and validation
# ---------------------------------------------------------------------------

def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _self_hashed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = dict(value)
    payload[field] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def _validate_reused_source_manifest() -> None:
    if not SOURCE_MANIFEST_TEMPLATE.is_file():
        raise FileNotFoundError(SOURCE_MANIFEST_TEMPLATE)
    if _sha256_file(SOURCE_MANIFEST_TEMPLATE) != SOURCE_MANIFEST_FILE_SHA256:
        raise RuntimeError("Pinned source-manifest file hash drifted")
    payload = _load_json(SOURCE_MANIFEST_TEMPLATE)
    if (
        payload.get("schema") != "interleaved-source-shards-v1"
        or payload.get("manifest_hash") != SOURCE_MANIFEST_HASH
        or int(payload.get("total_tokens", -1)) != SOURCE_TOKENS
        or len(payload.get("shards", [])) != SOURCE_SHARDS
    ):
        raise RuntimeError("Pinned source-manifest contract drifted")
    for shard_number in (0, 24_774, 24_775, 47_089):
        shard = SOURCE_DIR / f"raw.{shard_number:04d}.npy"
        if not shard.is_file() or shard.stat().st_size <= 128:
            raise FileNotFoundError(shard)


def _validate_sft_cache() -> None:
    from training.interleaved_data import SFTCache

    cache = SFTCache.load(SFT_CACHE_DIR, verify_large_files=False)
    metadata = _load_json(SFT_CACHE_DIR / "metadata.json")
    delimiter_counts = metadata.get("supervised_delimiter_counts")
    if (
        cache.cache_hash != SFT_CACHE_HASH
        or cache.num_rows != SFT_ROWS
        or cache.sequence_length != SEQUENCE_LENGTH
        or cache.prompt_field != SFT_PROMPT_FIELD
        or cache.cot_field != SFT_COT_FIELD
        or int(metadata.get("supervised_targets", -1)) != SFT_TARGETS_TOTAL
        or int(metadata.get("supervised_unk_targets", -1)) != 0
        or not isinstance(delimiter_counts, Mapping)
        or int(delimiter_counts.get("</T>", -1)) != SFT_ROWS
        or int(delimiter_counts.get("<call_env>", -1)) != 187_354
    ):
        raise RuntimeError("Pinned cleaned SFT cache contract drifted")


def _validate_manifest_set() -> dict[str, Any]:
    from training.interleaved_data import LegManifest, PretrainSelection

    payload = _load_json(MANIFEST_SET_PATH)
    unhashed = {
        key: value
        for key, value in payload.items()
        if key != "manifest_set_hash"
    }
    if payload.get("manifest_set_hash") != hashlib.sha256(
        _canonical_json(unhashed)
    ).hexdigest():
        raise RuntimeError("Manifest-set self hash drifted")
    if (
        payload.get("experiment_version") != DATA_ARTIFACT_VERSION
        or int(payload.get("pretrain_tokens", -1)) != PRETRAIN_TOTAL_TOKENS
        or int(payload.get("sft_rows", -1)) != SFT_ROWS
        or payload.get("source_manifest_hash") != SOURCE_MANIFEST_HASH
        or payload.get("selection_hash") != EXPECTED_SELECTION_HASH
        or payload.get("sft_cache_hash") != SFT_CACHE_HASH
    ):
        raise RuntimeError("20B manifest-set identity drifted")

    selection = PretrainSelection.load(PRETRAIN_SELECTION_PATH)
    if (
        selection.target_tokens != PRETRAIN_TOTAL_TOKENS
        or selection.source_tokens != PRETRAIN_TOTAL_TOKENS + 1
        or selection.selection_hash != EXPECTED_SELECTION_HASH
        or _sha256_file(PRETRAIN_SELECTION_PATH)
        != EXPECTED_SELECTION_FILE_SHA256
    ):
        raise RuntimeError("20B deterministic selection drifted")

    expected = {
        "p1": {
            "path": P1_METADATA_PATH,
            "target_start": 0,
            "sft_records": 38_858,
            "sft_targets": SFT_TARGETS_P1,
            "padding": P1_PADDING_RECORDS,
            "order_hash": EXPECTED_P1_ORDER_SHA256,
            "metadata_hash": EXPECTED_P1_METADATA_HASH,
        },
        "p2": {
            "path": P2_METADATA_PATH,
            "target_start": PRETRAIN_LEG_TOKENS,
            "sft_records": 38_859,
            "sft_targets": SFT_TARGETS_P2,
            "padding": P2_PADDING_RECORDS,
            "order_hash": EXPECTED_P2_ORDER_SHA256,
            "metadata_hash": EXPECTED_P2_METADATA_HASH,
        },
    }
    for name, spec in expected.items():
        manifest = LegManifest.load(spec["path"])
        if (
            manifest.leg != name
            or manifest.target_start != spec["target_start"]
            or manifest.target_count != PRETRAIN_LEG_TOKENS
            or manifest.pretrain_records != LEG_PT_RECORDS
            or manifest.sft_records != spec["sft_records"]
            or manifest.sft_supervised_targets != spec["sft_targets"]
            or manifest.padding_records != spec["padding"]
            or manifest.world_size != WORLD_SIZE
            or manifest.local_batch_size != LOCAL_BATCH_SIZE
            or manifest.physical_steps != LEG_STEPS
            or manifest.total_steps != LEG_STEPS
            or manifest.order_sha256 != spec["order_hash"]
            or manifest.metadata_hash != spec["metadata_hash"]
        ):
            raise RuntimeError(f"{name} manifest accounting drifted")

    exp2 = _load_json(EXP2_METADATA_PATH)
    if (
        exp2.get("schema") != "interleaved-composite-manifest-v1"
        or exp2.get("name") != "p1+p2"
        or int(exp2.get("total_steps", -1)) != MONOLITHIC_STEPS
        or exp2.get("metadata_hash") != EXPECTED_EXP2_METADATA_HASH
    ):
        raise RuntimeError("E2 composite manifest drifted")
    return payload


def _prepare_data_impl() -> dict[str, Any]:
    from training.interleaved_data import (
        build_leg_manifests,
        build_manifest_set,
        build_pretrain_selection,
    )

    _validate_reused_source_manifest()
    _validate_sft_cache()
    if MANIFEST_SET_PATH.is_file():
        return _validate_manifest_set()

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(SOURCE_MANIFEST_TEMPLATE, SOURCE_MANIFEST_PATH)
    build_pretrain_selection(
        source_manifest_path=SOURCE_MANIFEST_PATH,
        output_path=PRETRAIN_SELECTION_PATH,
        target_tokens=PRETRAIN_TOTAL_TOKENS,
        seed=DATA_SEED,
    )
    build_leg_manifests(
        source_manifest_path=SOURCE_MANIFEST_PATH,
        selection_manifest_path=PRETRAIN_SELECTION_PATH,
        sft_cache_dir=SFT_CACHE_DIR,
        output_root=LEGS_ROOT,
        sequence_length=SEQUENCE_LENGTH,
        leg_target_tokens=PRETRAIN_LEG_TOKENS,
        world_size=WORLD_SIZE,
        local_batch_size=LOCAL_BATCH_SIZE,
        split_seed=DATA_SEED,
        p1_seed=P1_SHUFFLE_SEED,
        p2_seed=P2_SHUFFLE_SEED,
        canary_world_size=1,
        canary_local_batch_size=2,
        canary_total_steps=LEG_STEPS,
        expected_sft_supervised_targets=(
            SFT_TARGETS_P1,
            SFT_TARGETS_P2,
        ),
    )
    build_manifest_set(
        output_path=MANIFEST_SET_PATH,
        source_manifest_path=SOURCE_MANIFEST_PATH,
        selection_manifest_path=PRETRAIN_SELECTION_PATH,
        sft_cache_dir=SFT_CACHE_DIR,
        legs_root=LEGS_ROOT,
        experiment_version=DATA_ARTIFACT_VERSION,
        source_repo=SOURCE_REPO,
        source_revision=SOURCE_REVISION,
        sft_repo=SFT_REPO,
        sft_revision=SFT_REVISION,
        pretrain_tokens=PRETRAIN_TOTAL_TOKENS,
        sft_rows=SFT_ROWS,
    )
    data_volume.commit()
    return _validate_manifest_set()


# ---------------------------------------------------------------------------
# Training plans and immutable execution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainPlan:
    stage: str
    manifest_leg: str
    output_dir: str
    run_name: str
    total_steps: int
    arc_steps: tuple[int, ...]
    manifest_metadata: str
    manifest_order: str
    sft_loss_weight: float
    num_gpus: int = WORLD_SIZE
    local_batch_size: int = LOCAL_BATCH_SIZE
    weights_only: str | None = None
    init_fingerprint: str | None = None
    max_steps: int | None = None
    canary: bool = False


def _validate_run_id(run_id: str) -> str:
    value = run_id.strip()
    if not _RUN_ID_RE.fullmatch(value):
        raise ValueError(
            "run_id must be 1-80 lowercase path-safe characters"
        )
    return value


def _checkpoint_fingerprint(path: Path) -> str:
    resolved = path.resolve(strict=True)
    allowed = Path("/checkpoints").resolve()
    if not resolved.is_relative_to(allowed):
        raise ValueError("Initialization checkpoint must be under /checkpoints")
    files = [
        child
        for child in resolved.rglob("*")
        if child.is_file()
        and (
            child.name.endswith(".safetensors")
            or child.name
            in {
                "config.json",
                "generation_config.json",
                "tokenizer.py",
                "tokenizer_config.json",
                "vocab.json",
                "interleaved_training_state.json",
            }
        )
    ]
    if not files or not any(
        child.name.endswith(".safetensors") for child in files
    ):
        raise ValueError(f"Not a complete HF checkpoint: {resolved}")
    digest = hashlib.sha256()
    for child in sorted(files, key=lambda item: str(item.relative_to(resolved))):
        relative = str(child.relative_to(resolved))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with child.open("rb") as handle:
            while chunk := handle.read(8 << 20):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _plan(stage: str, run_id: str = "", init_checkpoint: str = "") -> TrainPlan:
    if stage == "p1":
        return TrainPlan(
            stage="p1-20b",
            manifest_leg="p1",
            output_dir=str(CHECKPOINT_ROOT / "p1_shared"),
            run_name=f"20b-p1-shared-{EXPERIMENT_VERSION}",
            total_steps=LEG_STEPS,
            arc_steps=(LEG_STEPS,),
            manifest_metadata=str(P1_METADATA_PATH),
            manifest_order=str(P1_ORDER_PATH),
            sft_loss_weight=P1_SFT_LOSS_WEIGHT,
        )
    if stage == "exp2":
        return TrainPlan(
            stage="exp2-20b-one-cosine",
            manifest_leg="p1+p2",
            output_dir=str(CHECKPOINT_ROOT / "exp2_monolithic"),
            run_name=f"20b-exp2-one-cosine-{EXPERIMENT_VERSION}",
            total_steps=MONOLITHIC_STEPS,
            arc_steps=(MONOLITHIC_STEPS,),
            manifest_metadata=str(EXP2_METADATA_PATH),
            manifest_order="p1+p2",
            sft_loss_weight=MONOLITHIC_SFT_LOSS_WEIGHT,
        )
    if stage == "canary":
        return TrainPlan(
            stage="canary-20b",
            manifest_leg="canary",
            output_dir=str(
                CHECKPOINT_ROOT / "canary" / "mixed-step1-source-v2"
            ),
            run_name=f"20b-mixed-canary-source-v2-{EXPERIMENT_VERSION}",
            total_steps=LEG_STEPS,
            arc_steps=(LEG_STEPS,),
            manifest_metadata=str(CANARY_METADATA_PATH),
            manifest_order=str(LEGS_ROOT / "canary" / "order.npy"),
            sft_loss_weight=P1_SFT_LOSS_WEIGHT,
            num_gpus=1,
            local_batch_size=2,
            max_steps=1,
            canary=True,
        )
    if stage == "p2":
        safe_id = _validate_run_id(run_id)
        checkpoint = Path(init_checkpoint)
        fingerprint = _checkpoint_fingerprint(checkpoint)
        suffix = fingerprint[:12]
        return TrainPlan(
            stage="p2-20b",
            manifest_leg="p2",
            output_dir=str(
                CHECKPOINT_ROOT / "p2" / f"{safe_id}-from-{suffix}"
            ),
            run_name=f"20b-p2-{safe_id}-from-{suffix}",
            total_steps=LEG_STEPS,
            arc_steps=(LEG_STEPS,),
            manifest_metadata=str(P2_METADATA_PATH),
            manifest_order=str(P2_ORDER_PATH),
            sft_loss_weight=P2_SFT_LOSS_WEIGHT,
            weights_only=str(checkpoint),
            init_fingerprint=fingerprint,
        )
    raise ValueError(f"Unknown training stage: {stage}")


def _manifest_hash(payload: Mapping[str, Any], plan: TrainPlan) -> str:
    manifests = payload.get("manifests")
    if not isinstance(manifests, Mapping):
        raise RuntimeError("Manifest set lacks manifests")
    entry = manifests.get(plan.manifest_leg)
    if not isinstance(entry, Mapping):
        raise RuntimeError(f"Missing manifest {plan.manifest_leg}")
    value = entry.get("sha256")
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RuntimeError("Invalid manifest file hash")
    return value


def _overrides(plan: TrainPlan, manifest_hash: str) -> list[str]:
    values = [
        f"training.output_dir={plan.output_dir}",
        f"training.run_name={plan.run_name}",
        "training.seed=42",
        f"training.local_batch_size={plan.local_batch_size}",
        "training.gradient_accumulation_steps=1",
        f"training.total_steps={plan.total_steps}",
        f"training.arc_steps={list(plan.arc_steps)}",
        "training.reset_optimizer_between_arcs=true",
        "training.mixed_precision=bf16",
        f"training.sft_loss_weight={plan.sft_loss_weight}",
        "training.optimizer.lr=1e-3",
        "training.optimizer.weight_decay=0.1",
        "training.optimizer.betas=[0.9,0.95]",
        "training.scheduler.warmup_ratio=0.05",
        "training.scheduler.eta_min=1e-5",
        "training.torch_compile=none",
        "model.attn_implementation=sdpa",
        "model.flash_attention_version=2.8.3",
        f"data.source_root={SOURCE_DIR}",
        f"data.source_manifest_path={SOURCE_MANIFEST_PATH}",
        f"data.selection_manifest_path={PRETRAIN_SELECTION_PATH}",
        f"data.sft_cache_dir={SFT_CACHE_DIR}",
        f"data.leg_manifest_path={plan.manifest_metadata}",
        f"data.expected_manifest_hash={manifest_hash}",
        "data.num_workers=8",
        "training.num_workers=8",
        "training.persistent_workers=true",
        "training.save_interval=200",
        "training.log_interval=10",
        "logging.backend=none",
        f"logging.project={WANDB_PROJECT}",
        f"logging.entity={WANDB_ENTITY}",
        f"provenance.experiment_version={EXPERIMENT_VERSION}",
        f"provenance.data_artifact_version={DATA_ARTIFACT_VERSION}",
        f"provenance.contract_sha256={CONTRACT_SHA256}",
        f"provenance.source_tree_sha256={SOURCE_TREE_SHA256}",
        f"provenance.source_repo={SOURCE_REPO}",
        f"provenance.source_revision={SOURCE_REVISION}",
        f"provenance.sft_repo={SFT_REPO}",
        f"provenance.sft_revision={SFT_REVISION}",
        f"provenance.sft_loss_weight={plan.sft_loss_weight}",
        "provenance.sft_response_normalization="
        "strip-numeric-verify-score-pairs-normalize-whitespace-v1",
        "provenance.sft_supervised_unk_policy=reject-supervised-unk-v1",
    ]
    if plan.init_fingerprint:
        values.append(
            f"provenance.init_checkpoint_sha256={plan.init_fingerprint}"
        )
    if plan.canary:
        values.extend(
            [
                "training.max_steps=1",
                "training.allow_topology_override=true",
                "training.persistent_workers=false",
                "training.save_interval=1",
                "training.log_interval=1",
                "data.num_workers=0",
                "training.num_workers=0",
            ]
        )
    return values


def _command(
    plan: TrainPlan, manifest_hash: str, resume: str | None
) -> list[str]:
    command = ["accelerate", "launch"]
    if plan.num_gpus > 1:
        command.append("--multi_gpu")
    command.extend(
        [
            "--num_processes",
            str(plan.num_gpus),
            "--mixed_precision",
            "bf16",
            "--main_process_port",
            "29721" if plan.canary else "29711",
            TRAIN_CLI,
            "--config",
            BASE_CONFIG,
            "--override",
            *_overrides(plan, manifest_hash),
        ]
    )
    if resume:
        command.extend(["--resume", resume])
    elif plan.weights_only:
        command.extend(["--weights-only", plan.weights_only])
    return command


def _validate_run_config(
    plan: TrainPlan, manifest_hash: str
) -> Mapping[str, Any]:
    import yaml

    path = Path(plan.output_dir) / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        ("training", "output_dir"): plan.output_dir,
        ("training", "run_name"): plan.run_name,
        ("training", "total_steps"): plan.total_steps,
        ("training", "arc_steps"): list(plan.arc_steps),
        ("training", "local_batch_size"): plan.local_batch_size,
        ("training", "sft_loss_weight"): plan.sft_loss_weight,
        ("data", "expected_manifest_hash"): manifest_hash,
        ("provenance", "experiment_version"): EXPERIMENT_VERSION,
        ("provenance", "contract_sha256"): CONTRACT_SHA256,
        ("provenance", "source_tree_sha256"): SOURCE_TREE_SHA256,
    }
    for keys, expected_value in expected.items():
        current: Any = payload
        for key in keys:
            if not isinstance(current, Mapping) or key not in current:
                raise RuntimeError(f"Config lacks {'.'.join(keys)}")
            current = current[key]
        if keys == ("training", "sft_loss_weight"):
            matches = float(current) == float(expected_value)
        else:
            matches = current == expected_value
        if not matches:
            raise RuntimeError(
                f"Immutable run drift at {'.'.join(keys)}: "
                f"{current!r} != {expected_value!r}"
            )
    return payload


def _resolve_run(plan: TrainPlan, manifest_hash: str) -> tuple[str, str | None]:
    root = Path(plan.output_dir)
    if not root.exists() or not any(root.iterdir()):
        return "fresh", None
    _validate_run_config(plan, manifest_hash)
    expected_step = plan.max_steps or plan.total_steps
    final_state = root / "final" / "interleaved_training_state.json"
    if final_state.is_file():
        state = _load_json(final_state)
        if (
            int(state.get("global_step", -1)) != expected_step
            or int(state.get("manifest_cursor", -1)) != expected_step
            or state.get("manifest_hash") != manifest_hash
        ):
            raise RuntimeError("Completed run state does not match its plan")
        if not list((root / "final").glob("model*.safetensors")):
            raise RuntimeError("Completed run lacks HF weights")
        return "complete", str(root / "final")
    latest_state = root / "latest" / "trainer_state.json"
    if latest_state.is_file():
        state = _load_json(latest_state)
        if (
            state.get("manifest_hash") != manifest_hash
            or state.get("arc_steps") != list(plan.arc_steps)
            or float(state.get("sft_loss_weight", -1))
            != float(plan.sft_loss_weight)
        ):
            raise RuntimeError("Resume state does not match its plan")
        return "resume", str(root / "latest")
    allowed = {"config.yaml", "metrics.jsonl"}
    unexpected = [child.name for child in root.iterdir() if child.name not in allowed]
    if unexpected:
        raise RuntimeError(f"Unrecognized partial run artifacts: {unexpected}")
    return "fresh", None


def _validate_canary_gate(manifest_set: Mapping[str, Any]) -> None:
    if not CANARY_GATE_PATH.is_file():
        raise RuntimeError("20B mixed canary gate is missing")
    gate = _load_json(CANARY_GATE_PATH)
    recorded = gate.pop("gate_sha256", None)
    if recorded != hashlib.sha256(_canonical_json(gate)).hexdigest():
        raise RuntimeError("Canary gate self hash drifted")
    if (
        gate.get("schema") != "interleaved-20b-mixed-canary-gate-v2"
        or gate.get("decision") != "pass"
        or gate.get("contract_sha256") != CONTRACT_SHA256
        or gate.get("source_tree_sha256") != SOURCE_TREE_SHA256
        or gate.get("manifest_set_hash")
        != manifest_set.get("manifest_set_hash")
    ):
        raise RuntimeError("Canary gate is not valid for this 20B launch")


def _run_training(plan: TrainPlan) -> str:
    data_volume.reload()
    checkpoint_volume.reload()
    if _sha256_file(Path("/root/chess") / BASE_CONFIG) != BASE_CONFIG_SHA256:
        raise RuntimeError("Base config hash drifted inside Modal image")
    _validate_reused_source_manifest()
    _validate_sft_cache()
    manifest_set = _validate_manifest_set()
    if not plan.canary:
        _validate_canary_gate(manifest_set)
    manifest_hash = _manifest_hash(manifest_set, plan)
    state, resume = _resolve_run(plan, manifest_hash)
    if state == "complete":
        return plan.output_dir

    command = _command(plan, manifest_hash, resume)
    print("[20b-interleave] " + " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd="/root/chess",
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
    )
    checkpoint_volume.commit()
    if result.returncode != 0:
        raise RuntimeError(
            f"{plan.stage} failed with exit code {result.returncode}"
        )
    final_state = Path(plan.output_dir) / "final" / "interleaved_training_state.json"
    if not final_state.is_file():
        raise RuntimeError("Training returned without a final HF export")
    return plan.output_dir


@app.function(
    cpu=16.0,
    memory=64 * 1024,
    timeout=60 * 60 * 4,
    retries=0,
)
def prepare_data() -> str:
    data_volume.reload()
    payload = _prepare_data_impl()
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return str(MANIFEST_SET_PATH)


@app.function(
    gpu="H100:1",
    cpu=8.0,
    memory=32 * 1024,
    timeout=60 * 60 * 2,
    retries=0,
)
def train_canary() -> str:
    plan = _plan("canary")
    output = _run_training(plan)
    manifest_set = _validate_manifest_set()
    gate = {
        "schema": "interleaved-20b-mixed-canary-gate-v2",
        "decision": "pass",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": CONTRACT_SHA256,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "manifest_set_hash": manifest_set["manifest_set_hash"],
        "canary_output": output,
        "canary_steps": 1,
    }
    gate["gate_sha256"] = hashlib.sha256(_canonical_json(gate)).hexdigest()
    _atomic_json(CANARY_GATE_PATH, gate)
    data_volume.commit()
    return output


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}",
    cpu=32.0,
    memory=128 * 1024,
    timeout=60 * 60 * 48,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
    max_containers=4,
)
def train_production(
    stage: str,
    run_id: str = "",
    init_checkpoint: str = "",
) -> str:
    if stage in {"p1", "exp2"}:
        if run_id or init_checkpoint:
            raise ValueError(f"{stage} has a fixed immutable identity")
    elif stage == "p2":
        if not run_id or not init_checkpoint:
            raise ValueError("p2 requires run_id and init_checkpoint")
    else:
        raise ValueError(f"Unsupported production stage: {stage}")
    return _run_training(_plan(stage, run_id, init_checkpoint))


def _dry_plan(
    action: str, run_id: str, init_checkpoint: str
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "action": action,
        "app": APP_NAME,
        "experiment_version": EXPERIMENT_VERSION,
        "data_artifact_version": DATA_ARTIFACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "artifact_root": str(ARTIFACT_ROOT),
        "checkpoint_root": str(CHECKPOINT_ROOT),
    }
    if action in {"p1", "exp2", "canary", "p2"}:
        if action == "p2" and not Path(init_checkpoint).exists():
            payload["deferred_init_checkpoint"] = init_checkpoint
        else:
            plan = _plan(action, run_id, init_checkpoint)
            payload["plan"] = {
                "stage": plan.stage,
                "run_name": plan.run_name,
                "output_dir": plan.output_dir,
                "manifest_leg": plan.manifest_leg,
                "total_steps": plan.total_steps,
                "arc_steps": list(plan.arc_steps),
                "gpus": (
                    "H100:1"
                    if plan.canary
                    else f"{GPU_TYPE}:{WORLD_SIZE}"
                ),
                "local_batch_size": plan.local_batch_size,
                "sft_loss_weight": plan.sft_loss_weight,
                "weights_only": plan.weights_only,
            }
    elif action == "launch-initial":
        payload["plans"] = [
            _dry_plan("p1", "", "")["plan"],
            _dry_plan("exp2", "", "")["plan"],
        ]
    return payload


@app.local_entrypoint()
def main(
    action: str = "data-prep",
    run_id: str = "",
    init_checkpoint: str = "",
    dry_run: bool = False,
) -> None:
    action = action.strip().lower()
    allowed = {
        "data-prep",
        "canary",
        "p1",
        "exp2",
        "p2",
        "launch-initial",
    }
    if action not in allowed:
        raise ValueError(f"action must be one of {sorted(allowed)}")
    if dry_run:
        print(
            json.dumps(
                _dry_plan(action, run_id, init_checkpoint),
                indent=2,
                sort_keys=True,
            )
        )
        return

    if action == "data-prep":
        handle = prepare_data.spawn()
        calls = [{"stage": "data-prep", "function_call_id": handle.object_id}]
    elif action == "canary":
        handle = train_canary.spawn()
        calls = [{"stage": "canary", "function_call_id": handle.object_id}]
    elif action == "launch-initial":
        handles = [
            (stage, train_production.spawn(stage=stage))
            for stage in ("p1", "exp2")
        ]
        calls = [
            {"stage": stage, "function_call_id": handle.object_id}
            for stage, handle in handles
        ]
    elif action in {"p1", "exp2"}:
        handle = train_production.spawn(stage=action)
        calls = [{"stage": action, "function_call_id": handle.object_id}]
    else:
        handle = train_production.spawn(
            stage="p2",
            run_id=_validate_run_id(run_id),
            init_checkpoint=init_checkpoint,
        )
        calls = [{"stage": "p2", "function_call_id": handle.object_id}]

    for call in calls:
        if not _CALL_ID_RE.fullmatch(call["function_call_id"]):
            raise RuntimeError(f"Invalid Modal call id: {call}")
    print(
        json.dumps(
            {
                "action": action,
                "experiment_version": EXPERIMENT_VERSION,
                "contract_sha256": CONTRACT_SHA256,
                "calls": calls,
            },
            indent=2,
            sort_keys=True,
        )
    )
