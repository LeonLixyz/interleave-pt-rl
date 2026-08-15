"""Configuration-backed initial PT wave for the context-2,048 experiment matrix.

This launcher owns the two bit-identical PT parents shared by the seven 5B
experiments:

* 5B PT targets plus all three SFT exposure copies under one cosine schedule.
* the first 2.5B PT targets plus the first half of the SFT exposure set.

Downstream RL, trace training, and second-PT-stage launchers consume these
immutable HF exports.  The scientific graphs themselves live in YAML; this
file contains only reusable data preparation, training, gating, and exact-once
infrastructure.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import secrets
import signal
import subprocess
import sys
import sysconfig
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import modal

_IN_MODAL_IMAGE = bool(os.environ.get("MATRIX_SOURCE_TREE_SHA256"))
_LOCAL_PRETRAIN_ROOT = (
    Path("/root/chess")
    if _IN_MODAL_IMAGE
    else Path(__file__).resolve().parent.parent
)
if str(_LOCAL_PRETRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOCAL_PRETRAIN_ROOT))

from training.immutable_checkpoint import (
    LATEST_CHECKPOINT_POINTER,
    checkpoint_volume_commit_lock,
    inspect_accelerator_checkpoint_fp32,
    resolve_resume_checkpoint,
    validate_checkpoint_run_root,
    validate_completed_hf_export,
)
from training.interleaved_hf_trainer import (
    INITIAL_LAUNCH_COMMAND_ENV,
    INITIAL_LAUNCH_COMMAND_SHA256_ENV,
)
from training.tokenizer_contract import (
    EXPECTED_VOCAB_85,
    validate_hf_tokenizer_contract,
)


EXPERIMENT_VERSION = "context2048_pt_sft_trace_rl_fp32_master_v5_20260815"
APP_NAME = "chess-context2048-configured-pt-matrix-v5"
WANDB_ENTITY = "jingyanshen-new-york-university"
WANDB_PROJECT = "chess-47m-context2048-pt-sft-trace-rl-v1"
WANDB_SECRET = "wandb-interleave-pt-rl"

CONTEXT_LENGTH = 2_048
WORLD_SIZE = 8
LOCAL_BATCH_SIZE = 16
GLOBAL_SEQUENCES = WORLD_SIZE * LOCAL_BATCH_SIZE
GLOBAL_TOKEN_SLOTS = GLOBAL_SEQUENCES * CONTEXT_LENGTH
CANARY_TOTAL_STEPS = 2
GPU_TYPE = "H200"
SEED = 42

PT_FULL_TARGETS = 5_000_000_000
PT_HALF_TARGETS = 2_500_000_000
PT_FULL_RECORDS = math.ceil(PT_FULL_TARGETS / CONTEXT_LENGTH)
PT_HALF_RECORDS = math.ceil(PT_HALF_TARGETS / CONTEXT_LENGTH)
SFT_ROWS = 77_717
SFT_TOTAL_EXPOSURES = 3 * SFT_ROWS
SFT_FIRST_EXPOSURES = 116_575
SFT_SECOND_EXPOSURES = 116_576
SFT_TARGETS_PER_COPY = 52_482_753

SOURCE_REPO = "chess-pre-to-post/pretrain_v1_20b"
SOURCE_REVISION = "07dd1b7090ca5f0fb05ef624c26b20bff19483c8"
SOURCE_DIR = Path("/data/pretrain_v1_20b")
SOURCE_TEMPLATE_ROOT = Path(
    "/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate"
)
SOURCE_MANIFEST_TEMPLATE = SOURCE_TEMPLATE_ROOT / "source_manifest.json"
SOURCE_MANIFEST_FILE_SHA256 = (
    "7f144d2329628759f2529540bfb9b10692e374d0c8b1933ec43c7c634b979253"
)
SFT_CACHE_DIR = Path(
    "/data/context2048_vocab_mixing_fp32_master_v13_20260813/"
    "sft_cache_context2048"
)
SFT_CACHE_HASH = (
    "6e5b0553366d51ec3c95cb606b063919eed83efb79d11b187ba7d318b0fd60d5"
)

ARTIFACT_ROOT = Path(f"/data/{EXPERIMENT_VERSION}")
SOURCE_MANIFEST_PATH = ARTIFACT_ROOT / "source_manifest.json"
SELECTION_PATH = ARTIFACT_ROOT / "pretrain_selection_5b.json"
MANIFEST_SET_PATH = ARTIFACT_ROOT / "initial_pt_manifest_set.json"
GATE_ROOT = ARTIFACT_ROOT / "gates"
CHECKPOINT_ROOT = Path(f"/checkpoints/{EXPERIMENT_VERSION}")
DURABLE_LAUNCH_ROOT = CHECKPOINT_ROOT / "_launch_ledger"

BASE_CONFIG = "config/configs/interleaved_50m/context2048_vocab_mixing.yaml"
TRAIN_CLI = "scripts/train/train_interleaved_hf.py"
EXPECTED_TOKEN_IDS = {
    "<bos>": 0,
    "<eos>": 1,
    "<unk>": 2,
    "<T>": 81,
    "</T>": 82,
    "<sep>": 83,
    "<call_env>": 84,
}
EXPECTED_PRECISION = {
    "master_parameter_dtype": "float32",
    "optimizer_state_dtype": "float32",
    "forward_backward_dtype": "bfloat16",
    "gradient_dtype": "float32",
    "hf_export_dtype": "float32",
}
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

PRETRAIN_ROOT = _LOCAL_PRETRAIN_ROOT
REPOSITORY_ROOT = Path("/root") if _IN_MODAL_IMAGE else PRETRAIN_ROOT.parents[1]
CONFIG_SOURCE_ROOT = (
    Path("/root/experiment_configs")
    if _IN_MODAL_IMAGE
    else REPOSITORY_ROOT / "experiments/context2048_pt_sft_trace_rl_v1"
)
CONFIG_LOADER_PATH = (
    Path("/root/context2048_matrix_config.py")
    if _IN_MODAL_IMAGE
    else REPOSITORY_ROOT / "chess/experiments/context2048_matrix_config.py"
)
LOCAL_RECOVERY_ROOT = (
    PRETRAIN_ROOT / ".launch-recovery" / EXPERIMENT_VERSION
)

CUDA_BASE_IMAGE = (
    "nvidia/cuda:12.8.0-devel-ubuntu22.04@"
    "sha256:09d8951b943dee03cf8fc841b6ea1f201ad33f82f76567171394853c0f494054"
)
PINNED_PIP_PACKAGES = (
    "torch==2.9.0",
    "accelerate==1.10.1",
    "transformers==4.57.0",
    "datasets==4.2.0",
    "huggingface-hub==0.35.3",
    "numpy==2.2.6",
    "safetensors==0.6.2",
    "pyarrow==23.0.1",
    "pandas==3.0.1",
    "pyyaml==6.0.3",
    "omegaconf==2.3.0",
    "wandb==0.25.0",
    "einops==0.8.1",
    "tokenizers==0.22.1",
    "tqdm==4.67.3",
    "chess==1.11.2",
    "sentencepiece==0.2.1",
    "typing-extensions==4.16.0",
)
PINNED_RUNTIME_VERSIONS = {
    item.split("==", 1)[0]: item.split("==", 1)[1]
    for item in PINNED_PIP_PACKAGES
}
RUNTIME_SITE_PACKAGES = sysconfig.get_paths()["purelib"]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _source_entries(
    *,
    pretrain_root: Path,
    repository_root: Path,
    launcher_path: Path,
) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for relative in ("config", "llm_tokens", "scripts", "training"):
        root = pretrain_root / relative
        entries.extend(
            (
                f"chess/pretrain-sft/{path.relative_to(pretrain_root).as_posix()}",
                path,
            )
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    entries.append(
        (
            "chess/pretrain-sft/modal_scripts/modal_context2048_pt_matrix.py",
            launcher_path,
        )
    )
    loader = repository_root / "chess/experiments/context2048_matrix_config.py"
    if not loader.is_file():
        loader = repository_root / "context2048_matrix_config.py"
    entries.append(("chess/experiments/context2048_matrix_config.py", loader))
    config_root = repository_root / "experiments/context2048_pt_sft_trace_rl_v1"
    if not config_root.is_dir():
        config_root = repository_root / "experiment_configs"
    entries.extend(
        (
            f"experiments/context2048_pt_sft_trace_rl_v1/{path.relative_to(config_root).as_posix()}",
            path,
        )
        for path in config_root.rglob("*.yaml")
    )
    return sorted(entries, key=lambda item: item[0])


def _source_tree_sha256(
    *,
    pretrain_root: Path = PRETRAIN_ROOT,
    repository_root: Path = REPOSITORY_ROOT,
    launcher_path: Path = Path(__file__).resolve(),
) -> str:
    digest = hashlib.sha256()
    seen: set[str] = set()
    for label, path in _source_entries(
        pretrain_root=pretrain_root,
        repository_root=repository_root,
        launcher_path=launcher_path,
    ):
        if label in seen:
            raise RuntimeError(f"duplicate source label: {label}")
        seen.add(label)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


SOURCE_TREE_SHA256 = (
    os.environ.get("MATRIX_SOURCE_TREE_SHA256", "").strip()
    or _source_tree_sha256()
)

image = (
    modal.Image.from_registry(CUDA_BASE_IMAGE, add_python="3.11")
    .apt_install("curl", "git")
    .pip_install(*PINNED_PIP_PACKAGES)
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/root/chess",
            "TOKENIZERS_PARALLELISM": "false",
            "MATRIX_SOURCE_TREE_SHA256": SOURCE_TREE_SHA256,
            "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG,
        }
    )
    .add_local_dir(str(PRETRAIN_ROOT / "scripts"), remote_path="/root/chess/scripts")
    .add_local_dir(str(PRETRAIN_ROOT / "training"), remote_path="/root/chess/training")
    .add_local_dir(str(PRETRAIN_ROOT / "config"), remote_path="/root/chess/config")
    .add_local_dir(str(PRETRAIN_ROOT / "llm_tokens"), remote_path="/root/chess/llm_tokens")
    .add_local_file(str(Path(__file__).resolve()), remote_path="/root/matrix_launcher.py")
    .add_local_file(str(CONFIG_LOADER_PATH), remote_path="/root/context2048_matrix_config.py")
    .add_local_dir(str(CONFIG_SOURCE_ROOT), remote_path="/root/experiment_configs")
)

data_volume = modal.Volume.from_name(
    "rl-reasoning-training-data", create_if_missing=False
)
checkpoint_volume = modal.Volume.from_name(
    "rl-reasoning-checkpoints", create_if_missing=False
)
claim_store = modal.Dict.from_name(
    "chess-ctx2048-configured-pt-claims-"
    + hashlib.sha256(EXPERIMENT_VERSION.encode()).hexdigest()[:16],
    create_if_missing=True,
)
app = modal.App(
    APP_NAME,
    image=image,
    secrets=[modal.Secret.from_name(WANDB_SECRET)],
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
)


INITIAL_STAGES: dict[str, dict[str, Any]] = {
    "pt5b_sft3_full": {
        "description": "5B PT targets plus all 233151 ordinary SFT exposures",
        "manifest": "pt5b_sft3_full",
        "run_name": "ctx2048-pt5b-sft3-uniform-mixed-one-decay-fp32-master-v1",
        "dependent_config_files": (
            "5b/01_one_pt_decay_rl3000.yaml",
            "5b/03_full_pt_shuffled_trace_rl1500x2.yaml",
            "5b/04_full_pt_chronological_trace_rl1500x2.yaml",
        ),
    },
    "pt2p5b_sft3_first": {
        "description": "first 2.5B PT targets plus first 116575 ordinary SFT exposures",
        "manifest": "pt2p5b_sft3_first",
        "run_name": "ctx2048-pt2p5b-sft3-first-half-uniform-mixed-fp32-master-v1",
        "dependent_config_files": (
            "5b/02_two_pt_decays_rl3000.yaml",
            "5b/05_split_pt_shuffled_trace_rl1500x2.yaml",
            "5b/06_split_pt_chronological_trace_rl1500x2.yaml",
            "5b/07_split_pt_trace_mixed_rl1500x2.yaml",
        ),
    },
}


def _runtime_identity() -> dict[str, Any]:
    package_versions = {
        name: importlib.metadata.version(name)
        for name in sorted(PINNED_RUNTIME_VERSIONS)
    }
    if package_versions != PINNED_RUNTIME_VERSIONS:
        raise RuntimeError("runtime Python package versions drifted")
    inventory: dict[str, str] = {}
    for distribution in importlib.metadata.distributions(
        path=[RUNTIME_SITE_PACKAGES]
    ):
        raw = distribution.metadata.get("Name")
        if not raw:
            raise RuntimeError("installed distribution has no Name")
        name = re.sub(r"[-_.]+", "-", str(raw)).lower()
        version = str(distribution.version)
        prior = inventory.setdefault(name, version)
        if prior != version:
            raise RuntimeError(f"duplicate runtime distribution: {name}")
    app_id = app.app_id
    image_id = image.object_id
    if not str(app_id).startswith("ap-") or not str(image_id).startswith("im-"):
        raise RuntimeError("Modal app/image identity is not hydrated")
    return {
        "app_name": APP_NAME,
        "app_id": app_id,
        "image_id": image_id,
        "base_image": CUDA_BASE_IMAGE,
        "modal_client_version": str(
            getattr(modal, "__version__", "")
            or importlib.metadata.version("modal")
        ),
        "python": sys.version,
        "packages": package_versions,
        "distribution_count": len(inventory),
        "distribution_inventory_sha256": _canonical_sha256(inventory),
    }


def _load_resolved_configs(
    config_root: Path = Path("/root/experiment_configs/5b"),
) -> dict[str, dict[str, Any]]:
    if str(Path("/root")) not in sys.path:
        sys.path.insert(0, "/root")
    from context2048_matrix_config import load_experiment_config

    rows = {
        str(path.relative_to(config_root.parent)): load_experiment_config(path)
        for path in sorted(config_root.glob("*.yaml"))
    }
    if len(rows) != 7:
        raise RuntimeError("deployed image does not contain exactly seven 5B configs")
    return rows


def _validate_uploaded_source() -> dict[str, Any]:
    observed = _source_tree_sha256(
        pretrain_root=Path("/root/chess"),
        repository_root=Path("/root"),
        launcher_path=Path("/root/matrix_launcher.py"),
    )
    expected = os.environ.get("MATRIX_SOURCE_TREE_SHA256", "")
    if observed != expected or observed != SOURCE_TREE_SHA256:
        raise RuntimeError(
            f"uploaded source hash drifted: {observed} != {expected} != {SOURCE_TREE_SHA256}"
        )
    configs = _load_resolved_configs()
    return {
        "source_tree_sha256": observed,
        "config_hashes": {
            name: value["resolved_config_sha256"]
            for name, value in configs.items()
        },
    }


def _pad_order(order: Any) -> tuple[Any, int]:
    import numpy as np
    from training.interleaved_data import PAD_RECORD

    padding = (-len(order)) % GLOBAL_SEQUENCES
    if padding:
        order = np.concatenate(
            (order, np.full(padding, PAD_RECORD, dtype="<i8"))
        )
    return order.astype("<i8", copy=False), int(padding)


def _stable_mixed_order(pt_order: Any, sft_order: Any, *, seed: int) -> Any:
    import numpy as np

    flags = np.concatenate(
        (
            np.zeros(len(pt_order), dtype=np.int8),
            np.ones(len(sft_order), dtype=np.int8),
        )
    )
    np.random.Generator(np.random.PCG64(seed)).shuffle(flags)
    order = np.empty(len(flags), dtype="<i8")
    order[flags == 0] = pt_order
    order[flags == 1] = sft_order
    return order


def _build_canary_mixed_order() -> Any:
    """Build two updates with PT and SFT examples on every rank.

    Production uses the ordinary seeded uniform placement.  This small order
    exists only to make the objective and process-boundary gate deterministic:
    every contiguous per-rank batch alternates PT and SFT rows.
    """

    import numpy as np

    order = np.empty(CANARY_TOTAL_STEPS * GLOBAL_SEQUENCES, dtype="<i8")
    pt_index = 0
    sft_index = 0
    for update in range(CANARY_TOTAL_STEPS):
        update_start = update * GLOBAL_SEQUENCES
        for rank in range(WORLD_SIZE):
            rank_start = update_start + rank * LOCAL_BATCH_SIZE
            for local_index in range(LOCAL_BATCH_SIZE):
                position = rank_start + local_index
                if local_index % 2 == 0:
                    order[position] = pt_index
                    pt_index += 1
                else:
                    order[position] = -(sft_index + 1)
                    sft_index += 1
    if pt_index != sft_index or pt_index + sft_index != len(order):
        raise AssertionError("canary PT/SFT row accounting drifted")
    return order


def _prepare_data() -> dict[str, Any]:
    import numpy as np
    from training.interleaved_data import (
        PretrainSelection,
        SFTCache,
        _sft_supervised_targets_per_row,
        _write_leg_manifest,
        build_pretrain_selection,
    )

    source = _validate_uploaded_source()
    if _sha256_file(SOURCE_MANIFEST_TEMPLATE) != SOURCE_MANIFEST_FILE_SHA256:
        raise RuntimeError("pinned source manifest file drifted")
    cache = SFTCache.load(SFT_CACHE_DIR, verify_large_files=True)
    if cache.cache_hash != SFT_CACHE_HASH or cache.num_rows != SFT_ROWS:
        raise RuntimeError("context-2048 SFT cache identity drifted")
    per_row_targets = _sft_supervised_targets_per_row(cache)
    if int(per_row_targets.sum(dtype=np.int64)) != SFT_TARGETS_PER_COPY:
        raise RuntimeError("SFT supervised target inventory drifted")

    if MANIFEST_SET_PATH.is_file():
        payload = _load_json(MANIFEST_SET_PATH)
        recorded = payload.pop("set_sha256", None)
        if recorded != _canonical_sha256(payload):
            raise RuntimeError("initial PT manifest set self hash drifted")
        payload["set_sha256"] = recorded
        if payload.get("source_tree_sha256") != SOURCE_TREE_SHA256:
            raise RuntimeError("prepared data belongs to a different source tree")
        return payload

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=False)
    SOURCE_MANIFEST_PATH.write_bytes(SOURCE_MANIFEST_TEMPLATE.read_bytes())
    build_pretrain_selection(
        SOURCE_MANIFEST_PATH,
        SELECTION_PATH,
        target_tokens=PT_FULL_TARGETS,
        seed=SEED,
    )
    selection = PretrainSelection.load(SELECTION_PATH)
    if selection.target_tokens != PT_FULL_TARGETS:
        raise RuntimeError("5B selection target count drifted")

    sft_exposure_order = np.concatenate(
        [
            -(
                np.random.Generator(np.random.PCG64(seed)).permutation(SFT_ROWS)
                .astype("<i8")
                + 1
            )
            for seed in (43, 44, 45)
        ]
    )
    if len(sft_exposure_order) != SFT_TOTAL_EXPOSURES:
        raise AssertionError("SFT exposure order length drifted")
    exposure_slices = {
        "full": sft_exposure_order,
        "first": sft_exposure_order[:SFT_FIRST_EXPOSURES],
        "second": sft_exposure_order[SFT_FIRST_EXPOSURES:],
    }
    if len(exposure_slices["second"]) != SFT_SECOND_EXPOSURES:
        raise AssertionError("SFT exposure split drifted")

    definitions = {
        "pt5b_sft3_full": {
            "target_start": 0,
            "target_count": PT_FULL_TARGETS,
            "pretrain_records": PT_FULL_RECORDS,
            "sft_order": exposure_slices["full"],
            "placement_seed": 520_100,
        },
        "pt2p5b_sft3_first": {
            "target_start": 0,
            "target_count": PT_HALF_TARGETS,
            "pretrain_records": PT_HALF_RECORDS,
            "sft_order": exposure_slices["first"],
            "placement_seed": 520_200,
        },
        "pt2p5b_sft3_second": {
            "target_start": PT_HALF_TARGETS,
            "target_count": PT_HALF_TARGETS,
            "pretrain_records": PT_HALF_RECORDS,
            "sft_order": exposure_slices["second"],
            "placement_seed": 520_201,
        },
    }
    manifests: dict[str, Any] = {}
    for name, definition in definitions.items():
        pt_order = np.random.Generator(np.random.PCG64(SEED)).permutation(
            int(definition["pretrain_records"])
        ).astype("<i8")
        sft_order = definition["sft_order"]
        mixed = _stable_mixed_order(
            pt_order,
            sft_order,
            seed=int(definition["placement_seed"]),
        )
        padded, padding = _pad_order(mixed)
        absolute_sft_indices = -sft_order - 1
        sft_targets = int(
            per_row_targets[absolute_sft_indices].sum(dtype=np.int64)
        )
        order_provenance = {
            "schema": "context2048-configured-mixed-order-v1",
            "pt_permutation": {
                "bit_generator": "numpy.random.PCG64",
                "seed": SEED,
                "records": int(definition["pretrain_records"]),
            },
            "sft_exposures": {
                "epoch_permutation_seeds": [43, 44, 45],
                "split": (
                    [0, SFT_TOTAL_EXPOSURES]
                    if name.endswith("full")
                    else [0, SFT_FIRST_EXPOSURES]
                    if name.endswith("first")
                    else [SFT_FIRST_EXPOSURES, SFT_TOTAL_EXPOSURES]
                ),
            },
            "stable_binary_placement": {
                "bit_generator": "numpy.random.PCG64",
                "seed": int(definition["placement_seed"]),
                "preserves_pt_relative_order": True,
                "preserves_sft_relative_order": True,
            },
        }
        manifest = _write_leg_manifest(
            ARTIFACT_ROOT / "manifests" / name,
            leg=name,
            order=padded,
            target_start=int(definition["target_start"]),
            target_count=int(definition["target_count"]),
            sequence_length=CONTEXT_LENGTH,
            pretrain_records=int(definition["pretrain_records"]),
            sft_records=len(sft_order),
            sft_supervised_targets=sft_targets,
            padding_records=padding,
            world_size=WORLD_SIZE,
            local_batch_size=LOCAL_BATCH_SIZE,
            total_steps=len(padded) // GLOBAL_SEQUENCES,
            source_manifest_hash=selection.source_manifest_hash,
            selection_hash=selection.selection_hash,
            sft_cache_hash=cache.cache_hash,
            shuffle_seed=None,
            order_provenance=order_provenance,
        )
        manifests[name] = {
            "metadata_path": str(manifest.metadata_path),
            "metadata_sha256": _sha256_file(manifest.metadata_path),
            "order_sha256": manifest.order_sha256,
            "target_start": int(definition["target_start"]),
            "target_count": int(definition["target_count"]),
            "pretrain_records": int(definition["pretrain_records"]),
            "sft_records": len(sft_order),
            "sft_supervised_targets": sft_targets,
            "padding_records": padding,
            "total_steps": len(padded) // GLOBAL_SEQUENCES,
            "placement_seed": int(definition["placement_seed"]),
            "order_provenance": order_provenance,
        }

    canary_order = _build_canary_mixed_order()
    canary_pt_records = int((canary_order >= 0).sum())
    canary_sft_records = int((canary_order < 0).sum())
    canary_sft_indices = -canary_order[canary_order < 0] - 1
    canary_sft_targets = int(
        per_row_targets[canary_sft_indices].sum(dtype=np.int64)
    )
    canary_manifests: dict[str, Any] = {}
    for name, definition in definitions.items():
        canary_order_provenance = {
            "schema": "context2048-configured-mixed-canary-order-v1",
            "purpose": "preproduction-objective-and-resume-gate-only",
            "total_steps": CANARY_TOTAL_STEPS,
            "world_size": WORLD_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "per_rank_pattern": "alternating-pt-sft",
            "production_manifest": name,
        }
        manifest = _write_leg_manifest(
            ARTIFACT_ROOT / "canary_manifests" / name,
            leg=f"{name}_canary",
            order=canary_order,
            target_start=int(definition["target_start"]),
            target_count=int(definition["target_count"]),
            sequence_length=CONTEXT_LENGTH,
            pretrain_records=canary_pt_records,
            sft_records=canary_sft_records,
            sft_supervised_targets=canary_sft_targets,
            padding_records=0,
            world_size=WORLD_SIZE,
            local_batch_size=LOCAL_BATCH_SIZE,
            total_steps=CANARY_TOTAL_STEPS,
            source_manifest_hash=selection.source_manifest_hash,
            selection_hash=selection.selection_hash,
            sft_cache_hash=cache.cache_hash,
            shuffle_seed=None,
            order_provenance=canary_order_provenance,
        )
        canary_manifests[name] = {
            "metadata_path": str(manifest.metadata_path),
            "metadata_sha256": _sha256_file(manifest.metadata_path),
            "order_sha256": manifest.order_sha256,
            "target_start": int(definition["target_start"]),
            "target_count": int(definition["target_count"]),
            "pretrain_records": canary_pt_records,
            "sft_records": canary_sft_records,
            "sft_supervised_targets": canary_sft_targets,
            "padding_records": 0,
            "total_steps": CANARY_TOTAL_STEPS,
            "order_provenance": canary_order_provenance,
        }

    payload: dict[str, Any] = {
        "schema": "context2048-configured-initial-pt-manifest-set-v2",
        "experiment_version": EXPERIMENT_VERSION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "config_hashes": source["config_hashes"],
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_manifest_hash": selection.source_manifest_hash,
        "selection_path": str(SELECTION_PATH),
        "selection_hash": selection.selection_hash,
        "pt_total_target_tokens": PT_FULL_TARGETS,
        "sft_cache_dir": str(SFT_CACHE_DIR),
        "sft_cache_hash": cache.cache_hash,
        "manifests": manifests,
        "canary_manifests": canary_manifests,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["set_sha256"] = _canonical_sha256(payload)
    _atomic_json(MANIFEST_SET_PATH, payload)
    data_volume.commit()
    return payload


def _manifest(
    stage_key: str,
    *,
    canary: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if stage_key not in INITIAL_STAGES:
        raise ValueError(f"unknown initial PT stage: {stage_key}")
    payload = _load_json(MANIFEST_SET_PATH)
    recorded = payload.pop("set_sha256", None)
    if recorded != _canonical_sha256(payload):
        raise RuntimeError("initial PT manifest set hash drifted")
    payload["set_sha256"] = recorded
    if payload.get("source_tree_sha256") != SOURCE_TREE_SHA256:
        raise RuntimeError("manifest set source tree drifted")
    group = "canary_manifests" if canary else "manifests"
    manifest = payload[group][INITIAL_STAGES[stage_key]["manifest"]]
    if _sha256_file(Path(manifest["metadata_path"])) != manifest["metadata_sha256"]:
        raise RuntimeError(f"{stage_key} manifest metadata drifted")
    return payload, manifest


def _runtime_provenance_overrides() -> list[str]:
    runtime = _runtime_identity()
    return [
        f"provenance.modal_app_name={APP_NAME}",
        f"provenance.modal_app_id={runtime['app_id']}",
        f"provenance.modal_image_id={runtime['image_id']}",
        f"provenance.modal_base_image={runtime['base_image']}",
        f"provenance.modal_client_version={runtime['modal_client_version']}",
        "provenance.runtime_package_versions="
        + json.dumps(runtime["packages"], sort_keys=True),
        f"provenance.runtime_distribution_count={runtime['distribution_count']}",
        "provenance.runtime_distribution_inventory_sha256="
        + str(runtime["distribution_inventory_sha256"]),
    ]


def _stage_output(stage_key: str, *, canary_suffix: str = "") -> Path:
    suffix = f"_{canary_suffix}" if canary_suffix else ""
    return CHECKPOINT_ROOT / f"{stage_key}{suffix}"


def _random_initialization_identity() -> dict[str, Any]:
    return {
        "schema": "interleaved-random-initialization-v1",
        "mode": "random",
        "destination_seed": SEED,
    }


def _stage_command(
    stage_key: str,
    *,
    output_dir: Path,
    run_name: str,
    max_steps: int | None = None,
    resume: Path | None = None,
    canary: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    if canary != (max_steps is not None):
        raise ValueError("canary commands require an explicit max_steps value")
    payload, manifest = _manifest(stage_key, canary=canary)
    production_manifest = payload["manifests"][INITIAL_STAGES[stage_key]["manifest"]]
    total_steps = int(manifest["total_steps"])
    initialization_identity = _random_initialization_identity()
    resolved_config_hashes = {
        name: payload["config_hashes"][name]
        for name in INITIAL_STAGES[stage_key]["dependent_config_files"]
    }
    overrides = [
        "model.block_size=2048",
        "model.vocab_size=85",
        "tokenizer.name=LanTokenizerSFT",
        "tokenizer.include_env_tokens=true",
        "tokenizer.include_reward_tokens=false",
        f"training.output_dir={output_dir}",
        f"training.run_name={run_name}",
        f"training.seed={SEED}",
        f"training.local_batch_size={LOCAL_BATCH_SIZE}",
        "training.gradient_accumulation_steps=1",
        "training.allow_topology_override=true",
        "training.allow_weight_decay_override=false",
        "training.allow_vocab_expansion=false",
        f"training.total_steps={total_steps}",
        f"training.arc_steps=[{total_steps}]",
        "training.reset_optimizer_between_arcs=true",
        "training.mixed_precision=bf16",
        "training.sft_loss_weight=1.0",
        "training.optimizer.lr=0.001",
        "training.optimizer.weight_decay=0.1",
        "training.optimizer.betas=[0.9,0.95]",
        "training.scheduler.warmup_ratio=0.05",
        "training.scheduler.eta_min=0.00001",
        "training.torch_compile=none",
        "model.attn_implementation=sdpa",
        "model.flash_attention_version=2.8.3",
        f"data.source_root={SOURCE_DIR}",
        f"data.source_manifest_path={SOURCE_MANIFEST_PATH}",
        f"data.selection_manifest_path={SELECTION_PATH}",
        f"data.sft_cache_dir={SFT_CACHE_DIR}",
        f"data.leg_manifest_path={manifest['metadata_path']}",
        f"data.expected_manifest_hash={manifest['metadata_sha256']}",
        "data.sequence_length=2048",
        "data.num_workers=8",
        "training.num_workers=8",
        "training.persistent_workers=true",
        "training.save_interval=200",
        "training.log_interval=10",
        f"logging.backend={'none' if canary else 'wandb'}",
        f"logging.project={WANDB_PROJECT}",
        f"logging.entity={WANDB_ENTITY}",
        "logging.group=5b-total-pt-matrix-v1",
        "logging.job_type=pt-sft-mixed",
        f"logging.id={EXPERIMENT_VERSION}-{stage_key}",
        "logging.resume=allow",
        f"provenance.experiment_version={EXPERIMENT_VERSION}",
        f"provenance.source_tree_sha256={SOURCE_TREE_SHA256}",
        f"provenance.experiment={stage_key}",
        "provenance.stage=pt_sft_mixed",
        f"provenance.seed={SEED}",
        "provenance.initialization_identity="
        + json.dumps(initialization_identity, sort_keys=True),
        "provenance.context_length=2048",
        "provenance.vocab_size=85",
        "provenance.token_ids=" + json.dumps(EXPECTED_TOKEN_IDS, sort_keys=True),
        f"provenance.pt_target_tokens={production_manifest['target_count']}",
        f"provenance.pt_global_token_batch={GLOBAL_TOKEN_SLOTS}",
        f"provenance.sft_copies={production_manifest['sft_records'] / SFT_ROWS}",
        "provenance.sft_packing=one-row-per-sequence-right-padded",
        f"provenance.source_repo={SOURCE_REPO}",
        f"provenance.source_revision={SOURCE_REVISION}",
        "provenance.sft_repo=Pre-to-Post-2/200M_SFT_dataset",
        "provenance.sft_revision=fd343bd28f6a40fc3dab4dcfb6e74c11b7a20b90",
        "provenance.peak_lr=0.001",
        "provenance.eta_min=0.00001",
        "provenance.weight_decay=0.1",
        "provenance.master_parameter_dtype=float32",
        "provenance.optimizer_state_dtype=float32",
        "provenance.forward_backward_dtype=bfloat16",
        "provenance.gradient_dtype=float32",
        "provenance.hf_export_dtype=float32",
        "provenance.resolved_config_hashes="
        + json.dumps(resolved_config_hashes, sort_keys=True),
        *_runtime_provenance_overrides(),
    ]
    if canary:
        overrides.extend(
            [
                f"training.max_steps={int(max_steps)}",
                "training.save_interval=1",
                "training.log_interval=1",
                "training.persistent_workers=false",
                "data.num_workers=0",
                "training.num_workers=0",
                "provenance.canary_sample_contract=mixed-pt-sft",
                f"provenance.canary_manifest_sha256={manifest['metadata_sha256']}",
            ]
        )
    command = [
        "accelerate",
        "launch",
        "--multi_gpu",
        "--num_processes",
        str(WORLD_SIZE),
        "--mixed_precision",
        "bf16",
        "--main_process_port",
        "29651",
        TRAIN_CLI,
        "--config",
        BASE_CONFIG,
        "--override",
        *overrides,
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    identity = {
        "schema": "interleaved-initial-launch-command-v1",
        "argv": command if resume is None else command[: -2],
    }
    identity["sha256"] = _canonical_sha256(identity["argv"])
    return command, identity


def _commit_checkpoint(output_dir: Path, previous: str | None, label: str) -> str | None:
    with checkpoint_volume_commit_lock(output_dir):
        pointer = output_dir / LATEST_CHECKPOINT_POINTER
        if not pointer.is_file():
            return previous
        observed = _sha256_file(pointer)
        if observed == previous:
            return previous
        if (output_dir / "final").exists() or (output_dir / ".final.tmp").exists():
            return previous
        checkpoint = validate_checkpoint_run_root(output_dir)
        checkpoint_volume.commit()
    print(f"[{label}] committed {checkpoint}", flush=True)
    return observed


def _run_process(
    command: list[str],
    *,
    output_dir: Path,
    label: str,
    initial_command: Mapping[str, Any] | None,
) -> int:
    environment = dict(os.environ)
    if initial_command is not None:
        environment[INITIAL_LAUNCH_COMMAND_ENV] = json.dumps(
            initial_command["argv"], separators=(",", ":")
        )
        environment[INITIAL_LAUNCH_COMMAND_SHA256_ENV] = str(
            initial_command["sha256"]
        )
    process = subprocess.Popen(
        command,
        cwd="/root/chess",
        env=environment,
        stdout=sys.stdout,
        stderr=sys.stderr,
        start_new_session=True,
    )
    pointer = output_dir / LATEST_CHECKPOINT_POINTER
    pointer_hash = _sha256_file(pointer) if pointer.is_file() else None
    try:
        while True:
            try:
                code = int(process.wait(timeout=5))
                pointer_hash = _commit_checkpoint(output_dir, pointer_hash, label)
                return code
            except subprocess.TimeoutExpired:
                pointer_hash = _commit_checkpoint(output_dir, pointer_hash, label)
    except BaseException:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        raise


def _validate_state(
    stage_key: str,
    output_dir: Path,
    *,
    expected_step: int,
    canary: bool = False,
) -> dict[str, Any]:
    _, manifest = _manifest(stage_key, canary=canary)
    checkpoint = validate_checkpoint_run_root(
        output_dir,
        allowed_root_directories=(
            frozenset({"final"}) if (output_dir / "final").is_dir() else frozenset()
        ),
    )
    state = _load_json(checkpoint / "trainer_state.json")
    if int(state.get("global_step", -1)) != expected_step:
        raise RuntimeError(f"{stage_key} checkpoint step drifted")
    if int(state.get("manifest_cursor", -1)) != expected_step:
        raise RuntimeError(f"{stage_key} manifest cursor drifted")
    if state.get("manifest_hash") != manifest["metadata_sha256"]:
        raise RuntimeError(f"{stage_key} manifest identity drifted")
    if state.get("precision_contract") != EXPECTED_PRECISION:
        raise RuntimeError(f"{stage_key} precision contract drifted")
    configured = state.get("configured_provenance", {})
    expected = {
        "experiment_version": EXPERIMENT_VERSION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "experiment": stage_key,
        "stage": "pt_sft_mixed",
        "context_length": CONTEXT_LENGTH,
        "vocab_size": 85,
        "token_ids": EXPECTED_TOKEN_IDS,
        "peak_lr": 1e-3,
        "eta_min": 1e-5,
    }
    for key, value in expected.items():
        if configured.get(key) != value:
            raise RuntimeError(f"{stage_key} configured provenance {key} drifted")
    return state


def _validate_final(stage_key: str, output_dir: Path) -> dict[str, Any]:
    _, manifest = _manifest(stage_key)
    state = _validate_state(
        stage_key,
        output_dir,
        expected_step=int(manifest["total_steps"]),
    )
    final = output_dir / "final"
    final_state = _load_json(final / "interleaved_training_state.json")
    if int(final_state.get("global_step", -1)) != int(manifest["total_steps"]):
        raise RuntimeError(f"{stage_key} final state step drifted")
    tokenizer = validate_hf_tokenizer_contract(
        final,
        expected_vocab_size=85,
        expected_context_length=CONTEXT_LENGTH,
    )
    export = validate_completed_hf_export(final)
    return {
        "final": str(final),
        "global_step": int(manifest["total_steps"]),
        "state_sha256": _sha256_file(final / "interleaved_training_state.json"),
        "tokenizer": tokenizer,
        "export": export["marker"],
        "checkpoint_state": state,
    }


def _assert_models_equal(left: Path, right: Path) -> str:
    import torch
    from safetensors.torch import load_file

    left_state = load_file(str(left / "model.safetensors"), device="cpu")
    right_state = load_file(str(right / "model.safetensors"), device="cpu")
    if left_state.keys() != right_state.keys():
        raise RuntimeError("canary model key sets differ")
    for name, tensor in left_state.items():
        other = right_state[name]
        if tensor.dtype != torch.float32 or other.dtype != torch.float32:
            raise RuntimeError(f"canary tensor is not FP32: {name}")
        if not torch.equal(tensor, other):
            raise RuntimeError(f"resume/reference model mismatch: {name}")
    return _sha256_file(left / "model.safetensors")


def _bf16_inference(final: Path) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        final, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    try:
        dtypes = {parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()}
        if dtypes != {torch.bfloat16}:
            raise RuntimeError(f"BF16 inference parameter dtypes drifted: {dtypes}")
        input_ids = torch.tensor([[0, 0, 0, 0]], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits
        if logits.dtype != torch.bfloat16 or not torch.isfinite(logits.float()).all():
            raise RuntimeError("BF16 inference output contract drifted")
        return {"parameter_dtype": "bfloat16", "logits_dtype": "bfloat16", "finite": True}
    finally:
        del model
        torch.cuda.empty_cache()


def _gate_path(stage_key: str) -> Path:
    return GATE_ROOT / f"{stage_key}_{SOURCE_TREE_SHA256}.json"


def _validate_gate(stage_key: str) -> dict[str, Any]:
    marker = _load_json(_gate_path(stage_key))
    recorded = marker.pop("gate_sha256", None)
    if recorded != _canonical_sha256(marker):
        raise RuntimeError(f"{stage_key} canary gate hash drifted")
    marker["gate_sha256"] = recorded
    expected = {
        "schema": "context2048-configured-pt-canary-v1",
        "decision": "pass",
        "stage_key": stage_key,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "experiment_version": EXPERIMENT_VERSION,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise RuntimeError(f"{stage_key} canary gate {key} drifted")
    return marker


def _run_canary(stage_key: str) -> dict[str, Any]:
    _prepare_data()
    path = _gate_path(stage_key)
    if path.is_file():
        return _validate_gate(stage_key)
    root = CHECKPOINT_ROOT / "canary" / SOURCE_TREE_SHA256 / stage_key
    if root.exists():
        raise FileExistsError(f"incomplete canary root exists: {root}")
    first_root = root / "first"
    resumed_root = root / "resumed"
    reference_root = root / "reference"

    first_command, first_identity = _stage_command(
        stage_key,
        output_dir=first_root,
        run_name=f"{stage_key}-canary-first",
        max_steps=1,
        canary=True,
    )
    if _run_process(first_command, output_dir=first_root, label=f"{stage_key}:canary-first", initial_command=None):
        raise RuntimeError("first canary process failed")
    first_checkpoint = resolve_resume_checkpoint(first_root)
    first_precision = inspect_accelerator_checkpoint_fp32(first_checkpoint)
    first_state = _validate_state(
        stage_key, first_root, expected_step=1, canary=True
    )
    evidence = first_state.get("runtime_provenance", {}).get("canary_sample_evidence", {})
    if not (
        evidence.get("contract") == "mixed-pt-sft"
        and int(evidence.get("global_pretrain_rows", 0)) > 0
        and int(evidence.get("global_sft_rows", 0)) > 0
        and evidence.get("pt_leading_bos_validated") is True
        and evidence.get("sft_bos_and_mask_validated") is True
    ):
        raise RuntimeError(f"mixed PT/SFT canary evidence drifted: {evidence}")

    second_command, second_identity = _stage_command(
        stage_key,
        output_dir=resumed_root,
        run_name=f"{stage_key}-canary-resumed",
        max_steps=2,
        resume=first_checkpoint,
        canary=True,
    )
    if _run_process(second_command, output_dir=resumed_root, label=f"{stage_key}:canary-resumed", initial_command=None):
        raise RuntimeError("resumed canary process failed")
    reference_command, reference_identity = _stage_command(
        stage_key,
        output_dir=reference_root,
        run_name=f"{stage_key}-canary-reference",
        max_steps=2,
        canary=True,
    )
    if _run_process(reference_command, output_dir=reference_root, label=f"{stage_key}:canary-reference", initial_command=None):
        raise RuntimeError("reference canary process failed")
    resumed_state = _validate_state(
        stage_key, resumed_root, expected_step=2, canary=True
    )
    reference_state = _validate_state(
        stage_key, reference_root, expected_step=2, canary=True
    )
    for field in ("global_step", "manifest_cursor", "precision_contract", "determinism_contract"):
        if resumed_state.get(field) != reference_state.get(field):
            raise RuntimeError(f"canary resume/reference {field} differs")
    model_sha256 = _assert_models_equal(resumed_root / "final", reference_root / "final")
    validate_completed_hf_export(resumed_root / "final")
    inference = _bf16_inference(resumed_root / "final")
    marker: dict[str, Any] = {
        "schema": "context2048-configured-pt-canary-v1",
        "decision": "pass",
        "experiment_version": EXPERIMENT_VERSION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "stage_key": stage_key,
        "manifest_set_sha256": _load_json(MANIFEST_SET_PATH)["set_sha256"],
        "canary_manifest": _manifest(stage_key, canary=True)[1],
        "first_update_precision": first_precision,
        "sample_evidence": evidence,
        "resume_reference_bitwise_equal": True,
        "model_sha256": model_sha256,
        "bf16_inference": inference,
        "runtime_identity": _runtime_identity(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    marker["gate_sha256"] = _canonical_sha256(marker)
    _atomic_json(path, marker)
    data_volume.commit()
    return _validate_gate(stage_key)


def _wandb_gate_key() -> str:
    return f"wandb-gate:{SOURCE_TREE_SHA256}"


def _run_wandb_gate() -> dict[str, Any]:
    existing = claim_store.get(_wandb_gate_key(), None)
    if existing is not None:
        return dict(existing)
    import wandb

    run_id = f"infra-{SOURCE_TREE_SHA256[:24]}"
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        id=run_id,
        name=run_id,
        resume="allow",
        job_type="infra-gate",
        settings=wandb.Settings(
            disable_code=True,
            disable_git=True,
            x_disable_stats=True,
            x_disable_machine_info=True,
            console="off",
            silent=True,
        ),
    )
    if run is None:
        raise RuntimeError("W&B gate initialization failed")
    run.log({"infra_gate/write_marker": 1}, step=0)
    run.summary["source_tree_sha256"] = SOURCE_TREE_SHA256
    run.finish(exit_code=0)
    api = wandb.Api(timeout=30)
    remote = None
    for _ in range(12):
        try:
            remote = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
            if remote.state == "finished" and dict(remote.summary).get("source_tree_sha256") == SOURCE_TREE_SHA256:
                break
        except Exception:
            remote = None
        time.sleep(2)
    if remote is None or remote.state != "finished":
        raise RuntimeError("W&B gate read-after-write failed")
    marker = {
        "schema": "context2048-configured-pt-wandb-gate-v1",
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "run_id": run_id,
        "url": str(remote.url),
        "state": str(remote.state),
    }
    marker["gate_sha256"] = _canonical_sha256(marker)
    won = claim_store.put(_wandb_gate_key(), marker, skip_if_exists=True)
    observed = dict(claim_store[_wandb_gate_key()])
    if won and observed != marker:
        raise RuntimeError("W&B gate CAS winner drifted")
    return observed


def _launch_identity(stage_key: str) -> dict[str, Any]:
    payload, manifest = _manifest(stage_key)
    gate = _validate_gate(stage_key)
    wandb_gate = dict(claim_store[_wandb_gate_key()])
    configs = {
        name: payload["config_hashes"][name]
        for name in INITIAL_STAGES[stage_key]["dependent_config_files"]
    }
    return {
        "schema": "context2048-configured-pt-launch-identity-v1",
        "experiment_version": EXPERIMENT_VERSION,
        "stage_key": stage_key,
        "description": INITIAL_STAGES[stage_key]["description"],
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "resolved_config_hashes": configs,
        "manifest_set_sha256": payload["set_sha256"],
        "manifest": manifest,
        "gate_sha256": gate["gate_sha256"],
        "wandb_gate_sha256": wandb_gate["gate_sha256"],
        "output_root": str(_stage_output(stage_key)),
        "run_name": INITIAL_STAGES[stage_key]["run_name"],
        "runtime": _runtime_identity(),
    }


def _claim_key(stage_key: str) -> str:
    return f"claim:{stage_key}"


def _execution_key(stage_key: str) -> str:
    return f"execution:{stage_key}:0000"


def _completion_key(stage_key: str) -> str:
    return f"completion:{stage_key}"


def _token_sha256(token: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        raise RuntimeError("launch token must be 256-bit lowercase hexadecimal")
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    core = {key: item for key, item in value.items() if key != field}
    return {**core, field: _canonical_sha256(core)}


def _durable_anchor_path(stage_key: str) -> Path:
    return DURABLE_LAUNCH_ROOT / stage_key / "anchor.json"


def _validate_claim(stage_key: str, token: str, identity: Mapping[str, Any]) -> dict[str, Any]:
    claim = dict(claim_store[_claim_key(stage_key)])
    recorded = claim.pop("claim_sha256", None)
    if recorded != _canonical_sha256(claim):
        raise RuntimeError(f"{stage_key} claim hash drifted")
    claim["claim_sha256"] = recorded
    if claim.get("launch_token_sha256") != _token_sha256(token):
        raise RuntimeError(f"{stage_key} launch token does not match")
    if claim.get("launch_identity") != dict(identity):
        raise RuntimeError(f"{stage_key} launch identity differs from its claim")
    return claim


def _validate_anchor(stage_key: str, claim: Mapping[str, Any]) -> dict[str, Any]:
    anchor = _load_json(_durable_anchor_path(stage_key))
    recorded = anchor.pop("anchor_sha256", None)
    if recorded != _canonical_sha256(anchor):
        raise RuntimeError(f"{stage_key} durable anchor hash drifted")
    anchor["anchor_sha256"] = recorded
    expected = {
        "schema": "context2048-configured-pt-durable-anchor-v1",
        "stage_key": stage_key,
        "claim_sha256": claim["claim_sha256"],
        "launch_token_sha256": claim["launch_token_sha256"],
        "launch_identity_sha256": claim["launch_identity_sha256"],
        "output_root": str(_stage_output(stage_key)),
    }
    for key, value in expected.items():
        if anchor.get(key) != value:
            raise RuntimeError(f"{stage_key} durable anchor {key} drifted")
    return anchor


def _bind_worker(stage_key: str, claim: Mapping[str, Any], function_call_id: str) -> dict[str, Any]:
    proposed = _self_hash(
        {
            "schema": "context2048-configured-pt-execution-v1",
            "stage_key": stage_key,
            "generation": 0,
            "claim_sha256": claim["claim_sha256"],
            "function_call_id": function_call_id,
        },
        "execution_sha256",
    )
    claim_store.put(_execution_key(stage_key), proposed, skip_if_exists=True)
    observed = dict(claim_store[_execution_key(stage_key)])
    if observed != proposed:
        raise RuntimeError(f"{stage_key} is bound to another FunctionCall")
    return observed


def _run_production_stage(stage_key: str, token: str) -> dict[str, Any]:
    data_volume.reload()
    checkpoint_volume.reload()
    identity = _launch_identity(stage_key)
    claim = _validate_claim(stage_key, token, identity)
    _validate_anchor(stage_key, claim)
    call_id = modal.current_function_call_id()
    execution = _bind_worker(stage_key, claim, call_id)
    output_dir = _stage_output(stage_key)
    _, manifest = _manifest(stage_key)
    total_steps = int(manifest["total_steps"])
    resume = None
    if output_dir.exists() and any(output_dir.iterdir()):
        try:
            final = _validate_final(stage_key, output_dir)
            completion = _self_hash(
                {
                    "schema": "context2048-configured-pt-completion-v1",
                    "stage_key": stage_key,
                    "claim_sha256": claim["claim_sha256"],
                    "execution_sha256": execution["execution_sha256"],
                    "final": final,
                },
                "completion_sha256",
            )
            claim_store.put(_completion_key(stage_key), completion, skip_if_exists=True)
            return completion
        except Exception:
            resume = resolve_resume_checkpoint(output_dir)
            _validate_state(
                stage_key,
                output_dir,
                expected_step=int(_load_json(resume / "trainer_state.json")["global_step"]),
            )
    command, initial = _stage_command(
        stage_key,
        output_dir=output_dir,
        run_name=INITIAL_STAGES[stage_key]["run_name"],
        resume=resume,
    )
    print(f"[{stage_key}] " + " ".join(command), flush=True)
    code = _run_process(
        command,
        output_dir=output_dir,
        label=stage_key,
        initial_command=initial,
    )
    if code:
        raise RuntimeError(f"{stage_key} training failed with exit code {code}")
    final = _validate_final(stage_key, output_dir)
    checkpoint_volume.commit()
    checkpoint_volume.reload()
    final = _validate_final(stage_key, output_dir)
    completion = _self_hash(
        {
            "schema": "context2048-configured-pt-completion-v1",
            "stage_key": stage_key,
            "claim_sha256": claim["claim_sha256"],
            "execution_sha256": execution["execution_sha256"],
            "global_step": total_steps,
            "final": final,
        },
        "completion_sha256",
    )
    completion_path = DURABLE_LAUNCH_ROOT / stage_key / "completion.json"
    _atomic_json(completion_path, completion)
    checkpoint_volume.commit()
    checkpoint_volume.reload()
    if _load_json(completion_path) != completion:
        raise RuntimeError(f"{stage_key} durable completion did not survive commit")
    claim_store.put(_completion_key(stage_key), completion, skip_if_exists=True)
    return completion


@app.function(cpu=16.0, memory=64 * 1024, timeout=4 * 60 * 60, retries=0)
def prepare_data() -> dict[str, Any]:
    return _prepare_data()


@app.function(cpu=1.0, memory=1024, timeout=5 * 60, retries=0)
def deployment_identity() -> dict[str, Any]:
    return {
        "schema": "context2048-configured-pt-deployment-v1",
        "experiment_version": EXPERIMENT_VERSION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "runtime": _runtime_identity(),
    }


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}",
    cpu=32.0,
    memory=128 * 1024,
    timeout=4 * 60 * 60,
    retries=0,
    max_containers=1,
)
def run_canary(stage_key: str) -> dict[str, Any]:
    if stage_key not in INITIAL_STAGES:
        raise ValueError(stage_key)
    return _run_canary(stage_key)


@app.function(cpu=1.0, memory=1024, timeout=10 * 60, retries=0)
def wandb_write_gate() -> dict[str, Any]:
    return _run_wandb_gate()


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}",
    cpu=32.0,
    memory=128 * 1024,
    timeout=24 * 60 * 60,
    retries=modal.Retries(initial_delay=10.0, max_retries=3),
    single_use_containers=True,
    max_containers=2,
)
def run_stage(stage_key: str, launch_token: str) -> dict[str, Any]:
    return _run_production_stage(stage_key, launch_token)


@app.function(cpu=1.0, memory=1024, timeout=10 * 60, retries=0, max_containers=1)
def dispatch_stage(stage_key: str, launch_token: str) -> dict[str, Any]:
    if stage_key not in INITIAL_STAGES:
        raise ValueError(stage_key)
    _token_sha256(launch_token)
    data_volume.reload()
    checkpoint_volume.reload()
    identity = _launch_identity(stage_key)
    if _completion_key(stage_key) in claim_store:
        return {
            "outcome": "already_complete",
            "stage_key": stage_key,
            "completion": dict(claim_store[_completion_key(stage_key)]),
        }
    existing = claim_store.get(_claim_key(stage_key), None)
    if existing is None:
        core = {
            "schema": "context2048-configured-pt-claim-v1",
            "stage_key": stage_key,
            "launch_token_sha256": _token_sha256(launch_token),
            "launch_identity": identity,
            "launch_identity_sha256": _canonical_sha256(identity),
            "claimed_at": datetime.now(timezone.utc).isoformat(),
        }
        proposed = _self_hash(core, "claim_sha256")
        won = claim_store.put(_claim_key(stage_key), proposed, skip_if_exists=True)
        if not won:
            raise RuntimeError(f"{stage_key} lost the atomic launch claim")
    claim = _validate_claim(stage_key, launch_token, identity)
    anchor_path = _durable_anchor_path(stage_key)
    if not anchor_path.exists():
        anchor = _self_hash(
            {
                "schema": "context2048-configured-pt-durable-anchor-v1",
                "stage_key": stage_key,
                "claim_sha256": claim["claim_sha256"],
                "launch_token_sha256": claim["launch_token_sha256"],
                "launch_identity_sha256": claim["launch_identity_sha256"],
                "output_root": str(_stage_output(stage_key)),
                "created_at": claim["claimed_at"],
            },
            "anchor_sha256",
        )
        anchor_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(anchor_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(anchor, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        checkpoint_volume.commit()
        checkpoint_volume.reload()
    anchor = _validate_anchor(stage_key, claim)
    existing_execution = claim_store.get(_execution_key(stage_key), None)
    if existing_execution is not None:
        return {
            "outcome": "existing_claim",
            "stage_key": stage_key,
            "spawned": False,
            "claim_sha256": claim["claim_sha256"],
            "anchor_sha256": anchor["anchor_sha256"],
            "function_call_id": dict(existing_execution)["function_call_id"],
        }
    call = run_stage.spawn(stage_key, launch_token)
    execution = _bind_worker(stage_key, claim, call.object_id)
    return {
        "outcome": "spawned",
        "stage_key": stage_key,
        "spawned": True,
        "claim_sha256": claim["claim_sha256"],
        "anchor_sha256": anchor["anchor_sha256"],
        "execution_sha256": execution["execution_sha256"],
        "function_call_id": call.object_id,
        "output_root": str(_stage_output(stage_key)),
        "run_name": INITIAL_STAGES[stage_key]["run_name"],
    }


@app.function(cpu=1.0, memory=1024, timeout=5 * 60, retries=0)
def read_status(stage_keys: list[str]) -> list[dict[str, Any]]:
    checkpoint_volume.reload()
    rows = []
    for key in stage_keys:
        if key not in INITIAL_STAGES:
            raise ValueError(key)
        row: dict[str, Any] = {"stage_key": key}
        for label, dict_key in (
            ("claim", _claim_key(key)),
            ("execution", _execution_key(key)),
            ("completion", _completion_key(key)),
        ):
            value = claim_store.get(dict_key, None)
            if value is not None:
                row[label] = dict(value)
        output = _stage_output(key)
        if output.exists() and (output / LATEST_CHECKPOINT_POINTER).is_file():
            checkpoint = resolve_resume_checkpoint(output)
            row["checkpoint"] = _load_json(checkpoint / "trainer_state.json")
        rows.append(row)
    return rows


DEPLOYED_FUNCTIONS = frozenset(
    {
        "deployment_identity",
        "prepare_data",
        "run_canary",
        "wandb_write_gate",
        "dispatch_stage",
        "read_status",
    }
)


def _deployed(name: str):
    if name not in DEPLOYED_FUNCTIONS:
        raise ValueError(name)
    return modal.Function.from_name(APP_NAME, name)


def _require_deployment() -> dict[str, Any]:
    identity = _deployed("deployment_identity").remote()
    if not isinstance(identity, Mapping):
        raise RuntimeError("deployment identity is not an object")
    expected = {
        "schema": "context2048-configured-pt-deployment-v1",
        "experiment_version": EXPERIMENT_VERSION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise RuntimeError(
                "deployed source does not match local source; run "
                f"modal deploy {Path(__file__).resolve()}"
            )
    return dict(identity)


def _recovery_path(stage_key: str) -> Path:
    return LOCAL_RECOVERY_ROOT / f"{stage_key}.json"


def _write_recovery(stage_key: str, token: str) -> Path:
    path = _recovery_path(stage_key)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    record = {
        "schema": "context2048-configured-pt-local-recovery-v1",
        "experiment_version": EXPERIMENT_VERSION,
        "stage_key": stage_key,
        "launch_token": token,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return path


@app.local_entrypoint()
def main(action: str = "dry-run", stage: str = "") -> None:
    action = action.strip().lower()
    selected = [item.strip() for item in stage.split(",") if item.strip()] or list(INITIAL_STAGES)
    unknown = sorted(set(selected) - set(INITIAL_STAGES))
    if unknown:
        raise ValueError(f"unknown stages: {unknown}")
    if action == "dry-run":
        print(
            json.dumps(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "app": APP_NAME,
                    "source_tree_sha256": SOURCE_TREE_SHA256,
                    "initial_stages": INITIAL_STAGES,
                    "deployment_command": [
                        "modal",
                        "deploy",
                        str(Path(__file__).resolve()),
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    _require_deployment()
    if action == "prep":
        print(json.dumps(_deployed("prepare_data").remote(), indent=2, sort_keys=True))
        return
    if action == "canary":
        for key in selected:
            print(json.dumps(_deployed("run_canary").remote(key), indent=2, sort_keys=True))
        return
    if action == "wandb-gate":
        print(json.dumps(_deployed("wandb_write_gate").remote(), indent=2, sort_keys=True))
        return
    if action == "launch":
        dispatcher = _deployed("dispatch_stage")
        for key in selected:
            token = secrets.token_hex(32)
            recovery = _write_recovery(key, token)
            result = dispatcher.remote(key, token)
            print(
                json.dumps(
                    {
                        **dict(result),
                        "local_recovery_record": str(recovery),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return
    if action == "status":
        print(json.dumps(_deployed("read_status").remote(selected), indent=2, sort_keys=True))
        return
    raise ValueError("action must be dry-run, prep, canary, wandb-gate, launch, or status")
