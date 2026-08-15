"""Launch the four controlled 2,048-context PT/SFT experiments.

The four experiments are:

1. 81-token pretraining, expand to 85 tokens, then clean SFT for 3 epochs.
2. 85-token pretraining, then clean SFT for 3 epochs.
3. 85-token mixed training with one copy of every clean SFT row.
4. 85-token mixed training with three copies of every clean SFT row.

SFT rows are always independent records.  They are right-padded by the
collator and are never concatenated with another SFT row or PT tokens.
"""
from __future__ import annotations

import builtins
import functools
import hashlib
import importlib.metadata
import json
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import sysconfig
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import modal

from training.immutable_checkpoint import (
    LATEST_CHECKPOINT_POINTER,
    checkpoint_volume_commit_lock,
    inspect_accelerator_checkpoint_fp32,
    resolve_resume_checkpoint,
    sha256_file as _checkpoint_sha256_file,
    validate_checkpoint_run_root,
    validate_completed_hf_export,
)
from training.interleaved_hf_trainer import (
    INITIAL_LAUNCH_COMMAND_ENV,
    INITIAL_LAUNCH_COMMAND_SHA256_ENV,
    authenticated_weights_only_identity,
)
from training.tokenizer_contract import (
    EXPECTED_VOCAB_81,
    EXPECTED_VOCAB_85,
    validate_hf_tokenizer_contract as _shared_validate_hf_tokenizer_contract,
)


EXPERIMENT_VERSION = "context2048_vocab_mixing_fp32_master_v13_20260813"
APP_NAME = "chess-context2048-vocab-mixing-fp32-master-v13"
WANDB_ENTITY = "jingyanshen-new-york-university"
WANDB_PROJECT = "chess-47m-context2048-tokenizer-mixing-fp32-master-v13"
WANDB_SECRET = "wandb-interleave-pt-rl"
LAUNCH_CLAIM_DICT_NAME = (
    "chess-ctx2048-production-launch-claims-"
    + hashlib.sha256(EXPERIMENT_VERSION.encode("utf-8")).hexdigest()[:16]
)
LAUNCH_CLAIM_SCHEMA = "context2048-production-launch-claim-v1"
LAUNCH_ATTEMPT_SCHEMA = "context2048-production-launch-attempt-v1"
LAUNCH_EXECUTION_SCHEMA = "context2048-production-generation-resolution-v3"
LAUNCH_STATUS_SCHEMA = "context2048-production-launch-status-v2"
LAUNCH_ATOMICITY_GATE_SCHEMA = "context2048-launch-claim-atomicity-gate-v1"
WANDB_WRITE_GATE_SCHEMA = "context2048-wandb-write-gate-v1"
TERMINAL_CALL_EVIDENCE_SCHEMA = (
    "context2048-authoritative-terminal-call-result-v2"
)
DURABLE_LAUNCH_ANCHOR_SCHEMA = "context2048-durable-launch-anchor-v1"
DURABLE_LAUNCH_ANCHOR_INTENT_SCHEMA = (
    "context2048-durable-launch-anchor-intent-v1"
)
LAUNCH_HEARTBEAT_SECONDS = 5 * 60
MAX_LAUNCH_GENERATIONS = 1_000
PRODUCTION_FUNCTION_TIMEOUT_SECONDS = 24 * 60 * 60


PRODUCTION_TRAINING_TERMINAL_MARKER = (
    "CONTEXT2048_PRODUCTION_TRAINING_TERMINAL_V1"
)
PRODUCTION_DISPATCHER_TERMINAL_MARKER = (
    "CONTEXT2048_PRODUCTION_DISPATCHER_TERMINAL_V1"
)


def _wrap_deployed_terminal_failure(marker: str):
    """Prevent a user-raised TimeoutError from looking like a poll timeout."""

    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                # Do not serialize the original message into recovery evidence.
                # The exception chain remains available in retained Modal logs.
                raise RuntimeError(marker) from exc

        return wrapped

    return decorate

CONTEXT_LENGTH = 2_048
WORLD_SIZE = 8
PT_LOCAL_BATCH_SIZE = 16
PT_GLOBAL_SEQUENCES = WORLD_SIZE * PT_LOCAL_BATCH_SIZE
PT_GLOBAL_TOKEN_BATCH = PT_GLOBAL_SEQUENCES * CONTEXT_LENGTH
SFT_LOCAL_BATCH_SIZE = 32
SFT_GLOBAL_SEQUENCES = WORLD_SIZE * SFT_LOCAL_BATCH_SIZE
GRADIENT_ACCUMULATION_STEPS = 1
GPU_TYPE = "H200"

PT_TARGET_TOKENS = 9_181_735_000
PT_RECORDS = math.ceil(PT_TARGET_TOKENS / CONTEXT_LENGTH)
PT_STEPS = math.ceil(PT_RECORDS / PT_GLOBAL_SEQUENCES)
PT_PEAK_LR = 1e-3
PT_ETA_MIN = 1e-4
PT_WARMUP_RATIO = 0.05

SFT_ROWS = 77_717
SFT_SUPERVISED_TARGETS = 52_482_753
SFT_EPOCHS = 3
SFT_STAGE_RECORDS = SFT_ROWS * SFT_EPOCHS
SFT_STAGE_STEPS = math.ceil(SFT_STAGE_RECORDS / SFT_GLOBAL_SEQUENCES)
SFT_PEAK_LR = 3e-4
SFT_ETA_MIN = 1e-5
SFT_WARMUP_STEPS = 50
CANARY_TOTAL_STEPS = 2

SOURCE_REPO = "chess-pre-to-post/pretrain_v1_20b"
SOURCE_REVISION = "07dd1b7090ca5f0fb05ef624c26b20bff19483c8"
SOURCE_DIR = Path("/data/pretrain_v1_20b")
SFT_REPO = "Pre-to-Post-2/200M_SFT_dataset"
SFT_REVISION = "fd343bd28f6a40fc3dab4dcfb6e74c11b7a20b90"

V2R1_ROOT = Path("/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate")
SOURCE_MANIFEST_TEMPLATE = V2R1_ROOT / "source_manifest.json"
SOURCE_MANIFEST_FILE_SHA256 = (
    "7f144d2329628759f2529540bfb9b10692e374d0c8b1933ec43c7c634b979253"
)
SOURCE_SFT_CACHE = V2R1_ROOT / "sft_cache"
SOURCE_SFT_CACHE_HASH = (
    "d82378522d43d5db3e8333588c24b1f864bb9e8ecd46303e1d2cd2e31d31df98"
)
SOURCE_SFT_MAX_ALIGNED_LENGTH = 1_877

# These are the authenticated v3 data identities.  This production revision
# changes the numerical-precision and leading-BOS contracts while preserving
# the exact PT target selection, clean SFT cache, and row order from v3.
EXPECTED_SOURCE_MANIFEST_HASH = (
    "5e2bd529811066c0c9c264eaf39a820f139ad4a4b1e9c9395fca42118e95a275"
)
EXPECTED_SELECTION_HASH = (
    "c5440b93bcf6f35db143ff5b3c22ba91b021b3a01e02a4ec17ba2337c8d29823"
)
EXPECTED_CONTEXT2048_SFT_CACHE_HASH = (
    "6e5b0553366d51ec3c95cb606b063919eed83efb79d11b187ba7d318b0fd60d5"
)
EXPECTED_ORDER_SHA256 = {
    "pt": "4d2d43999220b1abd54c3fb775937323e9cb94ad3150b5467acbc02095d733dd",
    "sft3": "cf86894c99b23475237d741a1a02b527e56dd0bdab9664e38115fb681f108446",
    "mixed_sft1": (
        "8c4dd690c8d9795f041ce8cdb642916b575546fe01fb668e3f0c983cc3de1ec3"
    ),
    "mixed_sft3": (
        "49e805ea4296cff7fa919e5bb4793a865354039bc9a97ea91ad710ae4c6e2f99"
    ),
}
EXPECTED_PRECISION_CONTRACT = {
    "master_parameter_dtype": "float32",
    "optimizer_state_dtype": "float32",
    "forward_backward_dtype": "bfloat16",
    "gradient_dtype": "float32",
    "hf_export_dtype": "float32",
}
EXPECTED_DETERMINISM_CONTRACT = {
    "deterministic_algorithms": True,
    "warn_only": False,
    "cublas_workspace_config": ":4096:8",
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_tf32": False,
    "cudnn_tf32": False,
    "device_specific_seed": True,
}
EXPECTED_TOKEN_IDS_81 = {
    "<bos>": 0,
    "<eos>": 1,
    "<unk>": 2,
}
EXPECTED_TOKEN_IDS_85 = {
    **EXPECTED_TOKEN_IDS_81,
    "<T>": 81,
    "</T>": 82,
    "<sep>": 83,
    "<call_env>": 84,
}
CUDA_BASE_IMAGE = (
    "nvidia/cuda:12.8.0-devel-ubuntu22.04@"
    "sha256:09d8951b943dee03cf8fc841b6ea1f201ad33f82f76567171394853c0f494054"
)
PYTHON_VERSION = "3.11"
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
PINNED_RUNTIME_PACKAGE_VERSIONS = {
    requirement.partition("==")[0]: requirement.partition("==")[2]
    for requirement in PINNED_PIP_PACKAGES
}
RUNTIME_SITE_PACKAGES = sysconfig.get_paths()["purelib"]

DISTRIBUTION_METADATA_CLEANUP_COMMAND = r"""python - <<'PY'
from importlib import metadata
from pathlib import Path
import re
import shutil

site_root = Path('/usr/local/lib/python3.11/site-packages').resolve()
distributions = list(metadata.distributions(path=[str(site_root)]))
for distribution in distributions:
    raw_name = distribution.metadata.get('Name')
    if not raw_name:
        raise RuntimeError('installed distribution has no Name metadata')
    selected_version = metadata.version(str(raw_name))
    if str(distribution.version) == str(selected_version):
        continue
    metadata_path = Path(distribution._path).resolve()
    if metadata_path.parent != site_root or not re.search(
        r'\.(?:dist|egg)-info$', metadata_path.name
    ):
        raise RuntimeError(f'refusing unsafe metadata cleanup: {metadata_path}')
    if metadata_path.is_dir():
        shutil.rmtree(metadata_path)
    elif metadata_path.is_file():
        metadata_path.unlink()

inventory = {}
for distribution in metadata.distributions(path=[str(site_root)]):
    raw_name = distribution.metadata.get('Name')
    if not raw_name:
        raise RuntimeError('installed distribution has no Name metadata')
    name = re.sub(r'[-_.]+', '-', str(raw_name)).lower()
    version = str(distribution.version)
    previous = inventory.setdefault(name, version)
    if previous != version:
        raise RuntimeError(
            f'duplicate distribution survived cleanup: {name} {previous} {version}'
        )
PY"""

ARTIFACT_ROOT = Path(f"/data/{EXPERIMENT_VERSION}")
SOURCE_MANIFEST_PATH = ARTIFACT_ROOT / "source_manifest.json"
SELECTION_PATH = ARTIFACT_ROOT / "pretrain_selection.json"
SFT_CACHE_DIR = ARTIFACT_ROOT / "sft_cache_context2048"
MANIFEST_SET_PATH = ARTIFACT_ROOT / "manifest_set.json"
GATE_ROOT = ARTIFACT_ROOT / "canary_gates"
CHECKPOINT_ROOT = Path(f"/checkpoints/{EXPERIMENT_VERSION}")
DURABLE_LAUNCH_ROOT = CHECKPOINT_ROOT / "_production_launch_ledger"

BASE_CONFIG = "config/configs/interleaved_50m/context2048_vocab_mixing.yaml"
TRAIN_CLI = "scripts/train/train_interleaved_hf.py"
MOUNTED_REPO_DIR_ENV = "CONTEXT2048_MOUNTED_REPO_DIR"
REPO_DIR = Path(
    os.environ.get(MOUNTED_REPO_DIR_ENV)
    or Path(__file__).resolve().parent.parent
).resolve()


def _repository_root(repo_dir: Path) -> Path:
    """Resolve the local repository root without breaking Modal mounts.

    Modal mounts this launcher at ``/root/launch_context2048_vocab_mixing.py``.
    In that layout ``REPO_DIR`` is already ``/root`` and has only one parent.
    The repository dotenv is used only by the local entrypoint, but every
    deployed function still imports this module before it can run.
    """

    return repo_dir.parents[1] if len(repo_dir.parents) > 1 else repo_dir


REPOSITORY_ROOT = _repository_root(REPO_DIR)
REPOSITORY_DOTENV = REPOSITORY_ROOT / ".env"
LOCAL_LAUNCH_RECOVERY_ROOT = (
    REPO_DIR / ".launch-recovery" / EXPERIMENT_VERSION
)


def _require_repo_wandb_api_key(dotenv_path: Path = REPOSITORY_DOTENV) -> None:
    """Fail locally without logging or exporting the repository W&B key."""

    from dotenv import dotenv_values

    if not dotenv_path.is_file():
        raise FileNotFoundError(
            f"non-dry-run actions require repository credential file {dotenv_path}"
        )
    values = dotenv_values(dotenv_path)
    if not str(values.get("WANDB_API_KEY") or "").strip():
        raise RuntimeError(
            f"WANDB_API_KEY is missing or empty in repository credential file {dotenv_path}"
        )


def _wandb_secret_sync_command() -> list[str]:
    """Return the required local command without reading or exposing the key."""

    return [
        "modal",
        "secret",
        "create",
        "--force",
        "--from-dotenv",
        str(REPOSITORY_DOTENV),
        WANDB_SECRET,
    ]


def _steps(records: int, global_sequences: int) -> int:
    return math.ceil(int(records) / int(global_sequences))


MIXED_STEPS = {
    1: _steps(PT_RECORDS + SFT_ROWS, PT_GLOBAL_SEQUENCES),
    3: _steps(PT_RECORDS + 3 * SFT_ROWS, PT_GLOBAL_SEQUENCES),
}

PT_ORDER_SEED = 42
SFT_EPOCH_ORDER_SEEDS = (43, 44, 45)
MIXED_PLACEMENT_SEEDS = {
    1: 20_260_813,
    3: 20_260_815,
}
ORDER_PROVENANCE_SCHEMA = "context2048-production-order-provenance-v1"
PCG64_BIT_GENERATOR = "numpy.random.PCG64"
PERMUTATION_API = "numpy.random.Generator.permutation"
SHUFFLE_API = "numpy.random.Generator.shuffle"
if len(SFT_EPOCH_ORDER_SEEDS) != SFT_EPOCHS:
    raise AssertionError("SFT epoch-order seeds must match SFT_EPOCHS")


def _permutation_component(
    *,
    name: str,
    record_type: str,
    record_count: int,
    seed: int,
) -> dict[str, Any]:
    if record_type == "pretrain":
        encoding = "nonnegative-local-packed-record-index"
    elif record_type == "sft":
        encoding = "negative-one-based-global-sft-row-index"
    else:  # pragma: no cover - only the two authenticated types are valid
        raise ValueError(record_type)
    return {
        "name": name,
        "record_type": record_type,
        "record_count": int(record_count),
        "source_indices": {
            "algorithm": "integer-range",
            "start_inclusive": 0,
            "stop_exclusive": int(record_count),
            "dtype": "little-endian-int64",
        },
        "permutation": {
            "api": PERMUTATION_API,
            "bit_generator": PCG64_BIT_GENERATOR,
            "seed": int(seed),
        },
        "order_encoding": encoding,
    }


def _padding_provenance(
    *,
    unpadded_records: int,
    global_batch_size: int,
) -> dict[str, Any]:
    padding_records = (-int(unpadded_records)) % int(global_batch_size)
    return {
        "algorithm": "append-pad-record-to-next-global-batch-multiple",
        "input_record_count": int(unpadded_records),
        "global_batch_size": int(global_batch_size),
        "pad_record": -(2**63),
        "padding_records": padding_records,
        "output_record_count": int(unpadded_records) + padding_records,
    }


def _production_order_provenance(name: str) -> dict[str, Any]:
    """Return the complete deterministic construction graph for one order."""

    pt_component = _permutation_component(
        name="pretrain",
        record_type="pretrain",
        record_count=PT_RECORDS,
        seed=PT_ORDER_SEED,
    )
    sft_components = [
        _permutation_component(
            name=f"sft_epoch_{epoch + 1}",
            record_type="sft",
            record_count=SFT_ROWS,
            seed=seed,
        )
        for epoch, seed in enumerate(SFT_EPOCH_ORDER_SEEDS)
    ]
    common = {
        "schema": ORDER_PROVENANCE_SCHEMA,
        "numpy_version": PINNED_RUNTIME_PACKAGE_VERSIONS["numpy"],
    }
    if name == "pt":
        return {
            **common,
            "components": [pt_component],
            "composition": [
                {
                    "algorithm": "use-component-order",
                    "inputs": ["pretrain"],
                    "output": "unpadded_order",
                }
            ],
            "padding": _padding_provenance(
                unpadded_records=PT_RECORDS,
                global_batch_size=PT_GLOBAL_SEQUENCES,
            ),
        }
    if name == "sft3":
        return {
            **common,
            "components": sft_components,
            "composition": [
                {
                    "algorithm": "concatenate-in-listed-order",
                    "inputs": [component["name"] for component in sft_components],
                    "output": "unpadded_order",
                }
            ],
            "padding": _padding_provenance(
                unpadded_records=SFT_STAGE_RECORDS,
                global_batch_size=SFT_GLOBAL_SEQUENCES,
            ),
        }
    if name not in {"mixed_sft1", "mixed_sft3"}:
        raise ValueError(f"unknown production order {name!r}")
    copies = 1 if name == "mixed_sft1" else 3
    selected_sft_components = sft_components[:copies]
    sft_records = SFT_ROWS * copies
    return {
        **common,
        "components": [pt_component, *selected_sft_components],
        "composition": [
            {
                "algorithm": "concatenate-in-listed-order",
                "inputs": [
                    component["name"] for component in selected_sft_components
                ],
                "output": "concatenated_sft",
            },
            {
                "algorithm": "stable-binary-placement-from-shuffled-flags",
                "inputs": ["pretrain", "concatenated_sft"],
                "flag_counts": {
                    "pretrain": PT_RECORDS,
                    "sft": sft_records,
                },
                "shuffle": {
                    "api": SHUFFLE_API,
                    "bit_generator": PCG64_BIT_GENERATOR,
                    "seed": MIXED_PLACEMENT_SEEDS[copies],
                },
                "preserves_each_input_relative_order": True,
                "output": "unpadded_order",
            },
        ],
        "padding": _padding_provenance(
            unpadded_records=PT_RECORDS + sft_records,
            global_batch_size=PT_GLOBAL_SEQUENCES,
        ),
    }


EXPECTED_ORDER_PROVENANCE = {
    name: _production_order_provenance(name)
    for name in ("pt", "sft3", "mixed_sft1", "mixed_sft3")
}


@dataclass(frozen=True)
class Experiment:
    key: str
    description: str
    kind: str
    pt_vocab_size: int
    sft_copies: int

    @property
    def stages(self) -> tuple[str, ...]:
        return ("pt", "sft") if self.kind == "staged" else ("mixed",)


EXPERIMENTS: dict[str, Experiment] = {
    "vocab81_then_sft3": Experiment(
        "vocab81_then_sft3",
        "81-token PT, expand to 85 tokens, then clean SFT for 3 epochs",
        "staged",
        81,
        3,
    ),
    "vocab85_then_sft3": Experiment(
        "vocab85_then_sft3",
        "85-token PT, then clean SFT for 3 epochs",
        "staged",
        85,
        3,
    ),
    "mixed_sft1": Experiment(
        "mixed_sft1",
        "85-token mixed PT plus one copy of clean SFT",
        "mixed",
        85,
        1,
    ),
    "mixed_sft3": Experiment(
        "mixed_sft3",
        "85-token mixed PT plus three copies of clean SFT",
        "mixed",
        85,
        3,
    ),
}


def _source_tree_sha256(
    *,
    repo_dir: Path = REPO_DIR,
    launcher_path: Path = Path(__file__).resolve(),
) -> str:
    """Hash uploaded bytes under canonical repository-relative labels."""

    entries: list[tuple[str, Path]] = []
    for relative in ("config", "llm_tokens", "scripts", "training"):
        root = repo_dir / relative
        entries.extend(
            (path.relative_to(repo_dir).as_posix(), path)
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    entries.append(
        (
            "modal_scripts/launch_context2048_vocab_mixing.py",
            launcher_path,
        )
    )
    digest = hashlib.sha256()
    for label, path in sorted(set(entries)):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


COMPUTED_SOURCE_TREE_SHA256 = _source_tree_sha256()
ENV_SOURCE_TREE_SHA256 = os.environ.get(
    "CONTEXT2048_SOURCE_TREE_SHA256", ""
).strip()
if (
    ENV_SOURCE_TREE_SHA256
    and ENV_SOURCE_TREE_SHA256 != COMPUTED_SOURCE_TREE_SHA256
):
    raise RuntimeError(
        "CONTEXT2048_SOURCE_TREE_SHA256 does not match the uploaded source "
        f"tree: {ENV_SOURCE_TREE_SHA256} != {COMPUTED_SOURCE_TREE_SHA256}"
    )
SOURCE_TREE_SHA256 = COMPUTED_SOURCE_TREE_SHA256
LAUNCH_ATOMICITY_GATE_KEY = f"atomicity-gate:{SOURCE_TREE_SHA256}"
WANDB_WRITE_GATE_KEY = f"wandb-write-gate:{SOURCE_TREE_SHA256}"
PRECISION_RESUME_ROOT = (
    CHECKPOINT_ROOT / "precision_resume_canary" / SOURCE_TREE_SHA256
)
PRECISION_RESUME_GATE_PATH = (
    GATE_ROOT / f"precision_resume_{SOURCE_TREE_SHA256}.json"
)
STAGED_SFT_RESUME_VARIANTS = (
    "vocab81_then_sft3",
    "vocab85_then_sft3",
)
STAGED_SFT_RESUME_ROOT = (
    CHECKPOINT_ROOT / "staged_sft_resume_canary" / SOURCE_TREE_SHA256
)
STAGED_SFT_RESUME_GATE_PATH = (
    GATE_ROOT / f"staged_sft_resume_{SOURCE_TREE_SHA256}.json"
)


_LOCAL_UPLOAD_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "wandb",
    }
)


def _ignore_local_upload_artifact(path: Path) -> bool:
    """Keep generated and local-environment files out of the Modal mount."""

    return (
        any(part in _LOCAL_UPLOAD_EXCLUDED_PARTS for part in path.parts)
        or path.name == ".env"
        or path.suffix in {".pyc", ".pyo"}
    )

image = (
    modal.Image.from_registry(
        CUDA_BASE_IMAGE,
        add_python=PYTHON_VERSION,
    )
    .apt_install("curl", "git")
    .pip_install(*PINNED_PIP_PACKAGES)
    # The CUDA base can leave obsolete distribution metadata beside the
    # versions installed above. Remove metadata only; never remove imported
    # package code. The command fails if a path escapes site-packages.
    .run_commands(DISTRIBUTION_METADATA_CLEANUP_COMMAND)
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/root/chess",
            MOUNTED_REPO_DIR_ENV: "/root/chess",
            "WANDB_ENTITY": WANDB_ENTITY,
            "CONTEXT2048_SOURCE_TREE_SHA256": SOURCE_TREE_SHA256,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    .add_local_dir(
        str(REPO_DIR / "scripts"),
        remote_path="/root/chess/scripts",
        ignore=_ignore_local_upload_artifact,
    )
    .add_local_dir(
        str(REPO_DIR / "training"),
        remote_path="/root/chess/training",
        ignore=_ignore_local_upload_artifact,
    )
    .add_local_dir(
        str(REPO_DIR / "config"),
        remote_path="/root/chess/config",
        ignore=_ignore_local_upload_artifact,
    )
    .add_local_dir(
        str(REPO_DIR / "llm_tokens"),
        remote_path="/root/chess/llm_tokens",
        ignore=_ignore_local_upload_artifact,
    )
)

data_volume = modal.Volume.from_name("rl-reasoning-training-data", create_if_missing=False)
checkpoint_volume = modal.Volume.from_name("rl-reasoning-checkpoints", create_if_missing=False)
launch_claims = modal.Dict.from_name(
    LAUNCH_CLAIM_DICT_NAME,
    create_if_missing=True,
)

app = modal.App(
    APP_NAME,
    image=image,
    # Before gates or launches, refresh this stable remote secret from the
    # repository-root .env with `_wandb_secret_sync_command()`.  The .env is
    # never included in the image, source hash, or function inputs.
    secrets=[modal.Secret.from_name(WANDB_SECRET)],
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _modal_client_version() -> str:
    """Return the injected Modal SDK version with a local-wheel fallback."""

    value = str(getattr(modal, "__version__", "") or "").strip()
    if not value:
        value = str(importlib.metadata.version("modal") or "").strip()
    if not value:
        raise RuntimeError("Modal SDK version is unavailable")
    return value


def _modal_runtime_identity() -> dict[str, Any]:
    """Resolve immutable image/app identity only inside a hydrated Modal app."""

    try:
        image_id = image.object_id
        app_id = app.app_id
    except Exception as exc:
        raise RuntimeError(
            "Modal app/image handles are not hydrated; runtime provenance "
            "cannot be authenticated"
        ) from exc
    if not isinstance(image_id, str) or not image_id.startswith("im-"):
        raise RuntimeError(f"invalid hydrated Modal image ID: {image_id!r}")
    if not isinstance(app_id, str) or not app_id.startswith("ap-"):
        raise RuntimeError(f"invalid hydrated Modal app ID: {app_id!r}")
    observed_packages = {
        name: importlib.metadata.version(name)
        for name in sorted(PINNED_RUNTIME_PACKAGE_VERSIONS)
    }
    if observed_packages != PINNED_RUNTIME_PACKAGE_VERSIONS:
        raise RuntimeError(
            "runtime package identity drifted from the image recipe: "
            f"{observed_packages} != {PINNED_RUNTIME_PACKAGE_VERSIONS}"
        )
    full_inventory: dict[str, str] = {}
    for distribution in importlib.metadata.distributions(
        path=[RUNTIME_SITE_PACKAGES]
    ):
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            raise RuntimeError(
                "installed distribution lacks a canonical metadata Name"
            )
        name = re.sub(r"[-_.]+", "-", str(raw_name)).lower()
        version = str(distribution.version)
        previous = full_inventory.setdefault(name, version)
        if previous != version:
            raise RuntimeError(
                f"multiple installed versions for {name}: {previous}, {version}"
            )
    if not full_inventory:
        raise RuntimeError("installed Python distribution inventory is empty")
    inventory_sha256 = hashlib.sha256(
        _canonical_json(full_inventory)
    ).hexdigest()
    return {
        "modal_app_name": APP_NAME,
        "modal_app_id": app_id,
        "modal_image_id": image_id,
        "modal_base_image": CUDA_BASE_IMAGE,
        "modal_client_version": _modal_client_version(),
        "runtime_package_versions": observed_packages,
        "runtime_distribution_count": len(full_inventory),
        "runtime_distribution_inventory_sha256": inventory_sha256,
        "python_version": sys.version,
    }


def _validate_recorded_runtime_identity(recorded: Any) -> None:
    if not isinstance(recorded, Mapping):
        raise RuntimeError("authenticated artifact lacks runtime identity")
    current = _modal_runtime_identity()
    keys = (
        "modal_app_name",
        "modal_app_id",
        "modal_image_id",
        "modal_base_image",
        "modal_client_version",
        "runtime_package_versions",
        "runtime_distribution_count",
        "runtime_distribution_inventory_sha256",
        "python_version",
    )
    drift = {
        key: {"recorded": recorded.get(key), "current": current.get(key)}
        for key in keys
        if recorded.get(key) != current.get(key)
    }
    if drift:
        raise RuntimeError(f"authenticated runtime identity drifted: {drift}")


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _launch_recovery_path(experiment_key: str) -> Path:
    _launch_claim_key(experiment_key)
    return (
        LOCAL_LAUNCH_RECOVERY_ROOT
        / SOURCE_TREE_SHA256
        / f"{experiment_key}.json"
    )


def _write_local_launch_recovery_record(
    *,
    experiment_key: str,
    launch_token: str,
) -> Path:
    """Durably preserve the only raw token before making a remote claim."""

    path = _launch_recovery_path(experiment_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    core = {
        "schema": "context2048-local-launch-recovery-v1",
        "experiment_version": EXPERIMENT_VERSION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "experiment": experiment_key,
        "launch_token": _require_launch_token(launch_token),
        "created_at": _utc_now(),
    }
    record = _self_hash_record(core, hash_field="record_sha256")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"launch recovery record already exists: {path}; use "
            "--action recover-launch instead of creating another token"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _read_local_launch_recovery_record(experiment_key: str) -> dict[str, Any]:
    path = _launch_recovery_path(experiment_key)
    record = _validate_self_hash(
        _load_json(path),
        hash_field="record_sha256",
        label=f"{experiment_key} local launch recovery record",
    )
    expected = {
        "schema": "context2048-local-launch-recovery-v1",
        "experiment_version": EXPERIMENT_VERSION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "experiment": experiment_key,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(f"local launch recovery {key} drifted")
    _require_launch_token(record.get("launch_token"))
    return record


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _launch_claim_key(experiment_key: str) -> str:
    if experiment_key not in EXPERIMENTS:
        raise ValueError(f"unknown experiment: {experiment_key!r}")
    return f"production-claim:{experiment_key}"


def _durable_launch_anchor_path(experiment_key: str) -> Path:
    _launch_claim_key(experiment_key)
    # The output roots are experiment-version scoped, so the durable tombstone
    # must use the same stable namespace.  Source identity remains inside the
    # immutable anchor and a redeploy with changed source fails validation;
    # source-namespacing this path would hide the old anchor after Dict expiry.
    return DURABLE_LAUNCH_ROOT / EXPERIMENT_VERSION / f"{experiment_key}.json"


def _durable_launch_anchor_intent_key(experiment_key: str) -> str:
    _launch_claim_key(experiment_key)
    return f"production-anchor-intent:{experiment_key}"


def _durable_launch_anchor_record(
    *,
    experiment_key: str,
    claim: Mapping[str, Any],
    launch_identity: Mapping[str, Any],
    publisher_recovery: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_roots = launch_identity.get("output_roots")
    if not isinstance(output_roots, Mapping):
        raise RuntimeError(f"{experiment_key} launch identity lacks output roots")
    core = {
        "schema": DURABLE_LAUNCH_ANCHOR_SCHEMA,
        "app_name": APP_NAME,
        "experiment_version": EXPERIMENT_VERSION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "experiment": experiment_key,
        "claim_sha256": claim["claim_sha256"],
        "launch_token_sha256": claim["launch_token_sha256"],
        "launch_identity_sha256": claim["launch_identity_sha256"],
        "output_roots": dict(output_roots),
        # Deterministic across recovery contenders after claim-before-anchor
        # failure; concurrent writers therefore publish identical bytes.
        "claimed_at": claim["claimed_at"],
        "publisher_recovery": (
            dict(publisher_recovery)
            if publisher_recovery is not None
            else None
        ),
    }
    return _self_hash_record(core, hash_field="anchor_sha256")


def _validate_durable_launch_anchor(
    *,
    experiment_key: str,
    claim: Mapping[str, Any],
    launch_identity: Mapping[str, Any],
    launch_token: str | None = None,
) -> dict[str, Any]:
    path = _durable_launch_anchor_path(experiment_key)
    anchor = _validate_self_hash(
        _load_json(path),
        hash_field="anchor_sha256",
        label=f"{experiment_key} durable production launch anchor",
    )
    publisher_recovery = anchor.get("publisher_recovery")
    if publisher_recovery is not None:
        recovery = _validate_self_hash(
            publisher_recovery,
            hash_field="recovery_sha256",
            label=f"{experiment_key} durable anchor publisher recovery",
        )
        if set(recovery) != {
            "schema",
            "intent_sha256",
            "prior_dispatcher_function_call_id",
            "terminal_call",
            "recovery_sha256",
        } or recovery.get("schema") != (
            "context2048-durable-anchor-publisher-recovery-v1"
        ):
            raise RuntimeError(
                f"{experiment_key} durable anchor recovery fields drifted"
            )
        _validate_terminal_call_evidence_record(
            recovery.get("terminal_call"),
            expected_function_call_id=str(
                recovery.get("prior_dispatcher_function_call_id") or ""
            ),
        )
    expected = _durable_launch_anchor_record(
        experiment_key=experiment_key,
        claim=claim,
        launch_identity=launch_identity,
        publisher_recovery=(
            publisher_recovery if isinstance(publisher_recovery, Mapping) else None
        ),
    )
    if anchor != expected:
        raise RuntimeError(f"{experiment_key} durable launch anchor drifted")
    if launch_token is not None and anchor["launch_token_sha256"] != (
        _launch_token_sha256(launch_token)
    ):
        raise RuntimeError(f"{experiment_key} durable launch token does not match")
    return anchor


def _publish_durable_launch_anchor(
    *,
    experiment_key: str,
    claim: Mapping[str, Any],
    launch_identity: Mapping[str, Any],
    launch_token: str,
    volume: Any = None,
    publisher_recovery: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist the exact-once anchor before any GPU FunctionCall is spawned."""

    mounted_volume = checkpoint_volume if volume is None else volume
    proposed = _durable_launch_anchor_record(
        experiment_key=experiment_key,
        claim=claim,
        launch_identity=launch_identity,
        publisher_recovery=publisher_recovery,
    )
    path = _durable_launch_anchor_path(experiment_key)
    # Never trust a file that exists only in this container after an
    # indeterminate prior commit. Reconcile from the durable Volume first.
    mounted_volume.reload()
    if path.exists():
        return _validate_durable_launch_anchor(
            experiment_key=experiment_key,
            claim=claim,
            launch_identity=launch_identity,
            launch_token=launch_token,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        mounted_volume.reload()
        return _validate_durable_launch_anchor(
            experiment_key=experiment_key,
            claim=claim,
            launch_identity=launch_identity,
            launch_token=launch_token,
        )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(proposed, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        mounted_volume.commit()
        mounted_volume.reload()
    except BaseException:
        # A failed commit can be indeterminate. Leave the anchor in place so
        # every subsequent launch fails closed instead of deleting evidence.
        raise
    observed = _validate_durable_launch_anchor(
        experiment_key=experiment_key,
        claim=claim,
        launch_identity=launch_identity,
        launch_token=launch_token,
    )
    if observed != proposed:
        raise RuntimeError(f"{experiment_key} durable launch publication drifted")
    return observed


def _ensure_durable_launch_anchor(
    store: Any,
    *,
    experiment_key: str,
    claim: Mapping[str, Any],
    launch_identity: Mapping[str, Any],
    launch_token: str,
    dispatcher_function_call_id: str,
    recovery: bool,
    volume: Any = None,
) -> dict[str, Any]:
    """Choose one publisher with Dict CAS, then require durable readback."""

    if not re.fullmatch(r"fc-[0-9A-Za-z]+", dispatcher_function_call_id):
        raise RuntimeError("durable anchor publisher has invalid FunctionCall ID")
    mounted_volume = checkpoint_volume if volume is None else volume
    mounted_volume.reload()
    path = _durable_launch_anchor_path(experiment_key)
    if path.exists():
        return _validate_durable_launch_anchor(
            experiment_key=experiment_key,
            claim=claim,
            launch_identity=launch_identity,
            launch_token=launch_token,
        )
    intent = _self_hash_record(
        {
            "schema": DURABLE_LAUNCH_ANCHOR_INTENT_SCHEMA,
            "experiment": experiment_key,
            "claim_sha256": claim["claim_sha256"],
            "dispatcher_function_call_id": dispatcher_function_call_id,
        },
        hash_field="intent_sha256",
    )
    key = _durable_launch_anchor_intent_key(experiment_key)
    won = bool(store.put(key, intent, skip_if_exists=True))
    observed = _validate_self_hash(
        store.get(key, None),
        hash_field="intent_sha256",
        label=f"{experiment_key} durable anchor publication intent",
    )
    publisher_recovery: Mapping[str, Any] | None = None
    if observed != intent:
        mounted_volume.reload()
        if path.exists():
            return _validate_durable_launch_anchor(
                experiment_key=experiment_key,
                claim=claim,
                launch_identity=launch_identity,
                launch_token=launch_token,
            )
        if not recovery:
            raise RuntimeError(
                f"{experiment_key} durable anchor publication is owned by a "
                "different dispatcher and has not completed"
            )
        prior_dispatcher = str(
            observed.get("dispatcher_function_call_id") or ""
        )
        terminal_call = _inspect_terminal_unsuccessful_function_call(
            prior_dispatcher
        )
        publisher_recovery = _self_hash_record(
            {
                "schema": "context2048-durable-anchor-publisher-recovery-v1",
                "intent_sha256": observed["intent_sha256"],
                "prior_dispatcher_function_call_id": prior_dispatcher,
                "terminal_call": terminal_call,
            },
            hash_field="recovery_sha256",
        )
    return _publish_durable_launch_anchor(
        experiment_key=experiment_key,
        claim=claim,
        launch_identity=launch_identity,
        launch_token=launch_token,
        volume=mounted_volume,
        publisher_recovery=publisher_recovery,
    )


def _launch_attempt_key(experiment_key: str, generation: int) -> str:
    _launch_claim_key(experiment_key)
    if (
        isinstance(generation, bool)
        or not 0 <= int(generation) < MAX_LAUNCH_GENERATIONS
    ):
        raise ValueError(f"invalid launch generation: {generation!r}")
    return f"production-attempt:{experiment_key}:{int(generation):04d}"


def _launch_execution_key(experiment_key: str, generation: int = 0) -> str:
    _launch_claim_key(experiment_key)
    if (
        isinstance(generation, bool)
        or not 0 <= int(generation) < MAX_LAUNCH_GENERATIONS
    ):
        raise ValueError(f"invalid launch generation: {generation!r}")
    return f"production-execution:{experiment_key}:{int(generation):04d}"


def _launch_status_key(experiment_key: str, generation: int = 0) -> str:
    _launch_claim_key(experiment_key)
    if (
        isinstance(generation, bool)
        or not 0 <= int(generation) < MAX_LAUNCH_GENERATIONS
    ):
        raise ValueError(f"invalid launch generation: {generation!r}")
    return f"production-status:{experiment_key}:{int(generation):04d}"


def _launch_completion_key(experiment_key: str) -> str:
    return f"production-completion:{experiment_key}"


def _require_launch_token(launch_token: str) -> str:
    if not isinstance(launch_token, str) or not re.fullmatch(
        r"[0-9a-f]{64}", launch_token
    ):
        raise RuntimeError("production launch requires a 256-bit hexadecimal token")
    return launch_token


def _launch_token_sha256(launch_token: str) -> str:
    token = _require_launch_token(launch_token)
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _self_hash_record(
    record: Mapping[str, Any],
    *,
    hash_field: str,
) -> dict[str, Any]:
    core = {key: value for key, value in record.items() if key != hash_field}
    return {
        **core,
        hash_field: hashlib.sha256(_canonical_json(core)).hexdigest(),
    }


def _validate_self_hash(
    record: Any,
    *,
    hash_field: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"{label} is missing or is not an object")
    value = dict(record)
    recorded = value.pop(hash_field, None)
    expected = hashlib.sha256(_canonical_json(value)).hexdigest()
    if recorded != expected:
        raise RuntimeError(f"{label} self hash drifted")
    return {**value, hash_field: recorded}


def _new_launch_claim(
    *,
    experiment_key: str,
    launch_token: str,
    launch_identity: Mapping[str, Any],
    claimed_at: str | None = None,
) -> dict[str, Any]:
    identity = dict(launch_identity)
    core = {
        "schema": LAUNCH_CLAIM_SCHEMA,
        "experiment": experiment_key,
        "launch_token_sha256": _launch_token_sha256(launch_token),
        "launch_identity": identity,
        "launch_identity_sha256": hashlib.sha256(
            _canonical_json(identity)
        ).hexdigest(),
        "claimed_at": claimed_at or _utc_now(),
    }
    return _self_hash_record(core, hash_field="claim_sha256")


def _validate_launch_claim(
    record: Any,
    *,
    experiment_key: str,
    expected_identity: Mapping[str, Any],
    launch_token: str | None = None,
) -> dict[str, Any]:
    claim = _validate_self_hash(
        record,
        hash_field="claim_sha256",
        label=f"{experiment_key} production launch claim",
    )
    required_fields = {
        "schema",
        "experiment",
        "launch_token_sha256",
        "launch_identity",
        "launch_identity_sha256",
        "claimed_at",
        "claim_sha256",
    }
    if set(claim) != required_fields:
        raise RuntimeError(
            f"{experiment_key} production launch claim fields drifted"
        )
    if claim.get("schema") != LAUNCH_CLAIM_SCHEMA:
        raise RuntimeError(f"{experiment_key} production launch claim schema drifted")
    if claim.get("experiment") != experiment_key:
        raise RuntimeError(
            f"{experiment_key} production launch claim experiment drifted"
        )
    identity = claim.get("launch_identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError(f"{experiment_key} launch identity is missing")
    identity_sha256 = hashlib.sha256(_canonical_json(identity)).hexdigest()
    if claim.get("launch_identity_sha256") != identity_sha256:
        raise RuntimeError(f"{experiment_key} launch identity hash drifted")
    if dict(identity) != dict(expected_identity):
        raise RuntimeError(
            f"{experiment_key} has an existing claim for a different launch identity"
        )
    if launch_token is not None and claim.get("launch_token_sha256") != (
        _launch_token_sha256(launch_token)
    ):
        raise RuntimeError(f"{experiment_key} production launch token does not match")
    return claim


def _acquire_launch_claim(
    store: Any,
    *,
    experiment_key: str,
    launch_token: str,
    launch_identity: Mapping[str, Any],
    claimed_at: str | None = None,
) -> dict[str, Any]:
    """Atomically acquire one immutable production claim, without stealing."""

    key = _launch_claim_key(experiment_key)
    existing = store.get(key, None)
    if existing is not None:
        claim = _validate_launch_claim(
            existing,
            experiment_key=experiment_key,
            expected_identity=launch_identity,
        )
        return {"outcome": "existing_claim", "claim": claim}
    proposed = _new_launch_claim(
        experiment_key=experiment_key,
        launch_token=launch_token,
        launch_identity=launch_identity,
        claimed_at=claimed_at,
    )
    won = bool(store.put(key, proposed, skip_if_exists=True))
    observed = store.get(key, None)
    claim = _validate_launch_claim(
        observed,
        experiment_key=experiment_key,
        expected_identity=launch_identity,
    )
    if won:
        if claim != proposed:
            raise RuntimeError(
                f"{experiment_key} atomic claim winner differs from its write"
            )
        return {"outcome": "acquired", "claim": claim}
    return {"outcome": "existing_claim", "claim": claim}


def _new_launch_attempt(
    *,
    experiment_key: str,
    claim: Mapping[str, Any],
    generation: int,
    dispatcher_function_call_id: str,
    recovery_evidence: Mapping[str, Any] | None,
    created_at: str | None = None,
) -> dict[str, Any]:
    _launch_attempt_key(experiment_key, generation)
    if not isinstance(dispatcher_function_call_id, str) or not re.fullmatch(
        r"fc-[0-9A-Za-z]+", dispatcher_function_call_id
    ):
        raise RuntimeError(
            f"invalid dispatcher FunctionCall ID: {dispatcher_function_call_id!r}"
        )
    if int(generation) == 0 and recovery_evidence is not None:
        raise RuntimeError("initial launch attempt cannot contain recovery evidence")
    if int(generation) > 0 and not isinstance(recovery_evidence, Mapping):
        raise RuntimeError("recovery launch attempt requires terminal evidence")
    core = {
        "schema": LAUNCH_ATTEMPT_SCHEMA,
        "experiment": experiment_key,
        "generation": int(generation),
        "claim_sha256": claim["claim_sha256"],
        "launch_token_sha256": claim["launch_token_sha256"],
        "launch_identity_sha256": claim["launch_identity_sha256"],
        "dispatcher_function_call_id": dispatcher_function_call_id,
        "recovery_evidence": (
            dict(recovery_evidence) if recovery_evidence is not None else None
        ),
        "created_at": created_at or _utc_now(),
    }
    return _self_hash_record(core, hash_field="attempt_sha256")


def _validate_launch_attempt(
    record: Any,
    *,
    experiment_key: str,
    generation: int,
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    attempt = _validate_self_hash(
        record,
        hash_field="attempt_sha256",
        label=f"{experiment_key} generation {generation} launch attempt",
    )
    required = {
        "schema",
        "experiment",
        "generation",
        "claim_sha256",
        "launch_token_sha256",
        "launch_identity_sha256",
        "dispatcher_function_call_id",
        "recovery_evidence",
        "created_at",
        "attempt_sha256",
    }
    if set(attempt) != required:
        raise RuntimeError(f"{experiment_key} launch attempt fields drifted")
    expected = {
        "schema": LAUNCH_ATTEMPT_SCHEMA,
        "experiment": experiment_key,
        "generation": int(generation),
        "claim_sha256": claim["claim_sha256"],
        "launch_token_sha256": claim["launch_token_sha256"],
        "launch_identity_sha256": claim["launch_identity_sha256"],
    }
    for key, value in expected.items():
        if attempt.get(key) != value:
            raise RuntimeError(
                f"{experiment_key} generation {generation} attempt {key} drifted"
            )
    dispatcher_id = attempt.get("dispatcher_function_call_id")
    if not isinstance(dispatcher_id, str) or not re.fullmatch(
        r"fc-[0-9A-Za-z]+", dispatcher_id
    ):
        raise RuntimeError(f"{experiment_key} launch attempt dispatcher drifted")
    evidence = attempt.get("recovery_evidence")
    if generation == 0 and evidence is not None:
        raise RuntimeError(f"{experiment_key} initial attempt has recovery evidence")
    if generation > 0 and not isinstance(evidence, Mapping):
        raise RuntimeError(f"{experiment_key} recovery attempt lacks evidence")
    return attempt


def _validate_terminal_call_evidence_record(
    record: Any,
    *,
    expected_function_call_id: str,
) -> dict[str, Any]:
    evidence = _validate_self_hash(
        record,
        hash_field="evidence_sha256",
        label="terminal unsuccessful FunctionCall evidence",
    )
    if set(evidence) != {
        "schema",
        "function_call_id",
        "result_category",
        "exception_type",
        "evidence_sha256",
    }:
        raise RuntimeError("terminal FunctionCall evidence fields drifted")
    if evidence.get("schema") != TERMINAL_CALL_EVIDENCE_SCHEMA:
        raise RuntimeError("terminal FunctionCall evidence schema drifted")
    if evidence.get("function_call_id") != expected_function_call_id:
        raise RuntimeError("terminal FunctionCall evidence root drifted")
    allowed = {
        "application_failure": {
            "builtins.RuntimeError",
        },
        "function_timeout": {
            "modal.exception.FunctionTimeoutError",
        },
        "remote_terminal_failure": {
            "modal.exception.RemoteError",
        },
    }
    category = evidence.get("result_category")
    if category not in allowed:
        raise RuntimeError("terminal FunctionCall evidence is not unsuccessful")
    if evidence.get("exception_type") not in allowed[category]:
        raise RuntimeError("terminal FunctionCall exception type drifted")
    return evidence


def _validate_attempt_recovery_chain(
    store: Any,
    *,
    experiment_key: str,
    claim: Mapping[str, Any],
    attempt: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
) -> None:
    generation = int(attempt["generation"])
    if generation == 0:
        if previous is not None or attempt.get("recovery_evidence") is not None:
            raise RuntimeError(f"{experiment_key} initial attempt chain drifted")
        return
    if previous is None or int(previous["generation"]) != generation - 1:
        raise RuntimeError(f"{experiment_key} recovery attempt has no predecessor")
    recovery = _validate_self_hash(
        attempt.get("recovery_evidence"),
        hash_field="recovery_sha256",
        label=f"{experiment_key} generation {generation} recovery evidence",
    )
    if set(recovery) != {
        "schema",
        "prior_generation",
        "prior_attempt_sha256",
        "prior_execution_sha256",
        "prior_call_id",
        "terminal_call",
        "checkpoint_state",
        "recovery_sha256",
    }:
        raise RuntimeError(f"{experiment_key} recovery evidence fields drifted")
    expected = {
        "schema": "context2048-launch-recovery-evidence-v1",
        "prior_generation": generation - 1,
        "prior_attempt_sha256": previous["attempt_sha256"],
    }
    for key, value in expected.items():
        if recovery.get(key) != value:
            raise RuntimeError(f"{experiment_key} recovery evidence {key} drifted")
    prior_call_id = str(recovery.get("prior_call_id") or "")
    if not re.fullmatch(r"fc-[0-9A-Za-z]+", prior_call_id):
        raise RuntimeError(f"{experiment_key} recovery prior call ID drifted")
    recorded_execution_sha256 = recovery.get("prior_execution_sha256")
    if not isinstance(recorded_execution_sha256, str):
        raise RuntimeError(f"{experiment_key} recovery lacks a generation resolution")
    prior_resolution = _validate_generation_resolution(
        store.get(
            _launch_execution_key(experiment_key, generation - 1),
            None,
        ),
        experiment_key=experiment_key,
        claim=claim,
        attempt=previous,
        generation=generation - 1,
    )
    if prior_resolution.get("execution_sha256") != recorded_execution_sha256:
        raise RuntimeError(f"{experiment_key} recovery prior resolution drifted")
    if prior_resolution.get("function_call_id") != prior_call_id:
        raise RuntimeError(f"{experiment_key} recovery prior call drifted")
    terminal_call = _validate_terminal_call_evidence_record(
        recovery.get("terminal_call"),
        expected_function_call_id=prior_call_id,
    )
    checkpoint_state = _validate_recovery_checkpoint_state(
        recovery.get("checkpoint_state"),
        label=f"{experiment_key} generation {generation} recovery",
    )
    if checkpoint_state.get("state") != "resumable":
        raise RuntimeError(f"{experiment_key} recovery checkpoint is not resumable")
    if prior_resolution.get("kind") == "recovery_closed":
        if prior_resolution.get("terminal_call") != terminal_call:
            raise RuntimeError(f"{experiment_key} recovery closure terminal drifted")
        if prior_resolution.get("checkpoint_state") != checkpoint_state:
            raise RuntimeError(f"{experiment_key} recovery closure checkpoint drifted")


def _current_launch_attempt(
    store: Any,
    *,
    experiment_key: str,
    claim: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the highest contiguous immutable generation; gaps fail closed."""

    current: dict[str, Any] | None = None
    for generation in range(MAX_LAUNCH_GENERATIONS):
        record = store.get(_launch_attempt_key(experiment_key, generation), None)
        if record is None:
            if generation + 1 < MAX_LAUNCH_GENERATIONS and store.get(
                _launch_attempt_key(experiment_key, generation + 1), None
            ) is not None:
                raise RuntimeError(f"{experiment_key} launch generations contain a gap")
            return current
        observed = _validate_launch_attempt(
            record,
            experiment_key=experiment_key,
            generation=generation,
            claim=claim,
        )
        _validate_attempt_recovery_chain(
            store,
            experiment_key=experiment_key,
            claim=claim,
            attempt=observed,
            previous=current,
        )
        current = observed
    raise RuntimeError(f"{experiment_key} exceeded maximum launch generations")


def _acquire_launch_attempt(
    store: Any,
    *,
    experiment_key: str,
    claim: Mapping[str, Any],
    generation: int,
    dispatcher_function_call_id: str,
    recovery_evidence: Mapping[str, Any] | None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """CAS one immutable generation so concurrent dispatchers cannot duplicate."""

    key = _launch_attempt_key(experiment_key, generation)
    existing = store.get(key, None)
    if existing is not None:
        return {
            "outcome": "attempt_exists",
            "attempt": _validate_launch_attempt(
                existing,
                experiment_key=experiment_key,
                generation=generation,
                claim=claim,
            ),
        }
    current = _current_launch_attempt(
        store,
        experiment_key=experiment_key,
        claim=claim,
    )
    expected_generation = 0 if current is None else int(current["generation"]) + 1
    if int(generation) != expected_generation:
        raise RuntimeError(
            f"{experiment_key} attempt generation {generation} is not the next "
            f"generation {expected_generation}"
        )
    proposed = _new_launch_attempt(
        experiment_key=experiment_key,
        claim=claim,
        generation=generation,
        dispatcher_function_call_id=dispatcher_function_call_id,
        recovery_evidence=recovery_evidence,
        created_at=created_at,
    )
    won = bool(store.put(key, proposed, skip_if_exists=True))
    observed = _validate_launch_attempt(
        store.get(key, None),
        experiment_key=experiment_key,
        generation=generation,
        claim=claim,
    )
    if won and observed != proposed:
        raise RuntimeError(f"{experiment_key} launch-attempt CAS winner drifted")
    return {
        "outcome": "attempt_acquired" if won else "attempt_exists",
        "attempt": observed,
    }


def _execution_binding(
    *,
    experiment_key: str,
    claim: Mapping[str, Any],
    attempt: Mapping[str, Any],
    generation: int,
    function_call_id: str,
) -> dict[str, Any]:
    if not isinstance(function_call_id, str) or not re.fullmatch(
        r"fc-[0-9A-Za-z]+", function_call_id
    ):
        raise RuntimeError(f"invalid Modal FunctionCall ID: {function_call_id!r}")
    core = {
        "schema": LAUNCH_EXECUTION_SCHEMA,
        "experiment": experiment_key,
        "generation": int(generation),
        "claim_sha256": claim["claim_sha256"],
        "attempt_sha256": attempt["attempt_sha256"],
        "launch_token_sha256": claim["launch_token_sha256"],
        "kind": "worker",
        "function_call_id": function_call_id,
        "terminal_call": None,
        "checkpoint_state": None,
    }
    return _self_hash_record(core, hash_field="execution_sha256")


def _recovery_closure(
    *,
    experiment_key: str,
    claim: Mapping[str, Any],
    attempt: Mapping[str, Any],
    generation: int,
    dispatcher_function_call_id: str,
    terminal_call: Mapping[str, Any],
    checkpoint_state: Mapping[str, Any],
) -> dict[str, Any]:
    if dispatcher_function_call_id != attempt.get("dispatcher_function_call_id"):
        raise RuntimeError(
            f"{experiment_key} generation {generation} recovery closure "
            "dispatcher drifted"
        )
    _validate_terminal_call_evidence_record(
        terminal_call,
        expected_function_call_id=dispatcher_function_call_id,
    )
    _validate_recovery_checkpoint_state(
        checkpoint_state,
        label=f"{experiment_key} generation {generation} recovery closure",
    )
    core = {
        "schema": LAUNCH_EXECUTION_SCHEMA,
        "experiment": experiment_key,
        "generation": int(generation),
        "claim_sha256": claim["claim_sha256"],
        "attempt_sha256": attempt["attempt_sha256"],
        "launch_token_sha256": claim["launch_token_sha256"],
        "kind": "recovery_closed",
        "function_call_id": dispatcher_function_call_id,
        "terminal_call": dict(terminal_call),
        "checkpoint_state": dict(checkpoint_state),
    }
    return _self_hash_record(core, hash_field="execution_sha256")


def _validate_recovery_checkpoint_state(
    record: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"{label} checkpoint evidence drifted")
    checkpoint = dict(record)
    if set(checkpoint) != {"state", "has_output", "stages", "completion"}:
        raise RuntimeError(f"{label} checkpoint evidence fields drifted")
    state = checkpoint.get("state")
    if state not in {"resumable", "complete"}:
        raise RuntimeError(f"{label} checkpoint state drifted")
    if not isinstance(checkpoint.get("has_output"), bool):
        raise RuntimeError(f"{label} checkpoint output flag drifted")
    if not isinstance(checkpoint.get("stages"), Mapping):
        raise RuntimeError(f"{label} checkpoint stage evidence drifted")
    completion = checkpoint.get("completion")
    if state == "complete" and not isinstance(completion, Mapping):
        raise RuntimeError(f"{label} complete checkpoint lacks completion evidence")
    if state == "resumable" and completion is not None:
        raise RuntimeError(f"{label} resumable checkpoint claims completion")
    return checkpoint


def _validate_generation_resolution(
    record: Any,
    *,
    experiment_key: str,
    claim: Mapping[str, Any],
    attempt: Mapping[str, Any],
    generation: int,
) -> dict[str, Any]:
    execution = _validate_self_hash(
        record,
        hash_field="execution_sha256",
        label=f"{experiment_key} generation {generation} resolution",
    )
    required = {
        "schema",
        "experiment",
        "generation",
        "claim_sha256",
        "attempt_sha256",
        "launch_token_sha256",
        "kind",
        "function_call_id",
        "terminal_call",
        "checkpoint_state",
        "execution_sha256",
    }
    if set(execution) != required:
        raise RuntimeError(f"{experiment_key} generation resolution fields drifted")
    expected = {
        "schema": LAUNCH_EXECUTION_SCHEMA,
        "experiment": experiment_key,
        "generation": int(generation),
        "claim_sha256": claim["claim_sha256"],
        "attempt_sha256": attempt["attempt_sha256"],
        "launch_token_sha256": claim["launch_token_sha256"],
    }
    for key, value in expected.items():
        if execution.get(key) != value:
            raise RuntimeError(f"{experiment_key} generation resolution {key} drifted")
    function_call_id = execution.get("function_call_id")
    if not isinstance(function_call_id, str) or not re.fullmatch(
        r"fc-[0-9A-Za-z]+", function_call_id
    ):
        raise RuntimeError(
            f"{experiment_key} generation resolution FunctionCall ID drifted"
        )
    kind = execution.get("kind")
    if kind == "worker":
        if execution.get("terminal_call") is not None:
            raise RuntimeError(f"{experiment_key} worker resolution is terminal")
        if execution.get("checkpoint_state") is not None:
            raise RuntimeError(
                f"{experiment_key} worker resolution contains checkpoint evidence"
            )
    elif kind == "recovery_closed":
        if function_call_id != attempt.get("dispatcher_function_call_id"):
            raise RuntimeError(
                f"{experiment_key} generation {generation} recovery closure "
                "dispatcher drifted"
            )
        _validate_terminal_call_evidence_record(
            execution.get("terminal_call"),
            expected_function_call_id=function_call_id,
        )
        _validate_recovery_checkpoint_state(
            execution.get("checkpoint_state"),
            label=f"{experiment_key} generation {generation} recovery closure",
        )
    else:
        raise RuntimeError(f"{experiment_key} generation resolution kind drifted")
    return execution


def _validate_execution_binding(
    record: Any,
    *,
    experiment_key: str,
    claim: Mapping[str, Any],
    attempt: Mapping[str, Any],
    generation: int,
) -> dict[str, Any]:
    execution = _validate_generation_resolution(
        record,
        experiment_key=experiment_key,
        claim=claim,
        attempt=attempt,
        generation=generation,
    )
    if execution.get("kind") != "worker":
        raise RuntimeError(
            f"{experiment_key} generation {generation} is closed for recovery"
        )
    return execution


def _begin_claimed_worker(
    store: Any,
    *,
    experiment_key: str,
    launch_token: str,
    expected_identity: Mapping[str, Any],
    generation: int,
    function_call_id: str,
) -> dict[str, Any]:
    """Bind the claim to one Modal call; retries of that call remain valid."""

    claim = _validate_launch_claim(
        store.get(_launch_claim_key(experiment_key), None),
        experiment_key=experiment_key,
        expected_identity=expected_identity,
        launch_token=launch_token,
    )
    _validate_durable_launch_anchor(
        experiment_key=experiment_key,
        claim=claim,
        launch_identity=expected_identity,
        launch_token=launch_token,
    )
    current = _current_launch_attempt(
        store,
        experiment_key=experiment_key,
        claim=claim,
    )
    if current is None or int(current["generation"]) != int(generation):
        observed_generation = None if current is None else current["generation"]
        raise RuntimeError(
            f"{experiment_key} worker generation {generation} is not current "
            f"({observed_generation!r})"
        )
    proposed = _execution_binding(
        experiment_key=experiment_key,
        claim=claim,
        attempt=current,
        generation=generation,
        function_call_id=function_call_id,
    )
    key = _launch_execution_key(experiment_key, generation)
    won = bool(store.put(key, proposed, skip_if_exists=True))
    observed = _validate_execution_binding(
        store.get(key, None),
        experiment_key=experiment_key,
        claim=claim,
        attempt=current,
        generation=generation,
    )
    if observed != proposed:
        raise RuntimeError(
            f"{experiment_key} generation {generation} is already bound to a "
            "different Modal FunctionCall"
        )
    latest = _current_launch_attempt(
        store,
        experiment_key=experiment_key,
        claim=claim,
    )
    if latest is None or int(latest["generation"]) != int(generation):
        latest_generation = None if latest is None else latest["generation"]
        raise RuntimeError(
            f"{experiment_key} worker generation {generation} lost current "
            f"generation during binding ({latest_generation!r})"
        )
    return {
        "claim": claim,
        "attempt": current,
        "execution": observed,
        "new_binding": won,
    }


def _update_launch_status(
    store: Any,
    *,
    experiment_key: str,
    launch_token: str,
    expected_identity: Mapping[str, Any],
    generation: int,
    function_call_id: str,
    state: str,
    stage: str | None,
    detail: str,
    completion: Mapping[str, Any] | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    if state not in {"running", "failed", "complete"}:
        raise ValueError(f"invalid launch state: {state!r}")
    binding = _begin_claimed_worker(
        store,
        experiment_key=experiment_key,
        launch_token=launch_token,
        expected_identity=expected_identity,
        generation=generation,
        function_call_id=function_call_id,
    )
    if state == "complete" and not isinstance(completion, Mapping):
        raise RuntimeError("completion status requires authenticated completion evidence")
    if state != "complete" and completion is not None:
        raise RuntimeError("only complete status may contain completion evidence")
    core: dict[str, Any] = {
        "schema": LAUNCH_STATUS_SCHEMA,
        "experiment": experiment_key,
        "generation": int(generation),
        "claim_sha256": binding["claim"]["claim_sha256"],
        "attempt_sha256": binding["attempt"]["attempt_sha256"],
        "execution_sha256": binding["execution"]["execution_sha256"],
        "function_call_id": function_call_id,
        "state": state,
        "stage": stage,
        "detail": str(detail),
        "updated_at": updated_at or _utc_now(),
        "completion": dict(completion) if completion is not None else None,
    }
    status = _self_hash_record(core, hash_field="status_sha256")
    if state == "complete":
        key = _launch_completion_key(experiment_key)
        added = bool(store.put(key, status, skip_if_exists=True))
        observed = _validate_self_hash(
            store.get(key, None),
            hash_field="status_sha256",
            label=f"{experiment_key} immutable production completion",
        )
        if observed != status:
            raise RuntimeError(
                f"{experiment_key} already has different completion evidence"
            )
        return observed
    if store.get(_launch_completion_key(experiment_key), None) is not None:
        raise RuntimeError(
            f"{experiment_key} is already complete; refusing a status downgrade"
        )
    store.put(_launch_status_key(experiment_key, generation), status)
    observed = _validate_self_hash(
        store.get(_launch_status_key(experiment_key, generation), None),
        hash_field="status_sha256",
        label=f"{experiment_key} production launch status",
    )
    if observed != status:
        raise RuntimeError(f"{experiment_key} launch status write was not durable")
    return status


def _validate_source_artifacts() -> None:
    if _sha256_file(SOURCE_MANIFEST_TEMPLATE) != SOURCE_MANIFEST_FILE_SHA256:
        raise RuntimeError("Pinned source manifest file drifted")
    source_cache = _load_json(SOURCE_SFT_CACHE / "metadata.json")
    if source_cache.get("cache_hash") != SOURCE_SFT_CACHE_HASH:
        raise RuntimeError("Pinned clean SFT cache drifted")
    if int(source_cache.get("num_rows", -1)) != SFT_ROWS:
        raise RuntimeError("Pinned clean SFT row count drifted")
    if int(source_cache.get("supervised_targets", -1)) != SFT_SUPERVISED_TARGETS:
        raise RuntimeError("Pinned clean SFT supervised-target count drifted")


def _prepare_context2048_sft_cache() -> dict[str, Any]:
    import numpy as np
    from training.interleaved_data import SFTCache, _hash_dict

    if (SFT_CACHE_DIR / "metadata.json").is_file():
        cache = SFTCache.load(SFT_CACHE_DIR, verify_large_files=True)
        metadata = _load_json(SFT_CACHE_DIR / "metadata.json")
        if cache.sequence_length != CONTEXT_LENGTH:
            raise RuntimeError("Existing SFT cache has the wrong context length")
        if metadata.get("derived_from_cache_hash") != SOURCE_SFT_CACHE_HASH:
            raise RuntimeError("Existing SFT cache has the wrong source cache")
        return metadata

    source = SFTCache.load(SOURCE_SFT_CACHE, verify_large_files=True)
    offsets = np.load(
        source.directory / "offsets.npy", mmap_mode="r", allow_pickle=False
    )
    maximum = int(np.diff(offsets).max())
    if maximum != SOURCE_SFT_MAX_ALIGNED_LENGTH or maximum > CONTEXT_LENGTH:
        raise RuntimeError(
            f"SFT length audit drifted: max={maximum}, expected "
            f"{SOURCE_SFT_MAX_ALIGNED_LENGTH} and <= {CONTEXT_LENGTH}"
        )

    SFT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("input_ids.i32", "labels.i32", "offsets.npy"):
        shutil.copyfile(source.directory / name, SFT_CACHE_DIR / name)
    metadata = _load_json(source.directory / "metadata.json")
    metadata["sequence_length"] = CONTEXT_LENGTH
    metadata["derived_from_cache_hash"] = SOURCE_SFT_CACHE_HASH
    metadata["maximum_aligned_length"] = maximum
    metadata["packing"] = "one-sft-row-per-sequence-right-padded-by-collator"
    metadata["cache_hash"] = _hash_dict(metadata, "cache_hash")
    _atomic_json(SFT_CACHE_DIR / "metadata.json", metadata)
    SFTCache.load(SFT_CACHE_DIR, verify_large_files=True)
    return metadata


def _pad_order(order, global_batch_size: int):
    import numpy as np
    from training.interleaved_data import PAD_RECORD

    padding = (-len(order)) % int(global_batch_size)
    if padding:
        order = np.concatenate(
            (order, np.full(padding, PAD_RECORD, dtype="<i8"))
        )
    return order.astype("<i8", copy=False), int(padding)


def _stable_mixed_order(pt_order, sft_order, *, seed: int):
    """Mix two fixed subsequences without changing either relative order."""
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


def _build_canary_orders(per_row_targets) -> dict[str, tuple[Any, int, int, int]]:
    """Build two-update manifests that prove the objective on every rank."""

    import numpy as np

    per_row_targets = np.asarray(per_row_targets, dtype=np.int64)
    required_sft_rows = CANARY_TOTAL_STEPS * SFT_GLOBAL_SEQUENCES
    if per_row_targets.ndim != 1 or len(per_row_targets) < required_sft_rows:
        raise ValueError(
            "canary construction requires two global SFT batches of "
            "per-row target counts"
        )
    if bool((per_row_targets[:required_sft_rows] <= 0).any()):
        raise ValueError("canary SFT rows must each contain supervised targets")

    orders: dict[str, tuple[Any, int, int, int]] = {}
    required_pt_rows = CANARY_TOTAL_STEPS * PT_GLOBAL_SEQUENCES
    pt_order = np.arange(required_pt_rows, dtype="<i8")
    orders["pt"] = (pt_order, required_pt_rows, 0, 0)

    sft_indices = np.arange(required_sft_rows, dtype="<i8")
    sft_order = -(sft_indices + 1)
    orders["sft3"] = (
        sft_order,
        0,
        required_sft_rows,
        int(per_row_targets[sft_indices].sum(dtype=np.int64)),
    )

    # DistributedManifestBatchSampler gives rank r a contiguous local slice in
    # each global batch. Alternating within every such slice proves PT and SFT
    # participate in both optimizer updates on every rank.
    mixed_order = np.empty(required_pt_rows, dtype="<i8")
    pt_index = 0
    sft_index = 0
    for update in range(CANARY_TOTAL_STEPS):
        update_start = update * PT_GLOBAL_SEQUENCES
        for rank in range(WORLD_SIZE):
            rank_start = update_start + rank * PT_LOCAL_BATCH_SIZE
            for local_index in range(PT_LOCAL_BATCH_SIZE):
                position = rank_start + local_index
                if local_index % 2 == 0:
                    mixed_order[position] = pt_index
                    pt_index += 1
                else:
                    mixed_order[position] = -(sft_index + 1)
                    sft_index += 1
    if pt_index + sft_index != required_pt_rows:
        raise AssertionError("mixed canary accounting drifted")
    mixed_values = (
        mixed_order,
        pt_index,
        sft_index,
        int(per_row_targets[:sft_index].sum(dtype=np.int64)),
    )
    orders["mixed_sft1"] = mixed_values
    orders["mixed_sft3"] = mixed_values
    return orders


def _validate_data_identities(payload: Mapping[str, Any]) -> None:
    """Require this version to reuse the exact authenticated v3 training data."""

    expected_scalars = {
        "source_manifest_hash": EXPECTED_SOURCE_MANIFEST_HASH,
        "selection_hash": EXPECTED_SELECTION_HASH,
        "sft_cache_hash": EXPECTED_CONTEXT2048_SFT_CACHE_HASH,
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"Data identity drifted for {key}: "
                f"{payload.get(key)!r} != {expected!r}"
            )
    manifests = payload.get("manifests")
    if not isinstance(manifests, Mapping):
        raise RuntimeError("Manifest set has no manifests mapping")
    observed_orders = {
        key: value.get("order_sha256")
        for key, value in manifests.items()
        if isinstance(value, Mapping)
    }
    if observed_orders != EXPECTED_ORDER_SHA256:
        raise RuntimeError(
            "Training row order drifted from authenticated v3 data: "
            f"{observed_orders!r} != {EXPECTED_ORDER_SHA256!r}"
        )
    observed_order_provenance = {
        key: value.get("order_provenance")
        for key, value in manifests.items()
        if isinstance(value, Mapping)
    }
    if observed_order_provenance != EXPECTED_ORDER_PROVENANCE:
        raise RuntimeError(
            "Training order provenance drifted from the exact production "
            "construction graph: "
            f"{observed_order_provenance!r} != "
            f"{EXPECTED_ORDER_PROVENANCE!r}"
        )


def _prepare_impl() -> dict[str, Any]:
    import numpy as np
    from training.interleaved_data import (
        PretrainSelection,
        SFTCache,
        _hash_dict,
        _sft_supervised_targets_per_row,
        _write_leg_manifest,
        build_pretrain_selection,
    )

    _validate_source_artifacts()
    if MANIFEST_SET_PATH.is_file():
        payload = _load_json(MANIFEST_SET_PATH)
        body = {key: value for key, value in payload.items() if key != "set_hash"}
        if payload.get("set_hash") != hashlib.sha256(_canonical_json(body)).hexdigest():
            raise RuntimeError("Manifest-set self hash drifted")
        if payload.get("source_tree_sha256") != SOURCE_TREE_SHA256:
            raise RuntimeError(
                "Prepared artifact root belongs to different source code; "
                "use a fresh experiment version and artifact root"
            )
        _validate_data_identities(payload)
        return payload

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_MANIFEST_TEMPLATE, SOURCE_MANIFEST_PATH)
    source_manifest_hash = _load_json(SOURCE_MANIFEST_PATH)["manifest_hash"]
    build_pretrain_selection(
        source_manifest_path=SOURCE_MANIFEST_PATH,
        output_path=SELECTION_PATH,
        target_tokens=PT_TARGET_TOKENS,
        seed=42,
    )
    selection = PretrainSelection.load(SELECTION_PATH)
    if selection.target_tokens != PT_TARGET_TOKENS:
        raise RuntimeError("Pretraining selection token count drifted")

    cache_metadata = _prepare_context2048_sft_cache()
    cache = SFTCache.load(SFT_CACHE_DIR, verify_large_files=True)
    if cache.num_rows != SFT_ROWS:
        raise RuntimeError("Context-2048 SFT cache row count drifted")
    per_row_targets = _sft_supervised_targets_per_row(cache)
    if int(per_row_targets.sum(dtype=np.int64)) != SFT_SUPERVISED_TARGETS:
        raise RuntimeError("Context-2048 SFT target count drifted")

    # All experiments see PT records in this exact relative order.
    pt_order = np.random.Generator(
        np.random.PCG64(PT_ORDER_SEED)
    ).permutation(PT_RECORDS).astype("<i8")
    sft_epoch_orders = [
        -(
            np.random.Generator(np.random.PCG64(seed)).permutation(
                SFT_ROWS
            ).astype("<i8")
            + 1
        )
        for seed in SFT_EPOCH_ORDER_SEEDS
    ]

    orders: dict[str, tuple[Any, int, int, int, int]] = {}
    pt_padded, pt_padding = _pad_order(pt_order, PT_GLOBAL_SEQUENCES)
    orders["pt"] = (pt_padded, pt_padding, PT_RECORDS, 0, 0)

    staged_sft = np.concatenate(sft_epoch_orders)
    staged_sft_padded, staged_sft_padding = _pad_order(
        staged_sft, SFT_GLOBAL_SEQUENCES
    )
    orders["sft3"] = (
        staged_sft_padded,
        staged_sft_padding,
        0,
        SFT_STAGE_RECORDS,
        SFT_SUPERVISED_TARGETS * SFT_EPOCHS,
    )

    for copies in (1, 3):
        sft_order = np.concatenate(sft_epoch_orders[:copies])
        mixed = _stable_mixed_order(
            pt_order,
            sft_order,
            seed=MIXED_PLACEMENT_SEEDS[copies],
        )
        mixed_padded, mixed_padding = _pad_order(
            mixed, PT_GLOBAL_SEQUENCES
        )
        orders[f"mixed_sft{copies}"] = (
            mixed_padded,
            mixed_padding,
            PT_RECORDS,
            SFT_ROWS * copies,
            SFT_SUPERVISED_TARGETS * copies,
        )

    manifests: dict[str, Any] = {}
    for name, values in orders.items():
        order, padding, pretrain_records, sft_records, sft_targets = values
        local_batch_size = (
            SFT_LOCAL_BATCH_SIZE if name == "sft3" else PT_LOCAL_BATCH_SIZE
        )
        global_sequences = WORLD_SIZE * local_batch_size
        expected_steps = len(order) // global_sequences
        manifest = _write_leg_manifest(
            ARTIFACT_ROOT / "manifests" / name,
            leg=name,
            order=order,
            target_start=0,
            target_count=PT_TARGET_TOKENS,
            sequence_length=CONTEXT_LENGTH,
            pretrain_records=pretrain_records,
            sft_records=sft_records,
            sft_supervised_targets=sft_targets,
            padding_records=padding,
            world_size=WORLD_SIZE,
            local_batch_size=local_batch_size,
            total_steps=expected_steps,
            source_manifest_hash=source_manifest_hash,
            selection_hash=selection.selection_hash,
            sft_cache_hash=cache.cache_hash,
            # Production orders have multiple independent permutations and,
            # for mixed runs, a separate placement shuffle.  A scalar seed
            # cannot describe that construction truthfully.
            shuffle_seed=None,
            order_provenance=EXPECTED_ORDER_PROVENANCE[name],
        )
        if manifest.order_provenance != EXPECTED_ORDER_PROVENANCE[name]:
            raise RuntimeError(f"{name} manifest order provenance drifted")
        manifests[name] = {
            "metadata_path": str(manifest.metadata_path),
            "metadata_sha256": _sha256_file(manifest.metadata_path),
            "order_sha256": manifest.order_sha256,
            "order_provenance": manifest.order_provenance,
            "pretrain_records": pretrain_records,
            "sft_records": sft_records,
            "sft_supervised_targets": sft_targets,
            "padding_records": padding,
            "total_steps": expected_steps,
        }

    # Canary orders are separate authenticated two-update manifests. The
    # ordinary objective gate stops after update one; the process-boundary
    # resume gate consumes update two from the same deterministic PT manifest.
    canary_orders = _build_canary_orders(per_row_targets)

    canary_manifests: dict[str, Any] = {}
    for name, values in canary_orders.items():
        order, pretrain_records, sft_records, sft_targets = values
        local_batch_size = (
            SFT_LOCAL_BATCH_SIZE if name == "sft3" else PT_LOCAL_BATCH_SIZE
        )
        manifest = _write_leg_manifest(
            ARTIFACT_ROOT / "canary_manifests" / name,
            leg=f"{name}_canary",
            order=order,
            target_start=0,
            target_count=PT_TARGET_TOKENS,
            sequence_length=CONTEXT_LENGTH,
            pretrain_records=pretrain_records,
            sft_records=sft_records,
            sft_supervised_targets=sft_targets,
            padding_records=0,
            world_size=WORLD_SIZE,
            local_batch_size=local_batch_size,
            total_steps=CANARY_TOTAL_STEPS,
            source_manifest_hash=source_manifest_hash,
            selection_hash=selection.selection_hash,
            sft_cache_hash=cache.cache_hash,
            shuffle_seed=None,
        )
        canary_manifests[name] = {
            "metadata_path": str(manifest.metadata_path),
            "metadata_sha256": _sha256_file(manifest.metadata_path),
            "order_sha256": manifest.order_sha256,
            "pretrain_records": pretrain_records,
            "sft_records": sft_records,
            "sft_supervised_targets": sft_targets,
            "padding_records": 0,
            "total_steps": CANARY_TOTAL_STEPS,
        }

    expected = {
        "pt": PT_STEPS,
        "sft3": SFT_STAGE_STEPS,
        "mixed_sft1": MIXED_STEPS[1],
        "mixed_sft3": MIXED_STEPS[3],
    }
    observed = {key: int(value["total_steps"]) for key, value in manifests.items()}
    if observed != expected:
        raise RuntimeError(f"Step accounting drifted: {observed} != {expected}")

    payload: dict[str, Any] = {
        "schema": "context2048-vocab-mixing-manifest-set-v2",
        "experiment_version": EXPERIMENT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_manifest_hash": source_manifest_hash,
        "selection_path": str(SELECTION_PATH),
        "selection_hash": selection.selection_hash,
        "pt_target_tokens": PT_TARGET_TOKENS,
        "pt_records": PT_RECORDS,
        "context_length": CONTEXT_LENGTH,
        "pt_global_token_batch": PT_GLOBAL_TOKEN_BATCH,
        "sft_repo": SFT_REPO,
        "sft_revision": SFT_REVISION,
        "sft_cache_dir": str(SFT_CACHE_DIR),
        "sft_cache_hash": cache.cache_hash,
        "sft_cache_metadata_sha256": _sha256_file(SFT_CACHE_DIR / "metadata.json"),
        "sft_max_aligned_length": cache_metadata["maximum_aligned_length"],
        "sft_packing": cache_metadata["packing"],
        "manifests": manifests,
        "canary_manifests": canary_manifests,
    }
    payload["set_hash"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    _validate_data_identities(payload)
    _atomic_json(MANIFEST_SET_PATH, payload)
    data_volume.commit()
    return payload


@app.function(cpu=16.0, memory=64 * 1024, timeout=60 * 60 * 4)
def prepare_data() -> str:
    payload = _prepare_impl()
    return json.dumps(
        {
            "set_hash": payload["set_hash"],
            "selection_hash": payload["selection_hash"],
            "sft_cache_hash": payload["sft_cache_hash"],
            "manifests": {
                key: {
                    "sha256": value["metadata_sha256"],
                    "steps": value["total_steps"],
                }
                for key, value in payload["manifests"].items()
            },
        },
        indent=2,
    )


@app.function(cpu=1.0, memory=1024, timeout=5 * 60)
def deployment_identity() -> dict[str, Any]:
    """Return the identity of the persistent deployment serving this call."""

    return {
        "schema": "context2048-modal-deployment-identity-v1",
        "experiment_version": EXPERIMENT_VERSION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "runtime_identity": _modal_runtime_identity(),
    }


def _manifest(
    name: str,
    *,
    canary: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _load_json(MANIFEST_SET_PATH)
    body = {key: value for key, value in payload.items() if key != "set_hash"}
    if payload.get("set_hash") != hashlib.sha256(_canonical_json(body)).hexdigest():
        raise RuntimeError("Manifest-set self hash drifted")
    if payload.get("source_tree_sha256") != SOURCE_TREE_SHA256:
        raise RuntimeError("Prepared artifacts were built from different source code")
    _validate_data_identities(payload)
    manifest_group = "canary_manifests" if canary else "manifests"
    manifests = payload.get(manifest_group)
    if not isinstance(manifests, Mapping) or name not in manifests:
        raise RuntimeError(
            f"Manifest set lacks authenticated {manifest_group}.{name}"
        )
    manifest = manifests[name]
    if _sha256_file(Path(manifest["metadata_path"])) != manifest["metadata_sha256"]:
        raise RuntimeError(f"{name} manifest drifted on disk")
    if not canary:
        from training.interleaved_data import LegManifest

        loaded_manifest = LegManifest.load(manifest["metadata_path"])
        expected_provenance = EXPECTED_ORDER_PROVENANCE[name]
        if loaded_manifest.order_sha256 != manifest.get("order_sha256"):
            raise RuntimeError(f"{name} manifest-set order hash drifted")
        if loaded_manifest.order_provenance != expected_provenance:
            raise RuntimeError(
                f"{name} metadata has incorrect order provenance"
            )
        if manifest.get("order_provenance") != loaded_manifest.order_provenance:
            raise RuntimeError(
                f"{name} manifest-set order provenance differs from metadata"
            )
    return payload, manifest


def _stage_output(experiment: Experiment, stage: str, *, canary: bool) -> Path:
    suffix = "_canary" if canary else ""
    return CHECKPOINT_ROOT / f"{experiment.key}{suffix}" / stage


def _run_name(experiment: Experiment, stage: str, *, canary: bool) -> str:
    names = {
        ("vocab81_then_sft3", "pt"): "ctx2048-pt9p181735b-vocab81-pt",
        ("vocab81_then_sft3", "sft"): "ctx2048-pt9p181735b-vocab81-expand85-clean-sft3",
        ("vocab85_then_sft3", "pt"): "ctx2048-pt9p181735b-vocab85-pt",
        ("vocab85_then_sft3", "sft"): "ctx2048-pt9p181735b-vocab85-clean-sft3",
        ("mixed_sft1", "mixed"): "ctx2048-pt9p181735b-mixed-clean-sft1",
        ("mixed_sft3", "mixed"): "ctx2048-pt9p181735b-mixed-clean-sft3",
    }
    name = names[(experiment.key, stage)] + "-fp32-master-v13"
    return name + ("-canary" if canary else "")


def _stage_spec(experiment: Experiment, stage: str) -> dict[str, Any]:
    if stage == "pt":
        return {
            "manifest": "pt",
            "steps": PT_STEPS,
            "vocab_size": experiment.pt_vocab_size,
            "local_batch_size": PT_LOCAL_BATCH_SIZE,
            "mixed_precision": "bf16",
            "peak_lr": PT_PEAK_LR,
            "eta_min": PT_ETA_MIN,
            "warmup_ratio": PT_WARMUP_RATIO,
            "weight_decay": 0.1,
            "allow_vocab_expansion": False,
        }
    if stage == "sft":
        return {
            "manifest": "sft3",
            "steps": SFT_STAGE_STEPS,
            "vocab_size": 85,
            "local_batch_size": SFT_LOCAL_BATCH_SIZE,
            "mixed_precision": "bf16",
            "peak_lr": SFT_PEAK_LR,
            "eta_min": SFT_ETA_MIN,
            "warmup_ratio": SFT_WARMUP_STEPS / SFT_STAGE_STEPS,
            "weight_decay": 0.01,
            "allow_vocab_expansion": experiment.pt_vocab_size == 81,
        }
    if stage == "mixed":
        return {
            "manifest": experiment.key,
            "steps": MIXED_STEPS[experiment.sft_copies],
            "vocab_size": 85,
            "local_batch_size": PT_LOCAL_BATCH_SIZE,
            "mixed_precision": "bf16",
            "peak_lr": PT_PEAK_LR,
            "eta_min": 1e-5,
            "warmup_ratio": PT_WARMUP_RATIO,
            "weight_decay": 0.1,
            "allow_vocab_expansion": False,
        }
    raise ValueError(stage)


def _overrides(
    experiment: Experiment,
    stage: str,
    manifest: Mapping[str, Any],
    *,
    canary: bool,
    initialization_identity: Mapping[str, Any] | None = None,
) -> list[str]:
    spec = _stage_spec(experiment, stage)
    output_dir = _stage_output(experiment, stage, canary=canary)
    run_name = _run_name(experiment, stage, canary=canary)
    tokenizer_name = "LanTokenizer" if spec["vocab_size"] == 81 else "LanTokenizerSFT"
    include_env_tokens = str(spec["vocab_size"] == 85).lower()
    runtime_identity = _modal_runtime_identity()
    canary_sample_contract = {
        "pt": "pt-only",
        "sft": "sft-only",
        "mixed": "mixed-pt-sft",
    }[stage]
    if initialization_identity is None:
        if stage == "sft":
            raise RuntimeError(
                "staged SFT overrides require an authenticated PT parent identity"
            )
        initialization_identity = {
            "schema": "interleaved-random-initialization-v1",
            "mode": "random",
            "destination_seed": 42,
        }
    values = [
        "model.block_size=2048",
        f"model.vocab_size={spec['vocab_size']}",
        f"tokenizer.name={tokenizer_name}",
        f"tokenizer.include_env_tokens={include_env_tokens}",
        "tokenizer.include_reward_tokens=false",
        f"training.output_dir={output_dir}",
        f"training.run_name={run_name}",
        "training.seed=42",
        f"training.local_batch_size={spec['local_batch_size']}",
        "training.gradient_accumulation_steps=1",
        "training.allow_topology_override=true",
        f"training.allow_weight_decay_override={str(spec['weight_decay'] == 0.01).lower()}",
        f"training.allow_vocab_expansion={str(spec['allow_vocab_expansion']).lower()}",
        f"training.total_steps={spec['steps']}",
        f"training.arc_steps=[{spec['steps']}]",
        "training.reset_optimizer_between_arcs=true",
        f"training.mixed_precision={spec['mixed_precision']}",
        "training.sft_loss_weight=1.0",
        f"training.optimizer.lr={spec['peak_lr']}",
        f"training.optimizer.weight_decay={spec['weight_decay']}",
        "training.optimizer.betas=[0.9,0.95]",
        f"training.scheduler.warmup_ratio={spec['warmup_ratio']}",
        f"training.scheduler.eta_min={spec['eta_min']}",
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
        f"logging.group={experiment.key}",
        f"logging.job_type={stage}",
        f"logging.id={EXPERIMENT_VERSION}-{experiment.key}-{stage}",
        "logging.resume=allow",
        f"provenance.experiment_version={EXPERIMENT_VERSION}",
        f"provenance.source_tree_sha256={SOURCE_TREE_SHA256}",
        f"provenance.experiment={experiment.key}",
        f"provenance.stage={stage}",
        "provenance.seed=42",
        "provenance.initialization_identity="
        + json.dumps(dict(initialization_identity), sort_keys=True),
        f"provenance.context_length={CONTEXT_LENGTH}",
        f"provenance.vocab_size={spec['vocab_size']}",
        "provenance.token_ids="
        + json.dumps(
            EXPECTED_TOKEN_IDS_85
            if int(spec["vocab_size"]) == 85
            else EXPECTED_TOKEN_IDS_81,
            sort_keys=True,
        ),
        f"provenance.pt_target_tokens={PT_TARGET_TOKENS}",
        f"provenance.pt_global_token_batch={PT_GLOBAL_TOKEN_BATCH}",
        f"provenance.sft_copies={experiment.sft_copies}",
        "provenance.sft_packing=one-row-per-sequence-right-padded",
        f"provenance.source_repo={SOURCE_REPO}",
        f"provenance.source_revision={SOURCE_REVISION}",
        f"provenance.sft_repo={SFT_REPO}",
        f"provenance.sft_revision={SFT_REVISION}",
        f"provenance.peak_lr={spec['peak_lr']}",
        f"provenance.eta_min={spec['eta_min']}",
        f"provenance.weight_decay={spec['weight_decay']}",
        "provenance.master_parameter_dtype=float32",
        "provenance.optimizer_state_dtype=float32",
        "provenance.forward_backward_dtype=bfloat16",
        "provenance.gradient_dtype=float32",
        "provenance.hf_export_dtype=float32",
        f"provenance.modal_app_name={runtime_identity['modal_app_name']}",
        f"provenance.modal_app_id={runtime_identity['modal_app_id']}",
        f"provenance.modal_image_id={runtime_identity['modal_image_id']}",
        f"provenance.modal_base_image={runtime_identity['modal_base_image']}",
        f"provenance.modal_client_version={runtime_identity['modal_client_version']}",
        "provenance.runtime_package_versions="
        + json.dumps(runtime_identity["runtime_package_versions"], sort_keys=True),
        "provenance.runtime_distribution_count="
        f"{runtime_identity['runtime_distribution_count']}",
        "provenance.runtime_distribution_inventory_sha256="
        f"{runtime_identity['runtime_distribution_inventory_sha256']}",
    ]
    if canary:
        values = _replace_override(
            values,
            "training.total_steps",
            CANARY_TOTAL_STEPS,
        )
        values = _replace_override(
            values,
            "training.arc_steps",
            f"[{CANARY_TOTAL_STEPS}]",
        )
        values.extend(
            [
                "training.max_steps=1",
                "training.save_interval=1",
                "training.log_interval=1",
                "training.persistent_workers=false",
                "data.num_workers=0",
                "training.num_workers=0",
                f"provenance.canary_sample_contract={canary_sample_contract}",
            ]
        )
    return values


def _validate_final(
    path: Path,
    *,
    expected_step: int,
    experiment: Experiment,
    stage: str,
    initialization_identity: Mapping[str, Any],
    initial_launch_command: Mapping[str, Any],
) -> None:
    state_path = path / "final" / "interleaved_training_state.json"
    if not state_path.is_file():
        raise RuntimeError(f"Missing final state: {state_path}")
    state = _load_json(state_path)
    if int(state.get("global_step", -1)) != expected_step:
        raise RuntimeError(
            f"{experiment.key}/{stage} final step "
            f"{state.get('global_step')} != {expected_step}"
        )
    provenance = state.get("configured_provenance", {})
    if provenance.get("experiment_version") != EXPERIMENT_VERSION:
        raise RuntimeError(f"{experiment.key}/{stage} provenance version drifted")
    if provenance.get("source_tree_sha256") != SOURCE_TREE_SHA256:
        raise RuntimeError(f"{experiment.key}/{stage} source tree drifted")
    expected_vocab_size = int(_stage_spec(experiment, stage)["vocab_size"])
    expected_token_ids = (
        EXPECTED_TOKEN_IDS_85
        if expected_vocab_size == 85
        else EXPECTED_TOKEN_IDS_81
    )
    if provenance.get("token_ids") != expected_token_ids:
        raise RuntimeError(f"{experiment.key}/{stage} tokenizer IDs drifted")
    exact_provenance = {
        "seed": 42,
        "initialization_identity": dict(initialization_identity),
        "initial_launch_command": dict(initial_launch_command),
    }
    for key, expected in exact_provenance.items():
        if provenance.get(key) != expected:
            raise RuntimeError(
                f"{experiment.key}/{stage} final provenance {key} drifted"
            )
    runtime = state.get("runtime_provenance", {})
    if not isinstance(runtime, Mapping):
        raise RuntimeError(f"{experiment.key}/{stage} runtime provenance is missing")
    expected_runtime_identity = _modal_runtime_identity()
    runtime_checks = {
        "modal_app_name": expected_runtime_identity["modal_app_name"],
        "modal_app_id": expected_runtime_identity["modal_app_id"],
        "modal_image_id": expected_runtime_identity["modal_image_id"],
        "modal_base_image": expected_runtime_identity["modal_base_image"],
        "modal_client_version": expected_runtime_identity[
            "modal_client_version"
        ],
        "runtime_package_versions": expected_runtime_identity[
            "runtime_package_versions"
        ],
        "runtime_distribution_count": expected_runtime_identity[
            "runtime_distribution_count"
        ],
        "runtime_distribution_inventory_sha256": expected_runtime_identity[
            "runtime_distribution_inventory_sha256"
        ],
        "python_version": expected_runtime_identity["python_version"],
    }
    for key, value in runtime_checks.items():
        if runtime.get(key) != value:
            raise RuntimeError(
                f"{experiment.key}/{stage} runtime identity {key} drifted"
            )
    if state.get("precision_contract") != EXPECTED_PRECISION_CONTRACT:
        raise RuntimeError(
            f"{experiment.key}/{stage} precision contract drifted"
        )
    if state.get("determinism_contract") != EXPECTED_DETERMINISM_CONTRACT:
        raise RuntimeError(
            f"{experiment.key}/{stage} deterministic training contract drifted"
        )
    if expected_step == 1:
        sample_evidence = state.get("runtime_provenance", {}).get(
            "canary_sample_evidence"
        )
        _validate_canary_sample_evidence(sample_evidence, stage=stage)
        _validate_canary_metrics(
            path,
            stage=stage,
            sample_evidence=sample_evidence,
        )
    final = path / "final"
    _validate_fp32_hf_export(final)
    _validate_hf_tokenizer_contract(
        final,
        expected_vocab_size=expected_vocab_size,
    )


def _validate_canary_sample_evidence(
    evidence: Any,
    *,
    stage: str,
) -> None:
    if not isinstance(evidence, Mapping):
        raise RuntimeError(f"{stage} canary lacks sample evidence")
    expected_contract = {
        "pt": "pt-only",
        "sft": "sft-only",
        "mixed": "mixed-pt-sft",
    }[stage]
    pt_rows = int(evidence.get("global_pretrain_rows", 0))
    sft_rows = int(evidence.get("global_sft_rows", 0))
    pt_tokens = int(evidence.get("global_pretrain_supervised_tokens", 0))
    sft_tokens = int(evidence.get("global_sft_supervised_tokens", 0))
    if evidence.get("contract") != expected_contract:
        raise RuntimeError(f"{stage} canary sample contract drifted: {evidence}")
    if stage == "pt" and not (
        pt_rows > 0
        and pt_tokens > 0
        and sft_rows == 0
        and sft_tokens == 0
        and evidence.get("pt_leading_bos_validated") is True
    ):
        raise RuntimeError(f"PT canary sample evidence is invalid: {evidence}")
    if stage == "sft" and not (
        pt_rows == 0
        and pt_tokens == 0
        and sft_rows > 0
        and sft_tokens > 0
        and evidence.get("sft_bos_and_mask_validated") is True
    ):
        raise RuntimeError(f"SFT canary sample evidence is invalid: {evidence}")
    if stage == "mixed" and not (
        pt_rows > 0
        and pt_tokens > 0
        and sft_rows > 0
        and sft_tokens > 0
        and evidence.get("pt_leading_bos_validated") is True
        and evidence.get("sft_bos_and_mask_validated") is True
    ):
        raise RuntimeError(f"mixed canary sample evidence is invalid: {evidence}")


def _validate_canary_metrics(
    output_root: Path,
    *,
    stage: str,
    sample_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the one-update metric record to the proven canary batch."""

    metrics_path = output_root / "metrics.jsonl"
    if not metrics_path.is_file() or metrics_path.is_symlink():
        raise RuntimeError(f"{stage} canary lacks a regular metrics log")
    lines = [
        line
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        raise RuntimeError(
            f"{stage} one-update canary must have exactly one metric record"
        )
    try:
        record = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{stage} canary metrics are invalid JSON") from exc
    if not isinstance(record, Mapping):
        raise RuntimeError(f"{stage} canary metric record is not an object")
    if record.get("schema") != "interleaved-local-metrics-v1":
        raise RuntimeError(f"{stage} canary metric schema drifted")
    if int(record.get("step", -1)) != 1:
        raise RuntimeError(f"{stage} canary metric step drifted")
    runtime = record.get("runtime_provenance")
    if not isinstance(runtime, Mapping) or runtime.get(
        "canary_sample_evidence"
    ) != dict(sample_evidence):
        raise RuntimeError(
            f"{stage} metric record is not bound to its sample evidence"
        )
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise RuntimeError(f"{stage} canary metric payload is missing")

    expected_pt = int(sample_evidence["global_pretrain_supervised_tokens"])
    expected_sft = int(sample_evidence["global_sft_supervised_tokens"])
    expected_total = expected_pt + expected_sft
    exact_counts = {
        "train/global_pretrain_valid_tokens": expected_pt,
        "train/global_sft_valid_tokens": expected_sft,
        "train/global_valid_tokens": expected_total,
        "train/manifest_cursor": 1,
    }
    for key, expected in exact_counts.items():
        if int(metrics.get(key, -1)) != expected:
            raise RuntimeError(
                f"{stage} canary metric {key} drifted: "
                f"{metrics.get(key)!r} != {expected}"
            )
    weighted_tokens = metrics.get("train/global_weighted_valid_tokens")
    if (
        not isinstance(weighted_tokens, (int, float))
        or isinstance(weighted_tokens, bool)
        or not math.isfinite(float(weighted_tokens))
        or float(weighted_tokens) != float(expected_total)
    ):
        raise RuntimeError(
            f"{stage} canary weighted-token metric drifted: {weighted_tokens!r}"
        )
    if float(metrics.get("train/sft_loss_weight", float("nan"))) != 1.0:
        raise RuntimeError(f"{stage} canary SFT loss weight drifted")
    loss = metrics.get("train/loss")
    if (
        not isinstance(loss, (int, float))
        or isinstance(loss, bool)
        or not math.isfinite(float(loss))
        or float(loss) < 0.0
    ):
        raise RuntimeError(f"{stage} canary loss is invalid: {loss!r}")
    required_loss_keys = {
        "train/pretrain_token_loss": expected_pt > 0,
        "train/sft_token_loss": expected_sft > 0,
    }
    for key, required in required_loss_keys.items():
        value = metrics.get(key)
        if required:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise RuntimeError(f"{stage} canary {key} is invalid: {value!r}")
        elif key in metrics:
            raise RuntimeError(f"{stage} canary unexpectedly reports {key}")
    expected_share = expected_sft / expected_total
    actual_share = metrics.get("train/effective_sft_loss_mass_share")
    if (
        not isinstance(actual_share, (int, float))
        or isinstance(actual_share, bool)
        or not math.isclose(
            float(actual_share),
            float(expected_share),
            # The distributed statistics tensor is intentionally FP32.  Its
            # integer token counts are exact at this scale, but the division
            # that produces this diagnostic is rounded once to FP32.
            rel_tol=2e-7,
            abs_tol=1e-8,
        )
    ):
        raise RuntimeError(
            f"{stage} canary effective SFT share drifted: {actual_share!r}"
        )
    return {
        "metrics_file_sha256": _sha256_file(metrics_path),
        "step": 1,
        "global_pretrain_valid_tokens": expected_pt,
        "global_sft_valid_tokens": expected_sft,
        "global_valid_tokens": expected_total,
        "loss": float(loss),
        "pretrain_token_loss": (
            float(metrics["train/pretrain_token_loss"])
            if expected_pt > 0
            else None
        ),
        "sft_token_loss": (
            float(metrics["train/sft_token_loss"])
            if expected_sft > 0
            else None
        ),
        "effective_sft_loss_mass_share": float(actual_share),
    }


def _validate_hf_tokenizer_contract(
    path: Path,
    *,
    expected_vocab_size: int,
) -> dict[str, Any]:
    """Authenticate the complete tokenizer/model mapping at an HF handoff."""

    return _shared_validate_hf_tokenizer_contract(
        path,
        expected_vocab_size=int(expected_vocab_size),
        expected_context_length=CONTEXT_LENGTH,
    )


def _validate_fp32_hf_export(path: Path) -> dict[str, Any]:
    """Authenticate the full HF export and inspect actual tensor headers."""

    return validate_completed_hf_export(path)["marker"]["persisted_precision"]


def _stage_initialization_identity(
    experiment: Experiment,
    stage: str,
    *,
    weights_only: Path | None,
    canary: bool,
) -> dict[str, Any]:
    """Resolve the exact initialization identity before a stage may start."""

    if stage != "sft":
        if weights_only is not None:
            raise RuntimeError(f"{stage} may not accept a weights-only parent")
        return {
            "schema": "interleaved-random-initialization-v1",
            "mode": "random",
            "destination_seed": 42,
        }
    if weights_only is None:
        raise RuntimeError("staged SFT requires an authenticated PT HF export")
    identity = authenticated_weights_only_identity(
        weights_only,
        destination_vocab=EXPECTED_VOCAB_85,
        allow_vocab_expansion=bool(
            _stage_spec(experiment, stage)["allow_vocab_expansion"]
        ),
        context_length=CONTEXT_LENGTH,
    )
    identity["destination_seed"] = 42
    expected_source = {
        "source_experiment": experiment.key,
        "source_stage": "pt",
        "source_experiment_version": EXPERIMENT_VERSION,
        "source_global_step": 1 if canary else PT_STEPS,
        "source_seed": 42,
        "source_model_init_seed": 42,
    }
    for key, expected in expected_source.items():
        if identity.get(key) != expected:
            raise RuntimeError(
                f"staged SFT parent {key} drifted: "
                f"{identity.get(key)!r} != {expected!r}"
            )
    return identity


def _initial_launch_command_identity(command: list[str]) -> dict[str, Any]:
    if (
        not command
        or not all(isinstance(argument, str) and argument for argument in command)
    ):
        raise ValueError("launch command must be a nonempty string list")
    canonical = json.dumps(
        command,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return {
        "schema": "interleaved-initial-launch-command-v1",
        "argv": list(command),
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _validate_launch_command_identity(
    identity: Any,
    *,
    label: str,
) -> list[str]:
    """Authenticate a recorded outer launch command and return its argv."""

    if not isinstance(identity, Mapping):
        raise RuntimeError(f"{label} command identity is missing")
    argv = identity.get("argv")
    if (
        identity.get("schema") != "interleaved-initial-launch-command-v1"
        or not isinstance(argv, list)
        or not argv
        or not all(isinstance(argument, str) and argument for argument in argv)
    ):
        raise RuntimeError(f"{label} command identity is invalid")
    expected = _initial_launch_command_identity(argv)
    if dict(identity) != expected:
        raise RuntimeError(f"{label} command identity SHA-256 drifted")
    return list(argv)


def _validate_bf16_inference_from_fp32_hf_export(
    path: Path,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    """Load the canonical FP32 export as BF16 in memory and run inference."""

    import torch
    from transformers import AutoModelForCausalLM

    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("BF16 inference gate requires a CUDA GPU")
        device = "cuda"
    tokenizer_contract = _validate_hf_tokenizer_contract(
        path,
        expected_vocab_size=85,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    try:
        model.to(device)
        model.eval()
        floating_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.is_floating_point()
        ]
        if not floating_parameters:
            raise RuntimeError("BF16 inference model has no floating parameters")
        observed_parameter_dtypes = sorted(
            {str(parameter.dtype).removeprefix("torch.") for parameter in floating_parameters}
        )
        if observed_parameter_dtypes != ["bfloat16"]:
            raise RuntimeError(
                "in-memory inference parameters are not all BF16: "
                f"{observed_parameter_dtypes}"
            )
        bos_token_id = int(getattr(model.config, "bos_token_id", 0) or 0)
        input_ids = torch.full(
            (1, 4),
            bos_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
        if logits.dtype != torch.bfloat16:
            raise RuntimeError(
                f"BF16 inference produced {logits.dtype} logits instead of BF16"
            )
        if not torch.isfinite(logits.float()).all().item():
            raise RuntimeError("BF16 inference produced non-finite logits")
        return {
            "source_export_dtype": "float32",
            "in_memory_parameter_dtypes": observed_parameter_dtypes,
            "forward_logits_dtype": "bfloat16",
            "forward_logits_finite": True,
            "input_shape": list(input_ids.shape),
            "tokenizer_contract": tokenizer_contract,
        }
    finally:
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def _validate_stage_resume_state(
    state: Mapping[str, Any],
    *,
    experiment: Experiment,
    stage: str,
    manifest: Mapping[str, Any],
    expected_step: int,
    initialization_identity: Mapping[str, Any],
    initial_launch_command: Mapping[str, Any],
) -> None:
    spec = _stage_spec(experiment, stage)
    if state.get("manifest_hash") != manifest["metadata_sha256"]:
        raise RuntimeError(f"{experiment.key}/{stage} resume manifest drifted")
    if state.get("precision_contract") != EXPECTED_PRECISION_CONTRACT:
        raise RuntimeError(f"{experiment.key}/{stage} resume precision drifted")
    if state.get("determinism_contract") != EXPECTED_DETERMINISM_CONTRACT:
        raise RuntimeError(
            f"{experiment.key}/{stage} resume determinism drifted"
        )
    step = int(state.get("global_step", -1))
    if not 0 <= step <= int(expected_step):
        raise RuntimeError(
            f"{experiment.key}/{stage} invalid resume step {step} for target "
            f"{expected_step}"
        )
    if int(state.get("manifest_cursor", -1)) != step:
        raise RuntimeError(f"{experiment.key}/{stage} resume cursor drifted")
    if int(state.get("context_length", -1)) != CONTEXT_LENGTH:
        raise RuntimeError(f"{experiment.key}/{stage} resume context drifted")
    if int(state.get("vocab_size", -1)) != int(spec["vocab_size"]):
        raise RuntimeError(f"{experiment.key}/{stage} resume vocabulary drifted")
    provenance = state.get("configured_provenance", {})
    runtime_identity = _modal_runtime_identity()
    expected_provenance = {
        "experiment_version": EXPERIMENT_VERSION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "experiment": experiment.key,
        "stage": stage,
        "seed": 42,
        "initialization_identity": dict(initialization_identity),
        "initial_launch_command": dict(initial_launch_command),
        "vocab_size": int(spec["vocab_size"]),
        "token_ids": (
            EXPECTED_TOKEN_IDS_85
            if int(spec["vocab_size"]) == 85
            else EXPECTED_TOKEN_IDS_81
        ),
        "modal_app_name": APP_NAME,
        "modal_base_image": CUDA_BASE_IMAGE,
        "runtime_package_versions": PINNED_RUNTIME_PACKAGE_VERSIONS,
        "modal_app_id": runtime_identity["modal_app_id"],
        "modal_image_id": runtime_identity["modal_image_id"],
        "modal_client_version": runtime_identity["modal_client_version"],
        "runtime_distribution_count": runtime_identity[
            "runtime_distribution_count"
        ],
        "runtime_distribution_inventory_sha256": runtime_identity[
            "runtime_distribution_inventory_sha256"
        ],
    }
    for key, value in expected_provenance.items():
        if provenance.get(key) != value:
            raise RuntimeError(
                f"{experiment.key}/{stage} resume provenance {key} drifted"
            )


def _authenticate_stage_output_root(
    output_dir: Path,
    *,
    experiment: Experiment,
    stage: str,
    manifest: Mapping[str, Any],
    expected_step: int,
    initialization_identity: Mapping[str, Any],
    initial_launch_command: Mapping[str, Any],
) -> tuple[bool, Path | None]:
    """Classify a nonempty root only after authenticating all of its contents."""

    if not output_dir.is_dir():
        raise RuntimeError(f"stage output root is not a directory: {output_dir}")
    final_state_path = output_dir / "final" / "interleaved_training_state.json"
    has_final_directory = (output_dir / "final").exists()
    allowed = frozenset({"final"}) if has_final_directory else frozenset()
    latest = validate_checkpoint_run_root(
        output_dir,
        allowed_root_directories=allowed,
    )
    resume_state = _load_json(latest / "trainer_state.json")
    _validate_stage_resume_state(
        resume_state,
        experiment=experiment,
        stage=stage,
        manifest=manifest,
        expected_step=expected_step,
        initialization_identity=initialization_identity,
        initial_launch_command=initial_launch_command,
    )
    if has_final_directory:
        if not final_state_path.is_file():
            raise RuntimeError(
                f"unauthenticated or incomplete final export: {output_dir / 'final'}"
            )
        _validate_final(
            output_dir,
            expected_step=expected_step,
            experiment=experiment,
            stage=stage,
            initialization_identity=initialization_identity,
            initial_launch_command=initial_launch_command,
        )
        if int(resume_state["global_step"]) != expected_step:
            raise RuntimeError(
                f"{experiment.key}/{stage} final export has no matching final resume checkpoint"
            )
        return True, None
    return False, latest


def _stage_launch_contract(
    experiment: Experiment,
    stage: str,
    *,
    canary: bool,
    weights_only: Path | None = None,
) -> dict[str, Any]:
    """Build the exact immutable command/identity used to start a stage."""

    _, manifest = _manifest(
        _stage_spec(experiment, stage)["manifest"],
        canary=canary,
    )
    output_dir = _stage_output(experiment, stage, canary=canary)
    expected_step = 1 if canary else int(_stage_spec(experiment, stage)["steps"])
    initialization_identity = _stage_initialization_identity(
        experiment,
        stage,
        weights_only=weights_only,
        canary=canary,
    )
    launch_precision = _stage_spec(experiment, stage)["mixed_precision"]
    base_command = [
        "accelerate",
        "launch",
        "--multi_gpu",
        "--num_processes",
        str(WORLD_SIZE),
        "--mixed_precision",
        launch_precision,
        "--main_process_port",
        "29641",
        TRAIN_CLI,
        "--config",
        BASE_CONFIG,
        "--override",
        *_overrides(
            experiment,
            stage,
            manifest,
            canary=canary,
            initialization_identity=initialization_identity,
        ),
    ]
    initial_command = list(base_command)
    if weights_only is not None:
        initial_command.extend(("--weights-only", str(weights_only)))
    initial_launch_command = _initial_launch_command_identity(initial_command)
    return {
        "manifest": manifest,
        "output_dir": output_dir,
        "expected_step": expected_step,
        "initialization_identity": initialization_identity,
        "base_command": base_command,
        "initial_launch_command": initial_launch_command,
    }


def _run_stage(
    experiment: Experiment,
    stage: str,
    *,
    canary: bool,
    weights_only: Path | None = None,
    heartbeat: Callable[[str, str], None] | None = None,
) -> Path:
    contract = _stage_launch_contract(
        experiment,
        stage,
        canary=canary,
        weights_only=weights_only,
    )
    manifest = contract["manifest"]
    output_dir = contract["output_dir"]
    expected_step = int(contract["expected_step"])
    initialization_identity = contract["initialization_identity"]
    base_command = contract["base_command"]
    initial_launch_command = contract["initial_launch_command"]
    resume = None
    if output_dir.exists() and any(output_dir.iterdir()):
        complete, resume = _authenticate_stage_output_root(
            output_dir,
            experiment=experiment,
            stage=stage,
            manifest=manifest,
            expected_step=expected_step,
            initialization_identity=initialization_identity,
            initial_launch_command=initial_launch_command,
        )
        if complete:
            return output_dir / "final"

    command = list(base_command)
    if resume is not None:
        command.extend(("--resume", str(resume)))
    elif weights_only is not None:
        command.extend(("--weights-only", str(weights_only)))
    print(
        f"[{experiment.key}:{stage}] " + " ".join(command),
        flush=True,
    )
    returncode = _run_process_with_incremental_checkpoint_commits(
        command,
        label=f"{experiment.key}/{stage}",
        output_dir=output_dir,
        initial_launch_command=initial_launch_command,
        heartbeat=(
            (lambda: heartbeat(stage, "training"))
            if heartbeat is not None
            else None
        ),
    )
    if returncode != 0:
        raise RuntimeError(
            f"{experiment.key}/{stage} failed with exit code {returncode}"
        )
    _validate_final(
        output_dir,
        expected_step=expected_step,
        experiment=experiment,
        stage=stage,
        initialization_identity=initialization_identity,
        initial_launch_command=initial_launch_command,
    )
    checkpoint_volume.commit()
    print(
        f"[{experiment.key}:{stage}] validated final volume commit complete",
        flush=True,
    )
    return output_dir / "final"


def _replace_override(
    overrides: list[str],
    key: str,
    value: Any,
) -> list[str]:
    """Replace exactly one OmegaConf dot-list value, failing on drift."""

    prefix = f"{key}="
    matches = [
        index
        for index, item in enumerate(overrides)
        if item.startswith(prefix)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {key} override, found {len(matches)}"
        )
    result = list(overrides)
    result[matches[0]] = f"{key}={value}"
    return result


def _precision_resume_command(
    *,
    manifest: Mapping[str, Any],
    output_dir: Path,
    run_name: str,
    max_steps: int,
    resume: Path | None = None,
) -> list[str]:
    """Build one isolated process invocation for the resume canary."""

    experiment = EXPERIMENTS["vocab85_then_sft3"]
    overrides = _overrides(experiment, "pt", manifest, canary=True)
    for key, value in (
        ("training.output_dir", output_dir),
        ("training.run_name", run_name),
        ("training.max_steps", int(max_steps)),
        ("logging.id", f"{EXPERIMENT_VERSION}-{run_name}"),
    ):
        overrides = _replace_override(overrides, key, value)
    command = [
        "accelerate",
        "launch",
        "--multi_gpu",
        "--num_processes",
        str(WORLD_SIZE),
        "--mixed_precision",
        "bf16",
        "--main_process_port",
        "29642",
        TRAIN_CLI,
        "--config",
        BASE_CONFIG,
        "--override",
        *overrides,
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    return command


def _staged_sft_resume_variant_root(experiment: Experiment) -> Path:
    if experiment.key not in STAGED_SFT_RESUME_VARIANTS:
        raise ValueError(
            f"{experiment.key} is not a staged-SFT resume-gate variant"
        )
    return STAGED_SFT_RESUME_ROOT / experiment.key


def _staged_sft_resume_command(
    *,
    experiment: Experiment,
    manifest: Mapping[str, Any],
    initialization_identity: Mapping[str, Any],
    output_dir: Path,
    run_name: str,
    max_steps: int,
    weights_only: Path | None = None,
    resume: Path | None = None,
) -> list[str]:
    """Build one isolated staged-SFT process for the GPU restart gate."""

    if experiment.key not in STAGED_SFT_RESUME_VARIANTS:
        raise ValueError(
            f"{experiment.key} is not a staged-SFT resume-gate variant"
        )
    if (weights_only is None) == (resume is None):
        raise ValueError(
            "staged-SFT gate command requires exactly one of weights_only or resume"
        )
    if int(max_steps) not in (1, CANARY_TOTAL_STEPS):
        raise ValueError(
            "staged-SFT gate max_steps must be one or the complete canary length"
        )
    overrides = _overrides(
        experiment,
        "sft",
        manifest,
        canary=True,
        initialization_identity=initialization_identity,
    )
    for key, value in (
        ("training.output_dir", output_dir),
        ("training.run_name", run_name),
        ("training.max_steps", int(max_steps)),
        ("logging.id", f"{EXPERIMENT_VERSION}-{run_name}"),
    ):
        overrides = _replace_override(overrides, key, value)
    command = [
        "accelerate",
        "launch",
        "--multi_gpu",
        "--num_processes",
        str(WORLD_SIZE),
        "--mixed_precision",
        "bf16",
        "--main_process_port",
        "29643",
        TRAIN_CLI,
        "--config",
        BASE_CONFIG,
        "--override",
        *overrides,
    ]
    if weights_only is not None:
        command.extend(("--weights-only", str(weights_only)))
    else:
        command.extend(("--resume", str(resume)))
    return command


def _expected_training_process_argv(command: list[str]) -> list[str]:
    """Return the argv seen by the training script under Accelerate."""

    indices = [
        index for index, argument in enumerate(command) if argument == TRAIN_CLI
    ]
    if len(indices) != 1:
        raise RuntimeError(
            f"outer command must contain exactly one {TRAIN_CLI}: {command!r}"
        )
    return list(command[indices[0] :])


def _validate_training_process_argv(
    state: Mapping[str, Any],
    *,
    command: list[str],
    label: str,
) -> dict[str, Any]:
    """Bind persisted runtime argv to the exact gate process invocation."""

    runtime = state.get("runtime_provenance")
    if not isinstance(runtime, Mapping):
        raise RuntimeError(f"{label} lacks runtime provenance")
    expected = _expected_training_process_argv(command)
    observed = runtime.get("process_argv")
    if observed != expected:
        raise RuntimeError(
            f"{label} process argv drifted: {observed!r} != {expected!r}"
        )
    expected_sha256 = hashlib.sha256(
        json.dumps(
            expected,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if runtime.get("process_argv_sha256") != expected_sha256:
        raise RuntimeError(f"{label} process argv SHA-256 drifted")
    return {
        "schema": "interleaved-training-process-command-v1",
        "argv": expected,
        "sha256": expected_sha256,
    }


def _commit_new_authenticated_checkpoint(
    output_dir: Path,
    previous_pointer_sha256: str | None,
    *,
    label: str,
) -> str | None:
    """Commit a volume only after a new atomic pointer validates completely."""

    with checkpoint_volume_commit_lock(output_dir):
        pointer = output_dir / LATEST_CHECKPOINT_POINTER
        if not pointer.is_file():
            return previous_pointer_sha256
        pointer_sha256 = _sha256_file(pointer)
        if pointer_sha256 == previous_pointer_sha256:
            return previous_pointer_sha256
        checkpoints_root = output_dir / "resume_checkpoints"
        if checkpoints_root.is_dir() and any(
            child.name.startswith(".") for child in checkpoints_root.iterdir()
        ):
            # Defensive validation for output created by an older trainer that
            # did not participate in the commit lock protocol.
            return previous_pointer_sha256
        if any(
            path.exists() or path.is_symlink()
            for path in (output_dir / "final", output_dir / ".final.tmp")
        ):
            # Final export publication is authenticated and committed by the
            # caller after the child exits.  Never publish it incrementally.
            return previous_pointer_sha256
        checkpoint = validate_checkpoint_run_root(output_dir)
        checkpoint_volume.commit()
    print(
        f"[{label}] committed authenticated checkpoint {checkpoint}",
        flush=True,
    )
    return pointer_sha256


def _run_process_with_incremental_checkpoint_commits(
    command: list[str],
    *,
    label: str,
    output_dir: Path,
    poll_seconds: float = 5.0,
    initial_launch_command: Mapping[str, Any] | None = None,
    heartbeat: Callable[[], None] | None = None,
) -> int:
    """Run training while durably committing each published checkpoint."""

    if not math.isfinite(float(poll_seconds)) or float(poll_seconds) <= 0:
        raise ValueError("checkpoint polling interval must be positive and finite")
    pointer = output_dir / LATEST_CHECKPOINT_POINTER
    pointer_sha256 = _sha256_file(pointer) if pointer.is_file() else None
    process_environment = None
    if initial_launch_command is not None:
        if initial_launch_command.get("schema") != (
            "interleaved-initial-launch-command-v1"
        ):
            raise RuntimeError("invalid initial launch command identity")
        process_environment = dict(os.environ)
        process_environment[INITIAL_LAUNCH_COMMAND_ENV] = json.dumps(
            initial_launch_command.get("argv"),
            ensure_ascii=True,
            separators=(",", ":"),
        )
        process_environment[INITIAL_LAUNCH_COMMAND_SHA256_ENV] = str(
            initial_launch_command.get("sha256")
        )
    process = subprocess.Popen(
        command,
        cwd="/root/chess",
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=process_environment,
        start_new_session=True,
    )
    returncode: int | None = None
    last_heartbeat = time.monotonic()
    try:
        while returncode is None:
            try:
                returncode = int(process.wait(timeout=float(poll_seconds)))
            except subprocess.TimeoutExpired:
                pointer_sha256 = _commit_new_authenticated_checkpoint(
                    output_dir,
                    pointer_sha256,
                    label=label,
                )
                now = time.monotonic()
                if (
                    heartbeat is not None
                    and now - last_heartbeat >= LAUNCH_HEARTBEAT_SECONDS
                ):
                    heartbeat()
                    last_heartbeat = now
        pointer_sha256 = _commit_new_authenticated_checkpoint(
            output_dir,
            pointer_sha256,
            label=label,
        )
        return returncode
    except BaseException:
        if process.poll() is None:
            pid = getattr(process, "pid", None)
            try:
                if isinstance(pid, int) and pid > 0:
                    os.killpg(pid, signal.SIGTERM)
                else:
                    process.terminate()
            except (AttributeError, ProcessLookupError):
                pass
            try:
                process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                try:
                    if isinstance(pid, int) and pid > 0:
                        os.killpg(pid, signal.SIGKILL)
                    else:
                        process.kill()
                except (AttributeError, ProcessLookupError):
                    pass
                try:
                    process.wait(timeout=30.0)
                except (subprocess.TimeoutExpired, ChildProcessError):
                    pass
        raise


def _run_checked_process(
    command: list[str],
    *,
    label: str,
    output_dir: Path,
) -> None:
    print(f"[{label}] " + " ".join(command), flush=True)
    returncode = _run_process_with_incremental_checkpoint_commits(
        command,
        label=label,
        output_dir=output_dir,
    )
    if returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {returncode}")
    final = output_dir / "final"
    validate_checkpoint_run_root(
        output_dir,
        allowed_root_directories=frozenset({"final"}),
    )
    if not (final / "interleaved_training_state.json").is_file():
        raise RuntimeError(f"{label} did not produce a complete final export")
    _validate_fp32_hf_export(final)
    checkpoint_volume.commit()
    print(f"[{label}] validated final volume commit complete", flush=True)


def _run_rejected_process(
    command: list[str],
    *,
    label: str,
    expected_message: str,
) -> int:
    """Require a fail-closed process invocation to exit unsuccessfully."""

    print(f"[{label}] expecting rejection: " + " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd="/root/chess",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    print(result.stdout, end="", flush=True)
    if result.returncode == 0:
        raise RuntimeError(f"{label} unexpectedly accepted invalid state")
    if expected_message not in result.stdout:
        raise RuntimeError(
            f"{label} failed for the wrong reason; expected output containing "
            f"{expected_message!r}"
        )
    return int(result.returncode)


def _run_staged_sft_gate_process(
    command: list[str],
    *,
    label: str,
    output_dir: Path,
    experiment: Experiment,
    manifest: Mapping[str, Any],
    final_step: int,
    initialization_identity: Mapping[str, Any],
    initial_launch_command: Mapping[str, Any],
) -> tuple[Path, Mapping[str, Any], dict[str, Any]]:
    """Run and authenticate one process in the staged-SFT restart gate."""

    _validate_launch_command_identity(
        initial_launch_command,
        label=f"{label} initial launch",
    )
    print(f"[{label}] " + " ".join(command), flush=True)
    returncode = _run_process_with_incremental_checkpoint_commits(
        command,
        label=label,
        output_dir=output_dir,
        initial_launch_command=initial_launch_command,
    )
    if returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {returncode}")
    _validate_final(
        output_dir,
        expected_step=int(final_step),
        experiment=experiment,
        stage="sft",
        initialization_identity=initialization_identity,
        initial_launch_command=initial_launch_command,
    )
    checkpoint = validate_checkpoint_run_root(
        output_dir,
        allowed_root_directories=frozenset({"final"}),
    )
    state = _load_json(checkpoint / "trainer_state.json")
    _validate_stage_resume_state(
        state,
        experiment=experiment,
        stage="sft",
        manifest=manifest,
        expected_step=CANARY_TOTAL_STEPS,
        initialization_identity=initialization_identity,
        initial_launch_command=initial_launch_command,
    )
    if int(state.get("global_step", -1)) != int(final_step):
        raise RuntimeError(f"{label} checkpoint step drifted")
    process_command = _validate_training_process_argv(
        state,
        command=command,
        label=label,
    )
    checkpoint_volume.commit()
    print(f"[{label}] validated final volume commit complete", flush=True)
    return checkpoint, state, process_command


def _load_metric_trace(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [
        {
            "step": int(record["step"]),
            "lr": float(record["metrics"]["train/lr"]),
        }
        for record in records
    ]


def _assert_exact_fp32_models_equal(left: Path, right: Path) -> str:
    import torch
    from safetensors.torch import load_file

    left_file = left / "model.safetensors"
    right_file = right / "model.safetensors"
    left_state = load_file(str(left_file), device="cpu")
    right_state = load_file(str(right_file), device="cpu")
    if left_state.keys() != right_state.keys():
        raise RuntimeError("Resume canary model keys differ from reference")
    for name, left_tensor in left_state.items():
        right_tensor = right_state[name]
        if left_tensor.dtype != torch.float32 or right_tensor.dtype != torch.float32:
            raise RuntimeError(
                f"Resume canary export {name} is not FP32: "
                f"{left_tensor.dtype}, {right_tensor.dtype}"
            )
        if not torch.equal(left_tensor, right_tensor):
            maximum = float(
                (left_tensor - right_tensor).abs().max().item()
            )
            raise RuntimeError(
                f"Resume canary weight mismatch for {name}; max_abs={maximum}"
            )
    # The two files are expected to be byte-identical, but the authenticated
    # tensor equality above is the semantic requirement.
    return _sha256_file(left_file)


def _write_precision_resume_gate() -> dict[str, Any]:
    """Run a process-boundary resume and compare with uninterrupted training."""

    payload, manifest = _manifest("pt", canary=True)
    if PRECISION_RESUME_GATE_PATH.is_file():
        _validate_precision_resume_gate()
        return _load_json(PRECISION_RESUME_GATE_PATH)
    if PRECISION_RESUME_ROOT.exists():
        raise FileExistsError(
            "Incomplete or unauthenticated precision-resume canary artifacts "
            f"already exist at {PRECISION_RESUME_ROOT}; refusing to overwrite "
            "evidence"
        )
    first_root = PRECISION_RESUME_ROOT / "first_update"
    resumed_root = PRECISION_RESUME_ROOT / "resumed"
    reference_root = PRECISION_RESUME_ROOT / "reference"

    first = _precision_resume_command(
        manifest=manifest,
        output_dir=first_root,
        run_name="precision-resume-first-update-fp32-master-v13",
        max_steps=1,
    )
    _run_checked_process(
        first,
        label="precision-resume:update-1",
        output_dir=first_root,
    )
    first_checkpoint = resolve_resume_checkpoint(first_root)
    first_persisted_precision = inspect_accelerator_checkpoint_fp32(
        first_checkpoint
    )
    first_state_path = first_checkpoint / "trainer_state.json"
    first_state = _load_json(first_state_path)
    if (
        int(first_state.get("global_step", -1)) != 1
        or int(first_state.get("manifest_cursor", -1)) != 1
    ):
        raise RuntimeError("First resume-canary process did not persist update 1")
    first_sample_evidence = first_state["runtime_provenance"].get(
        "canary_sample_evidence"
    )
    _validate_canary_sample_evidence(first_sample_evidence, stage="pt")
    first_metric_evidence = _validate_canary_metrics(
        first_root,
        stage="pt",
        sample_evidence=first_sample_evidence,
    )

    incomplete_checkpoint = PRECISION_RESUME_ROOT / "interrupted_checkpoint"
    incomplete_checkpoint.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(first_state_path, incomplete_checkpoint / "trainer_state.json")
    (incomplete_checkpoint / "model_state.partial").write_bytes(
        b"simulated interrupted Accelerator.save_state"
    )
    rejected_output = PRECISION_RESUME_ROOT / "interrupted_resume_must_not_start"
    rejected = _precision_resume_command(
        manifest=manifest,
        output_dir=rejected_output,
        run_name="precision-resume-interrupted-rejection-fp32-master-v13",
        max_steps=2,
        resume=incomplete_checkpoint,
    )
    incomplete_rejection_exit_code = _run_rejected_process(
        rejected,
        label="precision-resume:reject-interrupted-checkpoint",
        expected_message="missing authenticated completion marker",
    )

    second = _precision_resume_command(
        manifest=manifest,
        output_dir=resumed_root,
        run_name="precision-resume-second-update-fp32-master-v13",
        max_steps=2,
        resume=first_checkpoint,
    )
    _run_checked_process(
        second,
        label="precision-resume:update-2",
        output_dir=resumed_root,
    )

    reference = _precision_resume_command(
        manifest=manifest,
        output_dir=reference_root,
        run_name="precision-resume-reference-fp32-master-v13",
        max_steps=2,
    )
    _run_checked_process(
        reference,
        label="precision-resume:reference",
        output_dir=reference_root,
    )

    resumed_checkpoint = resolve_resume_checkpoint(resumed_root)
    reference_checkpoint = resolve_resume_checkpoint(reference_root)
    resumed_state = _load_json(resumed_checkpoint / "trainer_state.json")
    reference_state = _load_json(reference_checkpoint / "trainer_state.json")
    for field in (
        "global_step",
        "manifest_cursor",
        "precision_contract",
        "determinism_contract",
    ):
        if resumed_state.get(field) != reference_state.get(field):
            raise RuntimeError(
                f"Resume canary {field} mismatch: "
                f"{resumed_state.get(field)!r} != {reference_state.get(field)!r}"
            )
    if int(resumed_state["global_step"]) != 2:
        raise RuntimeError("Resume canary did not reach update 2")
    resumed_metrics = _load_metric_trace(first_root / "metrics.jsonl")
    resumed_metrics.extend(_load_metric_trace(resumed_root / "metrics.jsonl"))
    reference_metrics = _load_metric_trace(reference_root / "metrics.jsonl")
    if resumed_metrics != reference_metrics:
        raise RuntimeError(
            f"Resume canary LR trace differs: "
            f"{resumed_metrics!r} != {reference_metrics!r}"
        )
    model_sha256 = _assert_exact_fp32_models_equal(
        resumed_root / "final",
        reference_root / "final",
    )
    _validate_fp32_hf_export(resumed_root / "final")
    bf16_inference = _validate_bf16_inference_from_fp32_hf_export(
        resumed_root / "final"
    )
    marker: dict[str, Any] = {
        "schema": "context2048-fp32-master-resume-canary-v1",
        "decision": "pass",
        "experiment_version": EXPERIMENT_VERSION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "manifest_set_hash": payload["set_hash"],
        "global_step": 2,
        "manifest_cursor": 2,
        "lr_trace": resumed_metrics,
        "precision_contract": resumed_state["precision_contract"],
        "determinism_contract": resumed_state["determinism_contract"],
        "persisted_update_1_precision": first_persisted_precision,
        "runtime_identity": first_state["runtime_provenance"],
        "sample_evidence": first_sample_evidence,
        "metric_evidence": first_metric_evidence,
        "model_sha256": model_sha256,
        "incomplete_checkpoint_rejected": True,
        "incomplete_checkpoint_rejection_exit_code": incomplete_rejection_exit_code,
        "incomplete_trainer_state_sha256": _checkpoint_sha256_file(
            incomplete_checkpoint / "trainer_state.json"
        ),
        "bf16_inference": bf16_inference,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    marker["gate_sha256"] = hashlib.sha256(_canonical_json(marker)).hexdigest()
    _atomic_json(PRECISION_RESUME_GATE_PATH, marker)
    data_volume.commit()
    return marker


def _validate_precision_resume_gate() -> None:
    if not PRECISION_RESUME_GATE_PATH.is_file():
        raise RuntimeError(
            "Missing FP32-master process-boundary resume gate; run "
            "--action precision-canary first"
        )
    gate = _load_json(PRECISION_RESUME_GATE_PATH)
    recorded = gate.pop("gate_sha256", None)
    if recorded != hashlib.sha256(_canonical_json(gate)).hexdigest():
        raise RuntimeError("FP32-master resume gate self hash drifted")
    manifest_set = _load_json(MANIFEST_SET_PATH)
    expected = {
        "schema": "context2048-fp32-master-resume-canary-v1",
        "decision": "pass",
        "experiment_version": EXPERIMENT_VERSION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "manifest_set_hash": manifest_set["set_hash"],
        "global_step": 2,
        "manifest_cursor": 2,
        "precision_contract": EXPECTED_PRECISION_CONTRACT,
        "determinism_contract": EXPECTED_DETERMINISM_CONTRACT,
        "incomplete_checkpoint_rejected": True,
    }
    for key, value in expected.items():
        if gate.get(key) != value:
            raise RuntimeError(
                f"FP32-master resume gate {key} drifted: "
                f"{gate.get(key)!r} != {value!r}"
            )
    inference = gate.get("bf16_inference")
    expected_inference = {
        "source_export_dtype": "float32",
        "in_memory_parameter_dtypes": ["bfloat16"],
        "forward_logits_dtype": "bfloat16",
        "forward_logits_finite": True,
        "input_shape": [1, 4],
        "tokenizer_contract": _validate_hf_tokenizer_contract(
            PRECISION_RESUME_ROOT / "resumed" / "final",
            expected_vocab_size=85,
        ),
    }
    if not isinstance(inference, Mapping) or inference != expected_inference:
        raise RuntimeError(f"FP32-export/BF16-inference evidence drifted: {inference!r}")
    if int(gate.get("incomplete_checkpoint_rejection_exit_code", 0)) == 0:
        raise RuntimeError("Interrupted-checkpoint rejection has no failing exit code")
    persisted = gate.get("persisted_update_1_precision")
    if not isinstance(persisted, Mapping) or (
        persisted.get("model_floating_dtype") != "float32"
        or persisted.get("adam_moment_dtype") != "float32"
        or int(persisted.get("adam_moment_tensor_count", 0)) <= 0
    ):
        raise RuntimeError(
            f"persisted Accelerator precision evidence drifted: {persisted!r}"
        )
    sample_evidence = gate.get("sample_evidence")
    if not isinstance(sample_evidence, Mapping) or not (
        sample_evidence.get("contract") == "pt-only"
        and int(sample_evidence.get("global_pretrain_rows", 0)) > 0
        and int(sample_evidence.get("global_pretrain_supervised_tokens", 0)) > 0
        and int(sample_evidence.get("global_sft_rows", -1)) == 0
        and int(sample_evidence.get("global_sft_supervised_tokens", -1)) == 0
        and sample_evidence.get("pt_leading_bos_validated") is True
    ):
        raise RuntimeError(f"PT canary sample evidence drifted: {sample_evidence!r}")
    observed_metrics = _validate_canary_metrics(
        PRECISION_RESUME_ROOT / "first_update",
        stage="pt",
        sample_evidence=sample_evidence,
    )
    if gate.get("metric_evidence") != observed_metrics:
        raise RuntimeError("PT precision-canary metric evidence drifted")
    _validate_recorded_runtime_identity(gate.get("runtime_identity"))


def _require_fp32_checkpoint_evidence(
    evidence: Any,
    *,
    label: str,
) -> None:
    model = evidence.get("model") if isinstance(evidence, Mapping) else None
    optimizer_files = (
        evidence.get("optimizer_files")
        if isinstance(evidence, Mapping)
        else None
    )
    model_tensor_count = (
        int(model.get("tensor_count", 0))
        if isinstance(model, Mapping)
        else 0
    )
    parameter_state_count = (
        sum(
            int(item.get("parameter_state_count", 0))
            for item in optimizer_files
            if isinstance(item, Mapping)
        )
        if isinstance(optimizer_files, list)
        else 0
    )
    if not isinstance(evidence, Mapping) or (
        evidence.get("schema")
        != "interleaved-accelerator-persisted-fp32-v1"
        or evidence.get("model_floating_dtype") != "float32"
        or evidence.get("adam_moment_dtype") != "float32"
        or not isinstance(model, Mapping)
        or model_tensor_count <= 0
        or model.get("dtype_counts") != {"F32": model_tensor_count}
        or not isinstance(optimizer_files, list)
        or not optimizer_files
        or parameter_state_count <= 0
        or int(evidence.get("adam_moment_tensor_count", 0))
        != 2 * parameter_state_count
        or int(evidence.get("optimizer_tensor_count", 0))
        != 3 * parameter_state_count
    ):
        raise RuntimeError(f"{label} FP32 checkpoint evidence drifted: {evidence!r}")


def _staged_sft_gate_run_name(experiment: Experiment, leg: str) -> str:
    if experiment.key not in STAGED_SFT_RESUME_VARIANTS:
        raise ValueError(
            f"{experiment.key} is not a staged-SFT resume-gate variant"
        )
    if leg not in {"first-update", "resumed", "reference"}:
        raise ValueError(f"unknown staged-SFT gate leg: {leg}")
    return (
        f"staged-sft-resume-{experiment.key}-{leg}-fp32-master-v13"
    )


def _write_staged_sft_resume_gate() -> dict[str, Any]:
    """Prove two-process staged SFT restart for both tokenizer parents."""

    payload, manifest = _manifest("sft3", canary=True)
    if STAGED_SFT_RESUME_GATE_PATH.is_file():
        _validate_staged_sft_resume_gate()
        return _load_json(STAGED_SFT_RESUME_GATE_PATH)
    if STAGED_SFT_RESUME_ROOT.exists():
        raise FileExistsError(
            "Incomplete or unauthenticated staged-SFT resume canary artifacts "
            f"already exist at {STAGED_SFT_RESUME_ROOT}; refusing to overwrite "
            "evidence"
        )

    variants: dict[str, Any] = {}
    for experiment_key in STAGED_SFT_RESUME_VARIANTS:
        experiment = EXPERIMENTS[experiment_key]
        parent_final = _run_stage(experiment, "pt", canary=True)
        initialization_identity = _stage_initialization_identity(
            experiment,
            "sft",
            weights_only=parent_final,
            canary=True,
        )
        variant_root = _staged_sft_resume_variant_root(experiment)
        first_root = variant_root / "first_update"
        resumed_root = variant_root / "resumed"
        reference_root = variant_root / "reference"

        first_command = _staged_sft_resume_command(
            experiment=experiment,
            manifest=manifest,
            initialization_identity=initialization_identity,
            output_dir=first_root,
            run_name=_staged_sft_gate_run_name(experiment, "first-update"),
            max_steps=1,
            weights_only=parent_final,
        )
        initial_launch_command = _initial_launch_command_identity(first_command)
        first_checkpoint, first_state, first_process_command = (
            _run_staged_sft_gate_process(
                first_command,
                label=f"staged-sft-resume:{experiment.key}:update-1",
                output_dir=first_root,
                experiment=experiment,
                manifest=manifest,
                final_step=1,
                initialization_identity=initialization_identity,
                initial_launch_command=initial_launch_command,
            )
        )
        if int(first_state.get("manifest_cursor", -1)) != 1:
            raise RuntimeError(
                f"{experiment.key} SFT update-1 manifest cursor drifted"
            )
        first_precision = inspect_accelerator_checkpoint_fp32(first_checkpoint)
        _require_fp32_checkpoint_evidence(
            first_precision,
            label=f"{experiment.key} SFT update 1",
        )
        sample_evidence = first_state.get("runtime_provenance", {}).get(
            "canary_sample_evidence"
        )
        _validate_canary_sample_evidence(sample_evidence, stage="sft")
        metric_evidence = _validate_canary_metrics(
            first_root,
            stage="sft",
            sample_evidence=sample_evidence,
        )

        resume_command = _staged_sft_resume_command(
            experiment=experiment,
            manifest=manifest,
            initialization_identity=initialization_identity,
            output_dir=resumed_root,
            run_name=_staged_sft_gate_run_name(experiment, "resumed"),
            max_steps=CANARY_TOTAL_STEPS,
            resume=first_checkpoint,
        )
        if "--weights-only" in resume_command or resume_command[-2] != "--resume":
            raise AssertionError("staged-SFT resume command contract drifted")
        resumed_checkpoint, resumed_state, resumed_process_command = (
            _run_staged_sft_gate_process(
                resume_command,
                label=f"staged-sft-resume:{experiment.key}:update-2",
                output_dir=resumed_root,
                experiment=experiment,
                manifest=manifest,
                final_step=CANARY_TOTAL_STEPS,
                initialization_identity=initialization_identity,
                initial_launch_command=initial_launch_command,
            )
        )
        if int(resumed_state.get("manifest_cursor", -1)) != CANARY_TOTAL_STEPS:
            raise RuntimeError(
                f"{experiment.key} resumed SFT manifest cursor drifted"
            )
        resumed_precision = inspect_accelerator_checkpoint_fp32(
            resumed_checkpoint
        )
        _require_fp32_checkpoint_evidence(
            resumed_precision,
            label=f"{experiment.key} resumed SFT update 2",
        )

        reference_command = _staged_sft_resume_command(
            experiment=experiment,
            manifest=manifest,
            initialization_identity=initialization_identity,
            output_dir=reference_root,
            run_name=_staged_sft_gate_run_name(experiment, "reference"),
            max_steps=CANARY_TOTAL_STEPS,
            weights_only=parent_final,
        )
        reference_initial_launch_command = _initial_launch_command_identity(
            reference_command
        )
        reference_checkpoint, reference_state, reference_process_command = (
            _run_staged_sft_gate_process(
                reference_command,
                label=f"staged-sft-resume:{experiment.key}:reference",
                output_dir=reference_root,
                experiment=experiment,
                manifest=manifest,
                final_step=CANARY_TOTAL_STEPS,
                initialization_identity=initialization_identity,
                initial_launch_command=reference_initial_launch_command,
            )
        )
        if int(reference_state.get("manifest_cursor", -1)) != CANARY_TOTAL_STEPS:
            raise RuntimeError(
                f"{experiment.key} reference SFT manifest cursor drifted"
            )
        reference_precision = inspect_accelerator_checkpoint_fp32(
            reference_checkpoint
        )
        _require_fp32_checkpoint_evidence(
            reference_precision,
            label=f"{experiment.key} reference SFT update 2",
        )

        resumed_lr_trace = _load_metric_trace(first_root / "metrics.jsonl")
        resumed_lr_trace.extend(
            _load_metric_trace(resumed_root / "metrics.jsonl")
        )
        reference_lr_trace = _load_metric_trace(
            reference_root / "metrics.jsonl"
        )
        if resumed_lr_trace != reference_lr_trace:
            raise RuntimeError(
                f"{experiment.key} staged-SFT resume LR trace differs: "
                f"{resumed_lr_trace!r} != {reference_lr_trace!r}"
            )
        model_sha256 = _assert_exact_fp32_models_equal(
            resumed_root / "final",
            reference_root / "final",
        )
        final_fp32_evidence = _validate_fp32_hf_export(
            resumed_root / "final"
        )
        final_tokenizer_contract = _validate_hf_tokenizer_contract(
            resumed_root / "final",
            expected_vocab_size=85,
        )
        for state_label, state in (
            ("first", first_state),
            ("resumed", resumed_state),
            ("reference", reference_state),
        ):
            _validate_recorded_runtime_identity(state.get("runtime_provenance"))
            if state["configured_provenance"].get(
                "initialization_identity"
            ) != initialization_identity:
                raise RuntimeError(
                    f"{experiment.key} {state_label} SFT parent identity drifted"
                )

        variants[experiment.key] = {
            "parent_final": str(parent_final),
            "initialization_identity": initialization_identity,
            "initial_launch_command": initial_launch_command,
            "reference_initial_launch_command": reference_initial_launch_command,
            "first_process_command": first_process_command,
            "resume_process_command": resumed_process_command,
            "reference_process_command": reference_process_command,
            "checkpoint_1_path": str(first_checkpoint),
            "checkpoint_1_completion_marker_sha256": _checkpoint_sha256_file(
                first_checkpoint / ".complete.json"
            ),
            "checkpoint_1_precision": first_precision,
            "resumed_checkpoint_precision": resumed_precision,
            "reference_checkpoint_precision": reference_precision,
            "sample_evidence": sample_evidence,
            "metric_evidence": metric_evidence,
            "resumed_step": int(resumed_state["global_step"]),
            "resumed_manifest_cursor": int(resumed_state["manifest_cursor"]),
            "resumed_lr_trace": resumed_lr_trace,
            "reference_lr_trace": reference_lr_trace,
            "model_sha256": model_sha256,
            "final_fp32_evidence": final_fp32_evidence,
            "final_tokenizer_contract": final_tokenizer_contract,
            "runtime_identity": first_state["runtime_provenance"],
        }

    marker: dict[str, Any] = {
        "schema": "context2048-staged-sft-resume-canary-v1",
        "decision": "pass",
        "experiment_version": EXPERIMENT_VERSION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "manifest_set_hash": payload["set_hash"],
        "variant_keys": list(STAGED_SFT_RESUME_VARIANTS),
        "variants": variants,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    marker["gate_sha256"] = hashlib.sha256(_canonical_json(marker)).hexdigest()
    _atomic_json(STAGED_SFT_RESUME_GATE_PATH, marker)
    data_volume.commit()
    _validate_staged_sft_resume_gate()
    return marker


def _validate_staged_sft_resume_gate() -> dict[str, Any]:
    """Recompute all staged-SFT restart evidence from immutable artifacts."""

    if not STAGED_SFT_RESUME_GATE_PATH.is_file():
        raise RuntimeError(
            "Missing staged-SFT process-boundary resume gate; run "
            "--action precision-canary first"
        )
    gate = _load_json(STAGED_SFT_RESUME_GATE_PATH)
    recorded_gate_sha256 = gate.pop("gate_sha256", None)
    if recorded_gate_sha256 != hashlib.sha256(_canonical_json(gate)).hexdigest():
        raise RuntimeError("staged-SFT resume gate self hash drifted")
    payload, manifest = _manifest("sft3", canary=True)
    expected_top_level = {
        "schema": "context2048-staged-sft-resume-canary-v1",
        "decision": "pass",
        "experiment_version": EXPERIMENT_VERSION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "manifest_set_hash": payload["set_hash"],
        "variant_keys": list(STAGED_SFT_RESUME_VARIANTS),
    }
    expected_top_level_fields = set(expected_top_level) | {
        "variants",
        "created_at",
    }
    if set(gate) != expected_top_level_fields:
        raise RuntimeError(
            f"staged-SFT resume gate fields drifted: {sorted(gate)}"
        )
    for key, expected in expected_top_level.items():
        if gate.get(key) != expected:
            raise RuntimeError(
                f"staged-SFT resume gate {key} drifted: "
                f"{gate.get(key)!r} != {expected!r}"
            )
    variants = gate.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != set(
        STAGED_SFT_RESUME_VARIANTS
    ):
        raise RuntimeError(
            f"staged-SFT resume variant inventory drifted: {variants!r}"
        )

    expected_variant_fields = {
        "parent_final",
        "initialization_identity",
        "initial_launch_command",
        "reference_initial_launch_command",
        "first_process_command",
        "resume_process_command",
        "reference_process_command",
        "checkpoint_1_path",
        "checkpoint_1_completion_marker_sha256",
        "checkpoint_1_precision",
        "resumed_checkpoint_precision",
        "reference_checkpoint_precision",
        "sample_evidence",
        "metric_evidence",
        "resumed_step",
        "resumed_manifest_cursor",
        "resumed_lr_trace",
        "reference_lr_trace",
        "model_sha256",
        "final_fp32_evidence",
        "final_tokenizer_contract",
        "runtime_identity",
    }
    for experiment_key in STAGED_SFT_RESUME_VARIANTS:
        experiment = EXPERIMENTS[experiment_key]
        record = variants[experiment_key]
        if not isinstance(record, Mapping) or set(record) != expected_variant_fields:
            raise RuntimeError(
                f"{experiment.key} staged-SFT gate fields drifted: {record!r}"
            )
        parent_final = _stage_output(
            experiment,
            "pt",
            canary=True,
        ) / "final"
        initialization_identity = _stage_initialization_identity(
            experiment,
            "sft",
            weights_only=parent_final,
            canary=True,
        )
        if record.get("parent_final") != str(parent_final):
            raise RuntimeError(f"{experiment.key} staged-SFT parent path drifted")
        if record.get("initialization_identity") != initialization_identity:
            raise RuntimeError(
                f"{experiment.key} staged-SFT full parent identity drifted"
            )

        variant_root = _staged_sft_resume_variant_root(experiment)
        first_root = variant_root / "first_update"
        resumed_root = variant_root / "resumed"
        reference_root = variant_root / "reference"
        first_command = _staged_sft_resume_command(
            experiment=experiment,
            manifest=manifest,
            initialization_identity=initialization_identity,
            output_dir=first_root,
            run_name=_staged_sft_gate_run_name(experiment, "first-update"),
            max_steps=1,
            weights_only=parent_final,
        )
        expected_initial_launch = _initial_launch_command_identity(first_command)
        _validate_launch_command_identity(
            record.get("initial_launch_command"),
            label=f"{experiment.key} staged-SFT initial launch",
        )
        if record.get("initial_launch_command") != expected_initial_launch:
            raise RuntimeError(
                f"{experiment.key} staged-SFT initial launch command drifted"
            )
        first_checkpoint = validate_checkpoint_run_root(
            first_root,
            allowed_root_directories=frozenset({"final"}),
        )
        if record.get("checkpoint_1_path") != str(first_checkpoint):
            raise RuntimeError(
                f"{experiment.key} staged-SFT checkpoint-1 path drifted"
            )
        marker_sha256 = _checkpoint_sha256_file(
            first_checkpoint / ".complete.json"
        )
        if record.get("checkpoint_1_completion_marker_sha256") != marker_sha256:
            raise RuntimeError(
                f"{experiment.key} staged-SFT checkpoint-1 marker drifted"
            )
        first_precision = inspect_accelerator_checkpoint_fp32(first_checkpoint)
        _require_fp32_checkpoint_evidence(
            first_precision,
            label=f"{experiment.key} staged-SFT update 1",
        )
        if record.get("checkpoint_1_precision") != first_precision:
            raise RuntimeError(
                f"{experiment.key} staged-SFT checkpoint-1 precision drifted"
            )
        first_state = _load_json(first_checkpoint / "trainer_state.json")
        _validate_stage_resume_state(
            first_state,
            experiment=experiment,
            stage="sft",
            manifest=manifest,
            expected_step=CANARY_TOTAL_STEPS,
            initialization_identity=initialization_identity,
            initial_launch_command=expected_initial_launch,
        )
        if (
            int(first_state.get("global_step", -1)) != 1
            or int(first_state.get("manifest_cursor", -1)) != 1
        ):
            raise RuntimeError(
                f"{experiment.key} staged-SFT update-1 state drifted"
            )
        expected_first_process = _validate_training_process_argv(
            first_state,
            command=first_command,
            label=f"{experiment.key} staged-SFT update 1",
        )
        if record.get("first_process_command") != expected_first_process:
            raise RuntimeError(
                f"{experiment.key} staged-SFT first process command drifted"
            )
        sample_evidence = first_state.get("runtime_provenance", {}).get(
            "canary_sample_evidence"
        )
        _validate_canary_sample_evidence(sample_evidence, stage="sft")
        if record.get("sample_evidence") != sample_evidence:
            raise RuntimeError(
                f"{experiment.key} staged-SFT sample evidence drifted"
            )
        metric_evidence = _validate_canary_metrics(
            first_root,
            stage="sft",
            sample_evidence=sample_evidence,
        )
        if record.get("metric_evidence") != metric_evidence:
            raise RuntimeError(
                f"{experiment.key} staged-SFT metric evidence drifted"
            )

        resume_command = _staged_sft_resume_command(
            experiment=experiment,
            manifest=manifest,
            initialization_identity=initialization_identity,
            output_dir=resumed_root,
            run_name=_staged_sft_gate_run_name(experiment, "resumed"),
            max_steps=CANARY_TOTAL_STEPS,
            resume=first_checkpoint,
        )
        if "--weights-only" in resume_command or resume_command[-2] != "--resume":
            raise RuntimeError(
                f"{experiment.key} staged-SFT resume invocation drifted"
            )
        _validate_final(
            resumed_root,
            expected_step=CANARY_TOTAL_STEPS,
            experiment=experiment,
            stage="sft",
            initialization_identity=initialization_identity,
            initial_launch_command=expected_initial_launch,
        )
        resumed_checkpoint = validate_checkpoint_run_root(
            resumed_root,
            allowed_root_directories=frozenset({"final"}),
        )
        resumed_state = _load_json(resumed_checkpoint / "trainer_state.json")
        _validate_stage_resume_state(
            resumed_state,
            experiment=experiment,
            stage="sft",
            manifest=manifest,
            expected_step=CANARY_TOTAL_STEPS,
            initialization_identity=initialization_identity,
            initial_launch_command=expected_initial_launch,
        )
        if (
            int(resumed_state.get("global_step", -1)) != CANARY_TOTAL_STEPS
            or int(resumed_state.get("manifest_cursor", -1))
            != CANARY_TOTAL_STEPS
            or int(record.get("resumed_step", -1)) != CANARY_TOTAL_STEPS
            or int(record.get("resumed_manifest_cursor", -1))
            != CANARY_TOTAL_STEPS
        ):
            raise RuntimeError(
                f"{experiment.key} staged-SFT resumed step/cursor drifted"
            )
        expected_resume_process = _validate_training_process_argv(
            resumed_state,
            command=resume_command,
            label=f"{experiment.key} staged-SFT resumed process",
        )
        if record.get("resume_process_command") != expected_resume_process:
            raise RuntimeError(
                f"{experiment.key} staged-SFT resume process command drifted"
            )
        resumed_precision = inspect_accelerator_checkpoint_fp32(
            resumed_checkpoint
        )
        _require_fp32_checkpoint_evidence(
            resumed_precision,
            label=f"{experiment.key} staged-SFT resumed update 2",
        )
        if record.get("resumed_checkpoint_precision") != resumed_precision:
            raise RuntimeError(
                f"{experiment.key} staged-SFT resumed precision drifted"
            )

        reference_command = _staged_sft_resume_command(
            experiment=experiment,
            manifest=manifest,
            initialization_identity=initialization_identity,
            output_dir=reference_root,
            run_name=_staged_sft_gate_run_name(experiment, "reference"),
            max_steps=CANARY_TOTAL_STEPS,
            weights_only=parent_final,
        )
        expected_reference_initial = _initial_launch_command_identity(
            reference_command
        )
        _validate_launch_command_identity(
            record.get("reference_initial_launch_command"),
            label=f"{experiment.key} staged-SFT reference launch",
        )
        if record.get("reference_initial_launch_command") != (
            expected_reference_initial
        ):
            raise RuntimeError(
                f"{experiment.key} staged-SFT reference launch command drifted"
            )
        _validate_final(
            reference_root,
            expected_step=CANARY_TOTAL_STEPS,
            experiment=experiment,
            stage="sft",
            initialization_identity=initialization_identity,
            initial_launch_command=expected_reference_initial,
        )
        reference_checkpoint = validate_checkpoint_run_root(
            reference_root,
            allowed_root_directories=frozenset({"final"}),
        )
        reference_state = _load_json(reference_checkpoint / "trainer_state.json")
        _validate_stage_resume_state(
            reference_state,
            experiment=experiment,
            stage="sft",
            manifest=manifest,
            expected_step=CANARY_TOTAL_STEPS,
            initialization_identity=initialization_identity,
            initial_launch_command=expected_reference_initial,
        )
        if (
            int(reference_state.get("global_step", -1)) != CANARY_TOTAL_STEPS
            or int(reference_state.get("manifest_cursor", -1))
            != CANARY_TOTAL_STEPS
        ):
            raise RuntimeError(
                f"{experiment.key} staged-SFT reference step/cursor drifted"
            )
        expected_reference_process = _validate_training_process_argv(
            reference_state,
            command=reference_command,
            label=f"{experiment.key} staged-SFT reference process",
        )
        if record.get("reference_process_command") != expected_reference_process:
            raise RuntimeError(
                f"{experiment.key} staged-SFT reference process command drifted"
            )
        reference_precision = inspect_accelerator_checkpoint_fp32(
            reference_checkpoint
        )
        _require_fp32_checkpoint_evidence(
            reference_precision,
            label=f"{experiment.key} staged-SFT reference update 2",
        )
        if record.get("reference_checkpoint_precision") != reference_precision:
            raise RuntimeError(
                f"{experiment.key} staged-SFT reference precision drifted"
            )

        resumed_lr_trace = _load_metric_trace(first_root / "metrics.jsonl")
        resumed_lr_trace.extend(
            _load_metric_trace(resumed_root / "metrics.jsonl")
        )
        reference_lr_trace = _load_metric_trace(
            reference_root / "metrics.jsonl"
        )
        if (
            resumed_lr_trace != reference_lr_trace
            or record.get("resumed_lr_trace") != resumed_lr_trace
            or record.get("reference_lr_trace") != reference_lr_trace
        ):
            raise RuntimeError(
                f"{experiment.key} staged-SFT LR trace evidence drifted"
            )
        model_sha256 = _assert_exact_fp32_models_equal(
            resumed_root / "final",
            reference_root / "final",
        )
        if record.get("model_sha256") != model_sha256:
            raise RuntimeError(
                f"{experiment.key} staged-SFT exact model equality drifted"
            )
        final_fp32_evidence = _validate_fp32_hf_export(
            resumed_root / "final"
        )
        if record.get("final_fp32_evidence") != final_fp32_evidence:
            raise RuntimeError(
                f"{experiment.key} staged-SFT final FP32 evidence drifted"
            )
        final_tokenizer_contract = _validate_hf_tokenizer_contract(
            resumed_root / "final",
            expected_vocab_size=85,
        )
        if record.get("final_tokenizer_contract") != final_tokenizer_contract:
            raise RuntimeError(
                f"{experiment.key} staged-SFT tokenizer evidence drifted"
            )
        first_runtime = first_state.get("runtime_provenance")
        if record.get("runtime_identity") != first_runtime:
            raise RuntimeError(
                f"{experiment.key} staged-SFT runtime identity drifted"
            )
        for runtime in (
            first_runtime,
            resumed_state.get("runtime_provenance"),
            reference_state.get("runtime_provenance"),
        ):
            _validate_recorded_runtime_identity(runtime)
    return {
        **gate,
        "gate_sha256": recorded_gate_sha256,
    }


def _gate_path(experiment: Experiment) -> Path:
    return GATE_ROOT / f"{experiment.key}.json"


def _write_gate(experiment: Experiment) -> None:
    payload = _load_json(MANIFEST_SET_PATH)
    staged_sft_resume_gate_sha256: str | None = None
    if experiment.kind == "staged":
        staged_gate = _validate_staged_sft_resume_gate()
        staged_sft_resume_gate_sha256 = str(staged_gate["gate_sha256"])
    sample_evidence: dict[str, Any] = {}
    metric_evidence: dict[str, Any] = {}
    runtime_evidence: dict[str, Any] | None = None
    for stage in experiment.stages:
        output_root = _stage_output(experiment, stage, canary=True)
        final = output_root / "final"
        state = validate_completed_hf_export(final)["state"]
        evidence = state.get("runtime_provenance", {}).get(
            "canary_sample_evidence"
        )
        _validate_canary_sample_evidence(evidence, stage=stage)
        sample_evidence[stage] = evidence
        metric_evidence[stage] = _validate_canary_metrics(
            output_root,
            stage=stage,
            sample_evidence=evidence,
        )
        if runtime_evidence is None:
            runtime_evidence = dict(state["runtime_provenance"])
        else:
            for key in (
                "modal_app_name",
                "modal_app_id",
                "modal_image_id",
                "modal_base_image",
                "modal_client_version",
                "runtime_package_versions",
                "runtime_distribution_count",
                "runtime_distribution_inventory_sha256",
                "python_version",
            ):
                if state["runtime_provenance"].get(key) != runtime_evidence.get(key):
                    raise RuntimeError(
                        f"canary stages used different runtime identity for {key}"
                    )
    _validate_recorded_runtime_identity(runtime_evidence)
    gate: dict[str, Any] = {
        "schema": "context2048-vocab-mixing-canary-gate-v1",
        "decision": "pass",
        "experiment_version": EXPERIMENT_VERSION,
        "experiment": experiment.key,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "manifest_set_hash": payload["set_hash"],
        "runtime_identity": runtime_evidence,
        "sample_evidence": sample_evidence,
        "metric_evidence": metric_evidence,
        "staged_sft_resume_gate_sha256": staged_sft_resume_gate_sha256,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    gate["gate_sha256"] = hashlib.sha256(_canonical_json(gate)).hexdigest()
    _atomic_json(_gate_path(experiment), gate)
    data_volume.commit()


def _validate_gate(experiment: Experiment) -> None:
    _validate_precision_resume_gate()
    staged_sft_resume_gate_sha256: str | None = None
    if experiment.kind == "staged":
        staged_gate = _validate_staged_sft_resume_gate()
        staged_sft_resume_gate_sha256 = str(staged_gate["gate_sha256"])
    path = _gate_path(experiment)
    if not path.is_file():
        raise RuntimeError(f"Missing canary gate for {experiment.key}")
    gate = _load_json(path)
    recorded = gate.pop("gate_sha256", None)
    if recorded != hashlib.sha256(_canonical_json(gate)).hexdigest():
        raise RuntimeError(f"Canary gate self hash drifted for {experiment.key}")
    manifest_set = _load_json(MANIFEST_SET_PATH)
    expected = {
        "decision": "pass",
        "experiment_version": EXPERIMENT_VERSION,
        "experiment": experiment.key,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "manifest_set_hash": manifest_set["set_hash"],
    }
    for key, value in expected.items():
        if gate.get(key) != value:
            raise RuntimeError(
                f"Canary gate {key} drifted for {experiment.key}: "
                f"{gate.get(key)!r} != {value!r}"
            )
    if gate.get("staged_sft_resume_gate_sha256") != (
        staged_sft_resume_gate_sha256
    ):
        raise RuntimeError(
            f"Canary gate staged-SFT resume binding drifted for {experiment.key}"
        )
    _validate_recorded_runtime_identity(gate.get("runtime_identity"))
    recorded_samples = gate.get("sample_evidence")
    if not isinstance(recorded_samples, Mapping) or set(recorded_samples) != set(
        experiment.stages
    ):
        raise RuntimeError(
            f"Canary gate sample evidence inventory drifted: {recorded_samples!r}"
        )
    for stage in experiment.stages:
        _validate_canary_sample_evidence(recorded_samples[stage], stage=stage)
    recorded_metrics = gate.get("metric_evidence")
    if not isinstance(recorded_metrics, Mapping) or set(recorded_metrics) != set(
        experiment.stages
    ):
        raise RuntimeError(
            f"Canary gate metric evidence inventory drifted: {recorded_metrics!r}"
        )
    for stage in experiment.stages:
        observed = _validate_canary_metrics(
            _stage_output(experiment, stage, canary=True),
            stage=stage,
            sample_evidence=recorded_samples[stage],
        )
        if observed != recorded_metrics[stage]:
            raise RuntimeError(
                f"Canary gate metric evidence drifted for {experiment.key}/{stage}"
            )


def _wandb_api_key() -> str:
    api_key = str(os.environ.get("WANDB_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("deployed W&B secret has no WANDB_API_KEY")
    return api_key


def _wandb_gate_run_id() -> str:
    return f"infra-gate-{SOURCE_TREE_SHA256[:24]}"


def _wandb_gate_tags() -> tuple[str, ...]:
    return ("infra-gate", "remote-write", EXPERIMENT_VERSION)


def _wandb_gate_summary() -> dict[str, Any]:
    return {
        "infra_gate/schema": WANDB_WRITE_GATE_SCHEMA,
        "infra_gate/write_marker": 1,
        "infra_gate/experiment_version": EXPERIMENT_VERSION,
        "infra_gate/source_tree_sha256": SOURCE_TREE_SHA256,
        "infra_gate/entity": WANDB_ENTITY,
        "infra_gate/project": WANDB_PROJECT,
    }


def _authenticate_wandb_remote_run(api: Any, *, run_id: str) -> dict[str, Any]:
    """Read back the exact remote run and its persisted write marker."""

    path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}"
    remote = api.run(path)
    observed_id = str(getattr(remote, "id", "") or "")
    observed_entity = str(getattr(remote, "entity", "") or "")
    observed_project = str(getattr(remote, "project", "") or "")
    observed_path = "/".join(
        str(item) for item in (getattr(remote, "path", None) or ())
    )
    if observed_id != run_id:
        raise RuntimeError(f"W&B write-gate run ID drifted: {observed_id!r}")
    if observed_entity and observed_entity != WANDB_ENTITY:
        raise RuntimeError(f"W&B write-gate entity drifted: {observed_entity!r}")
    if observed_project and observed_project != WANDB_PROJECT:
        raise RuntimeError(f"W&B write-gate project drifted: {observed_project!r}")
    if observed_path and observed_path != path:
        raise RuntimeError(f"W&B write-gate path drifted: {observed_path!r}")
    observed_job_type = str(getattr(remote, "job_type", "") or "")
    if observed_job_type != "infra-gate":
        raise RuntimeError(
            f"W&B write-gate job type drifted: {observed_job_type!r}"
        )
    observed_group = str(getattr(remote, "group", "") or "")
    if observed_group:
        raise RuntimeError(
            f"W&B write-gate polluted a production group: {observed_group!r}"
        )
    observed_tags = tuple(sorted(str(tag) for tag in (remote.tags or ())))
    expected_tags = tuple(sorted(_wandb_gate_tags()))
    if observed_tags != expected_tags:
        raise RuntimeError(f"W&B write-gate tags drifted: {observed_tags!r}")
    observed_state = str(getattr(remote, "state", "") or "")
    if observed_state != "finished":
        raise RuntimeError(f"W&B write-gate is not finished: {observed_state!r}")
    summary = dict(getattr(remote, "summary", {}) or {})
    expected_summary = _wandb_gate_summary()
    drift = {
        key: {"expected": value, "observed": summary.get(key)}
        for key, value in expected_summary.items()
        if summary.get(key) != value
    }
    if drift:
        raise RuntimeError(f"W&B write-gate marker drifted: {drift}")
    return {
        "run_id": run_id,
        "path": path,
        "url": str(getattr(remote, "url", "") or ""),
        "job_type": "infra-gate",
        "group": "",
        "tags": list(expected_tags),
        "state": "finished",
        "summary": expected_summary,
    }


def _wandb_api() -> Any:
    import wandb

    api = wandb.Api(api_key=_wandb_api_key(), timeout=30)
    viewer = api.viewer
    viewer_username = str(getattr(viewer, "username", "") or "")
    if not viewer_username:
        raise RuntimeError("W&B API key did not authenticate a viewer")
    team = api.team(WANDB_ENTITY)
    observed_team = str(getattr(team, "name", "") or "")
    if observed_team != WANDB_ENTITY:
        raise RuntimeError(
            f"W&B entity access drifted: {observed_team!r} != {WANDB_ENTITY!r}"
        )
    return api


def _validate_wandb_write_gate(store: Any) -> dict[str, Any]:
    marker = _validate_self_hash(
        store.get(WANDB_WRITE_GATE_KEY, None),
        hash_field="gate_sha256",
        label="W&B remote-write gate",
    )
    expected = {
        "schema": WANDB_WRITE_GATE_SCHEMA,
        "decision": "pass",
        "experiment_version": EXPERIMENT_VERSION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "run_id": _wandb_gate_run_id(),
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise RuntimeError(f"W&B write-gate {key} drifted")
    _validate_recorded_runtime_identity(marker.get("runtime_identity"))
    observed = _authenticate_wandb_remote_run(
        _wandb_api(),
        run_id=str(marker["run_id"]),
    )
    if marker.get("remote_evidence") != observed:
        raise RuntimeError("W&B write-gate remote evidence drifted")
    return marker


def _publish_wandb_write_gate(store: Any) -> dict[str, Any]:
    """Perform one minimal online write, finish, and read it back remotely."""

    existing = store.get(WANDB_WRITE_GATE_KEY, None)
    if existing is not None:
        return _validate_wandb_write_gate(store)
    import wandb

    run_id = _wandb_gate_run_id()
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        id=run_id,
        name=run_id,
        resume="allow",
        job_type="infra-gate",
        tags=_wandb_gate_tags(),
        config={
            "schema": WANDB_WRITE_GATE_SCHEMA,
            "source_tree_sha256": SOURCE_TREE_SHA256,
        },
        settings=wandb.Settings(
            api_key=_wandb_api_key(),
            disable_code=True,
            disable_git=True,
            x_disable_stats=True,
            x_disable_machine_info=True,
            console="off",
            silent=True,
        ),
    )
    if run is None:
        raise RuntimeError("W&B write-gate initialization returned no run")
    summary = _wandb_gate_summary()
    try:
        run.log({"infra_gate/write_marker": 1}, step=0)
        for key, value in summary.items():
            run.summary[key] = value
    finally:
        run.finish(exit_code=0)
    remote_evidence: dict[str, Any] | None = None
    last_error: Exception | None = None
    for _ in range(6):
        try:
            remote_evidence = _authenticate_wandb_remote_run(
                _wandb_api(),
                run_id=run_id,
            )
            break
        except Exception as exc:
            last_error = exc
            time.sleep(2.0)
    if remote_evidence is None:
        raise RuntimeError("W&B write-gate read-after-write failed") from last_error
    marker = _self_hash_record(
        {
            "schema": WANDB_WRITE_GATE_SCHEMA,
            "decision": "pass",
            "experiment_version": EXPERIMENT_VERSION,
            "source_tree_sha256": SOURCE_TREE_SHA256,
            "entity": WANDB_ENTITY,
            "project": WANDB_PROJECT,
            "run_id": run_id,
            "remote_evidence": remote_evidence,
            "runtime_identity": _modal_runtime_identity(),
            "created_at": _utc_now(),
        },
        hash_field="gate_sha256",
    )
    won = bool(store.put(WANDB_WRITE_GATE_KEY, marker, skip_if_exists=True))
    observed = store.get(WANDB_WRITE_GATE_KEY, None)
    if won and observed != marker:
        raise RuntimeError("W&B write-gate CAS winner differs from its write")
    return _validate_wandb_write_gate(store)


def _gate_marker_sha256(path: Path, *, label: str) -> str:
    marker = _load_json(path)
    value = marker.get("gate_sha256")
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RuntimeError(f"{label} has no authenticated gate SHA-256")
    return value


def _launch_identity(experiment: Experiment) -> dict[str, Any]:
    """Build every immutable input to one production FunctionCall."""

    manifest_set = _load_json(MANIFEST_SET_PATH)
    stage_manifests: dict[str, str] = {}
    stage_specs: dict[str, Any] = {}
    output_roots: dict[str, str] = {}
    wandb_runs: dict[str, Any] = {}
    for stage in experiment.stages:
        _, manifest = _manifest(_stage_spec(experiment, stage)["manifest"])
        stage_manifests[stage] = str(manifest["metadata_sha256"])
        stage_specs[stage] = _stage_spec(experiment, stage)
        output_roots[stage] = str(
            _stage_output(experiment, stage, canary=False)
        )
        wandb_runs[stage] = {
            "name": _run_name(experiment, stage, canary=False),
            "id": f"{EXPERIMENT_VERSION}-{experiment.key}-{stage}",
            "job_type": stage,
        }
    gate_hashes: dict[str, str | None] = {
        "precision_resume": _gate_marker_sha256(
            PRECISION_RESUME_GATE_PATH,
            label="precision-resume gate",
        ),
        "experiment_canary": _gate_marker_sha256(
            _gate_path(experiment),
            label=f"{experiment.key} canary gate",
        ),
        "staged_sft_resume": None,
    }
    if experiment.kind == "staged":
        gate_hashes["staged_sft_resume"] = _gate_marker_sha256(
            STAGED_SFT_RESUME_GATE_PATH,
            label="staged-SFT resume gate",
        )
    runtime_identity = _modal_runtime_identity()
    return {
        "schema": "context2048-production-launch-identity-v1",
        "experiment_version": EXPERIMENT_VERSION,
        "experiment": {
            "key": experiment.key,
            "description": experiment.description,
            "kind": experiment.kind,
            "pt_vocab_size": experiment.pt_vocab_size,
            "sft_copies": experiment.sft_copies,
        },
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "manifest_set_hash": manifest_set["set_hash"],
        "source_manifest_hash": manifest_set["source_manifest_hash"],
        "selection_hash": manifest_set["selection_hash"],
        "sft_cache_hash": manifest_set["sft_cache_hash"],
        "stage_manifest_sha256": stage_manifests,
        "gate_sha256": gate_hashes,
        "output_roots": output_roots,
        "wandb": {
            "entity": WANDB_ENTITY,
            "project": WANDB_PROJECT,
            "group": experiment.key,
            "runs": wandb_runs,
        },
        "runtime_identity": runtime_identity,
        "spec": {
            "context_length": CONTEXT_LENGTH,
            "world_size": WORLD_SIZE,
            "precision_contract": EXPECTED_PRECISION_CONTRACT,
            "determinism_contract": EXPECTED_DETERMINISM_CONTRACT,
            "stages": stage_specs,
        },
    }


def _authenticate_existing_completion(
    experiment: Experiment,
) -> dict[str, Any] | None:
    """Return complete evidence, or refuse any pre-claim partial output."""

    recovery = _authenticate_recovery_output(experiment)
    if recovery["state"] == "complete":
        return dict(recovery["completion"])
    if recovery["has_output"]:
        raise RuntimeError(
            f"{experiment.key} has partial output without a terminal failed "
            "attempt; refusing an unsafe initial launch"
        )
    return None


def _authenticate_recovery_output(experiment: Experiment) -> dict[str, Any]:
    """Classify only absent/empty or cryptographically valid resumable output."""

    stage_evidence: dict[str, Any] = {}
    parent: Path | None = None
    has_output = False
    found_incomplete = False
    for stage in experiment.stages:
        output_dir = _stage_output(experiment, stage, canary=False)
        if not output_dir.exists() or (
            output_dir.is_dir() and not any(output_dir.iterdir())
        ):
            has_output = has_output or output_dir.exists()
            found_incomplete = True
            stage_evidence[stage] = {
                "state": "empty" if output_dir.exists() else "absent",
                "output_root": str(output_dir),
            }
            continue
        if not output_dir.is_dir():
            raise RuntimeError(
                f"{experiment.key}/{stage} output root is not a directory"
            )
        if found_incomplete:
            raise RuntimeError(
                f"{experiment.key}/{stage} exists after an incomplete prior stage"
            )
        has_output = True
        if stage == "sft" and parent is None:
            raise RuntimeError(
                f"{experiment.key}/sft exists without an authenticated PT parent"
            )
        contract = _stage_launch_contract(
            experiment,
            stage,
            canary=False,
            weights_only=parent if stage == "sft" else None,
        )
        complete, resume = _authenticate_stage_output_root(
            output_dir,
            experiment=experiment,
            stage=stage,
            manifest=contract["manifest"],
            expected_step=int(contract["expected_step"]),
            initialization_identity=contract["initialization_identity"],
            initial_launch_command=contract["initial_launch_command"],
        )
        if complete:
            if resume is not None:
                raise RuntimeError(
                    f"{experiment.key}/{stage} is both complete and resumable"
                )
            final = output_dir / "final"
            export = validate_completed_hf_export(final)
            stage_evidence[stage] = {
                "state": "complete",
                "final": str(final),
                "global_step": int(export["state"]["global_step"]),
                "export_marker_sha256": export["marker"]["marker_sha256"],
                "trainer_state_sha256": export["marker"]["trainer_state_sha256"],
            }
            parent = final
            continue
        if resume is None:
            raise RuntimeError(
                f"{experiment.key}/{stage} has no authenticated resume checkpoint"
            )
        resume_state = _load_json(resume / "trainer_state.json")
        stage_evidence[stage] = {
            "state": "resumable",
            "checkpoint": str(resume),
            "global_step": int(resume_state["global_step"]),
            "trainer_state_sha256": _sha256_file(resume / "trainer_state.json"),
        }
        found_incomplete = True

    if not found_incomplete and parent is not None:
        completion = {
            "schema": "context2048-authenticated-production-completion-v1",
            "experiment_version": EXPERIMENT_VERSION,
            "experiment": experiment.key,
            "source_tree_sha256": SOURCE_TREE_SHA256,
            "stages": stage_evidence,
            "final": str(parent),
        }
        return {
            "state": "complete",
            "has_output": has_output,
            "stages": stage_evidence,
            "completion": completion,
        }
    return {
        "state": "resumable",
        "has_output": has_output,
        "stages": stage_evidence,
        "completion": None,
    }


def _validate_launch_atomicity_gate(store: Any) -> dict[str, Any]:
    marker = _validate_self_hash(
        store.get(LAUNCH_ATOMICITY_GATE_KEY, None),
        hash_field="gate_sha256",
        label="production launch-claim atomicity gate",
    )
    expected = {
        "schema": LAUNCH_ATOMICITY_GATE_SCHEMA,
        "decision": "pass",
        "experiment_version": EXPERIMENT_VERSION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "dict_name": LAUNCH_CLAIM_DICT_NAME,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise RuntimeError(f"launch-claim atomicity gate {key} drifted")
    if int(marker.get("contender_count", 0)) < 8:
        raise RuntimeError("launch-claim atomicity gate used too few contenders")
    if int(marker.get("winner_count", -1)) != 1:
        raise RuntimeError("launch-claim atomicity gate did not have one winner")
    _validate_recorded_runtime_identity(marker.get("runtime_identity"))
    return marker


def _finalize_launch_atomicity_gate(
    store: Any,
    *,
    nonce: str,
    results: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{32}", nonce):
        raise RuntimeError("invalid atomicity-gate nonce")
    if len(results) < 8:
        raise RuntimeError("atomicity gate requires at least eight contenders")
    probe_key = f"atomicity-probe:{SOURCE_TREE_SHA256}:{nonce}"
    winner = store.get(probe_key, None)
    won = [result for result in results if result.get("won") is True]
    if len(won) != 1 or won[0].get("proposed") != winner:
        raise RuntimeError("Modal Dict atomicity probe did not produce one winner")
    if any(result.get("observed") != winner for result in results):
        raise RuntimeError("Modal Dict contenders observed different winners")
    marker = _self_hash_record(
        {
            "schema": LAUNCH_ATOMICITY_GATE_SCHEMA,
            "decision": "pass",
            "experiment_version": EXPERIMENT_VERSION,
            "source_tree_sha256": SOURCE_TREE_SHA256,
            "dict_name": LAUNCH_CLAIM_DICT_NAME,
            "nonce": nonce,
            "contender_count": len(results),
            "winner_count": 1,
            "winner": winner,
            "runtime_identity": _modal_runtime_identity(),
            "created_at": _utc_now(),
        },
        hash_field="gate_sha256",
    )
    added = bool(
        store.put(LAUNCH_ATOMICITY_GATE_KEY, marker, skip_if_exists=True)
    )
    observed = store.get(LAUNCH_ATOMICITY_GATE_KEY, None)
    if not added and observed != marker:
        # A gate is immutable for one source tree.  Never replace or weaken it.
        return _validate_launch_atomicity_gate(store)
    return _validate_launch_atomicity_gate(store)


def _production_launch_preflight(
    store: Any,
    *,
    experiment_key: str,
) -> tuple[Experiment, dict[str, Any], dict[str, Any]]:
    """Authenticate every shared production input and both remote gates."""

    if experiment_key not in EXPERIMENTS:
        raise ValueError(f"unknown experiment: {experiment_key!r}")
    data_volume.reload()
    checkpoint_volume.reload()
    _validate_source_artifacts()
    experiment = EXPERIMENTS[experiment_key]
    _validate_gate(experiment)
    atomicity_gate = _validate_launch_atomicity_gate(store)
    wandb_gate = _validate_wandb_write_gate(store)
    launch_identity = _launch_identity(experiment)
    gate_evidence = {
        "launch_atomicity_gate_sha256": atomicity_gate["gate_sha256"],
        "wandb_write_gate_sha256": wandb_gate["gate_sha256"],
    }
    return experiment, launch_identity, gate_evidence


def _exception_type_name(exc: BaseException) -> str:
    exception_type = type(exc)
    return f"{exception_type.__module__}.{exception_type.__qualname__}"


def _authoritative_terminal_call_evidence(
    get_result: Callable[..., Any],
    *,
    function_call_id: str,
    allow_success: bool = False,
) -> dict[str, Any]:
    """Poll one retained result; ambiguous client states always fail closed."""

    if not isinstance(function_call_id, str) or not re.fullmatch(
        r"fc-[0-9A-Za-z]+", function_call_id
    ):
        raise RuntimeError(f"invalid Modal FunctionCall ID: {function_call_id!r}")
    result_category: str
    exception_type: str | None
    try:
        get_result(timeout=0)
    except modal.exception.OutputExpiredError as exc:
        raise RuntimeError(
            f"Modal FunctionCall {function_call_id} output expired; recovery "
            "cannot authenticate its terminal result"
        ) from exc
    except modal.exception.FunctionTimeoutError as exc:
        result_category = "function_timeout"
        exception_type = _exception_type_name(exc)
    except builtins.TimeoutError as exc:
        raise RuntimeError(
            f"Modal FunctionCall {function_call_id} is still pending"
        ) from exc
    except modal.exception.RemoteError as exc:
        # A retained non-success output without a serialized application
        # exception, including an explicitly terminated input, maps here.
        result_category = "remote_terminal_failure"
        exception_type = _exception_type_name(exc)
    except modal.exception.Error as exc:
        raise RuntimeError(
            f"Modal FunctionCall {function_call_id} terminal state is "
            f"ambiguous ({_exception_type_name(exc)})"
        ) from exc
    except RuntimeError as exc:
        if exc.args not in {
            (PRODUCTION_TRAINING_TERMINAL_MARKER,),
            (PRODUCTION_DISPATCHER_TERMINAL_MARKER,),
        }:
            raise RuntimeError(
                f"Modal FunctionCall {function_call_id} returned an unknown "
                f"exception type ({_exception_type_name(exc)})"
            ) from exc
        result_category = "application_failure"
        exception_type = _exception_type_name(exc)
    except Exception as exc:
        # Production entrypoints wrap their application failures in the two
        # stable exceptions above. Any other deserialized exception indicates
        # stale code or an unclassified failure and cannot authorize recovery.
        raise RuntimeError(
            f"Modal FunctionCall {function_call_id} returned an unknown "
            f"exception type ({_exception_type_name(exc)})"
        ) from exc
    else:
        if not allow_success:
            raise RuntimeError(
                f"Modal FunctionCall {function_call_id} completed successfully; "
                "recovery is forbidden"
            )
        result_category = "success"
        exception_type = None
    core = {
        "schema": TERMINAL_CALL_EVIDENCE_SCHEMA,
        "function_call_id": function_call_id,
        "result_category": result_category,
        "exception_type": exception_type,
    }
    return _self_hash_record(core, hash_field="evidence_sha256")


def _inspect_terminal_unsuccessful_function_call(
    function_call_id: str,
) -> dict[str, Any]:
    if not isinstance(function_call_id, str) or not re.fullmatch(
        r"fc-[0-9A-Za-z]+", function_call_id
    ):
        raise RuntimeError(f"invalid Modal FunctionCall ID: {function_call_id!r}")
    function_call = modal.FunctionCall.from_id(function_call_id)
    return _authoritative_terminal_call_evidence(
        function_call.get,
        function_call_id=function_call_id,
    )


def _bind_spawned_worker(
    store: Any,
    *,
    experiment_key: str,
    launch_token: str,
    launch_identity: Mapping[str, Any],
    generation: int,
    function_call_id: str,
) -> dict[str, Any]:
    """Bind immediately after spawn; the worker repeats this same CAS."""

    return _begin_claimed_worker(
        store,
        experiment_key=experiment_key,
        launch_token=launch_token,
        expected_identity=launch_identity,
        generation=generation,
        function_call_id=function_call_id,
    )


def _dispatch_production_launch(
    store: Any,
    *,
    experiment_key: str,
    launch_token: str,
    recovery: bool,
    dispatcher_function_call_id: str,
    spawn_worker: Callable[[str, str, int], Any],
) -> dict[str, Any]:
    """Claim, recover, and spawn server-side so acknowledgement loss is safe."""

    _require_launch_token(launch_token)
    experiment, launch_identity, gate_evidence = _production_launch_preflight(
        store,
        experiment_key=experiment_key,
    )
    if not recovery:
        completion = _authenticate_existing_completion(experiment)
        if completion is not None:
            return {
                "experiment": experiment_key,
                "outcome": "authenticated_completion",
                "spawned": False,
                "completion": completion,
            }
        output_evidence = _authenticate_recovery_output(experiment)
        if output_evidence["has_output"]:
            raise RuntimeError(
                f"{experiment_key} has authenticated partial output but no "
                "launch claim; use recovery and the original token"
            )
    recorded_claim = store.get(_launch_claim_key(experiment_key), None)
    anchor_path = _durable_launch_anchor_path(experiment_key)
    if recorded_claim is None and anchor_path.exists():
        raise RuntimeError(
            f"{experiment_key} durable launch anchor exists but its Modal "
            "Dict claim expired or is missing; refusing a fresh claim"
        )
    if recorded_claim is None:
        acquired = _acquire_launch_claim(
            store,
            experiment_key=experiment_key,
            launch_token=launch_token,
            launch_identity=launch_identity,
        )
        if acquired["outcome"] != "acquired":
            return {
                "experiment": experiment_key,
                "outcome": "claim_lost",
                "spawned": False,
            }
        claim = acquired["claim"]
    else:
        claim = _validate_launch_claim(
            recorded_claim,
            experiment_key=experiment_key,
            expected_identity=launch_identity,
            launch_token=launch_token,
        )
    _ensure_durable_launch_anchor(
        store,
        experiment_key=experiment_key,
        claim=claim,
        launch_identity=launch_identity,
        launch_token=launch_token,
        dispatcher_function_call_id=dispatcher_function_call_id,
        recovery=recovery,
    )
    if recorded_claim is not None and not recovery:
        replay_attempt = _current_launch_attempt(
            store,
            experiment_key=experiment_key,
            claim=claim,
        )
        if replay_attempt is not None and str(
            replay_attempt["dispatcher_function_call_id"]
        ) == dispatcher_function_call_id:
            replay_generation = int(replay_attempt["generation"])
            replay_resolution_record = store.get(
                _launch_execution_key(experiment_key, replay_generation),
                None,
            )
            if replay_resolution_record is None:
                raise RuntimeError(
                    f"{experiment_key} dispatcher replay found its unbound "
                    f"generation {replay_generation}; recovery must first "
                    "authenticate this dispatcher as terminal unsuccessful"
                )
            replay_resolution = _validate_generation_resolution(
                replay_resolution_record,
                experiment_key=experiment_key,
                claim=claim,
                attempt=replay_attempt,
                generation=replay_generation,
            )
            if replay_resolution["kind"] == "worker":
                return {
                    "experiment": experiment_key,
                    "outcome": "already_spawned_by_same_dispatcher",
                    "spawned": False,
                    "generation": replay_generation,
                    "function_call_id": replay_resolution["function_call_id"],
                }
        if replay_attempt is not None:
            return {
                "experiment": experiment_key,
                "outcome": "existing_claim",
                "spawned": False,
                "claim_sha256": claim["claim_sha256"],
            }

    current = _current_launch_attempt(
        store,
        experiment_key=experiment_key,
        claim=claim,
    )
    recovery_evidence: Mapping[str, Any] | None = None
    if current is None:
        generation = 0
        if recovery:
            output_evidence = _authenticate_recovery_output(experiment)
            if output_evidence["state"] == "complete":
                return {
                    "experiment": experiment_key,
                    "outcome": "authenticated_completion",
                    "spawned": False,
                    "completion": output_evidence["completion"],
                }
    else:
        if not recovery:
            return {
                "experiment": experiment_key,
                "outcome": "attempt_exists",
                "spawned": False,
                "generation": int(current["generation"]),
            }
        current_generation = int(current["generation"])
        resolution_key = _launch_execution_key(
            experiment_key,
            current_generation,
        )
        resolution_record = store.get(resolution_key, None)
        if resolution_record is None:
            # The dispatcher may have spawned a worker immediately before it
            # died.  Closing this same immutable key is the linearization
            # point: either that worker binds, or recovery closes the
            # generation, but both can never become authorized.
            dispatcher_call_id = str(current["dispatcher_function_call_id"])
            dispatcher_terminal = _inspect_terminal_unsuccessful_function_call(
                dispatcher_call_id
            )
            checkpoint_volume.reload()
            dispatcher_checkpoint = _authenticate_recovery_output(experiment)
            proposed_closure = _recovery_closure(
                experiment_key=experiment_key,
                claim=claim,
                attempt=current,
                generation=current_generation,
                dispatcher_function_call_id=dispatcher_call_id,
                terminal_call=dispatcher_terminal,
                checkpoint_state=dispatcher_checkpoint,
            )
            won_closure = bool(
                store.put(
                    resolution_key,
                    proposed_closure,
                    skip_if_exists=True,
                )
            )
            resolution_record = store.get(resolution_key, None)
            if won_closure and resolution_record != proposed_closure:
                raise RuntimeError(
                    f"{experiment_key} recovery-closure CAS winner drifted"
                )

        resolution = _validate_generation_resolution(
            resolution_record,
            experiment_key=experiment_key,
            claim=claim,
            attempt=current,
            generation=current_generation,
        )
        prior_call_id = str(resolution["function_call_id"])
        if resolution["kind"] == "worker":
            # A worker won the resolution CAS.  It may be training, so only
            # its own retained terminal-unsuccessful graph permits recovery.
            terminal_evidence = _inspect_terminal_unsuccessful_function_call(
                prior_call_id
            )
            checkpoint_volume.reload()
            output_evidence = _authenticate_recovery_output(experiment)
        else:
            # No worker can bind after this immutable close.  Re-authenticate
            # the volume and require it to match the state sealed at close.
            terminal_evidence = dict(resolution["terminal_call"])
            output_evidence = dict(resolution["checkpoint_state"])
            checkpoint_volume.reload()
            current_output = _authenticate_recovery_output(experiment)
            if current_output != output_evidence:
                raise RuntimeError(
                    f"{experiment_key} output changed after generation "
                    f"{current_generation} was closed for recovery"
                )
        if output_evidence["state"] == "complete":
            return {
                "experiment": experiment_key,
                "outcome": "authenticated_completion",
                "spawned": False,
                "completion": output_evidence["completion"],
                "closed_generation": current_generation,
                "resolution_sha256": resolution["execution_sha256"],
            }
        generation = current_generation + 1
        recovery_core = {
            "schema": "context2048-launch-recovery-evidence-v1",
            "prior_generation": current_generation,
            "prior_attempt_sha256": current["attempt_sha256"],
            "prior_execution_sha256": resolution["execution_sha256"],
            "prior_call_id": prior_call_id,
            "terminal_call": terminal_evidence,
            "checkpoint_state": output_evidence,
        }
        recovery_evidence = _self_hash_record(
            recovery_core,
            hash_field="recovery_sha256",
        )

    acquired_attempt = _acquire_launch_attempt(
        store,
        experiment_key=experiment_key,
        claim=claim,
        generation=generation,
        dispatcher_function_call_id=dispatcher_function_call_id,
        recovery_evidence=recovery_evidence,
    )
    if acquired_attempt["outcome"] != "attempt_acquired":
        return {
            "experiment": experiment_key,
            "outcome": "attempt_exists",
            "spawned": False,
            "generation": generation,
        }
    call = spawn_worker(experiment_key, launch_token, generation)
    function_call_id = str(getattr(call, "object_id", "") or "")
    _bind_spawned_worker(
        store,
        experiment_key=experiment_key,
        launch_token=launch_token,
        launch_identity=launch_identity,
        generation=generation,
        function_call_id=function_call_id,
    )
    return {
        "experiment": experiment_key,
        "outcome": "recovery_spawned" if generation else "spawned",
        "spawned": True,
        "generation": generation,
        "function_call_id": function_call_id,
        "claim_sha256": claim["claim_sha256"],
        "attempt_sha256": acquired_attempt["attempt"]["attempt_sha256"],
        "gate_evidence": gate_evidence,
        "wandb_project": WANDB_PROJECT,
        "wandb_group": experiment_key,
    }


def _run_experiment(
    experiment: Experiment,
    *,
    canary: bool,
    heartbeat: Callable[[str, str], None] | None = None,
) -> str:
    data_volume.reload()
    checkpoint_volume.reload()
    _validate_source_artifacts()
    if canary:
        if experiment.kind == "staged":
            pt_final = _run_stage(experiment, "pt", canary=True)
            _run_stage(
                experiment,
                "sft",
                canary=True,
                weights_only=pt_final,
            )
        else:
            _run_stage(experiment, "mixed", canary=True)
        _write_gate(experiment)
        return json.dumps({"experiment": experiment.key, "canary": "pass"})

    _validate_gate(experiment)
    if experiment.kind == "staged":
        if heartbeat is not None:
            heartbeat("pt", "starting")
        pt_final = _run_stage(
            experiment,
            "pt",
            canary=False,
            heartbeat=heartbeat,
        )
        if heartbeat is not None:
            heartbeat("pt", "complete")
            heartbeat("sft", "starting")
        final = _run_stage(
            experiment,
            "sft",
            canary=False,
            weights_only=pt_final,
            heartbeat=heartbeat,
        )
        if heartbeat is not None:
            heartbeat("sft", "complete")
    else:
        if heartbeat is not None:
            heartbeat("mixed", "starting")
        final = _run_stage(
            experiment,
            "mixed",
            canary=False,
            heartbeat=heartbeat,
        )
        if heartbeat is not None:
            heartbeat("mixed", "complete")
    return str(final)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}",
    cpu=32.0,
    memory=128 * 1024,
    timeout=60 * 60 * 2,
    retries=0,
    max_containers=4,
)
def run_canary(experiment_key: str) -> str:
    return _run_experiment(EXPERIMENTS[experiment_key], canary=True)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}",
    cpu=32.0,
    memory=128 * 1024,
    timeout=60 * 60 * 2,
    retries=0,
    max_containers=1,
)
def run_precision_resume_canary() -> str:
    data_volume.reload()
    checkpoint_volume.reload()
    _validate_source_artifacts()
    precision_marker = _write_precision_resume_gate()
    staged_sft_marker = _write_staged_sft_resume_gate()
    _validate_precision_resume_gate()
    _validate_staged_sft_resume_gate()
    return json.dumps(
        {
            "precision_resume": precision_marker,
            "staged_sft_resume": staged_sft_marker,
        },
        indent=2,
        sort_keys=True,
    )


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}",
    cpu=32.0,
    memory=128 * 1024,
    timeout=PRODUCTION_FUNCTION_TIMEOUT_SECONDS,
    retries=modal.Retries(initial_delay=10.0, max_retries=3),
    single_use_containers=True,
    max_containers=4,
)
@_wrap_deployed_terminal_failure(PRODUCTION_TRAINING_TERMINAL_MARKER)
def run_experiment(
    experiment_key: str,
    launch_token: str,
    generation: int,
) -> str:
    """Run only work authorized by the exact immutable CPU launch claim."""

    if experiment_key not in EXPERIMENTS:
        raise ValueError(f"unknown experiment: {experiment_key!r}")
    experiment = EXPERIMENTS[experiment_key]
    # The worker can remain queued longer than Modal Dict's entry retention.
    # Reload the durable pre-spawn anchor before trusting the volatile lease.
    checkpoint_volume.reload()
    function_call_id = modal.current_function_call_id()
    launch_identity = _launch_identity(experiment)
    _begin_claimed_worker(
        launch_claims,
        experiment_key=experiment_key,
        launch_token=launch_token,
        expected_identity=launch_identity,
        generation=generation,
        function_call_id=function_call_id,
    )

    def heartbeat(stage: str, detail: str) -> None:
        _update_launch_status(
            launch_claims,
            experiment_key=experiment_key,
            launch_token=launch_token,
            expected_identity=launch_identity,
            generation=generation,
            function_call_id=function_call_id,
            state="running",
            stage=stage,
            detail=detail,
        )

    heartbeat(experiment.stages[0], "worker-authenticated")
    try:
        final = Path(
            _run_experiment(
                experiment,
                canary=False,
                heartbeat=heartbeat,
            )
        )
        # _run_stage validates before committing. Reload and recompute immutable
        # completion evidence only after that final commit is remotely visible.
        checkpoint_volume.reload()
        completion = _authenticate_existing_completion(experiment)
        if completion is None or completion.get("final") != str(final):
            raise RuntimeError(
                f"{experiment_key} final completion did not survive volume commit"
            )
        _update_launch_status(
            launch_claims,
            experiment_key=experiment_key,
            launch_token=launch_token,
            expected_identity=launch_identity,
            generation=generation,
            function_call_id=function_call_id,
            state="complete",
            stage=experiment.stages[-1],
            detail="validated-after-volume-commit",
            completion=completion,
        )
        return str(final)
    except BaseException as exc:
        _update_launch_status(
            launch_claims,
            experiment_key=experiment_key,
            launch_token=launch_token,
            expected_identity=launch_identity,
            generation=generation,
            function_call_id=function_call_id,
            state="failed",
            stage=None,
            detail=f"{type(exc).__name__}: {exc}",
        )
        raise


@app.function(
    cpu=1.0,
    memory=1024,
    timeout=15 * 60,
    retries=0,
    max_containers=4,
)
@_wrap_deployed_terminal_failure(PRODUCTION_DISPATCHER_TERMINAL_MARKER)
def dispatch_launch(
    experiment_key: str,
    launch_token: str,
    recovery: bool = False,
) -> dict[str, Any]:
    """Perform CPU preflight, generation CAS, and GPU spawn server-side."""

    return _dispatch_production_launch(
        launch_claims,
        experiment_key=experiment_key,
        launch_token=launch_token,
        recovery=bool(recovery),
        dispatcher_function_call_id=modal.current_function_call_id(),
        spawn_worker=lambda key, token, generation: run_experiment.spawn(
            key,
            token,
            generation,
        ),
    )


@app.function(cpu=1.0, memory=1024, timeout=10 * 60, max_containers=1)
def wandb_write_gate() -> dict[str, Any]:
    """Publish and authenticate the source-scoped W&B remote-write gate."""

    return _publish_wandb_write_gate(launch_claims)


@app.function(cpu=0.25, memory=256, timeout=5 * 60, max_containers=32)
def launch_claim_atomicity_contender(nonce: str, contender: int) -> dict[str, Any]:
    """One deployed CPU contender for the Modal Dict CAS gate."""

    if not re.fullmatch(r"[0-9a-f]{32}", nonce):
        raise RuntimeError("invalid atomicity-gate nonce")
    if isinstance(contender, bool) or not 0 <= int(contender) < 10_000:
        raise RuntimeError("invalid atomicity-gate contender")
    key = f"atomicity-probe:{SOURCE_TREE_SHA256}:{nonce}"
    proposed = {
        "nonce": nonce,
        "contender": int(contender),
        "function_call_id": modal.current_function_call_id(),
    }
    won = bool(launch_claims.put(key, proposed, skip_if_exists=True))
    return {"won": won, "proposed": proposed, "observed": launch_claims[key]}


@app.function(cpu=1.0, memory=1024, timeout=5 * 60, max_containers=1)
def finalize_launch_claim_atomicity_gate(
    nonce: str,
    results: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Authenticate and publish the immutable real-Modal atomicity gate."""

    return _finalize_launch_atomicity_gate(
        launch_claims,
        nonce=nonce,
        results=results,
    )


@app.function(cpu=1.0, memory=1024, timeout=5 * 60)
def read_status(experiment_keys: list[str]) -> str:
    """Read run state inside the deployed app with its mounted volume."""

    checkpoint_volume.reload()
    unknown = sorted(set(experiment_keys) - set(EXPERIMENTS))
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}")
    result: list[dict[str, Any]] = []
    for key in experiment_keys:
        experiment_spec = EXPERIMENTS[key]
        rows = []
        for stage in experiment_spec.stages:
            output = _stage_output(experiment_spec, stage, canary=False)
            state_path = output / "final" / "interleaved_training_state.json"
            latest_path = None
            if output.is_dir() and (output / LATEST_CHECKPOINT_POINTER).is_file():
                latest_path = resolve_resume_checkpoint(output) / "trainer_state.json"
            state = (
                _load_json(state_path)
                if state_path.is_file()
                else _load_json(latest_path)
                if latest_path is not None and latest_path.is_file()
                else {}
            )
            rows.append(
                {
                    "stage": stage,
                    "step": state.get("global_step"),
                    "target_steps": _stage_spec(experiment_spec, stage)["steps"],
                    "complete": state_path.is_file(),
                }
            )
        launch_records: dict[str, Any] = {}
        for name, dict_key, hash_field in (
            ("claim", _launch_claim_key(key), "claim_sha256"),
            ("completion", _launch_completion_key(key), "status_sha256"),
        ):
            value = launch_claims.get(dict_key, None)
            if value is not None:
                launch_records[name] = _validate_self_hash(
                    value,
                    hash_field=hash_field,
                    label=f"{key} launch {name}",
                )
        claim = launch_records.get("claim")
        attempts: list[dict[str, Any]] = []
        if claim is not None:
            current = _current_launch_attempt(
                launch_claims,
                experiment_key=key,
                claim=claim,
            )
            if current is not None:
                for generation in range(int(current["generation"]) + 1):
                    attempt = _validate_launch_attempt(
                        launch_claims.get(
                            _launch_attempt_key(key, generation),
                            None,
                        ),
                        experiment_key=key,
                        generation=generation,
                        claim=claim,
                    )
                    entry: dict[str, Any] = {"attempt": attempt}
                    execution = launch_claims.get(
                        _launch_execution_key(key, generation),
                        None,
                    )
                    if execution is not None:
                        entry["execution"] = _validate_generation_resolution(
                            execution,
                            experiment_key=key,
                            claim=claim,
                            attempt=attempt,
                            generation=generation,
                        )
                    status = launch_claims.get(
                        _launch_status_key(key, generation),
                        None,
                    )
                    if status is not None:
                        entry["status"] = _validate_self_hash(
                            status,
                            hash_field="status_sha256",
                            label=f"{key} generation {generation} status",
                        )
                    attempts.append(entry)
        launch_records["attempts"] = attempts
        result.append(
            {"experiment": key, "stages": rows, "launch": launch_records}
        )
    return json.dumps(result, indent=2, sort_keys=True)


DEPLOYED_FUNCTION_NAMES = frozenset(
    {
        "deployment_identity",
        "dispatch_launch",
        "finalize_launch_claim_atomicity_gate",
        "launch_claim_atomicity_contender",
        "prepare_data",
        "read_status",
        "run_canary",
        "run_experiment",
        "run_precision_resume_canary",
        "wandb_write_gate",
    }
)


def _deployment_command() -> list[str]:
    return ["modal", "deploy", str(Path(__file__).resolve())]


def _deployed_function(name: str):
    """Resolve a named function only from the persistent deployment."""

    if name not in DEPLOYED_FUNCTION_NAMES:
        raise ValueError(f"unknown deployed function: {name}")
    return modal.Function.from_name(APP_NAME, name)


def _require_matching_deployment() -> dict[str, Any]:
    """Fail before work if the stable deployment differs from local source."""

    try:
        identity = _deployed_function("deployment_identity").remote()
    except Exception as exc:
        raise RuntimeError(
            "No matching persistent Modal deployment is available. Run "
            f"{' '.join(_deployment_command())} once, then retry this action."
        ) from exc
    if not isinstance(identity, Mapping):
        raise RuntimeError("deployed identity response is not an object")
    expected = {
        "schema": "context2048-modal-deployment-identity-v1",
        "experiment_version": EXPERIMENT_VERSION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise RuntimeError(
                "persistent Modal deployment does not match the local launch "
                f"source for {key}: {identity.get(key)!r} != {value!r}; "
                f"redeploy with {' '.join(_deployment_command())}"
            )
    runtime = identity.get("runtime_identity")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("persistent deployment lacks runtime identity")
    runtime_expected = {
        "modal_app_name": APP_NAME,
        "modal_base_image": CUDA_BASE_IMAGE,
        "runtime_package_versions": PINNED_RUNTIME_PACKAGE_VERSIONS,
    }
    for key, value in runtime_expected.items():
        if runtime.get(key) != value:
            raise RuntimeError(
                f"persistent deployment runtime identity {key} drifted"
            )
    if not str(runtime.get("modal_app_id", "")).startswith("ap-"):
        raise RuntimeError("persistent deployment has no authenticated app ID")
    if not str(runtime.get("modal_image_id", "")).startswith("im-"):
        raise RuntimeError("persistent deployment has no authenticated image ID")
    if int(runtime.get("runtime_distribution_count", 0)) <= 0 or not re.fullmatch(
        r"[0-9a-f]{64}",
        str(runtime.get("runtime_distribution_inventory_sha256", "")),
    ):
        raise RuntimeError(
            "persistent deployment has no complete distribution inventory identity"
        )
    return dict(identity)


def _dry_run_payload() -> dict[str, Any]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "app": APP_NAME,
        "wandb": {"entity": WANDB_ENTITY, "project": WANDB_PROJECT},
        "wandb_secret": {
            "name": WANDB_SECRET,
            "required_sync_command": _wandb_secret_sync_command(),
            "dotenv_uploaded": False,
            "dotenv_hashed": False,
        },
        "runtime_identity": {
            "cuda_base_image": CUDA_BASE_IMAGE,
            "python_version": PYTHON_VERSION,
            "pip_packages": list(PINNED_PIP_PACKAGES),
            "modal_app_name": APP_NAME,
            "hydrated_modal_app_and_image_ids_recorded_at_runtime": True,
        },
        "deployment": {
            "required_command": _deployment_command(),
            "execution_mode": "persistent-deployment-function-lookup",
            "all_non_dry_run_actions_require_matching_source_tree": True,
            "gate_and_production_app_image_ids_must_match": True,
            "launch_claim_dict": LAUNCH_CLAIM_DICT_NAME,
            "launch_claim_gate_action": "launch-claim-gate",
            "wandb_remote_write_gate_action": "wandb-write-gate",
            "production_spawn_requires_atomic_immutable_claim": True,
            "production_spawn_uses_server_side_dispatcher": True,
            "recovery_uses_immutable_generations": True,
            "claims_are_never_stolen_by_age": True,
        },
        "shared": {
            "native_context": CONTEXT_LENGTH,
            "sequence_length": CONTEXT_LENGTH,
            "pt_target_tokens": PT_TARGET_TOKENS,
            "pt_records": PT_RECORDS,
            "pt_global_sequences": PT_GLOBAL_SEQUENCES,
            "pt_global_token_batch": PT_GLOBAL_TOKEN_BATCH,
            "pt_steps": PT_STEPS,
            "sft_rows": SFT_ROWS,
            "sft_max_aligned_length": SOURCE_SFT_MAX_ALIGNED_LENGTH,
            "sft_packing": "one-row-per-sequence-right-padded",
            "precision_contract": EXPECTED_PRECISION_CONTRACT,
            "determinism_contract": EXPECTED_DETERMINISM_CONTRACT,
            "required_gate": "precision-canary",
            "staged_sft_resume_gate_variants": list(
                STAGED_SFT_RESUME_VARIANTS
            ),
        },
        "experiments": {
            key: {
                "description": experiment.description,
                "stages": {
                    stage: _stage_spec(experiment, stage)
                    for stage in experiment.stages
                },
            }
            for key, experiment in EXPERIMENTS.items()
        },
        "source_tree_sha256": SOURCE_TREE_SHA256,
    }


@app.local_entrypoint()
def main(action: str = "dry-run", experiment: str = "") -> None:
    action = action.strip().lower()
    allowed_actions = {
        "dry-run",
        "prep",
        "launch-claim-gate",
        "wandb-write-gate",
        "precision-canary",
        "canary",
        "launch",
        "recover-launch",
        "status",
    }
    if action not in allowed_actions:
        raise ValueError(
            "action must be dry-run, prep, launch-claim-gate, precision-canary, "
            "wandb-write-gate, canary, launch, recover-launch, or status"
        )
    if action == "dry-run":
        print(json.dumps(_dry_run_payload(), indent=2, sort_keys=True))
        return
    _require_matching_deployment()
    if action in {
        "prep",
        "launch-claim-gate",
        "wandb-write-gate",
        "precision-canary",
        "canary",
        "launch",
        "recover-launch",
    }:
        _require_repo_wandb_api_key()
    if action == "prep":
        print(_deployed_function("prepare_data").remote())
        return
    if action == "precision-canary":
        print(_deployed_function("run_precision_resume_canary").remote())
        return
    if action == "launch-claim-gate":
        nonce = secrets.token_hex(16)
        contender = _deployed_function("launch_claim_atomicity_contender")
        calls = [contender.spawn(nonce, index) for index in range(16)]
        results = [call.get() for call in calls]
        marker = _deployed_function(
            "finalize_launch_claim_atomicity_gate"
        ).remote(nonce, results)
        print(json.dumps(marker, indent=2, sort_keys=True))
        return
    if action == "wandb-write-gate":
        marker = _deployed_function("wandb_write_gate").remote()
        print(json.dumps(marker, indent=2, sort_keys=True))
        return
    selected = (
        [item.strip() for item in experiment.split(",") if item.strip()]
        if experiment
        else list(EXPERIMENTS)
    )
    unknown = sorted(set(selected) - set(EXPERIMENTS))
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}")
    if action == "canary":
        deployed_canary = _deployed_function("run_canary")
        calls = {key: deployed_canary.spawn(key) for key in selected}
        for key, call in calls.items():
            print(json.dumps({"experiment": key, "result": call.get()}))
        return
    if action in {"launch", "recover-launch"}:
        deployed_dispatcher = _deployed_function("dispatch_launch")
        for key in selected:
            if action == "launch":
                launch_token = secrets.token_hex(32)
                recovery_path = _write_local_launch_recovery_record(
                    experiment_key=key,
                    launch_token=launch_token,
                )
                print(
                    json.dumps(
                        {
                            "experiment": key,
                            "launch_recovery_record": str(recovery_path),
                            "claim_started": False,
                        }
                    ),
                    flush=True,
                )
            else:
                recovery = _read_local_launch_recovery_record(key)
                launch_token = str(recovery["launch_token"])
            print(
                json.dumps(
                    deployed_dispatcher.remote(
                        key,
                        launch_token,
                        action == "recover-launch",
                    )
                )
            )
        return
    if action == "status":
        print(_deployed_function("read_status").remote(selected))
        return
    raise AssertionError(f"unhandled action: {action}")
