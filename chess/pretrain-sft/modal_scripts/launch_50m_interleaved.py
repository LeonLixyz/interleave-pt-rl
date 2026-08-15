"""Prepare and launch the 50M mixed-pretraining/SFT experiment on Modal.

This launcher deliberately handles only the pretraining side of the controlled
experiment. RL endpoints are converted to clean Hugging Face exports by the
Miles launcher and written under ``/checkpoints/interleave_50m/rl_hf``; a P2
run consumes one of those exports with ``--weights-only``.

Examples (none of these actions are performed merely by importing this file):

  # Validate/cache both datasets and build immutable P1/P2 manifests.
  modal run --detach modal_scripts/launch_50m_interleaved.py \
    --action data-prep

  # One tiny mixed-data update on the configurable canary topology.
  CHESS_INTERLEAVE_CANARY_GPU_TYPE=H100 \
  CHESS_INTERLEAVE_CANARY_GPUS=1 \
    modal run --detach modal_scripts/launch_50m_interleaved.py \
    --action canary --run-id canary1

  # Shared first leg and monolithic Experiment 2 pretraining.
  modal run --detach modal_scripts/launch_50m_interleaved.py --action p1
  modal run --detach modal_scripts/launch_50m_interleaved.py --action exp2

  # Fresh optimizer/scheduler P2 initialized only from model weights.
  modal run --detach modal_scripts/launch_50m_interleaved.py \
    --action p2 --run-id exp1-u-after-rl1500 \
    --init-checkpoint /checkpoints/interleave_50m/rl_hf/exp1-u-rl1500

Use ``--dry-run`` with any action to validate and print the launch plan without
submitting a Modal function.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import modal

from training.v2r2_gate import (
    CONFIRMATION_FIRST_SEED as V2R2_CONFIRMATION_FIRST_SEED,
    CONTRACT_PLAN_SHA256 as V2R2_CONTRACT_PLAN_SHA256,
    CONTRACT_SCHEMA as V2R2_CONTRACT_SCHEMA,
    CONTRACT_VERSION as V2R2_EXPERIMENT_VERSION,
    ELIGIBLE_SFT_LOSS_WEIGHTS as V2R2_ELIGIBLE_SFT_LOSS_WEIGHTS,
    PRIMARY_SEED as V2R2_PRIMARY_SEED,
    PROTOCOL_CANDIDATE_STEP as V2R2_PROTOCOL_CANDIDATE_STEP,
    audit_rollout_rows as _audit_v2r2_rollout_rows,
    canonical_json_sha256 as _v2r2_json_sha256,
    select_first_disjoint_confirmation as _select_v2r2_confirmation,
    select_first_disjoint_prompt_confirmation as _select_v2r2_prompt_confirmation,
    select_first_disjoint_prompt_set as _select_v2r2_prompt_set,
    self_hash_marker as _self_hash_v2r2_marker,
    validate_monolithic_protocol_gate as _validate_v2r2_monolithic_gate,
    validate_protocol_audit as _validate_v2r2_protocol_audit,
    validate_confirmation_against_prompt_selection as _validate_v2r2_selected_confirmation,
    validate_self_hashed_marker as _validate_v2r2_self_hashed_marker,
)
from training.v2r3_diagnostics import (
    CONTRACT_SCHEMA as V2R3_CONTRACT_SCHEMA,
    CONTRACT_VERSION as V2R3_EXPERIMENT_VERSION,
    PRIMARY_SEED as V2R3_PRIMARY_SEED,
    audit_diagnostic_rows as _audit_v2r3_diagnostic_rows,
)


# ---------------------------------------------------------------------------
# Immutable experiment/data identity
# ---------------------------------------------------------------------------

EXPERIMENT_VERSION = "mix10b_sft90k_3072_v2r1_weighted_clean_20260730"
DATA_ARTIFACT_VERSION = "mix10b_sft90k_v2r1_clean_verify_gate"
APP_NAME = "chess-50m-interleaved-pretrain-v2"

# The repository/directory kept its historical ``20b`` name, but this pinned
# revision is the enlarged 47,090-shard, 53.97B-token Modal corpus.
SOURCE_REPO = "chess-pre-to-post/pretrain_v1_20b"
SOURCE_REVISION = "07dd1b7090ca5f0fb05ef624c26b20bff19483c8"
SOURCE_DIR = Path("/data/pretrain_v1_20b")
SOURCE_FIRST_SHARD = 0
SOURCE_LAST_SHARD = 47_089
SOURCE_SHARDS = 47_090
SOURCE_TOKENS = 53_970_293_905
SOURCE_BYTES = 215_887_203_140
SOURCE_FLAT_MANIFEST_SHA256 = (
    "07ae91cded540a00e9b6554d1d54ed46310715b7fd68e3520a64b7f5967f99aa"
)
SOURCE_NPY_DTYPE = "<u4"
SOURCE_NPY_HEADER_BYTES = 128
SOURCE_HEADER_CHECK_SHARDS = (0, 24_774, 24_775, 47_089)

SFT_REPO = "Pre-to-Post-2/200M_SFT_dataset"
SFT_REVISION = "fd343bd28f6a40fc3dab4dcfb6e74c11b7a20b90"
SFT_JSON_FILES = 180
SFT_ROWS = 77_717
SFT_BYTES = 8_416_392_280
SFT_FLAT_MANIFEST_SHA256 = (
    "80f917f59d9f51b8fd14e8d4335c37e917a67c1990626656bbe106a7236f018e"
)
SFT_PROMPT_FIELD = "pgn"
SFT_COT_FIELD = "cot_by_method.trajectory_sep.cot_format_no_labels"
SFT_RESPONSE_NORMALIZATION = (
    "strip-numeric-verify-score-pairs-normalize-whitespace-v1"
)
SFT_STRICT_AUDIT_SCHEMA = "interleaved-sft-strict-audit-v1"
SFT_SUPERVISED_UNK_POLICY = "reject-supervised-unk-v1"
CLEAN_SFT_TARGETS_P1 = 26_289_598
CLEAN_SFT_TARGETS_P2 = 26_193_155
CLEAN_SFT_TARGETS_TOTAL = CLEAN_SFT_TARGETS_P1 + CLEAN_SFT_TARGETS_P2
CLEAN_END_THINKING_TARGETS = SFT_ROWS
CLEAN_CALL_ENV_TARGETS = 187_354

SEQUENCE_LENGTH = 3_072
PRETRAIN_TOTAL_TOKENS = 10_000_000_000
PRETRAIN_LEG_TOKENS = 5_000_000_000
DATA_SEED = 42
P1_SHUFFLE_SEED = 42
P2_SHUFFLE_SEED = 43

PRODUCTION_GPU_TYPE = "H200"
PRODUCTION_GPUS = 8
PRODUCTION_LOCAL_BATCH = 21
PRODUCTION_GRADIENT_ACCUMULATION = 1
LEG_STEPS = 9_920
MONOLITHIC_STEPS = 19_840
PINNED_FLASH_ATTENTION_VERSION = "2.8.3"
PRODUCTION_ATTENTION_BACKEND = "sdpa"
PRODUCTION_TORCH_COMPILE_MODE = "none"
PRODUCTION_DATA_WORKERS = 8
PRODUCTION_TRACKER_BACKEND = "none"
PRODUCTION_METRICS_FORMAT = "local-jsonl-v1"
P1_SFT_LOSS_WEIGHT = 190.189290837
P2_SFT_LOSS_WEIGHT = 190.889566377
MONOLITHIC_SFT_LOSS_WEIGHT = 190.538785189
SFT_WEIGHT_CANARY_DEFAULT_STEPS = 500
SFT_WEIGHT_CANARY_MAX_STEPS = 2_000
BENCHMARK_STEPS = 30
BENCHMARK_WARMUP_STEPS = 10
BENCHMARK_OUTPUT_ROOT = Path("/tmp/chess-interleave-benchmarks")

CANARY_GPU_TYPE = os.environ.get(
    "CHESS_INTERLEAVE_CANARY_GPU_TYPE", "H100"
).strip()
CANARY_GPUS = int(os.environ.get("CHESS_INTERLEAVE_CANARY_GPUS", "1"))
CANARY_LOCAL_BATCH = int(
    os.environ.get("CHESS_INTERLEAVE_CANARY_LOCAL_BATCH", "2")
)

ARTIFACT_ROOT = Path(f"/data/50m_interleaved_{DATA_ARTIFACT_VERSION}")
# Reuse the already verified immutable 8.4 GB source snapshot from v1.  The
# normalized token cache and every derived manifest live in the new v2 root.
SFT_SNAPSHOT_DIR = Path(
    "/data/50m_interleaved_mix10b_sft90k_v1/sft_hf_snapshot"
)
SFT_SNAPSHOT_MARKER = SFT_SNAPSHOT_DIR / ".complete.json"
SFT_CACHE_DIR = ARTIFACT_ROOT / "sft_cache"
SOURCE_MANIFEST_PATH = ARTIFACT_ROOT / "source_manifest.json"
PRETRAIN_SELECTION_PATH = ARTIFACT_ROOT / "pretrain_selection.json"
LEGS_ROOT = ARTIFACT_ROOT / "legs"
P1_METADATA_PATH = LEGS_ROOT / "p1" / "metadata.json"
P1_ORDER_PATH = LEGS_ROOT / "p1" / "order.npy"
P2_METADATA_PATH = LEGS_ROOT / "p2" / "metadata.json"
P2_ORDER_PATH = LEGS_ROOT / "p2" / "order.npy"
EXP2_METADATA_PATH = LEGS_ROOT / "exp2" / "metadata.json"
CANARY_METADATA_PATH = LEGS_ROOT / "canary" / "metadata.json"
CANARY_ORDER_PATH = LEGS_ROOT / "canary" / "order.npy"
MANIFEST_SET_PATH = ARTIFACT_ROOT / "manifest_set.json"
PRODUCTION_GATE_PATH = ARTIFACT_ROOT / "production_gate.json"
V2R2_GATE_ROOT = ARTIFACT_ROOT / "v2r2_staged_gate_20260730"
V2R2_P1_GATE_PATH = V2R2_GATE_ROOT / "p1_protocol_gate.json"
V2R2_P1_REJECTION_PATH = V2R2_GATE_ROOT / "p1_protocol_rejection.json"
V2R2_EXP2_GATE_PATH = V2R2_GATE_ROOT / "exp2_monolithic_protocol_gate.json"
V2R2_CHECKPOINT_ROOT = Path(
    f"/checkpoints/interleave_50m/pretrain/{V2R2_EXPERIMENT_VERSION}"
)
V2R3_CHECKPOINT_ROOT = Path(
    f"/checkpoints/interleave_50m/pretrain/{V2R3_EXPERIMENT_VERSION}"
)
V2R3_DIAGNOSTIC_ROOT = (
    ARTIFACT_ROOT / "v2r3_diagnostic_20260730"
)
V2R3_DIAGNOSTIC_REPORT_PATH = (
    V2R3_DIAGNOSTIC_ROOT / "seed42_report.json"
)
V2R3_TRAJECTORY_SPECS = {
    190.189290837: {
        "max_steps": LEG_STEPS,
        "snapshot_steps": (1_000, 2_000, 4_000, 6_000, 8_000, LEG_STEPS),
    },
    256.0: {
        "max_steps": 2_000,
        "snapshot_steps": (1_000, 2_000),
    },
    384.0: {
        "max_steps": 2_000,
        "snapshot_steps": (1_000, 2_000),
    },
    768.0: {
        "max_steps": 2_000,
        "snapshot_steps": (1_000, 2_000),
    },
}
V2R3_SNAPSHOT_COUNT = 12
V2R3_SNAPSHOT_COMMIT_POLL_SECONDS = 5.0
V2R3_RL_CHESS_SOURCE_SHA256 = (
    "d7d24d523a8c34577f7c1f01cf2e3855a9092d580cef918c319ad249d6d0f6b9"
)
V2R3_SEED42_PROMPT_SET_SHA256 = (
    "9ab746d0039bcc15d3573296cbe4503650a10b9b3248ffae1e7bb4121663b7c7"
)
V2R1_AUDITED_SOURCE_TREE_SHA256 = (
    "8b8cea9bdba2408209a5abd942ee24cd5c179dc9899c97f1edfc0cb3080832ff"
)
RL_GATE_ROOT = Path("/rl-checkpoints/chess-rl-miles-interleave")
RL_GATE_BALANCED_PATH = Path(
    "/data/chess-rl-data/train_v4_dataset_balanced_multi_turn.parquet"
)
RL_GATE_BALANCED_ROWS = 53_225
RL_GATE_ELIGIBLE_PROMPT_ROWS = 53_157
RL_GATE_FILTERED_LONG_PROMPT_ROWS = 68
V2R2_PROMPT_SELECTION_SCHEMA = "miles-fixed-prompt-selection-proof-v1"
RL_GATE_BALANCED_SHA256 = (
    "bcf131d88cd9916f6ec12a12f8e85901a05e0dabc5d5a1fdebd8ea0b0ae86c30"
)
RL_GATE_CHESS_SOURCE_SHA256 = (
    "38f829c60815cb8f7a07776561af51cc22a8b6740c431c59bd3d9847ff4c019f"
)
RL_GATE_MILES_SOURCE_SHA256 = (
    "9aeb9274e3c4d38ed2d2bdf80cd37df6b5ad3cb0202775de443bf1c6d60b9f0d"
)

CHECKPOINT_ROOT = Path(
    f"/checkpoints/interleave_50m/pretrain/{EXPERIMENT_VERSION}"
)
BASE_CONFIG = "config/configs/interleaved_50m/base_3072.yaml"
TRAIN_CLI = "scripts/train/train_interleaved_hf.py"
WANDB_ENTITY = "jingyanshen-new-york-university"
WANDB_PROJECT = "chess-50m-interleaved-pretrain-rl"
CANARY_WANDB_PROJECT = f"{WANDB_PROJECT}-canary"

_RUN_ID_RE = re.compile(r"[a-z0-9][a-z0-9.-]{0,79}")
_SHARD_RE = re.compile(r"raw\.(\d+)\.npy")
_ALLOWED_ACTIONS = frozenset(
    {
        "data-prep",
        "canary",
        "production-canary",
        "sft-weight-canary",
        "benchmark",
        "approve-gate",
        "p1",
        "exp2",
        "p2",
        "v2r2-approve-p1",
        "v2r2-p1",
        "v2r2-monolithic-canary",
        "v2r2-approve-exp2",
        "v2r2-exp2",
        "v2r3-trajectory",
        "v2r3-launch-all",
        "v2r3-audit",
    }
)
_ALLOWED_ATTENTION_BACKENDS = frozenset({"sdpa", "flash_attention_2"})
_ALLOWED_COMPILE_MODES = frozenset(
    {"none", "default", "reduce-overhead", "max-autotune"}
)


def _validate_static_settings() -> None:
    if PRODUCTION_GPUS != 8 or PRODUCTION_GPU_TYPE != "H200":
        raise RuntimeError("Production topology is pinned to exactly 8 H200 GPUs")
    if PRODUCTION_LOCAL_BATCH != 21:
        raise RuntimeError("Production local batch is pinned to 21")
    if PRODUCTION_GRADIENT_ACCUMULATION != 1:
        raise RuntimeError("Production gradient accumulation must remain 1")
    if PRODUCTION_ATTENTION_BACKEND != "sdpa":
        raise RuntimeError("Experiment v1 attention backend is frozen to SDPA")
    if PRODUCTION_TORCH_COMPILE_MODE != "none":
        raise RuntimeError("Experiment v1 is frozen without torch.compile")
    if PRODUCTION_DATA_WORKERS != 8:
        raise RuntimeError("Experiment v1 is frozen to 8 data workers per rank")
    if PRODUCTION_TRACKER_BACKEND != "none":
        raise RuntimeError("Scientific training cannot depend on a remote tracker")
    if CANARY_GPUS < 1 or CANARY_LOCAL_BATCH < 1:
        raise ValueError("Canary GPU count and local batch must be positive")
    if not CANARY_GPU_TYPE:
        raise ValueError("CHESS_INTERLEAVE_CANARY_GPU_TYPE cannot be empty")


_validate_static_settings()


# ---------------------------------------------------------------------------
# Pure validation/command helpers (also exercised by focused unit tests)
# ---------------------------------------------------------------------------


def _canonical_name_size_digest(entries: Iterable[tuple[str, int]]) -> str:
    """Hash sorted ``name<TAB>size<NEWLINE>`` records."""

    rows = []
    for name, size in entries:
        if "\t" in name or "\n" in name:
            raise ValueError(f"Unsafe manifest filename: {name!r}")
        if int(size) < 0:
            raise ValueError(f"Negative file size for {name}: {size}")
        rows.append((str(name), int(size)))
    payload = "".join(
        f"{name}\t{size}\n" for name, size in sorted(rows)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_run_id(run_id: str) -> str:
    value = run_id.strip()
    if not _RUN_ID_RE.fullmatch(value):
        raise ValueError(
            "run_id must be 1-80 lowercase path-safe characters and start "
            "with an alphanumeric character"
        )
    return value


def _validate_action(action: str) -> str:
    value = action.strip().lower()
    if value not in _ALLOWED_ACTIONS:
        raise ValueError(
            f"Unknown action {action!r}; choose from "
            f"{', '.join(sorted(_ALLOWED_ACTIONS))}"
        )
    return value


def _validate_attention_backend(value: str) -> str:
    backend = value.strip().lower()
    if backend not in _ALLOWED_ATTENTION_BACKENDS:
        raise ValueError(
            f"attention_backend must be one of "
            f"{sorted(_ALLOWED_ATTENTION_BACKENDS)}, got {value!r}"
        )
    return backend


def _validate_compile_mode(value: str) -> str:
    mode = value.strip().lower()
    if mode not in _ALLOWED_COMPILE_MODES:
        raise ValueError(
            f"compile_mode must be one of "
            f"{sorted(_ALLOWED_COMPILE_MODES)}, got {value!r}"
        )
    return mode


def _validate_sft_loss_weight(value: float) -> float:
    weight = float(value)
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("sft_loss_weight must be finite and strictly positive")
    return weight


def _validate_sft_weight_canary_steps(value: int) -> int:
    steps = int(value)
    if not 1 <= steps <= SFT_WEIGHT_CANARY_MAX_STEPS:
        raise ValueError(
            "sft-weight canary steps must be in "
            f"[1, {SFT_WEIGHT_CANARY_MAX_STEPS}]"
        )
    return steps


def _weight_slug(value: float) -> str:
    weight = _validate_sft_loss_weight(value)
    # Exact IEEE-754 identity avoids decimal-rounding collisions in immutable
    # output paths while the readable value remains in config/provenance.
    return struct.pack(">d", weight).hex()


def _source_tree_digest(root: Path) -> str:
    """Fingerprint exactly the local source subtrees copied into the image."""

    candidates: list[Path] = []
    for relative in ("config", "llm_tokens", "scripts", "training"):
        base = root / relative
        if not base.is_dir():
            continue
        candidates.extend(
            path
            for path in base.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    launcher_path = root / "modal_scripts" / Path(__file__).name
    if launcher_path.is_file():
        candidates.append(launcher_path)

    digest = hashlib.sha256()
    for path in sorted(set(candidates), key=lambda item: str(item.relative_to(root))):
        relative = str(path.relative_to(root))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _effective_source_tree_digest(
    computed: str, override: str | None
) -> str:
    value = (override or computed).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RuntimeError(
            "CHESS_INTERLEAVE_SOURCE_TREE_SHA256 must be a full lowercase "
            "SHA-256"
        )
    return value


def _checkpoint_fingerprint(path: Path) -> str:
    """Return a content fingerprint for a clean HF checkpoint."""

    resolved = path.resolve(strict=True)
    checkpoint_root = Path("/checkpoints").resolve()
    if not resolved.is_relative_to(checkpoint_root):
        raise ValueError(
            f"Initialization checkpoint must be under /checkpoints: {resolved}"
        )

    if resolved.is_file():
        weight_files = [resolved]
        base = resolved.parent
    else:
        base = resolved
        weight_files = sorted(resolved.glob("model*.safetensors"))
        if not weight_files:
            weight_files = sorted(resolved.glob("pytorch_model*.bin"))
        if not weight_files:
            raise FileNotFoundError(
                f"No Hugging Face weight files found in {resolved}"
            )
        if not (resolved / "config.json").is_file():
            raise FileNotFoundError(
                f"Clean Hugging Face checkpoint lacks config.json: {resolved}"
            )

    fingerprint_files = list(weight_files)
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        candidate = base / name
        if candidate.is_file():
            fingerprint_files.append(candidate)

    digest = hashlib.sha256()
    for candidate in sorted(
        set(fingerprint_files), key=lambda item: str(item.relative_to(base))
    ):
        relative = str(candidate.relative_to(base))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with candidate.open("rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainPlan:
    stage: str
    manifest_leg: str
    output_dir: str
    run_name: str
    total_steps: int
    arc_steps: tuple[int, ...]
    num_gpus: int
    local_batch_size: int
    manifest_metadata: str
    manifest_order: str
    weights_only: str | None = None
    init_fingerprint: str | None = None
    canary: bool = False
    max_steps: int | None = None
    attention_backend: str = PRODUCTION_ATTENTION_BACKEND
    torch_compile_mode: str = PRODUCTION_TORCH_COMPILE_MODE
    benchmark_only: bool = False
    benchmark_warmup_steps: int = 0
    data_workers: int = PRODUCTION_DATA_WORKERS
    sft_loss_weight: float = 1.0
    structure_canary: bool = False
    diagnostic_only: bool = False
    snapshot_steps: tuple[int, ...] = ()
    experiment_version: str = EXPERIMENT_VERSION
    source_tree_sha256: str | None = None


def _plan_source_tree_sha256(plan: TrainPlan) -> str:
    value = plan.source_tree_sha256 or SOURCE_TREE_SHA256
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("plan source_tree_sha256 must be a full SHA-256")
    return value


def _fixed_plan(stage: str) -> TrainPlan:
    if stage == "p1":
        run_name = f"50m-interleaved-p1-shared-{EXPERIMENT_VERSION}"
        return TrainPlan(
            stage=stage,
            manifest_leg="p1",
            output_dir=str(CHECKPOINT_ROOT / "p1_shared"),
            run_name=run_name,
            total_steps=LEG_STEPS,
            arc_steps=(LEG_STEPS,),
            num_gpus=PRODUCTION_GPUS,
            local_batch_size=PRODUCTION_LOCAL_BATCH,
            manifest_metadata=str(P1_METADATA_PATH),
            manifest_order=str(P1_ORDER_PATH),
            sft_loss_weight=P1_SFT_LOSS_WEIGHT,
        )
    if stage == "exp2":
        run_name = f"50m-interleaved-exp2-monolithic-{EXPERIMENT_VERSION}"
        return TrainPlan(
            stage=stage,
            manifest_leg="p1+p2",
            output_dir=str(CHECKPOINT_ROOT / "exp2_monolithic"),
            run_name=run_name,
            total_steps=MONOLITHIC_STEPS,
            arc_steps=(MONOLITHIC_STEPS,),
            num_gpus=PRODUCTION_GPUS,
            local_batch_size=PRODUCTION_LOCAL_BATCH,
            manifest_metadata=str(EXP2_METADATA_PATH),
            manifest_order="p1+p2",
            sft_loss_weight=MONOLITHIC_SFT_LOSS_WEIGHT,
        )
    raise ValueError(f"No fixed plan for stage {stage!r}")


def _v2r2_plan(stage: str, *, selected_weight: float) -> TrainPlan:
    """Build one version-scoped production or monolithic-canary plan."""

    weight = _validate_sft_loss_weight(selected_weight)
    if weight not in V2R2_ELIGIBLE_SFT_LOSS_WEIGHTS:
        raise ValueError(f"v2r2 selected an ineligible SFT weight: {weight}")
    common = {
        "num_gpus": PRODUCTION_GPUS,
        "local_batch_size": PRODUCTION_LOCAL_BATCH,
        "sft_loss_weight": weight,
        "experiment_version": V2R2_EXPERIMENT_VERSION,
    }
    if stage == "v2r2-p1":
        return TrainPlan(
            stage=stage,
            manifest_leg="p1",
            output_dir=str(V2R2_CHECKPOINT_ROOT / "p1_shared"),
            run_name=f"50m-interleaved-v2r2-p1-w{_weight_slug(weight)}",
            total_steps=LEG_STEPS,
            arc_steps=(LEG_STEPS,),
            manifest_metadata=str(P1_METADATA_PATH),
            manifest_order=str(P1_ORDER_PATH),
            **common,
        )
    if stage == "v2r2-exp2-monolithic-canary":
        return TrainPlan(
            stage=stage,
            manifest_leg="p1+p2",
            output_dir=str(
                V2R2_CHECKPOINT_ROOT
                / "protocol_canary"
                / f"exp2-monolithic-w{_weight_slug(weight)}-s2000"
            ),
            run_name=(
                "50m-interleaved-v2r2-exp2-monolithic-canary-"
                f"w{_weight_slug(weight)}-s2000"
            ),
            total_steps=MONOLITHIC_STEPS,
            arc_steps=(MONOLITHIC_STEPS,),
            manifest_metadata=str(EXP2_METADATA_PATH),
            manifest_order="p1+p2",
            max_steps=V2R2_PROTOCOL_CANDIDATE_STEP,
            structure_canary=True,
            **common,
        )
    if stage == "v2r2-exp2":
        return TrainPlan(
            stage=stage,
            manifest_leg="p1+p2",
            output_dir=str(V2R2_CHECKPOINT_ROOT / "exp2_monolithic"),
            run_name=f"50m-interleaved-v2r2-exp2-w{_weight_slug(weight)}",
            total_steps=MONOLITHIC_STEPS,
            arc_steps=(MONOLITHIC_STEPS,),
            manifest_metadata=str(EXP2_METADATA_PATH),
            manifest_order="p1+p2",
            **common,
        )
    raise ValueError(f"No v2r2 plan for stage {stage!r}")


def _v2r3_plan(sft_loss_weight: float) -> TrainPlan:
    """Build one exact diagnostic-only continuous P1 trajectory."""

    weight = _validate_sft_loss_weight(sft_loss_weight)
    spec = V2R3_TRAJECTORY_SPECS.get(weight)
    if spec is None:
        raise ValueError(
            f"v2r3 weight must be one of "
            f"{list(V2R3_TRAJECTORY_SPECS)}, got {weight}"
        )
    max_steps = int(spec["max_steps"])
    snapshot_steps = tuple(int(step) for step in spec["snapshot_steps"])
    weight_slug = _weight_slug(weight)
    return TrainPlan(
        stage="v2r3-diagnostic",
        manifest_leg="p1",
        output_dir=str(
            V2R3_CHECKPOINT_ROOT / f"p1_w{weight_slug}"
        ),
        run_name=(
            f"50m-interleaved-v2r3-diagnostic-p1-w{weight_slug}"
        ),
        total_steps=LEG_STEPS,
        arc_steps=(LEG_STEPS,),
        num_gpus=PRODUCTION_GPUS,
        local_batch_size=PRODUCTION_LOCAL_BATCH,
        manifest_metadata=str(P1_METADATA_PATH),
        manifest_order=str(P1_ORDER_PATH),
        max_steps=max_steps,
        sft_loss_weight=weight,
        diagnostic_only=True,
        snapshot_steps=snapshot_steps,
        experiment_version=V2R3_EXPERIMENT_VERSION,
    )


def _canary_plan(run_id: str) -> TrainPlan:
    safe_id = _validate_run_id(run_id)
    run_name = f"50m-interleaved-canary-{safe_id}-{EXPERIMENT_VERSION}"
    return TrainPlan(
        stage="canary",
        manifest_leg="canary",
        output_dir=str(CHECKPOINT_ROOT / "canary" / safe_id),
        run_name=run_name,
        # Keep the real P1 LR arc and stop after one update via max_steps. This
        # exercises the production warmup rather than turning the canary's
        # only update into an eta_min endpoint.
        total_steps=LEG_STEPS,
        arc_steps=(LEG_STEPS,),
        num_gpus=CANARY_GPUS,
        local_batch_size=CANARY_LOCAL_BATCH,
        manifest_metadata=str(CANARY_METADATA_PATH),
        manifest_order=str(CANARY_ORDER_PATH),
        canary=True,
        max_steps=1,
        data_workers=0,
    )


def _production_canary_plan(run_id: str) -> TrainPlan:
    """Short warmed benchmark at the exact production 8×H200 topology."""

    safe_id = _validate_run_id(run_id)
    run_name = (
        f"50m-interleaved-production-canary-{safe_id}-"
        f"{EXPERIMENT_VERSION}"
    )
    return TrainPlan(
        stage="production-canary",
        manifest_leg="p1",
        output_dir=str(CHECKPOINT_ROOT / "production_canary" / safe_id),
        run_name=run_name,
        total_steps=LEG_STEPS,
        arc_steps=(LEG_STEPS,),
        num_gpus=PRODUCTION_GPUS,
        local_batch_size=PRODUCTION_LOCAL_BATCH,
        manifest_metadata=str(P1_METADATA_PATH),
        manifest_order=str(P1_ORDER_PATH),
        # Reuse the one-update/no-W&B override set. Unlike the lightweight
        # canary, this plan consumes the real P1 manifest and exact production
        # world/local batch sizes.
        canary=True,
        max_steps=20,
    )


def _sft_weight_canary_plan(
    run_id: str,
    *,
    sft_loss_weight: float,
    max_steps: int = SFT_WEIGHT_CANARY_DEFAULT_STEPS,
) -> TrainPlan:
    """Exact-topology v2 structure canary on the cleaned P1 stream."""

    safe_id = _validate_run_id(run_id)
    weight = _validate_sft_loss_weight(sft_loss_weight)
    steps = _validate_sft_weight_canary_steps(max_steps)
    weight_slug = _weight_slug(weight)
    identity = f"{safe_id}-w{weight_slug}-s{steps}"
    return TrainPlan(
        stage="sft-weight-canary",
        manifest_leg="p1",
        output_dir=str(CHECKPOINT_ROOT / "sft_weight_canary" / identity),
        run_name=(
            f"50m-interleaved-v2-sft-weight-canary-{identity}-"
            f"{EXPERIMENT_VERSION}"
        ),
        total_steps=LEG_STEPS,
        arc_steps=(LEG_STEPS,),
        num_gpus=PRODUCTION_GPUS,
        local_batch_size=PRODUCTION_LOCAL_BATCH,
        manifest_metadata=str(P1_METADATA_PATH),
        manifest_order=str(P1_ORDER_PATH),
        max_steps=steps,
        sft_loss_weight=weight,
        structure_canary=True,
    )


def _benchmark_plan(
    run_id: str,
    *,
    attention_backend: str,
    compile_mode: str,
) -> TrainPlan:
    """Exact-topology ephemeral benchmark that cannot touch checkpoints."""

    safe_id = _validate_run_id(run_id)
    backend = _validate_attention_backend(attention_backend)
    normalized_compile = _validate_compile_mode(compile_mode)
    variant = f"{backend}-{normalized_compile}"
    run_name = (
        f"50m-interleaved-benchmark-{safe_id}-{variant}-"
        f"{EXPERIMENT_VERSION}"
    )
    return TrainPlan(
        stage="benchmark",
        manifest_leg="p1",
        output_dir=str(BENCHMARK_OUTPUT_ROOT / safe_id / variant),
        run_name=run_name,
        total_steps=LEG_STEPS,
        arc_steps=(LEG_STEPS,),
        num_gpus=PRODUCTION_GPUS,
        local_batch_size=PRODUCTION_LOCAL_BATCH,
        manifest_metadata=str(P1_METADATA_PATH),
        manifest_order=str(P1_ORDER_PATH),
        max_steps=BENCHMARK_STEPS,
        attention_backend=backend,
        torch_compile_mode=normalized_compile,
        benchmark_only=True,
        benchmark_warmup_steps=BENCHMARK_WARMUP_STEPS,
    )


def _p2_plan(
    *,
    run_id: str,
    init_checkpoint: Path,
    init_fingerprint: str,
) -> TrainPlan:
    safe_id = _validate_run_id(run_id)
    if not re.fullmatch(r"[0-9a-f]{64}", init_fingerprint):
        raise ValueError("init_fingerprint must be a full SHA-256 digest")
    fingerprint_short = init_fingerprint[:12]
    output_name = f"{safe_id}-from-{fingerprint_short}"
    run_name = (
        f"50m-interleaved-p2-{safe_id}-from-{fingerprint_short}-"
        f"{EXPERIMENT_VERSION}"
    )
    return TrainPlan(
        stage="p2",
        manifest_leg="p2",
        output_dir=str(CHECKPOINT_ROOT / "p2" / output_name),
        run_name=run_name,
        total_steps=LEG_STEPS,
        arc_steps=(LEG_STEPS,),
        num_gpus=PRODUCTION_GPUS,
        local_batch_size=PRODUCTION_LOCAL_BATCH,
        manifest_metadata=str(P2_METADATA_PATH),
        manifest_order=str(P2_ORDER_PATH),
        weights_only=str(init_checkpoint),
        init_fingerprint=init_fingerprint,
        sft_loss_weight=P2_SFT_LOSS_WEIGHT,
    )


def _plan_overrides(plan: TrainPlan, manifest_hash: str) -> list[str]:
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_hash):
        raise ValueError("manifest_hash must be a full SHA-256 digest")
    weight = _validate_sft_loss_weight(plan.sft_loss_weight)
    project = (
        CANARY_WANDB_PROJECT
        if plan.canary or plan.structure_canary
        else WANDB_PROJECT
    )
    overrides = [
        f"training.output_dir={plan.output_dir}",
        f"training.run_name={plan.run_name}",
        "training.seed=42",
        f"training.local_batch_size={plan.local_batch_size}",
        "training.gradient_accumulation_steps=1",
        f"training.total_steps={plan.total_steps}",
        f"training.arc_steps={list(plan.arc_steps)}",
        f"model.attn_implementation={plan.attention_backend}",
        (
            "model.flash_attention_version="
            f"{PINNED_FLASH_ATTENTION_VERSION}"
        ),
        f"training.torch_compile={plan.torch_compile_mode}",
        f"training.sft_loss_weight={weight}",
        f"data.num_workers={plan.data_workers}",
        "training.reset_optimizer_between_arcs=true",
        "training.scheduler.warmup_ratio=0.05",
        "training.scheduler.eta_min=1e-5",
        "training.optimizer.lr=1e-3",
        "training.optimizer.weight_decay=0.1",
        "training.optimizer.betas=[0.9,0.95]",
        f"data.source_root={SOURCE_DIR}",
        f"data.source_manifest_path={SOURCE_MANIFEST_PATH}",
        f"data.selection_manifest_path={PRETRAIN_SELECTION_PATH}",
        f"data.sft_cache_dir={SFT_CACHE_DIR}",
        f"data.leg_manifest_path={plan.manifest_metadata}",
        f"data.expected_manifest_hash={manifest_hash}",
        f"logging.project={project}",
        f"logging.entity={WANDB_ENTITY}",
        f"logging.backend={PRODUCTION_TRACKER_BACKEND}",
        f"provenance.experiment_version={plan.experiment_version}",
        f"provenance.data_artifact_version={DATA_ARTIFACT_VERSION}",
        f"provenance.source_repo={SOURCE_REPO}",
        f"provenance.source_revision={SOURCE_REVISION}",
        (
            "provenance.source_flat_manifest_sha256="
            f"{SOURCE_FLAT_MANIFEST_SHA256}"
        ),
        f"provenance.sft_repo={SFT_REPO}",
        f"provenance.sft_revision={SFT_REVISION}",
        f"provenance.attention_backend={plan.attention_backend}",
        (
            "provenance.flash_attention_version="
            f"{PINNED_FLASH_ATTENTION_VERSION}"
        ),
        f"provenance.torch_compile_mode={plan.torch_compile_mode}",
        f"provenance.data_num_workers={plan.data_workers}",
        f"provenance.sft_loss_weight={weight}",
        (
            "provenance.sft_response_normalization="
            f"{SFT_RESPONSE_NORMALIZATION}"
        ),
        (
            "provenance.sft_supervised_unk_policy="
            f"{SFT_SUPERVISED_UNK_POLICY}"
        ),
        f"provenance.metrics_format={PRODUCTION_METRICS_FORMAT}",
        (
            "provenance.source_tree_sha256="
            f"{_plan_source_tree_sha256(plan)}"
        ),
    ]
    if plan.init_fingerprint:
        overrides.append(
            f"provenance.init_checkpoint_sha256={plan.init_fingerprint}"
        )
    if plan.canary:
        if plan.max_steps is None or plan.max_steps <= 0:
            raise ValueError("Canary plans require a positive max_steps")
        overrides.extend(
            [
                f"training.max_steps={plan.max_steps}",
                "training.allow_topology_override=true",
                "training.persistent_workers=false",
                "training.save_interval=1",
                "training.log_interval=1",
            ]
        )
    if plan.structure_canary:
        if plan.max_steps is None or plan.max_steps <= 0:
            raise ValueError(
                "SFT-weight structure canaries require a positive max_steps"
            )
        overrides.extend(
            [
                f"training.max_steps={plan.max_steps}",
                "training.persistent_workers=true",
                "training.save_interval=0",
                "training.export_interval=0",
                "training.log_interval=10",
            ]
        )
    if plan.diagnostic_only:
        if (
            plan.stage != "v2r3-diagnostic"
            or plan.max_steps is None
            or plan.max_steps <= 0
            or not plan.snapshot_steps
            or plan.snapshot_steps[-1] != plan.max_steps
            or tuple(sorted(set(plan.snapshot_steps)))
            != plan.snapshot_steps
        ):
            raise ValueError(
                "v2r3 diagnostics require an increasing immutable snapshot "
                "schedule ending exactly at max_steps"
            )
        overrides.extend(
            [
                f"training.max_steps={plan.max_steps}",
                f"training.snapshot_steps={list(plan.snapshot_steps)}",
                "training.persistent_workers=true",
                "training.save_interval=0",
                "training.export_interval=0",
                "training.log_interval=10",
            ]
        )
    if plan.benchmark_only:
        if plan.max_steps is None or plan.max_steps <= 0:
            raise ValueError("Benchmark plans require a positive max_steps")
        if not 0 <= plan.benchmark_warmup_steps < plan.max_steps:
            raise ValueError("Benchmark warmup must be shorter than max_steps")
        output = Path(plan.output_dir)
        if not output.is_relative_to(BENCHMARK_OUTPUT_ROOT):
            raise ValueError("Benchmark output escaped the ephemeral root")
        overrides.extend(
            [
                f"training.max_steps={plan.max_steps}",
                "training.benchmark_only=true",
                (
                    "training.benchmark_warmup_steps="
                    f"{plan.benchmark_warmup_steps}"
                ),
                "training.persistent_workers=true",
                "training.save_interval=0",
                "training.export_interval=0",
                "training.log_interval=1",
            ]
        )
    return overrides


def _build_training_command(
    plan: TrainPlan,
    *,
    manifest_hash: str,
    main_process_port: int,
    resume: str | None = None,
) -> list[str]:
    if plan.num_gpus < 1:
        raise ValueError("num_gpus must be positive")
    command = [
        "accelerate",
        "launch",
    ]
    if plan.num_gpus > 1:
        command.append("--multi_gpu")
    command.extend(
        [
            "--num_processes",
            str(plan.num_gpus),
            "--mixed_precision",
            "bf16",
            "--main_process_port",
            str(main_process_port),
            TRAIN_CLI,
            "--config",
            BASE_CONFIG,
            "--override",
            *_plan_overrides(plan, manifest_hash),
        ]
    )
    if resume:
        command.extend(["--resume", resume])
    elif plan.weights_only:
        command.extend(["--weights-only", plan.weights_only])
    return command


# ---------------------------------------------------------------------------
# Modal image, volumes, and remote implementation
# ---------------------------------------------------------------------------


repo_dir = Path(__file__).resolve().parent.parent
_computed_source_tree_sha256 = _source_tree_digest(repo_dir)
# Modal re-imports this launcher from ``/root/launch_50m_interleaved.py`` while
# the copied source tree lives under ``/root/chess``. The image carries the
# locally computed digest explicitly; prefer it in the remote import so
# checkpoint provenance cannot silently become SHA-256(empty).
SOURCE_TREE_SHA256 = _effective_source_tree_digest(
    _computed_source_tree_sha256,
    os.environ.get("CHESS_INTERLEAVE_SOURCE_TREE_SHA256"),
)

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
            "CHESS_INTERLEAVE_SOURCE_TREE_SHA256": SOURCE_TREE_SHA256,
        }
    )
    .add_local_dir(str(repo_dir / "scripts"), remote_path="/root/chess/scripts")
    .add_local_dir(str(repo_dir / "training"), remote_path="/root/chess/training")
    .add_local_dir(str(repo_dir / "config"), remote_path="/root/chess/config")
    .add_local_dir(
        str(repo_dir / "llm_tokens"), remote_path="/root/chess/llm_tokens"
    )
)

data_volume = modal.Volume.from_name(
    "rl-reasoning-training-data", create_if_missing=False
)
checkpoint_volume = modal.Volume.from_name(
    "rl-reasoning-checkpoints", create_if_missing=False
)
rl_checkpoint_volume = modal.Volume.from_name(
    "chess-rl-miles-checkpoints", create_if_missing=False
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
        "/rl-checkpoints": rl_checkpoint_volume,
    },
)


def _verify_source_corpus() -> dict[str, object]:
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Missing Modal source corpus: {SOURCE_DIR}")
    files = list(SOURCE_DIR.glob("raw.*.npy"))
    if len(files) != SOURCE_SHARDS:
        raise RuntimeError(
            f"Source shard count mismatch: {len(files)} != {SOURCE_SHARDS}"
        )

    shard_ids: list[int] = []
    entries: list[tuple[str, int]] = []
    total_bytes = 0
    for path in files:
        match = _SHARD_RE.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"Unexpected source filename: {path.name}")
        shard_ids.append(int(match.group(1)))
        size = path.stat().st_size
        entries.append((path.name, size))
        total_bytes += size

    expected_ids = list(range(SOURCE_FIRST_SHARD, SOURCE_LAST_SHARD + 1))
    if sorted(shard_ids) != expected_ids:
        actual_ids = set(shard_ids)
        missing = sorted(set(expected_ids) - actual_ids)
        duplicate_count = len(shard_ids) - len(actual_ids)
        raise RuntimeError(
            "Source shard IDs are not the pinned contiguous range; "
            f"missing={missing[:5]} duplicate_count={duplicate_count}"
        )
    digest = _canonical_name_size_digest(entries)
    if digest != SOURCE_FLAT_MANIFEST_SHA256:
        raise RuntimeError(
            f"Source manifest mismatch: {digest} != "
            f"{SOURCE_FLAT_MANIFEST_SHA256}"
        )
    if total_bytes != SOURCE_BYTES:
        raise RuntimeError(
            f"Source byte count mismatch: {total_bytes} != {SOURCE_BYTES}"
        )

    # The pinned corpus is NumPy v1, little-endian uint32, with 128-byte
    # headers. Checking representative boundary files catches a wrong format
    # without reopening all 47,090 files after the name/size digest passes.
    import numpy as np

    for shard_id in SOURCE_HEADER_CHECK_SHARDS:
        sample = SOURCE_DIR / f"raw.{shard_id:04d}.npy"
        array = np.load(sample, mmap_mode="r", allow_pickle=False)
        if (
            array.ndim != 1
            or array.dtype.str != SOURCE_NPY_DTYPE
            or int(array.offset) != SOURCE_NPY_HEADER_BYTES
            or int(array.offset) + int(array.nbytes) != sample.stat().st_size
        ):
            raise RuntimeError(
                f"Unexpected source array format at {sample}: "
                f"shape={array.shape} dtype={array.dtype} "
                f"offset={getattr(array, 'offset', None)} "
                f"nbytes={array.nbytes} file_bytes={sample.stat().st_size}"
            )

    return {
        "repo": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "path": str(SOURCE_DIR),
        "shards": SOURCE_SHARDS,
        "tokens": SOURCE_TOKENS,
        "bytes": SOURCE_BYTES,
        "flat_manifest_sha256": digest,
        "npy_dtype": SOURCE_NPY_DTYPE,
        "npy_header_bytes": SOURCE_NPY_HEADER_BYTES,
        "header_check_shards": list(SOURCE_HEADER_CHECK_SHARDS),
    }


def _sft_snapshot_entries() -> list[tuple[str, int]]:
    return [
        (str(path.relative_to(SFT_SNAPSHOT_DIR)), path.stat().st_size)
        for path in SFT_SNAPSHOT_DIR.rglob("*.json")
        if path.is_file()
        and path != SFT_SNAPSHOT_MARKER
        and ".cache" not in path.relative_to(SFT_SNAPSHOT_DIR).parts
    ]


def _validate_sft_snapshot() -> dict[str, object]:
    entries = _sft_snapshot_entries()
    if len(entries) != SFT_JSON_FILES:
        raise RuntimeError(
            f"SFT JSON file count mismatch: {len(entries)} != {SFT_JSON_FILES}"
        )
    total_bytes = sum(size for _, size in entries)
    digest = _canonical_name_size_digest(entries)
    if total_bytes != SFT_BYTES:
        raise RuntimeError(
            f"SFT byte count mismatch: {total_bytes} != {SFT_BYTES}"
        )
    if digest != SFT_FLAT_MANIFEST_SHA256:
        raise RuntimeError(
            f"SFT manifest mismatch: {digest} != "
            f"{SFT_FLAT_MANIFEST_SHA256}"
        )
    return {
        "repo": SFT_REPO,
        "revision": SFT_REVISION,
        "path": str(SFT_SNAPSHOT_DIR),
        "json_files": SFT_JSON_FILES,
        "expected_rows": SFT_ROWS,
        "bytes": SFT_BYTES,
        "flat_manifest_sha256": digest,
    }


def _cache_sft_snapshot() -> dict[str, object]:
    if SFT_SNAPSHOT_MARKER.is_file():
        marker = json.loads(SFT_SNAPSHOT_MARKER.read_text(encoding="utf-8"))
        if (
            marker.get("repo") != SFT_REPO
            or marker.get("revision") != SFT_REVISION
            or marker.get("flat_manifest_sha256")
            != SFT_FLAT_MANIFEST_SHA256
        ):
            raise RuntimeError(
                f"Invalid immutable SFT cache marker: {SFT_SNAPSHOT_MARKER}"
            )
        return _validate_sft_snapshot()

    from huggingface_hub import snapshot_download

    SFT_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=SFT_REPO,
        repo_type="dataset",
        revision=SFT_REVISION,
        local_dir=str(SFT_SNAPSHOT_DIR),
        allow_patterns=["*.json", ".gitattributes"],
    )
    metadata = _validate_sft_snapshot()
    marker_payload = {
        **metadata,
        "source_tree_sha256": SOURCE_TREE_SHA256,
    }
    temporary_marker = SFT_SNAPSHOT_MARKER.with_name(
        f"{SFT_SNAPSHOT_MARKER.name}.tmp-{os.getpid()}"
    )
    temporary_marker.write_text(
        json.dumps(marker_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # No complete marker existed at entry. If another prep won the race, only
    # accept it when it records the identical immutable snapshot.
    if SFT_SNAPSHOT_MARKER.exists():
        temporary_marker.unlink()
        existing = json.loads(
            SFT_SNAPSHOT_MARKER.read_text(encoding="utf-8")
        )
        if existing != marker_payload:
            raise RuntimeError("Concurrent SFT cache preparation disagreed")
    else:
        temporary_marker.rename(SFT_SNAPSHOT_MARKER)
    data_volume.commit()
    return metadata


def _manifest_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _call_builder(function, available: Mapping[str, object]):
    """Call a shared builder while rejecting unsatisfied required arguments."""

    signature = inspect.signature(function)
    kwargs = {
        name: available[name]
        for name in signature.parameters
        if name in available
    }
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and name not in kwargs
        and parameter.kind
        not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    ]
    if missing:
        raise TypeError(
            f"Launcher cannot satisfy {function.__name__} arguments: {missing}"
        )
    return function(**kwargs)


def _canonical_mapping_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_manifest_set() -> Mapping[str, object]:
    if not MANIFEST_SET_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {MANIFEST_SET_PATH}; run --action data-prep first"
        )
    payload = json.loads(MANIFEST_SET_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != "interleaved-manifest-set-v1":
        raise RuntimeError(
            f"Unexpected manifest-set schema: {payload.get('schema')!r}"
        )
    recorded_set_hash = payload.get("manifest_set_hash")
    unhashed_payload = {
        key: value
        for key, value in payload.items()
        if key != "manifest_set_hash"
    }
    actual_set_hash = hashlib.sha256(
        json.dumps(
            unhashed_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    if recorded_set_hash != actual_set_hash:
        raise RuntimeError(
            "Manifest-set self hash mismatch: "
            f"{recorded_set_hash!r} != {actual_set_hash}"
        )
    if payload.get("experiment_version") != DATA_ARTIFACT_VERSION:
        raise RuntimeError(
            "Manifest-set experiment version mismatch: "
            f"{payload.get('experiment_version')!r}"
        )
    if payload.get("source_revision") != SOURCE_REVISION:
        raise RuntimeError("Manifest-set source revision mismatch")
    if payload.get("sft_revision") != SFT_REVISION:
        raise RuntimeError("Manifest-set SFT revision mismatch")
    if int(payload.get("pretrain_tokens", -1)) != PRETRAIN_TOTAL_TOKENS:
        raise RuntimeError("Manifest-set pretraining token budget mismatch")
    if int(payload.get("sft_rows", -1)) != SFT_ROWS:
        raise RuntimeError("Manifest-set SFT row count mismatch")

    source_payload = json.loads(
        SOURCE_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    if (
        source_payload.get("schema") != "interleaved-source-shards-v1"
        or int(source_payload.get("total_tokens", -1)) != SOURCE_TOKENS
        or len(source_payload.get("shards", ())) != SOURCE_SHARDS
    ):
        raise RuntimeError("Pinned source-manifest inventory mismatch")
    if (
        payload.get("source_manifest_hash")
        != source_payload.get("manifest_hash")
    ):
        raise RuntimeError("Manifest-set/source-manifest hash mismatch")

    selection_payload = json.loads(
        PRETRAIN_SELECTION_PATH.read_text(encoding="utf-8")
    )
    if (
        selection_payload.get("schema")
        != "interleaved-pretrain-selection-v1"
        or int(selection_payload.get("target_tokens", -1))
        != PRETRAIN_TOTAL_TOKENS
        or int(selection_payload.get("source_tokens", -1))
        != PRETRAIN_TOTAL_TOKENS + 1
        or selection_payload.get("source_manifest_hash")
        != source_payload.get("manifest_hash")
    ):
        raise RuntimeError("Pinned 10B pretraining selection mismatch")
    if payload.get("selection_hash") != selection_payload.get("selection_hash"):
        raise RuntimeError("Manifest-set/pretraining-selection hash mismatch")

    sft_payload = json.loads(
        (SFT_CACHE_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    if (
        sft_payload.get("schema") != "interleaved-sft-cache-v1"
        or int(sft_payload.get("num_rows", -1)) != SFT_ROWS
        or int(sft_payload.get("sequence_length", -1)) != SEQUENCE_LENGTH
        or sft_payload.get("prompt_field") != SFT_PROMPT_FIELD
        or sft_payload.get("cot_field") != SFT_COT_FIELD
        or sft_payload.get("response_normalization")
        != SFT_RESPONSE_NORMALIZATION
        or sft_payload.get("supervised_unk_policy")
        != SFT_SUPERVISED_UNK_POLICY
        or int(sft_payload.get("supervised_unk_targets", -1)) != 0
    ):
        raise RuntimeError("Pinned SFT cache contract mismatch")
    strict_audit = sft_payload.get("strict_sft_audit")
    delimiter_counts = sft_payload.get("supervised_delimiter_counts")
    if (
        sft_payload.get("strict_sft_audit_required") is not True
        or int(sft_payload.get("supervised_targets", -1))
        != CLEAN_SFT_TARGETS_TOTAL
        or not isinstance(strict_audit, Mapping)
        or strict_audit.get("schema") != SFT_STRICT_AUDIT_SCHEMA
        or strict_audit.get("expected_supervised_targets")
        != CLEAN_SFT_TARGETS_TOTAL
        or strict_audit.get("t_end_rows_exactly_one") != SFT_ROWS
        or strict_audit.get("call_env_rows_at_least_one") != SFT_ROWS
        or not isinstance(delimiter_counts, Mapping)
        or delimiter_counts.get("</T>") != CLEAN_END_THINKING_TARGETS
        or delimiter_counts.get("<call_env>") != CLEAN_CALL_ENV_TARGETS
    ):
        raise RuntimeError("Pinned strict SFT audit totals mismatch")
    if payload.get("sft_cache_hash") != sft_payload.get("cache_hash"):
        raise RuntimeError("Manifest-set/SFT-cache hash mismatch")

    manifests = payload.get("manifests")
    if not isinstance(manifests, Mapping):
        raise RuntimeError("manifest_set.json lacks a manifests mapping")
    expected_paths = {
        "p1": P1_METADATA_PATH,
        "p2": P2_METADATA_PATH,
        "p1+p2": EXP2_METADATA_PATH,
        "canary": CANARY_METADATA_PATH,
    }
    expected_leg_sft_targets = {
        "p1": CLEAN_SFT_TARGETS_P1,
        "p2": CLEAN_SFT_TARGETS_P2,
    }
    for key, expected_path in expected_paths.items():
        entry = manifests.get(key)
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"manifest_set.json lacks manifest {key!r}")
        relative_path = entry.get("path")
        expected_hash = entry.get("sha256")
        if not isinstance(relative_path, str):
            raise RuntimeError(f"Invalid path for manifest {key!r}")
        resolved = (MANIFEST_SET_PATH.parent / relative_path).resolve()
        if resolved != expected_path.resolve():
            raise RuntimeError(
                f"Manifest {key!r} points to {resolved}, expected "
                f"{expected_path.resolve()}"
            )
        if (
            not isinstance(expected_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
        ):
            raise RuntimeError(
                f"Invalid manifest hash for {key!r}: {expected_hash!r}"
            )
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        actual_hash = _manifest_file_sha256(resolved)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Manifest file hash mismatch for {key!r}: "
                f"{actual_hash} != {expected_hash}"
            )
        if key in expected_leg_sft_targets:
            leg_payload = json.loads(
                resolved.read_text(encoding="utf-8")
            )
            if (
                int(leg_payload.get("sft_supervised_targets", -1))
                != expected_leg_sft_targets[key]
            ):
                raise RuntimeError(
                    f"Manifest {key!r} SFT target total drifted: "
                    f"{leg_payload.get('sft_supervised_targets')!r} != "
                    f"{expected_leg_sft_targets[key]}"
                )
    return payload


def _manifest_hash_for_plan(
    payload: Mapping[str, object], plan: TrainPlan
) -> str:
    manifests = payload.get("manifests")
    if not isinstance(manifests, Mapping):
        raise RuntimeError("manifest_set.json lacks a manifests mapping")
    key = plan.manifest_leg
    entry = manifests.get(key)
    if not isinstance(entry, Mapping):
        raise RuntimeError(f"manifest_set.json lacks manifest {key!r}")
    value = entry.get("sha256")
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RuntimeError(f"Invalid manifest hash for {key!r}: {value!r}")
    return value


def _verified_source_fast_path(
    source_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Authorize byte-size token inference only for the verified pinned tree."""

    expected: dict[str, object] = {
        "repo": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "path": str(SOURCE_DIR),
        "shards": SOURCE_SHARDS,
        "tokens": SOURCE_TOKENS,
        "bytes": SOURCE_BYTES,
        "flat_manifest_sha256": SOURCE_FLAT_MANIFEST_SHA256,
        "npy_dtype": SOURCE_NPY_DTYPE,
        "npy_header_bytes": SOURCE_NPY_HEADER_BYTES,
        "header_check_shards": list(SOURCE_HEADER_CHECK_SHARDS),
    }
    mismatches = {
        key: (source_metadata.get(key), value)
        for key, value in expected.items()
        if source_metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Refusing trusted source-manifest fast path because the verified "
            f"pinned-corpus metadata differs: {mismatches}"
        )
    return {
        "trusted_npy_dtype": SOURCE_NPY_DTYPE,
        "trusted_npy_header_bytes": SOURCE_NPY_HEADER_BYTES,
        "expected_total_tokens": SOURCE_TOKENS,
    }


def _build_manifests(
    source_metadata: Mapping[str, object],
) -> Mapping[str, object]:
    """Invoke the shared deterministic mixed-stream builders."""

    if MANIFEST_SET_PATH.is_file():
        return _load_manifest_set()

    from config import load_config
    from llm_tokens.chess.tokenizer_factory import init_tokenizer
    from training.interleaved_data import (
        build_leg_manifests,
        build_manifest_set,
        build_pretrain_selection,
        build_sft_cache,
        build_source_manifest,
    )

    config = load_config(f"/root/chess/{BASE_CONFIG}")
    tokenizer = init_tokenizer(config.tokenizer.name, config.tokenizer)
    # Reuse the validated immutable inventory.  A raw ``rglob("*.json")``
    # would also pick up our ``.complete.json`` marker (and Hugging Face cache
    # metadata), neither of which is an SFT data shard.
    sft_files = [
        str(SFT_SNAPSHOT_DIR / relative_path)
        for relative_path, _ in _sft_snapshot_entries()
    ]
    source_fast_path = _verified_source_fast_path(source_metadata)

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    _call_builder(
        build_source_manifest,
        {
            "source_root": SOURCE_DIR,
            "output_path": SOURCE_MANIFEST_PATH,
            "pattern": "raw.*.npy",
            "content_hashes": False,
            **source_fast_path,
        },
    )
    _call_builder(
        build_pretrain_selection,
        {
            "source_manifest_path": SOURCE_MANIFEST_PATH,
            "source_manifest": SOURCE_MANIFEST_PATH,
            "output_path": PRETRAIN_SELECTION_PATH,
            "target_tokens": PRETRAIN_TOTAL_TOKENS,
            "seed": DATA_SEED,
        },
    )
    _call_builder(
        build_sft_cache,
        {
            "sft_files": sft_files,
            "data_files": sft_files,
            "tokenizer": tokenizer,
            "output_dir": SFT_CACHE_DIR,
            "sft_cache_dir": SFT_CACHE_DIR,
            "cot_field": SFT_COT_FIELD,
            "prompt_field": SFT_PROMPT_FIELD,
            "seq_len": SEQUENCE_LENGTH,
            "sequence_length": SEQUENCE_LENGTH,
            "expected_rows": SFT_ROWS,
            "strip_verify_scores": True,
            "reject_supervised_unk": True,
            "strict_sft_audit": True,
            "expected_supervised_targets": CLEAN_SFT_TARGETS_TOTAL,
        },
    )
    _call_builder(
        build_leg_manifests,
        {
            "source_manifest_path": SOURCE_MANIFEST_PATH,
            "pretrain_selection_path": PRETRAIN_SELECTION_PATH,
            "selection_manifest_path": PRETRAIN_SELECTION_PATH,
            "sft_cache_dir": SFT_CACHE_DIR,
            "output_root": LEGS_ROOT,
            "legs_root": LEGS_ROOT,
            "world_size": PRODUCTION_GPUS,
            "local_batch_size": PRODUCTION_LOCAL_BATCH,
            "split_seed": DATA_SEED,
            "p1_seed": P1_SHUFFLE_SEED,
            "p2_seed": P2_SHUFFLE_SEED,
            "canary_world_size": CANARY_GPUS,
            "canary_local_batch_size": CANARY_LOCAL_BATCH,
            "canary_total_steps": LEG_STEPS,
            "expected_sft_supervised_targets": (
                CLEAN_SFT_TARGETS_P1,
                CLEAN_SFT_TARGETS_P2,
            ),
        },
    )
    _call_builder(
        build_manifest_set,
        {
            "output_path": MANIFEST_SET_PATH,
            "experiment_version": DATA_ARTIFACT_VERSION,
            "source_manifest_path": SOURCE_MANIFEST_PATH,
            "pretrain_selection_path": PRETRAIN_SELECTION_PATH,
            "selection_manifest_path": PRETRAIN_SELECTION_PATH,
            "sft_cache_dir": SFT_CACHE_DIR,
            "legs_root": LEGS_ROOT,
            "output_root": LEGS_ROOT,
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "sft_repo": SFT_REPO,
            "sft_revision": SFT_REVISION,
            "pretrain_tokens": PRETRAIN_TOTAL_TOKENS,
            "sft_rows": SFT_ROWS,
        },
    )
    data_volume.commit()
    return _load_manifest_set()


def _nested_mapping_value(
    payload: Mapping[str, object],
    *keys: str,
) -> object:
    value: object = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(
                f"config snapshot lacks {'.'.join(keys)}"
            )
        value = value[key]
    return value


def _validate_existing_run_identity(
    plan: TrainPlan,
    *,
    manifest_hash: str,
) -> Mapping[str, object]:
    """Bind any retry/completed fast path to this exact v2 source/config."""

    import yaml

    output = Path(plan.output_dir)
    config_path = output / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"immutable run output lacks config snapshot: {config_path}"
        )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid config snapshot: {config_path}")
    expected = {
        ("training", "output_dir"): plan.output_dir,
        ("training", "run_name"): plan.run_name,
        ("training", "total_steps"): plan.total_steps,
        ("training", "arc_steps"): list(plan.arc_steps),
        ("training", "local_batch_size"): plan.local_batch_size,
        ("training", "gradient_accumulation_steps"): 1,
        ("training", "sft_loss_weight"): plan.sft_loss_weight,
        ("data", "expected_manifest_hash"): manifest_hash,
        ("provenance", "experiment_version"): plan.experiment_version,
        ("provenance", "data_artifact_version"): DATA_ARTIFACT_VERSION,
        ("provenance", "source_tree_sha256"): (
            _plan_source_tree_sha256(plan)
        ),
        ("provenance", "sft_response_normalization"): (
            SFT_RESPONSE_NORMALIZATION
        ),
        ("provenance", "sft_supervised_unk_policy"): (
            SFT_SUPERVISED_UNK_POLICY
        ),
    }
    if plan.diagnostic_only:
        expected.update(
            {
                ("training", "max_steps"): plan.max_steps,
                ("training", "snapshot_steps"): list(
                    plan.snapshot_steps
                ),
                ("training", "save_interval"): 0,
                ("training", "export_interval"): 0,
                ("training", "seed"): 42,
                ("training", "optimizer", "lr"): 1e-3,
                ("training", "scheduler", "warmup_ratio"): 0.05,
                ("training", "scheduler", "eta_min"): 1e-5,
            }
        )
    for keys, expected_value in expected.items():
        observed = _nested_mapping_value(payload, *keys)
        if keys == ("training", "sft_loss_weight"):
            matches = float(observed) == float(expected_value)
        else:
            matches = observed == expected_value
        if not matches:
            raise ValueError(
                f"immutable run config drift at {'.'.join(keys)}: "
                f"{observed!r} != {expected_value!r}"
            )
    return payload


def _validate_resume_checkpoint_identity(
    plan: TrainPlan,
    *,
    manifest_hash: str,
    state_dir: Path | None = None,
) -> Mapping[str, object]:
    state_path = (
        state_dir or (Path(plan.output_dir) / "latest")
    ) / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, Mapping):
        raise ValueError(f"invalid trainer state: {state_path}")
    expected = {
        "manifest_hash": manifest_hash,
        "arc_steps": list(plan.arc_steps),
        "local_batch_size": plan.local_batch_size,
        "world_size": plan.num_gpus,
        "gradient_accumulation_steps": 1,
        "sft_loss_weight": plan.sft_loss_weight,
        "attention_backend": plan.attention_backend,
        "torch_compile_mode": plan.torch_compile_mode,
    }
    if plan.diagnostic_only:
        expected["snapshot_steps"] = list(plan.snapshot_steps)
    for key, expected_value in expected.items():
        observed = state.get(key)
        if key == "sft_loss_weight":
            matches = (
                observed is not None
                and float(observed) == float(expected_value)
            )
        else:
            matches = observed == expected_value
        if not matches:
            raise ValueError(
                f"immutable trainer-state drift at {key}: "
                f"{observed!r} != {expected_value!r}"
            )
    return state


def _validate_final_export_identity(
    plan: TrainPlan,
    *,
    manifest_hash: str,
) -> Mapping[str, object]:
    final = Path(plan.output_dir) / "final"
    required_files = (
        "config.json",
        "generation_config.json",
        "tokenizer.py",
        "tokenizer_config.json",
        "vocab.json",
        "interleaved_training_state.json",
    )
    missing = [
        name
        for name in required_files
        if not (final / name).is_file()
    ]
    weight_files = sorted(final.glob("model*.safetensors"))
    if missing or not weight_files:
        raise RuntimeError(
            f"incomplete final HF export under {final}: "
            f"missing={missing}, weights={len(weight_files)}"
        )
    if any(path.stat().st_size <= 0 for path in weight_files):
        raise RuntimeError("final HF export contains an empty weight file")
    latest_state = _validate_resume_checkpoint_identity(
        plan, manifest_hash=manifest_hash
    )
    final_state = json.loads(
        (final / "interleaved_training_state.json").read_text(
            encoding="utf-8"
        )
    )
    if final_state != latest_state:
        raise RuntimeError(
            "final HF training state differs from the resumable latest state"
        )
    tokenizer_config = json.loads(
        (final / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    auto_map = tokenizer_config.get("auto_map")
    if (
        not isinstance(auto_map, Mapping)
        or auto_map.get("AutoTokenizer")
        != ["tokenizer.HFTokenizerWrapper", None]
    ):
        raise RuntimeError(
            "final HF tokenizer is not the required self-contained wrapper"
        )
    return final_state


def _validate_v2r3_snapshot_identity(
    plan: TrainPlan,
    *,
    step: int,
    manifest_hash: str,
    previous_ce_cumulative: Mapping[str, object],
) -> Mapping[str, object]:
    if not plan.diagnostic_only or step not in plan.snapshot_steps:
        raise ValueError("snapshot validation requires a declared v2r3 step")
    root = Path(plan.output_dir) / "snapshots" / f"step_{step}"
    marker_path = root / ".complete.json"
    resume_dir = root / "resume"
    hf_dir = root / "hf"
    if not marker_path.is_file():
        raise FileNotFoundError(marker_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker_core = {
        key: value
        for key, value in marker.items()
        if key != "marker_sha256"
    }
    if (
        marker_core.get("schema")
        != "interleaved-diagnostic-snapshot-v1"
        or marker_core.get("global_step") != step
        or marker.get("marker_sha256")
        != _canonical_mapping_sha256(marker_core)
    ):
        raise RuntimeError(f"invalid diagnostic snapshot marker: {root}")
    observed_resume_identity = _directory_file_identity(resume_dir)
    observed_hf_identity = _directory_file_identity(hf_dir)
    if marker_core.get("resume_identity") != observed_resume_identity:
        raise RuntimeError(
            "diagnostic snapshot resume-file inventory/hash mismatch"
        )
    if marker_core.get("hf_identity") != observed_hf_identity:
        raise RuntimeError(
            "diagnostic snapshot HF-file inventory/hash mismatch"
        )
    resume_state = _validate_resume_checkpoint_identity(
        plan,
        manifest_hash=manifest_hash,
        state_dir=resume_dir,
    )
    if (
        int(resume_state.get("global_step", -1)) != step
        or int(resume_state.get("manifest_cursor", -1)) != step
    ):
        raise RuntimeError(
            f"diagnostic snapshot state does not end at step {step}"
        )
    hf_required = (
        "config.json",
        "generation_config.json",
        "tokenizer.py",
        "tokenizer_config.json",
        "vocab.json",
        "interleaved_training_state.json",
    )
    missing = [name for name in hf_required if not (hf_dir / name).is_file()]
    weights = sorted(hf_dir.glob("model*.safetensors"))
    if missing or not weights or any(path.stat().st_size <= 0 for path in weights):
        raise RuntimeError(
            f"incomplete diagnostic HF snapshot {hf_dir}: "
            f"missing={missing}, weights={len(weights)}"
        )
    hf_state = json.loads(
        (hf_dir / "interleaved_training_state.json").read_text(
            encoding="utf-8"
        )
    )
    if hf_state != resume_state:
        raise RuntimeError(
            "diagnostic snapshot clean-HF and resume states differ"
        )
    interval = resume_state.get("diagnostic_last_ce_interval")
    cumulative = resume_state.get("diagnostic_ce_cumulative")
    interval_base = resume_state.get("diagnostic_ce_interval_base")
    last_interval_base = resume_state.get(
        "diagnostic_last_ce_interval_base"
    )
    snapshot_index = plan.snapshot_steps.index(step)
    previous_step = (
        plan.snapshot_steps[snapshot_index - 1]
        if snapshot_index > 0
        else 0
    )
    if (
        not isinstance(interval, Mapping)
        or not isinstance(cumulative, Mapping)
        or not isinstance(interval_base, Mapping)
        or not isinstance(last_interval_base, Mapping)
        or interval_base != cumulative
        or cumulative.get("through_step") != step
        or cumulative.get("schema")
        != "interleaved-diagnostic-ce-cumulative-v1"
        or last_interval_base.get("schema")
        != "interleaved-diagnostic-ce-cumulative-v1"
        or last_interval_base != previous_ce_cumulative
        or last_interval_base.get("through_step") != previous_step
        or marker_core.get("interval_unweighted_ce") != interval
        or interval.get("schema")
        != "interleaved-diagnostic-ce-interval-v1"
        or interval.get("measurement_semantics")
        != "token_weighted_training_stream_pre_update_batch_logits"
        or interval.get("held_out") is not False
        or interval.get("endpoint_checkpoint_evaluation") is not False
        or interval.get("start_step") != previous_step + 1
        or interval.get("end_step") != step
        or interval.get("optimizer_steps") != step - previous_step
    ):
        raise RuntimeError(
            "diagnostic snapshot CE interval/state boundary mismatch"
        )
    for prefix in ("pretrain", "sft"):
        loss_sum = interval.get(f"{prefix}_loss_sum")
        token_count = interval.get(f"{prefix}_token_count")
        contributing_steps = interval.get(
            f"{prefix}_contributing_steps"
        )
        token_ce = interval.get(f"{prefix}_token_ce")
        expected_loss_sum = float(
            cumulative[f"{prefix}_loss_sum"]
        ) - float(previous_ce_cumulative[f"{prefix}_loss_sum"])
        expected_token_count = int(
            cumulative[f"{prefix}_token_count"]
        ) - int(previous_ce_cumulative[f"{prefix}_token_count"])
        expected_contributing_steps = int(
            cumulative[f"{prefix}_contributing_steps"]
        ) - int(
            previous_ce_cumulative[f"{prefix}_contributing_steps"]
        )
        if (
            isinstance(loss_sum, bool)
            or not isinstance(loss_sum, (int, float))
            or not math.isfinite(float(loss_sum))
            or float(loss_sum) < 0
            or isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count <= 0
            or isinstance(contributing_steps, bool)
            or not isinstance(contributing_steps, int)
            or contributing_steps <= 0
            or contributing_steps > int(interval["optimizer_steps"])
            or float(loss_sum) != expected_loss_sum
            or token_count != expected_token_count
            or contributing_steps != expected_contributing_steps
            or isinstance(token_ce, bool)
            or not isinstance(token_ce, (int, float))
            or not math.isclose(
                float(token_ce),
                float(loss_sum) / token_count,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
        ):
            raise RuntimeError(
                f"diagnostic snapshot {prefix} CE interval is invalid"
            )
    if marker_core.get("trainer_state_sha256") != _canonical_mapping_sha256(
        resume_state
    ):
        raise RuntimeError("diagnostic snapshot trainer-state hash mismatch")
    return {
        "step": step,
        "root": str(root),
        "resume_path": str(resume_dir),
        "hf_path": str(hf_dir),
        "trainer_state_sha256": marker_core["trainer_state_sha256"],
        "hf_manifest_sha256": _directory_manifest_sha256(hf_dir),
        "resume_identity": observed_resume_identity,
        "hf_identity": observed_hf_identity,
        "interval_unweighted_ce": dict(interval),
        "diagnostic_ce_cumulative": dict(cumulative),
    }


def _initial_v2r3_ce_cumulative() -> Mapping[str, object]:
    return {
        "schema": "interleaved-diagnostic-ce-cumulative-v1",
        "through_step": 0,
        "pretrain_loss_sum": 0.0,
        "pretrain_token_count": 0,
        "pretrain_contributing_steps": 0,
        "sft_loss_sum": 0.0,
        "sft_token_count": 0,
        "sft_contributing_steps": 0,
    }


def _validate_v2r3_snapshot_prefix(
    plan: TrainPlan,
    *,
    manifest_hash: str,
    steps: Sequence[int] | None = None,
) -> list[Mapping[str, object]]:
    """Validate an exact declared prefix and every CE delta between snapshots."""

    selected_steps = tuple(plan.snapshot_steps if steps is None else steps)
    if selected_steps != plan.snapshot_steps[: len(selected_steps)]:
        raise ValueError(
            "v2r3 snapshot validation requires an exact declared prefix"
        )
    previous_cumulative = _initial_v2r3_ce_cumulative()
    snapshots: list[Mapping[str, object]] = []
    for step in selected_steps:
        snapshot = _validate_v2r3_snapshot_identity(
            plan,
            step=step,
            manifest_hash=manifest_hash,
            previous_ce_cumulative=previous_cumulative,
        )
        snapshots.append(snapshot)
        previous_cumulative = snapshot["diagnostic_ce_cumulative"]
    return snapshots


def _directory_manifest_sha256(root: Path) -> str:
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        rows.append(
            f"{relative}\t{path.stat().st_size}\t"
            f"{_manifest_file_sha256(path)}\n"
        )
    if not rows:
        raise RuntimeError(f"identity directory is empty: {root}")
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def _directory_file_identity(root: Path) -> Mapping[str, object]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    files: list[Mapping[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _manifest_file_sha256(path),
            }
        )
    if not files:
        raise RuntimeError(f"identity directory is empty: {root}")
    return {
        "file_count": len(files),
        "total_bytes": sum(int(row["bytes"]) for row in files),
        "files": files,
        "manifest_sha256": _v2r2_json_sha256(files),
    }


def _require_successful_modal_call(call_id: str) -> object:
    if not re.fullmatch(r"fc-[A-Za-z0-9]+", call_id):
        raise ValueError(f"invalid Modal function-call ID: {call_id!r}")
    call = modal.functions.FunctionCall.from_id(call_id)
    statuses = [int(node.status) for node in call.get_call_graph()]
    if not statuses or any(status != 1 for status in statuses):
        raise RuntimeError(
            f"Modal call {call_id} is not completely successful: {statuses}"
        )
    return call.get(timeout=10)


def _validate_gate_metrics(metrics: Mapping[str, object]) -> None:
    exact = {
        "rollout_rows": 2_048,
        "prompt_groups": 256,
        "samples_per_group": 8,
    }
    for key, expected in exact.items():
        if int(metrics.get(key, -1)) != expected:
            raise RuntimeError(
                f"structure gate metric {key} != {expected}: "
                f"{metrics.get(key)!r}"
            )
    positive_keys = (
        "outputs_with_end_thinking",
        "outputs_with_call_env",
        "rows_with_parsed_moves",
        "positive_samples",
        "nonzero_variance_groups",
    )
    for key in positive_keys:
        if int(metrics.get(key, 0)) <= 0:
            raise RuntimeError(
                f"structure gate requires positive {key}, got "
                f"{metrics.get(key)!r}"
            )


def _validate_production_gate(
    manifest_set: Mapping[str, object],
) -> Mapping[str, object]:
    if not PRODUCTION_GATE_PATH.is_file():
        raise FileNotFoundError(
            "v2 production training is blocked until the exact 500-update "
            f"structure/rollout gate is approved at {PRODUCTION_GATE_PATH}"
        )
    gate = json.loads(PRODUCTION_GATE_PATH.read_text(encoding="utf-8"))
    if not isinstance(gate, Mapping):
        raise RuntimeError("production gate marker is not an object")
    recorded_hash = gate.get("gate_sha256")
    unhashed = {
        str(key): value
        for key, value in gate.items()
        if key != "gate_sha256"
    }
    if recorded_hash != _canonical_mapping_sha256(unhashed):
        raise RuntimeError("production gate marker self-hash mismatch")
    expected = {
        "schema": "interleaved-v2-production-gate-v1",
        "approved": True,
        "experiment_version": EXPERIMENT_VERSION,
        "data_artifact_version": DATA_ARTIFACT_VERSION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "manifest_set_hash": manifest_set.get("manifest_set_hash"),
        "candidate_sft_loss_weight": P1_SFT_LOSS_WEIGHT,
        "candidate_steps": SFT_WEIGHT_CANARY_DEFAULT_STEPS,
    }
    for key, expected_value in expected.items():
        observed = gate.get(key)
        if key == "candidate_sft_loss_weight":
            matches = float(observed) == float(expected_value)
        else:
            matches = observed == expected_value
        if not matches:
            raise RuntimeError(
                f"production gate drift at {key}: "
                f"{observed!r} != {expected_value!r}"
            )
    metrics = gate.get("metrics")
    if not isinstance(metrics, Mapping):
        raise RuntimeError("production gate lacks metrics")
    _validate_gate_metrics(metrics)
    artifact_hashes = gate.get("rollout_artifact_sha256")
    if (
        not isinstance(artifact_hashes, Mapping)
        or set(artifact_hashes)
        != {
            "run_provenance.json",
            "rollout_0.jsonl",
            "all_attempts_positive_rollout_0.jsonl",
            "all_attempts_positive_rollout_0.summary.json",
        }
        or any(
            not isinstance(value, str)
            or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in artifact_hashes.values()
        )
    ):
        raise RuntimeError(
            "production gate lacks exact rollout artifact hashes"
        )
    return gate


def _inspect_rollout_gate(
    *,
    candidate_final: Path,
    rollout_run_name: str,
    expected_seed: int = 42,
    require_legacy_positive_gate: bool = True,
    expected_chess_source_sha256: str = RL_GATE_CHESS_SOURCE_SHA256,
    expected_deterministic_inference: bool | None = None,
    expected_rollout_only: bool | None = None,
    allowed_rollout_statuses: Sequence[str] = ("completed",),
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, str],
]:
    run_root = RL_GATE_ROOT / _validate_run_id(rollout_run_name)
    provenance_path = run_root / "run_provenance.json"
    rollout_path = (
        run_root / "rollouts" / "training" / "rollout_0.jsonl"
    )
    positive_summary_path = (
        run_root
        / "rollouts"
        / "all_attempts_positive"
        / "rollout_0.summary.json"
    )
    positive_rows_path = (
        run_root
        / "rollouts"
        / "all_attempts_positive"
        / "rollout_0.jsonl"
    )
    for path in (
        provenance_path,
        rollout_path,
        positive_rows_path,
        positive_summary_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    identity = provenance.get("identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("RL gate provenance lacks identity")
    if provenance.get("identity_sha256") != _canonical_mapping_sha256(
        identity
    ):
        raise RuntimeError("RL gate provenance identity hash mismatch")
    initial_command = provenance.get("initial_command")
    if (
        not isinstance(initial_command, list)
        or not all(isinstance(value, str) for value in initial_command)
    ):
        raise RuntimeError("RL gate provenance lacks its initial command")
    command_sha256 = hashlib.sha256(
        json.dumps(
            initial_command,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if provenance.get("initial_command_sha256") != command_sha256:
        raise RuntimeError("RL gate provenance command hash mismatch")
    run = identity.get("run")
    policy = identity.get("policy_update_profile")
    semantics = identity.get("fixed_rl_semantics")
    balanced = identity.get("balanced_data")
    origin = identity.get("origin_hf")
    sources = identity.get("sources")
    if not all(
        isinstance(value, Mapping)
        for value in (run, policy, semantics, balanced, origin, sources)
    ):
        raise RuntimeError("RL gate provenance identity is incomplete")
    expected_run = {
        "run_name": rollout_run_name,
        "app_name": "chess-interleave-rl",
        "model_id": "interleave_47m_qwen3",
        "num_rollout": 1,
        "dynamic_filter": False,
        "rollout_seed": expected_seed,
        "save_interval": 0,
        "eval_interval": 0,
        "canary": True,
    }
    for key, expected_value in expected_run.items():
        if run.get(key) != expected_value:
            raise RuntimeError(
                f"RL gate provenance drift at run.{key}: "
                f"{run.get(key)!r} != {expected_value!r}"
            )
    deterministic_flag = "--sglang-enable-deterministic-inference"
    if expected_deterministic_inference is not None:
        if (
            run.get("deterministic_inference")
            is not expected_deterministic_inference
            or (deterministic_flag in initial_command)
            is not expected_deterministic_inference
        ):
            raise RuntimeError(
                "RL gate deterministic-inference provenance/command drift"
            )
    rollout_only_flag = "--debug-rollout-only"
    if expected_rollout_only is not None:
        if (
            run.get("rollout_only") is not expected_rollout_only
            or (rollout_only_flag in initial_command)
            is not expected_rollout_only
        ):
            raise RuntimeError(
                "RL gate rollout-only provenance/command drift"
            )
    expected_policy = {
        "name": "small-model-h200",
        "max_tokens_per_gpu": 131_072,
        "gradient_checkpointing": False,
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 8,
        "gpu_type": "H200",
        "host_memory_gb": 192,
        "sglang_server_concurrency": 128,
    }
    for key, expected_value in expected_policy.items():
        if policy.get(key) != expected_value:
            raise RuntimeError(
                f"RL gate provenance drift at policy.{key}: "
                f"{policy.get(key)!r} != {expected_value!r}"
            )
    expected_semantics = {
        "rollout_batch_size": 256,
        "samples_per_prompt": 8,
        "global_batch_size": 2_048,
        "policy_loss_agg_mode": "token-mean",
        "advantage_estimator": "grpo",
        "cispo": False,
        "lr": 1e-5,
        "rollout_max_prompt_len": 512,
        "rollout_max_response_len": 2_560,
        "rollout_max_context_len": 3_072,
    }
    for key, expected_value in expected_semantics.items():
        if semantics.get(key) != expected_value:
            raise RuntimeError(
                f"RL gate provenance drift at semantics.{key}: "
                f"{semantics.get(key)!r} != {expected_value!r}"
            )
    if expected_deterministic_inference is not None:
        expected_seed_rule = (
            "rollout_seed_plus_sibling_sample_index_0_to_7"
            if expected_deterministic_inference
            else "backend_default"
        )
        if semantics.get("sampling_seed_rule") != expected_seed_rule:
            raise RuntimeError(
                "RL gate deterministic sampling-seed rule drift"
            )
    if expected_rollout_only is not None:
        expected_update_mode = (
            "disabled_rollout_only"
            if expected_rollout_only
            else "one_or_more_optimizer_updates"
        )
        if semantics.get("policy_update_mode") != expected_update_mode:
            raise RuntimeError("RL gate policy-update mode drift")
    if (
        balanced.get("logical_path")
        != "/data/chess-rl-data/train_v4_dataset_balanced_multi_turn.parquet"
    ):
        raise RuntimeError("RL gate used the wrong balanced dataset path")
    if balanced.get("sha256") != RL_GATE_BALANCED_SHA256:
        raise RuntimeError("RL gate used the wrong balanced dataset")
    expected_sources = {
        "chess_rl_miles": expected_chess_source_sha256,
        "miles": RL_GATE_MILES_SOURCE_SHA256,
    }
    for source_name, expected_sha256 in expected_sources.items():
        source = sources.get(source_name)
        if (
            not isinstance(source, Mapping)
            or source.get("manifest_sha256") != expected_sha256
        ):
            raise RuntimeError(
                f"RL gate source drift at {source_name}: {source!r}"
            )
    candidate_manifest = _directory_manifest_sha256(candidate_final)
    if origin.get("manifest_sha256") != candidate_manifest:
        raise RuntimeError(
            "RL gate origin-HF identity does not match the canary final"
        )
    expected_origin_path = str(candidate_final).replace(
        "/checkpoints/", "/pretrain-checkpoints/", 1
    )
    if origin.get("logical_path") != expected_origin_path:
        raise RuntimeError(
            "RL gate origin-HF logical path does not match the candidate"
        )

    if isinstance(allowed_rollout_statuses, (str, bytes)):
        raise ValueError(
            "allowed_rollout_statuses must be a sequence of statuses"
        )
    allowed_status_tuple = tuple(allowed_rollout_statuses)
    if (
        not allowed_status_tuple
        or len(set(allowed_status_tuple)) != len(allowed_status_tuple)
        or any(
            not isinstance(status, str) or not status
            for status in allowed_status_tuple
        )
    ):
        raise ValueError(
            "allowed_rollout_statuses must be unique nonempty strings"
        )
    allowed_status_set = set(allowed_status_tuple)
    status_counts = {status: 0 for status in allowed_status_tuple}
    rollout_rows = 0
    outputs_with_end_thinking = 0
    outputs_with_call_env = 0
    rows_with_parsed_moves = 0
    positive_samples = 0
    positive_completed_samples = 0
    group_scores: dict[int, list[float]] = {}
    with rollout_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise RuntimeError(
                    f"rollout gate row {line_number} is not an object"
                )
            status = row.get("status")
            if status not in allowed_status_set:
                if allowed_status_tuple == ("completed",):
                    raise RuntimeError(
                        f"rollout gate row {line_number} is not completed"
                    )
                raise RuntimeError(
                    f"rollout gate row {line_number} has disallowed status "
                    f"{status!r}"
                )
            status_counts[str(status)] += 1
            output = str(row.get("output") or "")
            outputs_with_end_thinking += int("</T>" in output)
            outputs_with_call_env += int("<call_env>" in output)
            rows_with_parsed_moves += int(
                bool(str(row.get("extracted_moves") or "").strip())
            )
            score = float(row.get("score", 0.0))
            if not math.isfinite(score):
                raise RuntimeError("rollout gate contains a non-finite reward")
            positive_samples += int(score == 1.0)
            positive_completed_samples += int(
                score == 1.0 and status == "completed"
            )
            group_index = int(row["group_index"])
            group_scores.setdefault(group_index, []).append(score)
            rollout_rows += 1

    group_sizes = {len(scores) for scores in group_scores.values()}
    if group_sizes != {8}:
        raise RuntimeError(
            f"rollout gate group sizes are not exactly 8: {group_sizes}"
        )
    nonzero_variance_groups = sum(
        1
        for scores in group_scores.values()
        if min(scores) < max(scores)
    )
    positive_summary = json.loads(
        positive_summary_path.read_text(encoding="utf-8")
    )
    if (
        int(positive_summary.get("attempted_groups", -1))
        != len(group_scores)
        or int(positive_summary.get("attempted_samples", -1))
        != rollout_rows
        or int(positive_summary.get("completed_samples", -1))
        != status_counts.get("completed", 0)
        or int(positive_summary.get("positive_completed_samples", -1))
        != positive_completed_samples
    ):
        raise RuntimeError(
            "all-attempts-positive summary disagrees with the fixed rollout"
        )
    metrics = {
        "rollout_rows": rollout_rows,
        "prompt_groups": len(group_scores),
        "samples_per_group": next(iter(group_sizes), 0),
        "outputs_with_end_thinking": outputs_with_end_thinking,
        "outputs_with_call_env": outputs_with_call_env,
        "rows_with_parsed_moves": rows_with_parsed_moves,
        "allowed_statuses": list(allowed_status_tuple),
        "status_counts": status_counts,
        "positive_samples": positive_samples,
        "positive_completed_samples": positive_completed_samples,
        "nonzero_variance_groups": nonzero_variance_groups,
    }
    if require_legacy_positive_gate:
        _validate_gate_metrics(metrics)
    return (
        metrics,
        provenance,
        {
            "run_provenance.json": _manifest_file_sha256(
                provenance_path
            ),
            "rollout_0.jsonl": _manifest_file_sha256(rollout_path),
            "all_attempts_positive_rollout_0.jsonl": (
                _manifest_file_sha256(positive_rows_path)
            ),
            "all_attempts_positive_rollout_0.summary.json": (
                _manifest_file_sha256(positive_summary_path)
            ),
        },
    )


def _inspect_v2r2_rollout_audit(
    *,
    candidate_final: Path,
    rollout_run_name: str,
    seed: int,
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, str],
]:
    """Authenticate a Miles rollout and apply the frozen joint-row audit."""

    _, provenance, artifact_hashes = _inspect_rollout_gate(
        candidate_final=candidate_final,
        rollout_run_name=rollout_run_name,
        expected_seed=seed,
        require_legacy_positive_gate=False,
    )
    rollout_path = (
        RL_GATE_ROOT
        / _validate_run_id(rollout_run_name)
        / "rollouts"
        / "training"
        / "rollout_0.jsonl"
    )
    rows: list[Mapping[str, object]] = []
    with rollout_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise RuntimeError(
                    f"rollout audit row {line_number} is not an object"
                )
            rows.append(row)
    metrics = _audit_v2r2_rollout_rows(rows, seed=seed)
    return metrics, provenance, artifact_hashes


def _v2r3_rollout_run_name(weight: float, step: int) -> str:
    plan = _v2r3_plan(weight)
    if step not in plan.snapshot_steps:
        raise ValueError(
            f"step {step} is not declared for v2r3 weight {weight}"
        )
    return (
        f"v2r3-diag-w{_weight_slug(weight)}-s{step}-seed42-rollout"
    )


def _inspect_v2r3_rollout_audit(
    *,
    candidate_hf: Path,
    rollout_run_name: str,
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, str],
]:
    """Authenticate one seed-42 diagnostic against the audited new source."""

    _, provenance, artifact_hashes = _inspect_rollout_gate(
        candidate_final=candidate_hf,
        rollout_run_name=rollout_run_name,
        expected_seed=V2R3_PRIMARY_SEED,
        require_legacy_positive_gate=False,
        expected_chess_source_sha256=V2R3_RL_CHESS_SOURCE_SHA256,
        expected_deterministic_inference=True,
        expected_rollout_only=True,
        allowed_rollout_statuses=("completed", "truncated"),
    )
    rollout_path = (
        RL_GATE_ROOT
        / _validate_run_id(rollout_run_name)
        / "rollouts"
        / "training"
        / "rollout_0.jsonl"
    )
    rows: list[Mapping[str, object]] = []
    with rollout_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise RuntimeError(
                    f"v2r3 rollout row {line_number} is not an object"
                )
            rows.append(row)
    metrics = _audit_v2r3_diagnostic_rows(
        rows, seed=V2R3_PRIMARY_SEED
    )
    if (
        metrics.get("prompt_set_sha256")
        != V2R3_SEED42_PROMPT_SET_SHA256
    ):
        raise RuntimeError(
            "v2r3 seed-42 rollout does not use the frozen prompt set"
        )
    return metrics, provenance, artifact_hashes


def _unweighted_ce_at_step(
    plan: TrainPlan,
    *,
    expected_step: int,
    manifest_hash: str,
    expected_runtime_provenance: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Return exact-step unweighted PT/SFT token CE from the append-only log."""

    metrics_path = Path(plan.output_dir) / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    if (
        isinstance(expected_step, bool)
        or not isinstance(expected_step, int)
        or expected_step <= 0
        or expected_step > (plan.max_steps or plan.total_steps)
    ):
        raise ValueError("expected CE step is outside the trajectory")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_hash):
        raise ValueError("metric manifest_hash must be a full SHA-256")
    selected: Mapping[str, object] | None = None
    last_step = -1
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise RuntimeError(
                    f"metrics row {line_number} is not an object"
                )
            if record.get("schema") != "interleaved-local-metrics-v1":
                raise RuntimeError(
                    f"metrics row {line_number} has the wrong schema"
                )
            if record.get("manifest_hash") != manifest_hash:
                raise RuntimeError(
                    f"metrics row {line_number} manifest hash drifted"
                )
            raw_step = record.get("step")
            if isinstance(raw_step, bool) or not isinstance(raw_step, int):
                raise RuntimeError(
                    f"metrics row {line_number} step is not an exact integer"
                )
            step = raw_step
            if step <= last_step:
                raise RuntimeError("metric steps are not strictly increasing")
            last_step = step
            runtime = record.get("runtime_provenance")
            if not isinstance(runtime, Mapping):
                raise RuntimeError(
                    f"metrics row {line_number} lacks runtime provenance"
                )
            configured = runtime.get("configured_provenance")
            if not isinstance(configured, Mapping):
                raise RuntimeError(
                    f"metrics row {line_number} lacks configured provenance"
                )
            expected_runtime = {
                "attention_backend": plan.attention_backend,
                "torch_compile_mode": plan.torch_compile_mode,
                "data_num_workers": plan.data_workers,
                "sft_loss_weight": plan.sft_loss_weight,
            }
            for key, expected_value in expected_runtime.items():
                observed = runtime.get(key)
                matches = (
                    float(observed) == float(expected_value)
                    if key == "sft_loss_weight" and observed is not None
                    else observed == expected_value
                )
                if not matches:
                    raise RuntimeError(
                        f"metrics row {line_number} runtime {key} drifted"
                    )
            expected_configured = {
                "experiment_version": plan.experiment_version,
                "data_artifact_version": DATA_ARTIFACT_VERSION,
                "source_repo": SOURCE_REPO,
                "source_revision": SOURCE_REVISION,
                "source_flat_manifest_sha256": (
                    SOURCE_FLAT_MANIFEST_SHA256
                ),
                "sft_repo": SFT_REPO,
                "sft_revision": SFT_REVISION,
                "attention_backend": plan.attention_backend,
                "torch_compile_mode": plan.torch_compile_mode,
                "data_num_workers": plan.data_workers,
                "sft_loss_weight": plan.sft_loss_weight,
                "sft_response_normalization": (
                    SFT_RESPONSE_NORMALIZATION
                ),
                "sft_supervised_unk_policy": (
                    SFT_SUPERVISED_UNK_POLICY
                ),
                "metrics_format": PRODUCTION_METRICS_FORMAT,
                "source_tree_sha256": _plan_source_tree_sha256(plan),
            }
            for key, expected_value in expected_configured.items():
                observed = configured.get(key)
                matches = (
                    float(observed) == float(expected_value)
                    if key == "sft_loss_weight" and observed is not None
                    else observed == expected_value
                )
                if not matches:
                    raise RuntimeError(
                        f"metrics row {line_number} configured {key} drifted"
                    )
            if (
                expected_runtime_provenance is not None
                and runtime != expected_runtime_provenance
            ):
                raise RuntimeError(
                    f"metrics row {line_number} runtime provenance differs "
                    "from the authenticated snapshot"
                )
            if step == expected_step:
                selected = record
    if selected is None:
        raise RuntimeError(
            f"metrics lack the exact candidate/requested step {expected_step}"
        )
    values = selected.get("metrics")
    if not isinstance(values, Mapping):
        raise RuntimeError("candidate metric row lacks metrics")
    result: dict[str, object] = {
        "step": expected_step,
        "metrics_file_sha256": _manifest_file_sha256(metrics_path),
        "metric_record_sha256": _v2r2_json_sha256(selected),
        "manifest_hash": manifest_hash,
        "runtime_provenance_sha256": _v2r2_json_sha256(
            _require_mapping(
                selected.get("runtime_provenance"),
                label="selected metric runtime_provenance",
            )
        ),
    }
    for key in ("train/pretrain_token_loss", "train/sft_token_loss"):
        raw = values.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RuntimeError(f"candidate metrics lack numeric {key}")
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise RuntimeError(f"candidate metric {key} is invalid: {raw!r}")
        result[key] = value
    return result


def _last_unweighted_ce(
    plan: TrainPlan,
    *,
    manifest_hash: str,
) -> Mapping[str, object]:
    return _unweighted_ce_at_step(
        plan,
        expected_step=plan.max_steps or plan.total_steps,
        manifest_hash=manifest_hash,
    )


def _tokenizer_manifest_sha256(candidate_final: Path) -> str:
    files = (
        "special_tokens_map.json",
        "tokenizer.py",
        "tokenizer_config.json",
        "vocab.json",
    )
    rows: list[str] = []
    for name in files:
        path = candidate_final / name
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            f"{name}\t{path.stat().st_size}\t"
            f"{_manifest_file_sha256(path)}\n"
        )
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def _v2r2_prompt_selection_proof(
    *,
    candidate_final: Path,
    primary_metrics: Mapping[str, object],
) -> Mapping[str, object]:
    """Recompute Miles epoch-0 shuffle and prove the first disjoint seed."""

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    if _manifest_file_sha256(RL_GATE_BALANCED_PATH) != RL_GATE_BALANCED_SHA256:
        raise RuntimeError("balanced parquet drifted during prompt selection")
    table = pq.read_table(RL_GATE_BALANCED_PATH)
    if table.num_rows != RL_GATE_BALANCED_ROWS:
        raise RuntimeError(
            f"balanced parquet rows drifted: {table.num_rows}"
        )
    rows = table.to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(
        candidate_final,
        trust_remote_code=True,
        local_files_only=True,
    )
    prompts = [str(row.get("prompt") or "") for row in rows]
    tokenized = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    eligible = [
        row
        for row, input_ids in zip(rows, tokenized, strict=True)
        if len(input_ids) <= 512
    ]
    if len(eligible) != RL_GATE_ELIGIBLE_PROMPT_ROWS:
        raise RuntimeError(
            f"Miles-eligible prompt rows drifted: {len(eligible)}"
        )
    if len(rows) - len(eligible) != RL_GATE_FILTERED_LONG_PROMPT_ROWS:
        raise RuntimeError("Miles long-prompt filter count drifted")

    fingerprints: list[str] = []
    for row in eligible:
        label = row.get("reward_model")
        metadata = row.get("extra_info")
        if not isinstance(label, Mapping) or not isinstance(metadata, Mapping):
            raise RuntimeError("balanced row lacks reward/metadata identity")
        fingerprints.append(
            _v2r2_json_sha256(
                {
                    "input": str(row.get("prompt") or ""),
                    "FEN": str(metadata.get("FEN") or ""),
                    "PuzzleId": str(metadata.get("PuzzleId") or ""),
                    "ground_truth": str(label.get("ground_truth") or ""),
                }
            )
        )

    def selected_prompt_set(seed: int) -> list[str]:
        indices = list(range(len(fingerprints)))
        random.Random(seed).shuffle(indices)
        selected = sorted(fingerprints[index] for index in indices[:256])
        if len(set(selected)) != 256:
            raise RuntimeError(f"Miles seed {seed} selected duplicate prompts")
        return selected

    primary_prompts = selected_prompt_set(V2R2_PRIMARY_SEED)
    if (
        primary_metrics.get("prompt_fingerprints") != primary_prompts
        or primary_metrics.get("prompt_set_sha256")
        != _v2r2_json_sha256(primary_prompts)
    ):
        raise RuntimeError(
            "actual primary rollout prompts differ from Miles selection"
        )
    candidates: list[Mapping[str, object]] = []
    primary_set = set(primary_prompts)
    seed = V2R2_CONFIRMATION_FIRST_SEED
    while seed < V2R2_CONFIRMATION_FIRST_SEED + 10_000:
        prompts_for_seed = selected_prompt_set(seed)
        candidates.append(
            {
                "seed": seed,
                "prompt_fingerprints": prompts_for_seed,
                "prompt_set_sha256": _v2r2_json_sha256(prompts_for_seed),
            }
        )
        if not primary_set.intersection(prompts_for_seed):
            break
        seed += 1
    selection = _select_v2r2_prompt_set(primary_metrics, candidates)
    return {
        "schema": V2R2_PROMPT_SELECTION_SCHEMA,
        "balanced_data_path": str(RL_GATE_BALANCED_PATH),
        "balanced_data_sha256": RL_GATE_BALANCED_SHA256,
        "balanced_data_rows": RL_GATE_BALANCED_ROWS,
        "eligible_prompt_rows": RL_GATE_ELIGIBLE_PROMPT_ROWS,
        "filtered_long_prompt_rows": RL_GATE_FILTERED_LONG_PROMPT_ROWS,
        "candidate_tokenizer_manifest_sha256": (
            _tokenizer_manifest_sha256(candidate_final)
        ),
        "reference_chess_source_sha256": RL_GATE_CHESS_SOURCE_SHA256,
        "reference_miles_source_sha256": RL_GATE_MILES_SOURCE_SHA256,
        "selector_source_tree_sha256": SOURCE_TREE_SHA256,
        "selector_function_sha256": hashlib.sha256(
            inspect.getsource(_v2r2_prompt_selection_proof).encode("utf-8")
        ).hexdigest(),
        "primary_prompt_set_sha256": _v2r2_json_sha256(primary_prompts),
        "candidate_prompt_sets": candidates,
        "selection": selection,
    }


def _decode_json_object(raw: str, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must decode to an object")
    return value


def _decode_json_object_list(
    raw: str, *, label: str
) -> list[Mapping[str, object]]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, Mapping) for item in value)
    ):
        raise ValueError(f"{label} must decode to a nonempty object list")
    return list(value)


def _required_spec_string(
    spec: Mapping[str, object], key: str, *, label: str
) -> str:
    value = spec.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{key} must be a nonempty string")
    return value


def _v2r1_candidate_plan(spec: Mapping[str, object]) -> TrainPlan:
    weight = float(spec.get("weight", float("nan")))
    if weight not in V2R2_ELIGIBLE_SFT_LOSS_WEIGHTS:
        raise ValueError(f"candidate has ineligible weight {weight}")
    plan = _sft_weight_canary_plan(
        _required_spec_string(spec, "run_id", label="candidate"),
        sft_loss_weight=weight,
        max_steps=V2R2_PROTOCOL_CANDIDATE_STEP,
    )
    return TrainPlan(
        **{
            **plan.__dict__,
            "source_tree_sha256": V2R1_AUDITED_SOURCE_TREE_SHA256,
        }
    )


def _inspect_v2r2_rollout_spec(
    *,
    candidate_final: Path,
    spec: Mapping[str, object],
    expected_seed: int,
    label: str,
) -> Mapping[str, object]:
    seed = spec.get("seed")
    if seed != expected_seed:
        raise ValueError(
            f"{label}.seed must equal {expected_seed}, got {seed!r}"
        )
    run_name = _validate_run_id(
        _required_spec_string(spec, "run_name", label=label)
    )
    call_id = _required_spec_string(spec, "call_id", label=label)
    call_result = _require_successful_modal_call(call_id)
    if not isinstance(call_result, Mapping):
        raise RuntimeError(f"{label} successful call returned no identity")
    metrics, provenance, hashes = _inspect_v2r2_rollout_audit(
        candidate_final=candidate_final,
        rollout_run_name=run_name,
        seed=expected_seed,
    )
    expected_result = {
        "run_name": run_name,
        "checkpoint_root": str(RL_GATE_ROOT / run_name),
        "num_rollout": 1,
        "dynamic_filter": False,
        "rollout_seed": expected_seed,
    }
    for key, expected_value in expected_result.items():
        if call_result.get(key) != expected_value:
            raise RuntimeError(
                f"{label} call-result drift at {key}: "
                f"{call_result.get(key)!r} != {expected_value!r}"
            )
    result_provenance = call_result.get("provenance")
    if (
        not isinstance(result_provenance, Mapping)
        or result_provenance.get("identity_sha256")
        != provenance.get("identity_sha256")
    ):
        raise RuntimeError(
            f"{label} call result is not bound to rollout provenance"
        )
    return {
        "run_name": run_name,
        "call_id": call_id,
        "seed": expected_seed,
        "identity_sha256": provenance.get("identity_sha256"),
        "call_result_sha256": _v2r2_json_sha256(call_result),
        "artifact_sha256": dict(hashes),
        "metrics": dict(metrics),
    }


def _confirmation_spec(
    spec: Mapping[str, object], *, label: str
) -> Mapping[str, object] | None:
    value = spec.get("confirmation")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}.confirmation must be an object")
    return value


def _inspect_v2r2_audit_pair(
    *,
    candidate_final: Path,
    spec: Mapping[str, object],
    label: str,
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
]:
    primary_spec = spec.get("primary")
    if not isinstance(primary_spec, Mapping):
        raise ValueError(f"{label}.primary must be an object")
    primary_record = _inspect_v2r2_rollout_spec(
        candidate_final=candidate_final,
        spec=primary_spec,
        expected_seed=V2R2_PRIMARY_SEED,
        label=f"{label}.primary",
    )
    primary_metrics = primary_record["metrics"]
    if not isinstance(primary_metrics, Mapping):
        raise RuntimeError("primary audit record lacks metrics")
    prompt_selection = _v2r2_prompt_selection_proof(
        candidate_final=candidate_final,
        primary_metrics=primary_metrics,
    )
    selection = _require_mapping(
        prompt_selection.get("selection"),
        label=f"{label}.prompt_selection.selection",
    )
    expected_seed = int(selection["confirmation_seed"])
    confirmation_spec = _confirmation_spec(spec, label=label)
    if confirmation_spec is None:
        raise ValueError(f"{label} lacks the selected confirmation audit")
    confirmation_record = _inspect_v2r2_rollout_spec(
        candidate_final=candidate_final,
        spec=confirmation_spec,
        expected_seed=expected_seed,
        label=f"{label}.confirmation",
    )
    confirmation_metrics = _require_mapping(
        confirmation_record.get("metrics"),
        label=f"{label}.confirmation.metrics",
    )
    pair = _validate_v2r2_selected_confirmation(
        primary_metrics,
        _require_mapping_list(
            prompt_selection.get("candidate_prompt_sets"),
            label=f"{label}.candidate_prompt_sets",
        ),
        confirmation_metrics,
    )
    return (
        primary_record,
        confirmation_record,
        prompt_selection,
        pair,
    )


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be an object")
    return value


def _require_mapping_list(
    value: object, *, label: str
) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise RuntimeError(f"{label} must be an object list")
    return list(value)


def _reinspect_v2r2_rollout_record(
    *,
    candidate_final: Path,
    record: Mapping[str, object],
    expected_seed: int,
    label: str,
) -> Mapping[str, object]:
    observed = _inspect_v2r2_rollout_spec(
        candidate_final=candidate_final,
        spec={
            "seed": record.get("seed"),
            "run_name": record.get("run_name"),
            "call_id": record.get("call_id"),
        },
        expected_seed=expected_seed,
        label=label,
    )
    if _v2r2_json_sha256(observed) != _v2r2_json_sha256(record):
        raise RuntimeError(f"{label} recorded evidence drifted")
    return observed


def _reinspect_v2r2_confirmations(
    *,
    candidate_final: Path,
    records: Sequence[Mapping[str, object]],
    label: str,
) -> list[Mapping[str, object]]:
    observed: list[Mapping[str, object]] = []
    for offset, record in enumerate(records):
        observed.append(
            _reinspect_v2r2_rollout_record(
                candidate_final=candidate_final,
                record=record,
                expected_seed=V2R2_CONFIRMATION_FIRST_SEED + offset,
                label=f"{label}[{offset}]",
            )
        )
    return observed


def _validate_v2r2_p1_candidate_record(
    record: Mapping[str, object],
    *,
    expected_weight: float,
    expected_status: str,
    manifest_set: Mapping[str, object],
    label: str,
) -> None:
    if float(record.get("weight", float("nan"))) != expected_weight:
        raise RuntimeError(f"{label} weight drifted")
    if record.get("status") != expected_status:
        raise RuntimeError(f"{label} status drifted")
    plan = _v2r1_candidate_plan(
        {
            "weight": expected_weight,
            "run_id": record.get("candidate_run_id"),
        }
    )
    manifest_hash = _manifest_hash_for_plan(manifest_set, plan)
    expected = {
        "candidate_output_dir": plan.output_dir,
        "candidate_source_tree_sha256": V2R1_AUDITED_SOURCE_TREE_SHA256,
        "candidate_manifest_sha256": manifest_hash,
        "candidate_step": V2R2_PROTOCOL_CANDIDATE_STEP,
    }
    for key, expected_value in expected.items():
        if record.get(key) != expected_value:
            raise RuntimeError(f"{label}.{key} drifted")
    call_id = _required_spec_string(record, "candidate_call_id", label=label)
    call_result = _require_successful_modal_call(call_id)
    if call_result != plan.output_dir:
        raise RuntimeError(f"{label} candidate call result drifted")
    if record.get("candidate_call_result_sha256") != _v2r2_json_sha256(
        call_result
    ):
        raise RuntimeError(f"{label} candidate call-result hash drifted")
    run_state, final_path = _resolve_existing_run(
        plan, manifest_hash=manifest_hash
    )
    if run_state != "complete" or final_path is None:
        raise RuntimeError(f"{label} candidate endpoint is not complete")
    candidate_final = Path(final_path)
    if record.get(
        "candidate_hf_manifest_sha256"
    ) != _directory_manifest_sha256(candidate_final):
        raise RuntimeError(f"{label} candidate HF manifest drifted")
    if _v2r2_json_sha256(record.get("unweighted_ce")) != _v2r2_json_sha256(
        _last_unweighted_ce(plan, manifest_hash=manifest_hash)
    ):
        raise RuntimeError(f"{label} unweighted CE drifted")

    primary = _require_mapping(record.get("primary"), label=f"{label}.primary")
    observed_primary = _reinspect_v2r2_rollout_record(
        candidate_final=candidate_final,
        record=primary,
        expected_seed=V2R2_PRIMARY_SEED,
        label=f"{label}.primary",
    )
    primary_metrics = _require_mapping(
        observed_primary.get("metrics"), label=f"{label}.primary.metrics"
    )
    if expected_status == "protocol_rejected":
        confirmation = record.get("confirmation")
        if confirmation is not None:
            raise RuntimeError(
                f"{label} primary rejection cannot carry a confirmation"
            )
        try:
            _validate_v2r2_protocol_audit(primary_metrics)
        except ValueError as exc:
            if record.get("reason") != str(exc):
                raise RuntimeError(f"{label} rejection reason drifted") from exc
            return
        raise RuntimeError(f"{label} primary unexpectedly passes protocol")

    _validate_v2r2_protocol_audit(primary_metrics)
    prompt_selection = _v2r2_prompt_selection_proof(
        candidate_final=candidate_final,
        primary_metrics=primary_metrics,
    )
    if _v2r2_json_sha256(
        record.get("prompt_selection")
    ) != _v2r2_json_sha256(prompt_selection):
        raise RuntimeError(f"{label} prompt selection drifted")
    selection = _require_mapping(
        prompt_selection.get("selection"),
        label=f"{label}.prompt_selection.selection",
    )
    confirmation_record = _require_mapping(
        record.get("confirmation"), label=f"{label}.confirmation"
    )
    observed_confirmation = _reinspect_v2r2_rollout_record(
        candidate_final=candidate_final,
        record=confirmation_record,
        expected_seed=int(selection["confirmation_seed"]),
        label=f"{label}.confirmation",
    )
    selected_metrics = _require_mapping(
        observed_confirmation.get("metrics"),
        label=f"{label}.confirmation.metrics",
    )
    _validate_v2r2_selected_confirmation(
        primary_metrics,
        _require_mapping_list(
            prompt_selection.get("candidate_prompt_sets"),
            label=f"{label}.candidate_prompt_sets",
        ),
        selected_metrics,
        require_protocol=False,
    )
    if expected_status == "confirmation_protocol_rejected":
        try:
            _validate_v2r2_protocol_audit(selected_metrics)
        except ValueError as exc:
            if record.get("reason") != str(exc):
                raise RuntimeError(f"{label} rejection reason drifted") from exc
            return
        raise RuntimeError(
            f"{label} rejected confirmation unexpectedly passes protocol"
        )
    if expected_status != "protocol_approved":
        raise RuntimeError(f"{label} has an unsupported status")
    _validate_v2r2_protocol_audit(selected_metrics)


def _validate_v2r2_p1_marker_evidence(
    marker: Mapping[str, object],
    *,
    manifest_set: Mapping[str, object],
) -> None:
    eligible = marker.get("eligible_sft_loss_weights")
    if eligible != list(V2R2_ELIGIBLE_SFT_LOSS_WEIGHTS):
        raise RuntimeError("P1 gate eligible-weight list drifted")
    if marker.get("selection_rule") != "smallest_eligible_weight":
        raise RuntimeError("P1 gate selection rule drifted")
    if marker.get("protocol_candidate_step") != V2R2_PROTOCOL_CANDIDATE_STEP:
        raise RuntimeError("P1 gate candidate step drifted")
    selected_weight = float(marker["selected_sft_loss_weight"])
    selected_index = list(V2R2_ELIGIBLE_SFT_LOSS_WEIGHTS).index(
        selected_weight
    )
    rejected = _require_mapping_list(
        marker.get("rejected_lower_weights"),
        label="P1 rejected_lower_weights",
    )
    if len(rejected) != selected_index:
        raise RuntimeError("P1 gate does not prove every lower weight failed")
    for index, record in enumerate(rejected):
        status = record.get("status")
        if status not in {
            "protocol_rejected",
            "confirmation_protocol_rejected",
        }:
            raise RuntimeError("P1 lower-weight rejection status is invalid")
        _validate_v2r2_p1_candidate_record(
            record,
            expected_weight=V2R2_ELIGIBLE_SFT_LOSS_WEIGHTS[index],
            expected_status=str(status),
            manifest_set=manifest_set,
            label=f"P1 rejected_lower_weights[{index}]",
        )
    selected = _require_mapping(
        marker.get("selected_candidate"), label="P1 selected_candidate"
    )
    _validate_v2r2_p1_candidate_record(
        selected,
        expected_weight=selected_weight,
        expected_status="protocol_approved",
        manifest_set=manifest_set,
        label="P1 selected_candidate",
    )


def _validate_v2r2_exp2_marker_evidence(
    marker: Mapping[str, object],
    *,
    manifest_set: Mapping[str, object],
) -> None:
    selected_weight = float(marker["selected_sft_loss_weight"])
    plan = _v2r2_plan(
        "v2r2-exp2-monolithic-canary",
        selected_weight=selected_weight,
    )
    manifest_hash = _manifest_hash_for_plan(manifest_set, plan)
    expected = {
        "candidate_output_dir": plan.output_dir,
        "candidate_manifest_sha256": manifest_hash,
        "candidate_step": V2R2_PROTOCOL_CANDIDATE_STEP,
        "schedule_total_steps": MONOLITHIC_STEPS,
        "manifest_leg": "p1+p2",
    }
    for key, expected_value in expected.items():
        if marker.get(key) != expected_value:
            raise RuntimeError(f"Exp2 gate {key} drifted")
    parent_hash = marker.get("parent_p1_gate_sha256")
    if not isinstance(parent_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", parent_hash
    ):
        raise RuntimeError("Exp2 gate parent P1 hash is invalid")
    call_id = _required_spec_string(
        marker, "candidate_call_id", label="Exp2 gate"
    )
    call_result = _require_successful_modal_call(call_id)
    if call_result != plan.output_dir:
        raise RuntimeError("Exp2 candidate call result drifted")
    if marker.get("candidate_call_result_sha256") != _v2r2_json_sha256(
        call_result
    ):
        raise RuntimeError("Exp2 candidate call-result hash drifted")
    run_state, final_path = _resolve_existing_run(
        plan, manifest_hash=manifest_hash
    )
    if run_state != "complete" or final_path is None:
        raise RuntimeError("Exp2 monolithic candidate is not complete")
    candidate_final = Path(final_path)
    if marker.get(
        "candidate_hf_manifest_sha256"
    ) != _directory_manifest_sha256(candidate_final):
        raise RuntimeError("Exp2 candidate HF manifest drifted")
    if _v2r2_json_sha256(marker.get("unweighted_ce")) != _v2r2_json_sha256(
        _last_unweighted_ce(plan, manifest_hash=manifest_hash)
    ):
        raise RuntimeError("Exp2 unweighted CE drifted")
    primary = _require_mapping(marker.get("primary"), label="Exp2 primary")
    observed_primary = _reinspect_v2r2_rollout_record(
        candidate_final=candidate_final,
        record=primary,
        expected_seed=V2R2_PRIMARY_SEED,
        label="Exp2 primary",
    )
    prompt_selection = _v2r2_prompt_selection_proof(
        candidate_final=candidate_final,
        primary_metrics=_require_mapping(
            observed_primary.get("metrics"), label="Exp2 primary.metrics"
        ),
    )
    if _v2r2_json_sha256(
        marker.get("prompt_selection")
    ) != _v2r2_json_sha256(prompt_selection):
        raise RuntimeError("Exp2 prompt selection drifted")
    selection = _require_mapping(
        prompt_selection.get("selection"),
        label="Exp2 prompt_selection.selection",
    )
    confirmation_record = _require_mapping(
        marker.get("confirmation"), label="Exp2 confirmation"
    )
    observed_confirmation = _reinspect_v2r2_rollout_record(
        candidate_final=candidate_final,
        record=confirmation_record,
        expected_seed=int(selection["confirmation_seed"]),
        label="Exp2 confirmation",
    )
    primary_metrics = _require_mapping(
        observed_primary.get("metrics"), label="Exp2 primary.metrics"
    )
    confirmation_metrics = _require_mapping(
        observed_confirmation.get("metrics"),
        label="Exp2 confirmation.metrics",
    )
    pair = _validate_v2r2_selected_confirmation(
        primary_metrics,
        _require_mapping_list(
            prompt_selection.get("candidate_prompt_sets"),
            label="Exp2 candidate_prompt_sets",
        ),
        confirmation_metrics,
    )
    _validate_v2r2_monolithic_gate(
        selected_weight=selected_weight,
        candidate_weight=plan.sft_loss_weight,
        candidate_step=V2R2_PROTOCOL_CANDIDATE_STEP,
        schedule_total_steps=plan.total_steps,
        manifest_leg=plan.manifest_leg,
        primary=primary_metrics,
        confirmation=pair["confirmation"],
    )


def _validate_v2r2_gate_marker(
    path: Path,
    *,
    gate_stage: str,
    manifest_set: Mapping[str, object],
) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{gate_stage} is blocked until its v2r2 gate exists at {path}"
        )
    marker = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(marker, Mapping):
        raise RuntimeError(f"{gate_stage} marker is not an object")
    _validate_v2r2_self_hashed_marker(marker)
    stable_core = {
        str(key): value
        for key, value in marker.items()
        if key
        not in {
            "gate_sha256",
            "approval_fingerprint",
            "approved_at",
        }
    }
    if marker.get("approval_fingerprint") != _v2r2_json_sha256(stable_core):
        raise RuntimeError(f"{gate_stage} approval fingerprint mismatch")
    expected = {
        "contract_schema": V2R2_CONTRACT_SCHEMA,
        "contract_version": V2R2_EXPERIMENT_VERSION,
        "contract_plan_sha256": V2R2_CONTRACT_PLAN_SHA256,
        "approved": True,
        "gate_stage": gate_stage,
        "manifest_set_hash": manifest_set.get("manifest_set_hash"),
        "data_artifact_version": DATA_ARTIFACT_VERSION,
        "production_source_tree_sha256": SOURCE_TREE_SHA256,
    }
    for key, expected_value in expected.items():
        if marker.get(key) != expected_value:
            raise RuntimeError(
                f"{gate_stage} marker drift at {key}: "
                f"{marker.get(key)!r} != {expected_value!r}"
            )
    weight = float(marker.get("selected_sft_loss_weight", float("nan")))
    if weight not in V2R2_ELIGIBLE_SFT_LOSS_WEIGHTS:
        raise RuntimeError(f"{gate_stage} marker selected invalid weight")
    if gate_stage == "p1_protocol":
        _validate_v2r2_p1_marker_evidence(
            marker, manifest_set=manifest_set
        )
    elif gate_stage == "exp2_monolithic_protocol":
        _validate_v2r2_exp2_marker_evidence(
            marker, manifest_set=manifest_set
        )
    else:
        raise RuntimeError(f"unsupported v2r2 gate stage: {gate_stage}")
    return marker


def _write_v2r2_marker(
    path: Path,
    *,
    core: Mapping[str, object],
    manifest_set: Mapping[str, object],
) -> str:
    stable_core = {
        "contract_schema": V2R2_CONTRACT_SCHEMA,
        "contract_version": V2R2_EXPERIMENT_VERSION,
        "contract_plan_sha256": V2R2_CONTRACT_PLAN_SHA256,
        "approved": True,
        "manifest_set_hash": manifest_set.get("manifest_set_hash"),
        "data_artifact_version": DATA_ARTIFACT_VERSION,
        "production_source_tree_sha256": SOURCE_TREE_SHA256,
        **dict(core),
    }
    approval_fingerprint = _v2r2_json_sha256(stable_core)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, Mapping):
            raise RuntimeError("existing v2r2 gate is not an object")
        _validate_v2r2_self_hashed_marker(existing)
        if existing.get("approval_fingerprint") != approval_fingerprint:
            raise FileExistsError(
                f"immutable v2r2 gate already differs: {path}"
            )
        return str(path)
    payload = _self_hash_v2r2_marker(
        {
            **stable_core,
            "approval_fingerprint": approval_fingerprint,
            "approved_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _atomic_json(path, payload)
    data_volume.commit()
    return str(path)


def _validate_v2r2_p1_rejection(
    path: Path,
    *,
    manifest_set: Mapping[str, object],
) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("v2r2 rejection report is not an object")
    _validate_v2r2_self_hashed_marker(payload)
    stable_core = {
        str(key): value
        for key, value in payload.items()
        if key
        not in {
            "gate_sha256",
            "decision_fingerprint",
            "decided_at",
        }
    }
    if payload.get("decision_fingerprint") != _v2r2_json_sha256(stable_core):
        raise RuntimeError("v2r2 rejection fingerprint mismatch")
    expected = {
        "contract_schema": V2R2_CONTRACT_SCHEMA,
        "contract_version": V2R2_EXPERIMENT_VERSION,
        "contract_plan_sha256": V2R2_CONTRACT_PLAN_SHA256,
        "approved": False,
        "gate_stage": "p1_protocol",
        "decision": "rejected_all_eligible_weights",
        "manifest_set_hash": manifest_set.get("manifest_set_hash"),
        "data_artifact_version": DATA_ARTIFACT_VERSION,
        "decision_source_tree_sha256": SOURCE_TREE_SHA256,
        "eligible_sft_loss_weights": list(V2R2_ELIGIBLE_SFT_LOSS_WEIGHTS),
        "full_p1_authorized": False,
        "full_exp2_authorized": False,
    }
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise RuntimeError(f"v2r2 rejection {key} drifted")
    rejected = _require_mapping_list(
        payload.get("rejected_candidates"),
        label="v2r2 rejected_candidates",
    )
    if len(rejected) != len(V2R2_ELIGIBLE_SFT_LOSS_WEIGHTS):
        raise RuntimeError("v2r2 rejection lacks all eligible weights")
    for index, record in enumerate(rejected):
        status = record.get("status")
        if status not in {
            "protocol_rejected",
            "confirmation_protocol_rejected",
        }:
            raise RuntimeError("v2r2 rejection candidate status is invalid")
        _validate_v2r2_p1_candidate_record(
            record,
            expected_weight=V2R2_ELIGIBLE_SFT_LOSS_WEIGHTS[index],
            expected_status=str(status),
            manifest_set=manifest_set,
            label=f"v2r2 rejected_candidates[{index}]",
        )
    return payload


def _write_v2r2_p1_rejection(
    *,
    rejected: Sequence[Mapping[str, object]],
    manifest_set: Mapping[str, object],
) -> str:
    stable_core = {
        "contract_schema": V2R2_CONTRACT_SCHEMA,
        "contract_version": V2R2_EXPERIMENT_VERSION,
        "contract_plan_sha256": V2R2_CONTRACT_PLAN_SHA256,
        "approved": False,
        "gate_stage": "p1_protocol",
        "decision": "rejected_all_eligible_weights",
        "manifest_set_hash": manifest_set.get("manifest_set_hash"),
        "data_artifact_version": DATA_ARTIFACT_VERSION,
        "decision_source_tree_sha256": SOURCE_TREE_SHA256,
        "eligible_sft_loss_weights": list(V2R2_ELIGIBLE_SFT_LOSS_WEIGHTS),
        "rejected_candidates": list(rejected),
        "full_p1_authorized": False,
        "full_exp2_authorized": False,
    }
    decision_fingerprint = _v2r2_json_sha256(stable_core)
    if V2R2_P1_REJECTION_PATH.is_file():
        existing = _validate_v2r2_p1_rejection(
            V2R2_P1_REJECTION_PATH,
            manifest_set=manifest_set,
        )
        if existing.get("decision_fingerprint") != decision_fingerprint:
            raise FileExistsError(
                "immutable v2r2 rejection already records different evidence"
            )
        return str(V2R2_P1_REJECTION_PATH)
    payload = _self_hash_v2r2_marker(
        {
            **stable_core,
            "decision_fingerprint": decision_fingerprint,
            "decided_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _atomic_json(V2R2_P1_REJECTION_PATH, payload)
    data_volume.commit()
    _validate_v2r2_p1_rejection(
        V2R2_P1_REJECTION_PATH,
        manifest_set=manifest_set,
    )
    return str(V2R2_P1_REJECTION_PATH)


def _resolve_existing_run(
    plan: TrainPlan,
    *,
    manifest_hash: str | None = None,
) -> tuple[str, str | None]:
    """Return ``fresh``, ``resume``, or ``complete`` for an immutable run."""

    path = Path(plan.output_dir)
    if not path.exists() or not any(path.iterdir()):
        return "fresh", None
    if manifest_hash is None:
        raise ValueError(
            "manifest_hash is required to inspect an existing immutable run"
        )
    _validate_existing_run_identity(plan, manifest_hash=manifest_hash)
    if plan.diagnostic_only:
        snapshots_root = path / "snapshots"
        if snapshots_root.is_dir():
            allowed_temporary_names = {
                f".step_{step}.tmp" for step in plan.snapshot_steps
            }
            for child in snapshots_root.iterdir():
                if child.name.startswith(".step_") and child.name.endswith(
                    ".tmp"
                ):
                    if (
                        child.name not in allowed_temporary_names
                        or not child.is_dir()
                    ):
                        raise RuntimeError(
                            "unrecognized v2r3 temporary snapshot artifact: "
                            f"{child}"
                        )
                    shutil.rmtree(child)
        for forbidden_name in ("latest", "final"):
            forbidden = path / forbidden_name
            if forbidden.exists():
                raise RuntimeError(
                    "v2r3 diagnostic trajectories forbid mutable/duplicate "
                    f"{forbidden_name} artifacts: {forbidden}"
                )

    final_dir = path / "final"
    final_weights = list(final_dir.glob("model*.safetensors"))
    if (final_dir / "config.json").is_file() and final_weights:
        state = _validate_final_export_identity(
            plan, manifest_hash=manifest_hash
        )
        expected_step = (
            plan.max_steps if plan.max_steps is not None else plan.total_steps
        )
        if (
            int(state.get("global_step", -1)) != expected_step
            or int(state.get("manifest_cursor", -1)) != expected_step
        ):
            raise ValueError(
                "completed run trainer state does not match its target step"
            )
        if plan.diagnostic_only:
            snapshots = _validate_v2r3_snapshot_prefix(
                plan,
                manifest_hash=manifest_hash,
            )
            observed_names = {
                child.name
                for child in (path / "snapshots").iterdir()
            }
            expected_names = {
                f"step_{step}" for step in plan.snapshot_steps
            }
            if (
                len(snapshots) != len(plan.snapshot_steps)
                or observed_names != expected_names
            ):
                raise RuntimeError(
                    "completed v2r3 trajectory lacks its exact snapshot "
                    "inventory"
                )
        return "complete", str(final_dir)
    if final_dir.exists():
        raise RuntimeError(f"Incomplete final export blocks run: {final_dir}")

    latest_dir = path / "latest"
    state_path = latest_dir / "trainer_state.json"
    if plan.diagnostic_only and latest_dir.exists():
        raise RuntimeError(
            "v2r3 forbids mutable latest state; resume is allowed only from "
            "an authenticated immutable snapshot prefix"
        )
    if not plan.diagnostic_only:
        if state_path.is_file():
            _validate_resume_checkpoint_identity(
                plan, manifest_hash=manifest_hash
            )
            return "resume", str(latest_dir)
        if latest_dir.exists():
            raise RuntimeError(
                f"Incomplete resume checkpoint blocks run: {latest_dir}"
            )

    if plan.diagnostic_only:
        snapshots_root = path / "snapshots"
        completed: list[tuple[int, Mapping[str, object]]] = []
        if snapshots_root.exists() and not snapshots_root.is_dir():
            raise RuntimeError(
                f"v2r3 snapshot root is not a directory: {snapshots_root}"
            )
        if snapshots_root.is_dir():
            allowed_names = {
                f"step_{step}" for step in plan.snapshot_steps
            }
            observed_names = {child.name for child in snapshots_root.iterdir()}
            unexpected_names = observed_names - allowed_names
            if unexpected_names:
                raise RuntimeError(
                    "unrecognized v2r3 snapshot directories: "
                    f"{sorted(unexpected_names)}"
                )
            completed_steps: list[int] = []
            for step in plan.snapshot_steps:
                snapshot_root = snapshots_root / f"step_{step}"
                if not snapshot_root.exists():
                    break
                completed_steps.append(step)
            completed = list(
                zip(
                    completed_steps,
                    _validate_v2r3_snapshot_prefix(
                        plan,
                        manifest_hash=manifest_hash,
                        steps=completed_steps,
                    ),
                    strict=True,
                )
            )
            expected_prefix = {
                f"step_{step}"
                for step in plan.snapshot_steps[: len(completed)]
            }
            if observed_names != expected_prefix:
                raise RuntimeError(
                    "v2r3 snapshots are not a complete ordered prefix: "
                    f"{sorted(observed_names)} != "
                    f"{sorted(expected_prefix)}"
                )
        if (
            len(completed) == len(plan.snapshot_steps)
            and completed[-1][0] == plan.max_steps
        ):
            return "complete", str(completed[-1][1]["hf_path"])
        if completed:
            return "resume", str(completed[-1][1]["resume_path"])

    unexpected = sorted(
        child.name
        for child in path.iterdir()
        if child.name
        not in {
            "config.yaml",
            "metrics.jsonl",
            *({"snapshots"} if plan.diagnostic_only else set()),
        }
    )
    if unexpected:
        raise FileExistsError(
            f"Immutable output contains unrecognized partial artifacts: "
            f"{path}: {unexpected[:10]}"
        )
    # A container can fail before the first scheduled checkpoint, leaving only
    # the deterministic config snapshot. Starting the same immutable run from
    # update zero is safe; no learned state is being discarded.
    return "fresh", None


def _run_training(plan: TrainPlan, *, main_process_port: int) -> str:
    data_volume.reload()
    checkpoint_volume.reload()
    _verify_source_corpus()
    _validate_sft_snapshot()
    manifest_set = _load_manifest_set()
    if plan.stage in {"p1", "exp2", "p2"}:
        _validate_production_gate(manifest_set)
    manifest_hash = _manifest_hash_for_plan(manifest_set, plan)
    run_state, state_path = _resolve_existing_run(
        plan, manifest_hash=manifest_hash
    )
    if run_state == "complete":
        print(f"[interleaved] Final already complete: {state_path}", flush=True)
        return plan.output_dir

    command = _build_training_command(
        plan,
        manifest_hash=manifest_hash,
        main_process_port=main_process_port,
        resume=state_path if run_state == "resume" else None,
    )
    print("[interleaved] " + " ".join(command), flush=True)
    if plan.diagnostic_only:
        process = subprocess.Popen(
            command,
            cwd="/root/chess",
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        published_steps = {
            step
            for step in plan.snapshot_steps
            if (
                Path(plan.output_dir)
                / "snapshots"
                / f"step_{step}"
                / ".complete.json"
            ).is_file()
        }
        while True:
            try:
                returncode = process.wait(
                    timeout=V2R3_SNAPSHOT_COMMIT_POLL_SECONDS
                )
            except subprocess.TimeoutExpired:
                returncode = None
            for step in plan.snapshot_steps:
                if step in published_steps:
                    continue
                marker = (
                    Path(plan.output_dir)
                    / "snapshots"
                    / f"step_{step}"
                    / ".complete.json"
                )
                if not marker.is_file():
                    break
                snapshot_index = plan.snapshot_steps.index(step)
                _validate_v2r3_snapshot_prefix(
                    plan,
                    manifest_hash=manifest_hash,
                    steps=plan.snapshot_steps[: snapshot_index + 1],
                )
                try:
                    checkpoint_volume.commit()
                except Exception as exc:
                    print(
                        "[v2r3-snapshot-publish-retry] "
                        f"weight={plan.sft_loss_weight} step={step} "
                        f"error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    break
                published_steps.add(step)
                print(
                    "[v2r3-snapshot-published] "
                    f"weight={plan.sft_loss_weight} step={step}",
                    flush=True,
                )
            if returncode is not None:
                break
        result_returncode = returncode
    else:
        result = subprocess.run(
            command,
            cwd="/root/chess",
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=False,
        )
        result_returncode = result.returncode
    checkpoint_volume.commit()
    if result_returncode != 0:
        raise RuntimeError(
            f"{plan.stage} training failed with exit {result_returncode}"
        )
    return plan.output_dir


def _run_benchmark(plan: TrainPlan, *, main_process_port: int) -> dict[str, object]:
    if not plan.benchmark_only:
        raise ValueError("_run_benchmark requires a benchmark-only plan")
    output = Path(plan.output_dir)
    if not output.is_relative_to(BENCHMARK_OUTPUT_ROOT):
        raise ValueError("Benchmark output escaped the ephemeral root")
    if output.exists():
        raise FileExistsError(
            f"Benchmark output already exists in this container: {output}"
        )

    data_volume.reload()
    _verify_source_corpus()
    _validate_sft_snapshot()
    manifest_set = _load_manifest_set()
    manifest_hash = _manifest_hash_for_plan(manifest_set, plan)
    command = _build_training_command(
        plan,
        manifest_hash=manifest_hash,
        main_process_port=main_process_port,
    )
    print("[interleaved-benchmark] " + " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd="/root/chess",
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{plan.attention_backend}/{plan.torch_compile_mode} benchmark "
            f"failed with exit {result.returncode}"
        )
    result_path = output / "benchmark_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(
            f"Benchmark completed without its result: {result_path}"
        )
    with result_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("attention_backend") != plan.attention_backend:
        raise RuntimeError("Benchmark result backend provenance drifted")
    if value.get("torch_compile_mode") != plan.torch_compile_mode:
        raise RuntimeError("Benchmark result compile provenance drifted")
    return value


@app.function(
    cpu=16.0,
    memory=64 * 1024,
    timeout=60 * 60 * 24,
    retries=0,
)
def prepare_data() -> str:
    data_volume.reload()
    source_metadata = _verify_source_corpus()
    sft_metadata = _cache_sft_snapshot()
    manifest_set = _build_manifests(source_metadata)
    print(
        json.dumps(
            {
                "source": source_metadata,
                "sft": sft_metadata,
                "manifest_set": manifest_set,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return str(MANIFEST_SET_PATH)


@app.function(
    cpu=8.0,
    memory=32 * 1024,
    timeout=60 * 30,
    retries=0,
)
def approve_production_gate(
    candidate_run_id: str,
    candidate_call_id: str,
    rollout_run_name: str,
    rollout_call_id: str,
) -> str:
    """Approve v2 only from the exact-weight 500-update + 2,048 rollout gate."""

    data_volume.reload()
    checkpoint_volume.reload()
    rl_checkpoint_volume.reload()
    _verify_source_corpus()
    _validate_sft_snapshot()
    manifest_set = _load_manifest_set()
    plan = _sft_weight_canary_plan(
        candidate_run_id,
        sft_loss_weight=P1_SFT_LOSS_WEIGHT,
        max_steps=SFT_WEIGHT_CANARY_DEFAULT_STEPS,
    )
    p1_manifest_hash = _manifest_hash_for_plan(manifest_set, plan)
    run_state, final_path = _resolve_existing_run(
        plan, manifest_hash=p1_manifest_hash
    )
    if run_state != "complete" or final_path is None:
        raise RuntimeError(
            f"gate candidate is not complete: {run_state} {final_path}"
        )
    _require_successful_modal_call(candidate_call_id)
    _require_successful_modal_call(rollout_call_id)
    candidate_final = Path(final_path)
    metrics, rollout_provenance, rollout_artifact_sha256 = (
        _inspect_rollout_gate(
            candidate_final=candidate_final,
            rollout_run_name=rollout_run_name,
        )
    )
    core: dict[str, object] = {
        "schema": "interleaved-v2-production-gate-v1",
        "approved": True,
        "experiment_version": EXPERIMENT_VERSION,
        "data_artifact_version": DATA_ARTIFACT_VERSION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "manifest_set_hash": manifest_set.get("manifest_set_hash"),
        "p1_manifest_sha256": p1_manifest_hash,
        "candidate_run_id": _validate_run_id(candidate_run_id),
        "candidate_call_id": candidate_call_id,
        "candidate_output_dir": plan.output_dir,
        "candidate_hf_manifest_sha256": (
            _directory_manifest_sha256(candidate_final)
        ),
        "candidate_sft_loss_weight": P1_SFT_LOSS_WEIGHT,
        "candidate_steps": SFT_WEIGHT_CANARY_DEFAULT_STEPS,
        "rollout_run_name": _validate_run_id(rollout_run_name),
        "rollout_call_id": rollout_call_id,
        "rollout_artifact_sha256": dict(rollout_artifact_sha256),
        "rollout_identity_sha256": rollout_provenance.get(
            "identity_sha256"
        ),
        "metrics": metrics,
    }
    approval_fingerprint = _canonical_mapping_sha256(core)
    if PRODUCTION_GATE_PATH.is_file():
        existing = _validate_production_gate(manifest_set)
        if existing.get("approval_fingerprint") != approval_fingerprint:
            raise FileExistsError(
                "an immutable production gate already approves a different "
                "candidate"
            )
        return str(PRODUCTION_GATE_PATH)
    payload: dict[str, object] = {
        **core,
        "approval_fingerprint": approval_fingerprint,
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["gate_sha256"] = _canonical_mapping_sha256(payload)
    _atomic_json(PRODUCTION_GATE_PATH, payload)
    data_volume.commit()
    _validate_production_gate(manifest_set)
    print(
        "[interleaved-v2-gate] "
        + json.dumps(payload, sort_keys=True),
        flush=True,
    )
    return str(PRODUCTION_GATE_PATH)


@app.function(
    cpu=8.0,
    memory=32 * 1024,
    timeout=60 * 60,
    retries=0,
)
def approve_v2r2_p1_gate(candidate_specs_json: str) -> str:
    """Select the smallest protocol-qualified 2,000-step P1 candidate."""

    data_volume.reload()
    checkpoint_volume.reload()
    rl_checkpoint_volume.reload()
    _verify_source_corpus()
    _validate_sft_snapshot()
    manifest_set = _load_manifest_set()
    specs = _decode_json_object_list(
        candidate_specs_json, label="candidate_specs_json"
    )
    if len(specs) > len(V2R2_ELIGIBLE_SFT_LOSS_WEIGHTS):
        raise ValueError("too many v2r2 weight candidates")
    observed_weights = [float(spec.get("weight", float("nan"))) for spec in specs]
    expected_prefix = list(V2R2_ELIGIBLE_SFT_LOSS_WEIGHTS[: len(specs)])
    if observed_weights != expected_prefix:
        raise ValueError(
            "candidate weights must be the exact ordered eligible prefix: "
            f"{observed_weights!r} != {expected_prefix!r}"
        )

    rejected: list[Mapping[str, object]] = []
    selected_record: Mapping[str, object] | None = None
    for index, spec in enumerate(specs):
        plan = _v2r1_candidate_plan(spec)
        manifest_hash = _manifest_hash_for_plan(manifest_set, plan)
        run_state, final_path = _resolve_existing_run(
            plan, manifest_hash=manifest_hash
        )
        if run_state != "complete" or final_path is None:
            raise RuntimeError(
                f"candidate {plan.output_dir} is not complete: {run_state}"
            )
        candidate_call_id = _required_spec_string(
            spec, "call_id", label=f"candidate[{index}]"
        )
        candidate_call_result = _require_successful_modal_call(
            candidate_call_id
        )
        if candidate_call_result != plan.output_dir:
            raise RuntimeError(
                "candidate call result does not match its output directory"
            )
        candidate_final = Path(final_path)
        primary_spec = spec.get("primary")
        if not isinstance(primary_spec, Mapping):
            raise ValueError(f"candidate[{index}].primary must be an object")
        primary = _inspect_v2r2_rollout_spec(
            candidate_final=candidate_final,
            spec=primary_spec,
            expected_seed=V2R2_PRIMARY_SEED,
            label=f"candidate[{index}].primary",
        )
        primary_metrics = primary.get("metrics")
        if not isinstance(primary_metrics, Mapping):
            raise RuntimeError("candidate primary audit lacks metrics")
        confirmation_spec = _confirmation_spec(
            spec, label=f"candidate[{index}]"
        )
        base_record: dict[str, object] = {
            "weight": plan.sft_loss_weight,
            "candidate_run_id": _validate_run_id(
                _required_spec_string(
                    spec, "run_id", label=f"candidate[{index}]"
                )
            ),
            "candidate_call_id": candidate_call_id,
            "candidate_call_result_sha256": _v2r2_json_sha256(
                candidate_call_result
            ),
            "candidate_output_dir": plan.output_dir,
            "candidate_hf_manifest_sha256": (
                _directory_manifest_sha256(candidate_final)
            ),
            "candidate_source_tree_sha256": (
                _plan_source_tree_sha256(plan)
            ),
            "candidate_manifest_sha256": manifest_hash,
            "candidate_step": V2R2_PROTOCOL_CANDIDATE_STEP,
            "unweighted_ce": dict(
                _last_unweighted_ce(
                    plan, manifest_hash=manifest_hash
                )
            ),
            "primary": primary,
        }
        try:
            _validate_v2r2_protocol_audit(primary_metrics)
        except ValueError as exc:
            if confirmation_spec is not None:
                raise ValueError(
                    "a protocol-rejected primary must not include "
                    "a confirmation audit"
                ) from exc
            rejected.append(
                {
                    **base_record,
                    "status": "protocol_rejected",
                    "reason": str(exc),
                }
            )
            continue

        prompt_selection = _v2r2_prompt_selection_proof(
            candidate_final=candidate_final,
            primary_metrics=primary_metrics,
        )
        selection = _require_mapping(
            prompt_selection.get("selection"),
            label=f"candidate[{index}].prompt_selection.selection",
        )
        if confirmation_spec is None:
            raise ValueError(
                f"candidate[{index}] passed primary protocol but lacks "
                "the selected confirmation audit"
            )
        confirmation = _inspect_v2r2_rollout_spec(
            candidate_final=candidate_final,
            spec=confirmation_spec,
            expected_seed=int(selection["confirmation_seed"]),
            label=f"candidate[{index}].confirmation",
        )
        selected_metrics = _require_mapping(
            confirmation.get("metrics"),
            label=f"candidate[{index}].confirmation.metrics",
        )
        _validate_v2r2_selected_confirmation(
            primary_metrics,
            _require_mapping_list(
                prompt_selection.get("candidate_prompt_sets"),
                label=f"candidate[{index}].candidate_prompt_sets",
            ),
            selected_metrics,
            require_protocol=False,
        )
        try:
            _validate_v2r2_protocol_audit(selected_metrics)
        except ValueError as exc:
            rejected.append(
                {
                    **base_record,
                    "status": "confirmation_protocol_rejected",
                    "prompt_selection": prompt_selection,
                    "confirmation": confirmation,
                    "reason": str(exc),
                }
            )
            continue
        if index != len(specs) - 1:
            raise ValueError(
                "candidate_specs_json must end at the first passing weight"
            )
        selected_record = {
            **base_record,
            "status": "protocol_approved",
            "prompt_selection": prompt_selection,
            "confirmation": confirmation,
        }
        break

    if selected_record is None:
        if len(specs) != len(V2R2_ELIGIBLE_SFT_LOSS_WEIGHTS):
            raise RuntimeError(
                "no supplied weight passed, but all eligible weights were "
                "not evaluated"
            )
        path = _write_v2r2_p1_rejection(
            rejected=rejected,
            manifest_set=manifest_set,
        )
        print(
            "[v2r2-p1-rejection] "
            + json.dumps(
                {
                    "path": path,
                    "rejected_candidates": rejected,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return path
    selected_weight = float(selected_record["weight"])
    core = {
        "gate_stage": "p1_protocol",
        "selected_sft_loss_weight": selected_weight,
        "selection_rule": "smallest_eligible_weight",
        "eligible_sft_loss_weights": list(V2R2_ELIGIBLE_SFT_LOSS_WEIGHTS),
        "protocol_candidate_step": V2R2_PROTOCOL_CANDIDATE_STEP,
        "rejected_lower_weights": rejected,
        "selected_candidate": selected_record,
    }
    path = _write_v2r2_marker(
        V2R2_P1_GATE_PATH,
        core=core,
        manifest_set=manifest_set,
    )
    _validate_v2r2_gate_marker(
        V2R2_P1_GATE_PATH,
        gate_stage="p1_protocol",
        manifest_set=manifest_set,
    )
    print("[v2r2-p1-gate] " + json.dumps(core, sort_keys=True), flush=True)
    return path


@app.function(
    cpu=8.0,
    memory=32 * 1024,
    timeout=60 * 60,
    retries=0,
)
def audit_v2r3_diagnostics(audit_specs_json: str) -> str:
    """Write an immutable diagnostic report with no authorization semantics."""

    data_volume.reload()
    checkpoint_volume.reload()
    rl_checkpoint_volume.reload()
    manifest_set = _load_manifest_set()
    specs = _decode_json_object_list(
        audit_specs_json, label="audit_specs_json"
    )
    expected_pairs = [
        (weight, step)
        for weight, trajectory in V2R3_TRAJECTORY_SPECS.items()
        for step in trajectory["snapshot_steps"]
    ]
    observed_pairs = [
        (float(spec.get("weight", float("nan"))), int(spec.get("step", -1)))
        for spec in specs
    ]
    if len(specs) != V2R3_SNAPSHOT_COUNT or observed_pairs != expected_pairs:
        raise ValueError(
            "v2r3 audit specs must be the exact frozen ordered 12-snapshot "
            f"inventory: {observed_pairs!r} != {expected_pairs!r}"
        )

    training_calls: dict[float, str] = {}
    trajectory_snapshots: dict[
        float, Mapping[int, Mapping[str, object]]
    ] = {}
    records: list[Mapping[str, object]] = []
    for index, (spec, (weight, step)) in enumerate(
        zip(specs, expected_pairs, strict=True)
    ):
        plan = _v2r3_plan(weight)
        training_call_id = _required_spec_string(
            spec, "training_call_id", label=f"audit[{index}]"
        )
        previous_call = training_calls.setdefault(weight, training_call_id)
        if previous_call != training_call_id:
            raise ValueError(
                "all snapshots from one continuous trajectory must bind the "
                "same training call"
            )
        training_result = _require_successful_modal_call(training_call_id)
        if training_result != plan.output_dir:
            raise RuntimeError(
                "v2r3 training call result does not match trajectory output"
            )
        manifest_hash = _manifest_hash_for_plan(manifest_set, plan)
        state, resolved = _resolve_existing_run(
            plan, manifest_hash=manifest_hash
        )
        expected_final_hf = str(
            Path(plan.output_dir)
            / "snapshots"
            / f"step_{plan.snapshot_steps[-1]}"
            / "hf"
        )
        if state != "complete" or resolved != expected_final_hf:
            raise RuntimeError(
                f"v2r3 trajectory is not complete: {weight} -> "
                f"{state}, {resolved}"
            )
        if weight not in trajectory_snapshots:
            validated_prefix = _validate_v2r3_snapshot_prefix(
                plan,
                manifest_hash=manifest_hash,
            )
            trajectory_snapshots[weight] = {
                int(snapshot["step"]): snapshot
                for snapshot in validated_prefix
            }
        snapshot = trajectory_snapshots[weight][step]
        expected_run_name = _v2r3_rollout_run_name(weight, step)
        rollout_run_name = _required_spec_string(
            spec, "rollout_run_name", label=f"audit[{index}]"
        )
        if rollout_run_name != expected_run_name:
            raise ValueError(
                f"v2r3 rollout name drift: {rollout_run_name!r} != "
                f"{expected_run_name!r}"
            )
        rollout_call_id = _required_spec_string(
            spec, "rollout_call_id", label=f"audit[{index}]"
        )
        rollout_result = _require_successful_modal_call(rollout_call_id)
        if (
            not isinstance(rollout_result, Mapping)
            or rollout_result.get("run_name") != rollout_run_name
            or rollout_result.get("checkpoint_root")
            != str(RL_GATE_ROOT / rollout_run_name)
            or rollout_result.get("num_rollout") != 1
            or rollout_result.get("dynamic_filter") is not False
            or rollout_result.get("rollout_seed") != V2R3_PRIMARY_SEED
            or rollout_result.get("deterministic_inference") is not True
            or rollout_result.get("rollout_only") is not True
        ):
            raise RuntimeError(
                "v2r3 rollout call result does not match its exact artifact"
            )
        metrics, provenance, artifact_hashes = (
            _inspect_v2r3_rollout_audit(
                candidate_hf=Path(str(snapshot["hf_path"])),
                rollout_run_name=rollout_run_name,
            )
        )
        records.append(
            {
                "weight": weight,
                "step": step,
                "training_call_id": training_call_id,
                "training_call_result_sha256": _v2r2_json_sha256(
                    training_result
                ),
                "trajectory_output_dir": plan.output_dir,
                "trajectory_source_tree_sha256": (
                    _plan_source_tree_sha256(plan)
                ),
                "trajectory_manifest_sha256": manifest_hash,
                "snapshot": snapshot,
                "interval_unweighted_ce": snapshot[
                    "interval_unweighted_ce"
                ],
                "rollout_run_name": rollout_run_name,
                "rollout_call_id": rollout_call_id,
                "rollout_call_result_sha256": _v2r2_json_sha256(
                    rollout_result
                ),
                "rollout_provenance_identity_sha256": provenance.get(
                    "identity_sha256"
                ),
                "rollout_artifact_sha256": artifact_hashes,
                "metrics": metrics,
            }
        )
    if len(set(training_calls.values())) != len(V2R3_TRAJECTORY_SPECS):
        raise RuntimeError(
            "v2r3 must use exactly four distinct continuous trajectory calls"
        )

    stable_core = {
        "schema": "interleaved-v2r3-diagnostic-report-v1",
        "contract_schema": V2R3_CONTRACT_SCHEMA,
        "contract_version": V2R3_EXPERIMENT_VERSION,
        "diagnostic_only": True,
        "production_authorized": False,
        "p1_authorized": False,
        "exp2_authorized": False,
        "seed": V2R3_PRIMARY_SEED,
        "seed42_prompt_set_sha256": V2R3_SEED42_PROMPT_SET_SHA256,
        "balanced_data_sha256": RL_GATE_BALANCED_SHA256,
        "pretrain_source_tree_sha256": SOURCE_TREE_SHA256,
        "rollout_chess_source_sha256": V2R3_RL_CHESS_SOURCE_SHA256,
        "rollout_miles_source_sha256": RL_GATE_MILES_SOURCE_SHA256,
        "manifest_set_hash": manifest_set.get("manifest_set_hash"),
        "ce_measurement_semantics": {
            "name": "token_weighted_training_stream_pre_update_batch_logits",
            "held_out": False,
            "endpoint_checkpoint_evaluation": False,
            "claim_scope": "stability_and_optimization_diagnostic_only",
            "final_pretrain_performance_claim_supported": False,
            "required_followup": (
                "frozen_endpoint_benchmark_or_heldout_evaluation"
            ),
        },
        "records": records,
    }
    report_sha256 = _v2r2_json_sha256(stable_core)
    payload = {
        **stable_core,
        "reported_at": datetime.now(timezone.utc).isoformat(),
        "report_sha256": report_sha256,
    }
    if V2R3_DIAGNOSTIC_REPORT_PATH.is_file():
        existing = json.loads(
            V2R3_DIAGNOSTIC_REPORT_PATH.read_text(encoding="utf-8")
        )
        existing_core = {
            key: value
            for key, value in existing.items()
            if key not in {"reported_at", "report_sha256"}
        }
        if (
            existing.get("report_sha256")
            != _v2r2_json_sha256(existing_core)
            or existing.get("report_sha256") != report_sha256
        ):
            raise FileExistsError(
                "immutable v2r3 diagnostic report already differs"
            )
        return str(V2R3_DIAGNOSTIC_REPORT_PATH)
    _atomic_json(V2R3_DIAGNOSTIC_REPORT_PATH, payload)
    data_volume.commit()
    return str(V2R3_DIAGNOSTIC_REPORT_PATH)


@app.function(
    cpu=8.0,
    memory=32 * 1024,
    timeout=60 * 60,
    retries=0,
)
def approve_v2r2_exp2_gate(gate_spec_json: str) -> str:
    """Approve full Exp2 only from its selected-weight monolithic canary."""

    data_volume.reload()
    checkpoint_volume.reload()
    rl_checkpoint_volume.reload()
    _verify_source_corpus()
    _validate_sft_snapshot()
    manifest_set = _load_manifest_set()
    p1_gate = _validate_v2r2_gate_marker(
        V2R2_P1_GATE_PATH,
        gate_stage="p1_protocol",
        manifest_set=manifest_set,
    )
    selected_weight = float(p1_gate["selected_sft_loss_weight"])
    spec = _decode_json_object(gate_spec_json, label="gate_spec_json")
    plan = _v2r2_plan(
        "v2r2-exp2-monolithic-canary",
        selected_weight=selected_weight,
    )
    manifest_hash = _manifest_hash_for_plan(manifest_set, plan)
    run_state, final_path = _resolve_existing_run(
        plan, manifest_hash=manifest_hash
    )
    if run_state != "complete" or final_path is None:
        raise RuntimeError(
            f"monolithic candidate is not complete: {run_state}"
        )
    candidate_call_id = _required_spec_string(
        spec, "call_id", label="monolithic_candidate"
    )
    candidate_call_result = _require_successful_modal_call(candidate_call_id)
    if candidate_call_result != plan.output_dir:
        raise RuntimeError(
            "monolithic call result does not match its output directory"
        )
    candidate_final = Path(final_path)
    primary, confirmation, prompt_selection, pair = (
        _inspect_v2r2_audit_pair(
        candidate_final=candidate_final,
        spec=spec,
        label="monolithic_candidate",
        )
    )
    primary_metrics = primary.get("metrics")
    confirmation_metrics = confirmation.get("metrics")
    if (
        not isinstance(primary_metrics, Mapping)
        or not isinstance(confirmation_metrics, Mapping)
    ):
        raise RuntimeError("monolithic audit pair is incomplete")
    _validate_v2r2_monolithic_gate(
        selected_weight=selected_weight,
        candidate_weight=plan.sft_loss_weight,
        candidate_step=V2R2_PROTOCOL_CANDIDATE_STEP,
        schedule_total_steps=plan.total_steps,
        manifest_leg=plan.manifest_leg,
        primary=primary_metrics,
        confirmation=confirmation_metrics,
    )
    core = {
        "gate_stage": "exp2_monolithic_protocol",
        "selected_sft_loss_weight": selected_weight,
        "parent_p1_gate_sha256": p1_gate.get("gate_sha256"),
        "candidate_call_id": candidate_call_id,
        "candidate_call_result_sha256": _v2r2_json_sha256(
            candidate_call_result
        ),
        "candidate_output_dir": plan.output_dir,
        "candidate_hf_manifest_sha256": (
            _directory_manifest_sha256(candidate_final)
        ),
        "candidate_manifest_sha256": manifest_hash,
        "candidate_step": V2R2_PROTOCOL_CANDIDATE_STEP,
        "schedule_total_steps": MONOLITHIC_STEPS,
        "manifest_leg": "p1+p2",
        "unweighted_ce": dict(
            _last_unweighted_ce(
                plan, manifest_hash=manifest_hash
            )
        ),
        "primary": primary,
        "prompt_selection": prompt_selection,
        "confirmation": confirmation,
        "validated_prompt_intersection": pair["prompt_intersection"],
    }
    path = _write_v2r2_marker(
        V2R2_EXP2_GATE_PATH,
        core=core,
        manifest_set=manifest_set,
    )
    _validate_v2r2_gate_marker(
        V2R2_EXP2_GATE_PATH,
        gate_stage="exp2_monolithic_protocol",
        manifest_set=manifest_set,
    )
    print("[v2r2-exp2-gate] " + json.dumps(core, sort_keys=True), flush=True)
    return path


@app.function(
    gpu=f"{CANARY_GPU_TYPE}:{CANARY_GPUS}",
    cpu=8.0,
    memory=32 * 1024,
    timeout=60 * 60 * 2,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
)
def train_canary(run_id: str) -> str:
    return _run_training(_canary_plan(run_id), main_process_port=29641)


@app.function(
    gpu=f"{PRODUCTION_GPU_TYPE}:{PRODUCTION_GPUS}",
    cpu=32.0,
    memory=128 * 1024,
    timeout=60 * 60 * 6,
    retries=0,
    max_containers=3,
)
def train_sft_weight_canary(
    run_id: str,
    sft_loss_weight: float,
    max_steps: int = SFT_WEIGHT_CANARY_DEFAULT_STEPS,
) -> str:
    plan = _sft_weight_canary_plan(
        run_id,
        sft_loss_weight=sft_loss_weight,
        max_steps=max_steps,
    )
    return _run_training(plan, main_process_port=29671)


@app.function(
    gpu=f"{PRODUCTION_GPU_TYPE}:{PRODUCTION_GPUS}",
    cpu=32.0,
    memory=128 * 1024,
    timeout=60 * 60 * 2,
    retries=0,
    max_containers=3,
)
def benchmark_production(
    run_id: str,
    attention_backend: str = "sdpa",
    compile_mode: str = "none",
) -> dict[str, object]:
    plan = _benchmark_plan(
        run_id,
        attention_backend=attention_backend,
        compile_mode=compile_mode,
    )
    return _run_benchmark(plan, main_process_port=29661)


@app.function(
    gpu=f"{PRODUCTION_GPU_TYPE}:{PRODUCTION_GPUS}",
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
    normalized_stage = _validate_action(stage)
    if normalized_stage in {"p1", "exp2"}:
        if run_id or init_checkpoint:
            raise ValueError(
                f"{normalized_stage} has a fixed immutable identity and does "
                "not accept run_id/init_checkpoint"
            )
        plan = _fixed_plan(normalized_stage)
    elif normalized_stage == "production-canary":
        if not run_id or init_checkpoint:
            raise ValueError(
                "production-canary requires run_id and does not accept "
                "init_checkpoint"
            )
        plan = _production_canary_plan(run_id)
    elif normalized_stage == "p2":
        if not run_id or not init_checkpoint:
            raise ValueError("p2 requires run_id and init_checkpoint")
        checkpoint_path = Path(init_checkpoint)
        fingerprint = _checkpoint_fingerprint(checkpoint_path)
        plan = _p2_plan(
            run_id=run_id,
            init_checkpoint=checkpoint_path,
            init_fingerprint=fingerprint,
        )
    else:
        raise ValueError(
            f"train_production cannot execute stage {normalized_stage!r}"
        )
    return _run_training(plan, main_process_port=29651)


@app.function(
    gpu=f"{PRODUCTION_GPU_TYPE}:{PRODUCTION_GPUS}",
    cpu=32.0,
    memory=128 * 1024,
    timeout=60 * 60 * 48,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
    max_containers=3,
)
def train_v2r2(stage: str) -> str:
    """Run only a version-scoped stage authorized by its exact gate marker."""

    data_volume.reload()
    checkpoint_volume.reload()
    rl_checkpoint_volume.reload()
    manifest_set = _load_manifest_set()
    p1_gate = _validate_v2r2_gate_marker(
        V2R2_P1_GATE_PATH,
        gate_stage="p1_protocol",
        manifest_set=manifest_set,
    )
    selected_weight = float(p1_gate["selected_sft_loss_weight"])
    if stage in {"v2r2-p1", "v2r2-exp2-monolithic-canary"}:
        plan = _v2r2_plan(stage, selected_weight=selected_weight)
    elif stage == "v2r2-exp2":
        exp2_gate = _validate_v2r2_gate_marker(
            V2R2_EXP2_GATE_PATH,
            gate_stage="exp2_monolithic_protocol",
            manifest_set=manifest_set,
        )
        if (
            exp2_gate.get("parent_p1_gate_sha256")
            != p1_gate.get("gate_sha256")
        ):
            raise RuntimeError("Exp2 gate does not descend from the P1 gate")
        if float(exp2_gate["selected_sft_loss_weight"]) != selected_weight:
            raise RuntimeError("Exp2 gate selected a different SFT weight")
        plan = _v2r2_plan(stage, selected_weight=selected_weight)
    else:
        raise ValueError(f"unsupported v2r2 training stage: {stage!r}")
    return _run_training(plan, main_process_port=29681)


@app.function(
    gpu=f"{PRODUCTION_GPU_TYPE}:{PRODUCTION_GPUS}",
    cpu=32.0,
    memory=128 * 1024,
    timeout=60 * 60 * 48,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
    max_containers=4,
)
def train_v2r3_diagnostic(sft_loss_weight: float) -> str:
    """Run one frozen diagnostic trajectory; this cannot authorize P1/Exp2."""

    plan = _v2r3_plan(sft_loss_weight)
    if not plan.diagnostic_only:
        raise RuntimeError("v2r3 execution requires diagnostic_only=true")
    return _run_training(plan, main_process_port=29691)


def _dry_run_plan(
    action: str,
    *,
    run_id: str,
    init_checkpoint: str,
    attention_backend: str = "sdpa",
    compile_mode: str = "none",
    sft_loss_weight: float = 0.0,
    max_steps: int = SFT_WEIGHT_CANARY_DEFAULT_STEPS,
    candidate_call_id: str = "",
    rollout_run_name: str = "",
    rollout_call_id: str = "",
    gate_spec_json: str = "",
) -> dict[str, object]:
    if action in {"v2r3-trajectory", "v2r3-launch-all"}:
        if run_id or init_checkpoint or gate_spec_json:
            raise ValueError(
                f"{action} accepts no run/checkpoint/gate-spec overrides"
            )
        plans = (
            [_v2r3_plan(sft_loss_weight)]
            if action == "v2r3-trajectory"
            else [
                _v2r3_plan(weight)
                for weight in V2R3_TRAJECTORY_SPECS
            ]
        )
        return {
            "action": action,
            "contract_schema": V2R3_CONTRACT_SCHEMA,
            "contract_version": V2R3_EXPERIMENT_VERSION,
            "diagnostic_only": True,
            "production_authorized": False,
            "gpus_per_trajectory": (
                f"{PRODUCTION_GPU_TYPE}:{PRODUCTION_GPUS}"
            ),
            "source_tree_sha256": SOURCE_TREE_SHA256,
            "trajectories": [
                {
                    "weight": plan.sft_loss_weight,
                    "run_name": plan.run_name,
                    "output_dir": plan.output_dir,
                    "max_steps": plan.max_steps,
                    "total_schedule_steps": plan.total_steps,
                    "snapshot_steps": list(plan.snapshot_steps),
                    "snapshot_hf_paths": [
                        str(
                            Path(plan.output_dir)
                            / "snapshots"
                            / f"step_{step}"
                            / "hf"
                        )
                        for step in plan.snapshot_steps
                    ],
                    "rollout_run_names": [
                        _v2r3_rollout_run_name(
                            plan.sft_loss_weight, step
                        )
                        for step in plan.snapshot_steps
                    ],
                }
                for plan in plans
            ],
        }
    if action == "v2r3-audit":
        if run_id or init_checkpoint or not gate_spec_json:
            raise ValueError(
                "v2r3-audit requires only --gate-spec-json"
            )
        specs = _decode_json_object_list(
            gate_spec_json, label="gate_spec_json"
        )
        return {
            "action": action,
            "contract_schema": V2R3_CONTRACT_SCHEMA,
            "contract_version": V2R3_EXPERIMENT_VERSION,
            "diagnostic_only": True,
            "production_authorized": False,
            "spec_count": len(specs),
            "report_path": str(V2R3_DIAGNOSTIC_REPORT_PATH),
            "rollout_chess_source_sha256": (
                V2R3_RL_CHESS_SOURCE_SHA256
            ),
        }
    if action == "data-prep":
        return {
            "action": action,
            "source": f"{SOURCE_REPO}@{SOURCE_REVISION}",
            "source_path": str(SOURCE_DIR),
            "source_shards": SOURCE_SHARDS,
            "source_tokens": SOURCE_TOKENS,
            "sft": f"{SFT_REPO}@{SFT_REVISION}",
            "artifact_root": str(ARTIFACT_ROOT),
            "manifest_set": str(MANIFEST_SET_PATH),
        }
    if action == "approve-gate":
        if (
            not run_id
            or init_checkpoint
            or not candidate_call_id
            or not rollout_run_name
            or not rollout_call_id
        ):
            raise ValueError(
                "approve-gate requires run_id, candidate_call_id, "
                "rollout_run_name, and rollout_call_id"
            )
        for call_id in (candidate_call_id, rollout_call_id):
            if not re.fullmatch(r"fc-[A-Za-z0-9]+", call_id):
                raise ValueError(f"invalid Modal function-call ID: {call_id}")
        plan = _sft_weight_canary_plan(
            run_id,
            sft_loss_weight=P1_SFT_LOSS_WEIGHT,
            max_steps=SFT_WEIGHT_CANARY_DEFAULT_STEPS,
        )
        return {
            "action": action,
            "candidate_run_id": _validate_run_id(run_id),
            "candidate_call_id": candidate_call_id,
            "candidate_output_dir": plan.output_dir,
            "candidate_sft_loss_weight": plan.sft_loss_weight,
            "candidate_steps": plan.max_steps,
            "rollout_run_name": _validate_run_id(rollout_run_name),
            "rollout_call_id": rollout_call_id,
            "gate_path": str(PRODUCTION_GATE_PATH),
            "source_tree_sha256": SOURCE_TREE_SHA256,
        }
    if action == "v2r2-approve-p1":
        specs = _decode_json_object_list(
            gate_spec_json, label="gate_spec_json"
        )
        return {
            "action": action,
            "candidate_count": len(specs),
            "candidate_weights": [
                float(spec.get("weight", float("nan"))) for spec in specs
            ],
            "gate_path": str(V2R2_P1_GATE_PATH),
            "contract_version": V2R2_EXPERIMENT_VERSION,
            "contract_plan_sha256": V2R2_CONTRACT_PLAN_SHA256,
            "source_tree_sha256": SOURCE_TREE_SHA256,
        }
    if action == "v2r2-approve-exp2":
        spec = _decode_json_object(
            gate_spec_json, label="gate_spec_json"
        )
        return {
            "action": action,
            "candidate_call_id": spec.get("call_id"),
            "gate_path": str(V2R2_EXP2_GATE_PATH),
            "parent_gate_path": str(V2R2_P1_GATE_PATH),
            "contract_version": V2R2_EXPERIMENT_VERSION,
            "source_tree_sha256": SOURCE_TREE_SHA256,
        }
    if action in {
        "v2r2-p1",
        "v2r2-monolithic-canary",
        "v2r2-exp2",
    }:
        if run_id or init_checkpoint or gate_spec_json:
            raise ValueError(
                f"{action} accepts no run/checkpoint/gate-spec overrides"
            )
        stage = {
            "v2r2-p1": "v2r2-p1",
            "v2r2-monolithic-canary": (
                "v2r2-exp2-monolithic-canary"
            ),
            "v2r2-exp2": "v2r2-exp2",
        }[action]
        return {
            "action": action,
            "remote_stage": stage,
            "checkpoint_root": str(V2R2_CHECKPOINT_ROOT),
            "required_gate": str(
                V2R2_EXP2_GATE_PATH
                if action == "v2r2-exp2"
                else V2R2_P1_GATE_PATH
            ),
            "selected_weight_source": str(V2R2_P1_GATE_PATH),
            "gpus": f"{PRODUCTION_GPU_TYPE}:{PRODUCTION_GPUS}",
            "source_tree_sha256": SOURCE_TREE_SHA256,
        }
    if action == "canary":
        plan = _canary_plan(run_id)
    elif action == "production-canary":
        if not run_id or init_checkpoint:
            raise ValueError(
                "production-canary requires run_id and does not accept "
                "init_checkpoint"
            )
        plan = _production_canary_plan(run_id)
    elif action == "sft-weight-canary":
        if not run_id or init_checkpoint:
            raise ValueError(
                "sft-weight-canary requires run_id and does not accept "
                "init_checkpoint"
            )
        plan = _sft_weight_canary_plan(
            run_id,
            sft_loss_weight=sft_loss_weight,
            max_steps=max_steps,
        )
    elif action == "benchmark":
        if not run_id or init_checkpoint:
            raise ValueError(
                "benchmark requires run_id and does not accept init_checkpoint"
            )
        plan = _benchmark_plan(
            run_id,
            attention_backend=attention_backend,
            compile_mode=compile_mode,
        )
    elif action in {"p1", "exp2"}:
        if run_id or init_checkpoint:
            raise ValueError(
                f"{action} does not accept run_id or init_checkpoint"
            )
        plan = _fixed_plan(action)
    elif action == "p2":
        if not run_id or not init_checkpoint:
            raise ValueError("p2 requires run_id and init_checkpoint")
        safe_id = _validate_run_id(run_id)
        return {
            "action": action,
            "run_id": safe_id,
            "init_checkpoint": init_checkpoint,
            "output_prefix": str(CHECKPOINT_ROOT / "p2" / safe_id),
            "note": (
                "The immutable output name receives the first 12 characters "
                "of the checkpoint content SHA-256 inside Modal."
            ),
            "gpus": f"{PRODUCTION_GPU_TYPE}:{PRODUCTION_GPUS}",
            "local_batch_size": PRODUCTION_LOCAL_BATCH,
            "gradient_accumulation_steps": 1,
            "total_steps": LEG_STEPS,
        }
    else:
        raise AssertionError(action)

    return {
        "action": action,
        "run_name": plan.run_name,
        "output_dir": plan.output_dir,
        "manifest_leg": plan.manifest_leg,
        "gpus": (
            f"{CANARY_GPU_TYPE}:{CANARY_GPUS}"
            if action == "canary"
            else f"{PRODUCTION_GPU_TYPE}:{PRODUCTION_GPUS}"
        ),
        "local_batch_size": plan.local_batch_size,
        "gradient_accumulation_steps": 1,
        "total_steps": plan.total_steps,
        "arc_steps": list(plan.arc_steps),
        "max_steps": plan.max_steps,
        "attention_backend": plan.attention_backend,
        "torch_compile_mode": plan.torch_compile_mode,
        "data_num_workers": plan.data_workers,
        "tracker_backend": PRODUCTION_TRACKER_BACKEND,
        "metrics_format": PRODUCTION_METRICS_FORMAT,
        "metrics_path": str(Path(plan.output_dir) / "metrics.jsonl"),
        "benchmark_only": plan.benchmark_only,
        "benchmark_warmup_steps": plan.benchmark_warmup_steps,
        "sft_loss_weight": plan.sft_loss_weight,
        "structure_canary": plan.structure_canary,
        "diagnostic_only": plan.diagnostic_only,
        "snapshot_steps": list(plan.snapshot_steps),
        "source_tree_sha256": SOURCE_TREE_SHA256,
    }


@app.local_entrypoint()
def main(
    action: str = "data-prep",
    run_id: str = "",
    init_checkpoint: str = "",
    attention_backend: str = "sdpa",
    compile_mode: str = "none",
    sft_loss_weight: float = 0.0,
    max_steps: int = SFT_WEIGHT_CANARY_DEFAULT_STEPS,
    candidate_call_id: str = "",
    rollout_run_name: str = "",
    rollout_call_id: str = "",
    gate_spec_json: str = "",
    dry_run: bool = False,
) -> None:
    normalized_action = _validate_action(action)
    if dry_run:
        print(
            json.dumps(
                _dry_run_plan(
                    normalized_action,
                    run_id=run_id,
                    init_checkpoint=init_checkpoint,
                    attention_backend=attention_backend,
                    compile_mode=compile_mode,
                    sft_loss_weight=sft_loss_weight,
                    max_steps=max_steps,
                    candidate_call_id=candidate_call_id,
                    rollout_run_name=rollout_run_name,
                    rollout_call_id=rollout_call_id,
                    gate_spec_json=gate_spec_json,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return

    if normalized_action == "v2r3-launch-all":
        if run_id or init_checkpoint or gate_spec_json:
            raise ValueError(
                "v2r3-launch-all accepts no identity overrides"
            )
        handles = [
            (
                weight,
                train_v2r3_diagnostic.spawn(
                    sft_loss_weight=weight
                ),
            )
            for weight in V2R3_TRAJECTORY_SPECS
        ]
        print(
            json.dumps(
                {
                    "action": normalized_action,
                    "contract_version": V2R3_EXPERIMENT_VERSION,
                    "diagnostic_only": True,
                    "production_authorized": False,
                    "calls": [
                        {
                            "weight": weight,
                            "function_call_id": handle.object_id,
                        }
                        for weight, handle in handles
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if normalized_action == "v2r3-trajectory":
        if run_id or init_checkpoint or gate_spec_json:
            raise ValueError(
                "v2r3-trajectory accepts only --sft-loss-weight"
            )
        plan = _v2r3_plan(sft_loss_weight)
        handle = train_v2r3_diagnostic.spawn(
            sft_loss_weight=plan.sft_loss_weight
        )
    elif normalized_action == "v2r3-audit":
        if run_id or init_checkpoint or not gate_spec_json:
            raise ValueError(
                "v2r3-audit requires only --gate-spec-json"
            )
        _decode_json_object_list(
            gate_spec_json, label="gate_spec_json"
        )
        handle = audit_v2r3_diagnostics.spawn(
            audit_specs_json=gate_spec_json
        )
    elif normalized_action == "data-prep":
        if run_id or init_checkpoint:
            raise ValueError(
                "data-prep does not accept run_id or init_checkpoint"
            )
        handle = prepare_data.spawn()
    elif normalized_action == "approve-gate":
        if (
            not run_id
            or init_checkpoint
            or not candidate_call_id
            or not rollout_run_name
            or not rollout_call_id
        ):
            raise ValueError(
                "approve-gate requires --run-id, --candidate-call-id, "
                "--rollout-run-name, and --rollout-call-id"
            )
        for call_id in (candidate_call_id, rollout_call_id):
            if not re.fullmatch(r"fc-[A-Za-z0-9]+", call_id):
                raise ValueError(
                    f"invalid Modal function-call ID: {call_id}"
                )
        handle = approve_production_gate.spawn(
            candidate_run_id=_validate_run_id(run_id),
            candidate_call_id=candidate_call_id,
            rollout_run_name=_validate_run_id(rollout_run_name),
            rollout_call_id=rollout_call_id,
        )
    elif normalized_action == "v2r2-approve-p1":
        if run_id or init_checkpoint or not gate_spec_json:
            raise ValueError(
                "v2r2-approve-p1 requires only --gate-spec-json"
            )
        _decode_json_object_list(
            gate_spec_json, label="gate_spec_json"
        )
        handle = approve_v2r2_p1_gate.spawn(
            candidate_specs_json=gate_spec_json
        )
    elif normalized_action == "v2r2-approve-exp2":
        if run_id or init_checkpoint or not gate_spec_json:
            raise ValueError(
                "v2r2-approve-exp2 requires only --gate-spec-json"
            )
        _decode_json_object(gate_spec_json, label="gate_spec_json")
        handle = approve_v2r2_exp2_gate.spawn(
            gate_spec_json=gate_spec_json
        )
    elif normalized_action in {
        "v2r2-p1",
        "v2r2-monolithic-canary",
        "v2r2-exp2",
    }:
        if run_id or init_checkpoint or gate_spec_json:
            raise ValueError(
                f"{normalized_action} accepts no identity overrides"
            )
        stage = {
            "v2r2-p1": "v2r2-p1",
            "v2r2-monolithic-canary": (
                "v2r2-exp2-monolithic-canary"
            ),
            "v2r2-exp2": "v2r2-exp2",
        }[normalized_action]
        handle = train_v2r2.spawn(stage=stage)
    elif normalized_action == "canary":
        if not run_id:
            raise ValueError("canary requires a unique --run-id")
        if init_checkpoint:
            raise ValueError("canary does not accept init_checkpoint")
        handle = train_canary.spawn(run_id=_validate_run_id(run_id))
    elif normalized_action == "production-canary":
        if not run_id:
            raise ValueError(
                "production-canary requires a unique --run-id"
            )
        if init_checkpoint:
            raise ValueError(
                "production-canary does not accept init_checkpoint"
            )
        handle = train_production.spawn(
            stage=normalized_action,
            run_id=_validate_run_id(run_id),
        )
    elif normalized_action == "sft-weight-canary":
        if not run_id:
            raise ValueError(
                "sft-weight-canary requires a unique --run-id"
            )
        if init_checkpoint:
            raise ValueError(
                "sft-weight-canary does not accept init_checkpoint"
            )
        handle = train_sft_weight_canary.spawn(
            run_id=_validate_run_id(run_id),
            sft_loss_weight=_validate_sft_loss_weight(sft_loss_weight),
            max_steps=_validate_sft_weight_canary_steps(max_steps),
        )
    elif normalized_action == "benchmark":
        if not run_id:
            raise ValueError("benchmark requires a unique --run-id")
        if init_checkpoint:
            raise ValueError(
                "benchmark does not accept --init-checkpoint"
            )
        handle = benchmark_production.spawn(
            run_id=_validate_run_id(run_id),
            attention_backend=_validate_attention_backend(
                attention_backend
            ),
            compile_mode=_validate_compile_mode(compile_mode),
        )
    elif normalized_action in {"p1", "exp2"}:
        if run_id or init_checkpoint:
            raise ValueError(
                f"{normalized_action} does not accept run_id/init_checkpoint"
            )
        handle = train_production.spawn(stage=normalized_action)
    else:
        if not run_id or not init_checkpoint:
            raise ValueError("p2 requires --run-id and --init-checkpoint")
        handle = train_production.spawn(
            stage="p2",
            run_id=_validate_run_id(run_id),
            init_checkpoint=init_checkpoint,
        )
    print(
        f"SPAWNED {normalized_action} "
        f"(function call id: {handle.object_id})"
    )
