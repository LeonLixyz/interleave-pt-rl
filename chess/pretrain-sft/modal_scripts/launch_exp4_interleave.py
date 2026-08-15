"""Immutable Modal launcher for Exp 4 positive-rollout transfer.

This app is intentionally separate from the controlled Exp 1--3 launcher.
Exp 4 is FLOP-unbounded and has three post-RL-pretraining methods, each run for
both the unfiltered (U) and dynamic-filtered (D) first RL leg:

* hard SFT on one selected positive trajectory per prompt group, then P2;
* full-vocabulary forward-KL distillation on that identical order, then P2;
* random-initialized scratch training on one unified P2 + replay shuffle.

Importing this module never submits work. Use ``--dry-run`` to inspect a local
plan. Actual functions refuse missing/incomplete RL1 artifacts, authenticate
all replay inputs, and derive output directories from content fingerprints.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal

APP_NAME = "chess-50m-interleaved-exp4"
EXP4_SCHEMA_VERSION = 1
EXP4_VERSION = "positive-rollout-transfer-v1-20260730"
EXPERIMENT_VERSION = "mix10b_sft90k_3072_v1_20260730"

GPU_TYPE = "H200"
NUM_GPUS = 8
LOCAL_BATCH_SIZE = 21
GRADIENT_ACCUMULATION_STEPS = 1
GLOBAL_BATCH_SIZE = NUM_GPUS * LOCAL_BATCH_SIZE
SEQUENCE_LENGTH = 3_072
VOCAB_SIZE = 85
P2_STEPS = 9_920
TRANSFER_LR = 1e-5
TRANSFER_EPOCHS = 1
TRANSFER_SEED = 42
TRANSFER_CONTRACT_PROVENANCE = (
    "approved_fail_closed_exp4_v1_after_plan_clarification;"
    "not_an_originally_specified_hyperparameter"
)
SCRATCH_SHUFFLE_SEED = 44
MODEL_INIT_SEED = 42
SAVE_INTERVAL = 200
MAX_RL_STEP = 1_500
EXPECTED_ROLLOUT_FILES = 1_500
BALANCED_DATA_SHA256 = (
    "bcf131d88cd9916f6ec12a12f8e85901a05e0dabc5d5a1fdebd8ea0b0ae86c30"
)
UPSTREAM_PRETRAIN_SOURCE_TREE_SHA256 = (
    "98db54b40e6af5bbbbca526b890c5cf19a96924c08c0c3e92cf0ea7edc6aba49"
)
EXP4_V1_ATTENTION_BACKEND = "sdpa"
EXP4_V1_TORCH_COMPILE_MODE = "none"
EXP4_V1_FLASH_ATTENTION_VERSION = "2.8.3"

DATA_ROOT = Path("/data/50m_interleaved_mix10b_sft90k_v1")
SOURCE_ROOT = Path("/data/pretrain_v1_20b")
SOURCE_MANIFEST = DATA_ROOT / "source_manifest.json"
SELECTION_MANIFEST = DATA_ROOT / "pretrain_selection.json"
SFT_CACHE_DIR = DATA_ROOT / "sft_cache"
P2_MANIFEST = DATA_ROOT / "legs/p2/metadata.json"
MANIFEST_SET = DATA_ROOT / "manifest_set.json"

CHECKPOINT_MOUNT = Path("/checkpoints")
PRETRAIN_ROOT = CHECKPOINT_MOUNT / "interleave_50m/pretrain" / EXPERIMENT_VERSION
P1_CHECKPOINT = PRETRAIN_ROOT / "p1_shared/final"
RL_HF_ROOT = CHECKPOINT_MOUNT / "interleave_50m/rl_hf"
EXP4_ROOT = CHECKPOINT_MOUNT / "interleave_50m/exp4" / EXP4_VERSION
RAW_RL_ROOT = Path("/rl-checkpoints/chess-rl-miles-interleave")

BASE_CONFIG = "config/configs/interleaved_50m/base_3072.yaml"
TRANSFER_CLI = "scripts/train/train_positive_transfer.py"
INTERLEAVED_CLI = "scripts/train/train_interleaved_hf.py"
WANDB_ENTITY = "jingyanshen-new-york-university"
WANDB_PROJECT = "chess-50m-interleaved-exp4"

FILTER_INPUTS: Mapping[str, Mapping[str, str]] = {
    "U": {
        "run_name": "core-e1-u-rl1-seed42",
        "teacher_name": "core-e1-u-rl1-step1500",
    },
    "D": {
        "run_name": "core-e1-d-rl1-seed42",
        "teacher_name": "core-e1-d-rl1-step1500",
    },
}
METHODS = frozenset({"hard-sft", "soft-kl", "scratch-replay"})
_SAFE_COMPONENT_RE = re.compile(r"[a-z0-9][a-z0-9.-]{0,79}")
_ROLLOUT_RE = re.compile(r"rollout_(\d+)\.jsonl")
_SUMMARY_RE = re.compile(r"rollout_(\d+)\.summary\.json")
_SUPPORTED_ATTENTION_BACKENDS = frozenset({"sdpa", "flash_attention_2"})
_SUPPORTED_COMPILE_MODES = frozenset(
    {"none", "default", "reduce-overhead", "max-autotune"}
)
RUNTIME_PACKAGE_CONTRACT = {
    "python": "3.11",
    "cuda_base": "nvidia/cuda:12.8.0-devel-ubuntu22.04",
    "torch": "2.9.0",
    "accelerate": "1.10.1",
    "transformers": "4.57.0",
    "tokenizers": "0.22.1",
    "safetensors": "0.6.2",
}
EXPECTED_MODEL_CONFIG = {
    "model_type": "qwen3",
    "vocab_size": VOCAB_SIZE,
    "max_position_embeddings": SEQUENCE_LENGTH,
    "hidden_size": 512,
    "head_dim": 128,
    "num_hidden_layers": 12,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "intermediate_size": 1536,
    "tie_word_embeddings": True,
}


def _read_main_launcher_backend(path: Path) -> tuple[str, str]:
    """Assert that the controlled launcher still matches frozen Exp4 v1."""

    if not path.is_file():
        backend = os.environ.get("CHESS_EXP4_ATTENTION_BACKEND", "").strip()
        version = os.environ.get("CHESS_EXP4_FLASH_ATTENTION_VERSION", "").strip()
        compile_mode = os.environ.get("CHESS_EXP4_TORCH_COMPILE_MODE", "").strip()
        if not backend or not version or not compile_mode:
            raise RuntimeError(
                "remote Exp4 import lacks the embedded attention contract"
            )
    else:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        values: dict[str, str] = {}
        wanted = {
            "PRODUCTION_ATTENTION_BACKEND",
            "PINNED_FLASH_ATTENTION_VERSION",
            "PRODUCTION_TORCH_COMPILE_MODE",
        }
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if (
                isinstance(target, ast.Name)
                and target.id in wanted
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                values[target.id] = node.value.value
        if set(values) != wanted:
            raise RuntimeError(
                f"controlled launcher lacks frozen runtime constants: {path}"
            )
        backend = values["PRODUCTION_ATTENTION_BACKEND"]
        version = values["PINNED_FLASH_ATTENTION_VERSION"]
        compile_mode = values["PRODUCTION_TORCH_COMPILE_MODE"]
    observed = (backend, compile_mode, version)
    expected = (
        EXP4_V1_ATTENTION_BACKEND,
        EXP4_V1_TORCH_COMPILE_MODE,
        EXP4_V1_FLASH_ATTENTION_VERSION,
    )
    if observed != expected:
        raise RuntimeError(
            "controlled launcher drifted from frozen Exp4 v1 runtime: "
            f"{observed} != {expected}"
        )
    return backend, version


_LOCAL_REPO_DIR = Path(__file__).resolve().parent.parent
_read_main_launcher_backend(_LOCAL_REPO_DIR / "modal_scripts/launch_50m_interleaved.py")
PRODUCTION_ATTENTION_BACKEND = EXP4_V1_ATTENTION_BACKEND
PINNED_FLASH_ATTENTION_VERSION = EXP4_V1_FLASH_ATTENTION_VERSION
PRODUCTION_TORCH_COMPILE_MODE = EXP4_V1_TORCH_COMPILE_MODE
if PRODUCTION_ATTENTION_BACKEND not in _SUPPORTED_ATTENTION_BACKENDS:
    raise RuntimeError(
        f"unsupported production attention backend {PRODUCTION_ATTENTION_BACKEND!r}"
    )
if not re.fullmatch(r"\d+\.\d+\.\d+", PINNED_FLASH_ATTENTION_VERSION):
    raise RuntimeError("FlashAttention config compatibility version must be explicit")
if PRODUCTION_TORCH_COMPILE_MODE not in _SUPPORTED_COMPILE_MODES:
    raise RuntimeError(
        f"unsupported production torch compile mode {PRODUCTION_TORCH_COMPILE_MODE!r}"
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _normalize_filter(value: str) -> str:
    normalized = value.strip().upper()
    if normalized not in FILTER_INPUTS:
        raise ValueError("filter_setting must be U or D")
    return normalized


def _normalize_method(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    if normalized not in METHODS:
        raise ValueError(f"method must be one of {', '.join(sorted(METHODS))}")
    return normalized


def _safe_component(value: str, *, name: str) -> str:
    normalized = value.strip().lower()
    if not _SAFE_COMPONENT_RE.fullmatch(normalized):
        raise ValueError(f"{name} is not a safe lowercase path component")
    return normalized


def _content_fingerprint(kind: str, contract: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "schema_version": EXP4_SCHEMA_VERSION,
                "kind": kind,
                "contract": dict(contract),
            }
        )
    )


def _source_tree_digest(root: Path) -> str:
    candidates: list[Path] = []
    for relative in ("config", "llm_tokens", "scripts", "training"):
        base = root / relative
        if base.is_dir():
            candidates.extend(
                path
                for path in base.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}
            )
    launcher = root / "modal_scripts" / Path(__file__).name
    if launcher.is_file():
        candidates.append(launcher)
    digest = hashlib.sha256()
    for path in sorted(set(candidates), key=lambda item: str(item.relative_to(root))):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _effective_source_digest(computed: str, override: str | None) -> str:
    value = (override or computed).strip()
    if not _is_sha256(value):
        raise RuntimeError("CHESS_EXP4_SOURCE_TREE_SHA256 must be a lowercase SHA-256")
    return value


def _require_under(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    root_resolved = root.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(f"{label} must be under {root_resolved}: {resolved}")
    return resolved


def _checkpoint_files(checkpoint: Path) -> list[Path]:
    checkpoint = _require_under(checkpoint, CHECKPOINT_MOUNT, label="HF checkpoint")
    if not checkpoint.is_dir():
        raise NotADirectoryError(checkpoint)
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"missing config.json under {checkpoint}")
    weights = sorted(checkpoint.glob("model*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"missing safetensors weights under {checkpoint}")
    state = checkpoint / "interleaved_training_state.json"
    if not state.is_file():
        raise FileNotFoundError(
            f"missing interleaved_training_state.json under {checkpoint}"
        )
    for required_tokenizer_asset in ("tokenizer.py", "vocab.json"):
        if not (checkpoint / required_tokenizer_asset).is_file():
            raise FileNotFoundError(
                f"missing required custom tokenizer asset "
                f"{required_tokenizer_asset} under {checkpoint}"
            )
    config = json.loads((checkpoint / "config.json").read_text())
    mismatches = {
        key: (config.get(key), value)
        for key, value in EXPECTED_MODEL_CONFIG.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"checkpoint is not the exact 47.245M architecture: {mismatches}"
        )
    files = list(weights)
    for name in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "interleaved_training_state.json",
    ):
        candidate = checkpoint / name
        if candidate.is_file():
            files.append(candidate)
    for pattern in (
        "tokenizer*",
        "vocab*",
        "merges*",
        "special_tokens_map.json",
        "added_tokens.json",
        "sentencepiece*",
        "spiece*",
    ):
        files.extend(path for path in checkpoint.glob(pattern) if path.is_file())
    return sorted(set(files), key=lambda item: str(item.relative_to(checkpoint)))


def _checkpoint_fingerprint(checkpoint: Path) -> str:
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    files = _checkpoint_files(checkpoint)
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(checkpoint)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(8 << 20):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_p1_pretrain_contract(checkpoint: Path | None = None) -> dict[str, Any]:
    """Authenticate P1 against the frozen production pretraining contract."""

    from omegaconf import OmegaConf

    checkpoint = P1_CHECKPOINT if checkpoint is None else checkpoint
    checkpoint = _require_under(checkpoint, CHECKPOINT_MOUNT, label="P1 checkpoint")
    config_path = checkpoint.parent / "config.yaml"
    state_path = checkpoint / "interleaved_training_state.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"P1 production config snapshot is missing: {config_path}"
        )
    if not state_path.is_file():
        raise FileNotFoundError(f"P1 clean HF trainer state is missing: {state_path}")
    loaded = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(loaded, Mapping):
        raise ValueError("P1 production config snapshot is not a mapping")
    model = loaded.get("model")
    training = loaded.get("training")
    provenance = loaded.get("provenance")
    if not all(isinstance(value, Mapping) for value in (model, training, provenance)):
        raise ValueError("P1 production config snapshot lacks runtime provenance")
    config_expected = {
        ("model", "attn_implementation"): EXP4_V1_ATTENTION_BACKEND,
        ("model", "flash_attention_version"): EXP4_V1_FLASH_ATTENTION_VERSION,
        ("training", "torch_compile"): EXP4_V1_TORCH_COMPILE_MODE,
        ("training", "total_steps"): P2_STEPS,
        ("training", "local_batch_size"): LOCAL_BATCH_SIZE,
        ("training", "gradient_accumulation_steps"): (GRADIENT_ACCUMULATION_STEPS),
        ("provenance", "experiment_version"): EXPERIMENT_VERSION,
        ("provenance", "attention_backend"): EXP4_V1_ATTENTION_BACKEND,
        ("provenance", "flash_attention_version"): (EXP4_V1_FLASH_ATTENTION_VERSION),
        ("provenance", "torch_compile_mode"): EXP4_V1_TORCH_COMPILE_MODE,
        ("provenance", "source_tree_sha256"): (UPSTREAM_PRETRAIN_SOURCE_TREE_SHA256),
    }
    sections = {
        "model": model,
        "training": training,
        "provenance": provenance,
    }
    config_mismatches = {
        f"{section}.{key}": (sections[section].get(key), expected)
        for (section, key), expected in config_expected.items()
        if sections[section].get(key) != expected
    }
    if config_mismatches:
        raise ValueError(f"P1 frozen production config mismatch: {config_mismatches}")

    state = json.loads(state_path.read_text())
    runtime = state.get("runtime_provenance")
    if not isinstance(runtime, Mapping):
        raise ValueError("P1 clean HF state lacks runtime provenance")
    state_expected = {
        "global_step": P2_STEPS,
        "local_batch_size": LOCAL_BATCH_SIZE,
        "world_size": NUM_GPUS,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "attention_backend": EXP4_V1_ATTENTION_BACKEND,
        "torch_compile_mode": EXP4_V1_TORCH_COMPILE_MODE,
    }
    state_mismatches = {
        key: (state.get(key), expected)
        for key, expected in state_expected.items()
        if state.get(key) != expected
    }
    runtime_expected = {
        "attention_backend": EXP4_V1_ATTENTION_BACKEND,
        "torch_compile_mode": EXP4_V1_TORCH_COMPILE_MODE,
        # SDPA must not silently activate/import FlashAttention at runtime.
        "flash_attention_version": None,
    }
    runtime_mismatches = {
        key: (runtime.get(key), expected)
        for key, expected in runtime_expected.items()
        if runtime.get(key) != expected
    }
    if state_mismatches or runtime_mismatches:
        raise ValueError(
            "P1 frozen production state mismatch: "
            f"state={state_mismatches}, runtime={runtime_mismatches}"
        )
    contract = {
        "source_tree_sha256": UPSTREAM_PRETRAIN_SOURCE_TREE_SHA256,
        "attention_backend": EXP4_V1_ATTENTION_BACKEND,
        "torch_compile_mode": EXP4_V1_TORCH_COMPILE_MODE,
        "flash_attention_config_version": EXP4_V1_FLASH_ATTENTION_VERSION,
        "flash_attention_runtime_version": None,
        "config_sha256": _sha256_file(config_path),
        "clean_hf_state_sha256": _sha256_file(state_path),
    }
    return {
        **contract,
        "contract_sha256": _content_fingerprint(
            "exp4-upstream-p1-production-contract", contract
        ),
    }


def _teacher_checkpoint(filter_setting: str) -> Path:
    setting = _normalize_filter(filter_setting)
    return RL_HF_ROOT / FILTER_INPUTS[setting]["teacher_name"]


def _rollout_source(filter_setting: str) -> Path:
    setting = _normalize_filter(filter_setting)
    return (
        RAW_RL_ROOT
        / FILTER_INPUTS[setting]["run_name"]
        / "rollouts/all_attempts_positive"
    )


def _rollout_inventory(
    source: Path,
    *,
    expected_files: int = EXPECTED_ROLLOUT_FILES,
) -> dict[str, Any]:
    """Authenticate a complete, paired RL1 all-attempt positive stream."""

    source = _require_under(source, RAW_RL_ROOT, label="rollout source")
    jsonls: dict[int, Path] = {}
    summaries: dict[int, Path] = {}
    for path in source.iterdir():
        if not path.is_file():
            continue
        if match := _ROLLOUT_RE.fullmatch(path.name):
            jsonls[int(match.group(1))] = path
        elif match := _SUMMARY_RE.fullmatch(path.name):
            summaries[int(match.group(1))] = path
        elif path.suffix == ".tmp":
            raise RuntimeError(f"incomplete rollout temporary file: {path}")
    expected_ids = set(range(expected_files))
    if set(jsonls) != expected_ids or set(summaries) != expected_ids:
        raise RuntimeError(
            "RL1 positive artifacts are incomplete: "
            f"jsonl={len(jsonls)} summaries={len(summaries)} "
            f"expected={expected_files}"
        )

    files: list[dict[str, Any]] = []
    total_positive_rows = 0
    total_attempted_groups = 0
    for rollout_id in sorted(expected_ids):
        summary_path = summaries[rollout_id]
        summary = json.loads(summary_path.read_text())
        if (
            int(summary.get("rollout_id", -1)) != rollout_id
            or int(summary.get("step", -1)) != rollout_id
            or summary.get("sampling_scope")
            != "all_completed_attempts_before_dynamic_filter"
        ):
            raise ValueError(f"invalid positive summary {summary_path}")
        positive_rows = int(summary.get("positive_completed_samples", -1))
        attempted_groups = int(summary.get("attempted_groups", -1))
        if positive_rows < 0 or attempted_groups <= 0:
            raise ValueError(f"invalid counts in {summary_path}")
        total_positive_rows += positive_rows
        total_attempted_groups += attempted_groups
        for kind, path in (
            ("jsonl", jsonls[rollout_id]),
            ("summary", summary_path),
        ):
            files.append(
                {
                    "kind": kind,
                    "rollout_id": rollout_id,
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    contract = {
        "source": str(source),
        "rollout_ids": [0, expected_files - 1],
        "rollout_files": expected_files,
        "summary_files": expected_files,
        "total_positive_completed_rows": total_positive_rows,
        "total_attempted_groups": total_attempted_groups,
        "files": files,
    }
    return {
        **contract,
        "inventory_sha256": _content_fingerprint(
            "exp4-rollout-source-inventory", contract
        ),
    }


@dataclass(frozen=True)
class RLRunProvenance:
    root_manifest_path: Path
    launch_manifest_paths: tuple[Path, ...]
    root_manifest_sha256: str
    launch_manifests: tuple[Mapping[str, Any], ...]
    identity_sha256: str
    bundle_sha256: str
    bundle_contract: Mapping[str, Any]


def _provenance_identity_sha256(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(identity),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()


def _command_sha256(command: Sequence[Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            [str(value) for value in command],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _validate_origin_hf_identity(
    recorded: Mapping[str, Any],
    actual_root: Path,
) -> None:
    """Match every RL-recorded origin-HF file to the mounted P1 checkpoint."""

    actual_root = _require_under(actual_root, CHECKPOINT_MOUNT, label="RL origin HF")
    files = recorded.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("RL provenance has no origin-HF file inventory")
    expected: dict[str, tuple[int, str]] = {}
    manifest_rows: list[str] = []
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("invalid RL origin-HF file record")
        relative = str(item.get("path", ""))
        size = int(item.get("bytes", -1))
        digest = item.get("sha256")
        if (
            not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or size < 0
            or not _is_sha256(digest)
            or relative in expected
        ):
            raise ValueError("invalid RL origin-HF inventory entry")
        expected[relative] = (size, str(digest))
        manifest_rows.append(f"{relative}\t{size}\t{digest}\n")
    actual_files = {
        path.relative_to(actual_root).as_posix(): path
        for path in actual_root.rglob("*")
        if path.is_file()
        and not any(
            part
            in {
                ".git",
                ".mypy_cache",
                ".pytest_cache",
                ".ruff_cache",
                ".venv",
                "__pycache__",
                "wandb",
            }
            for part in path.relative_to(actual_root).parts
        )
    }
    if set(actual_files) != set(expected):
        raise ValueError("RL provenance origin-HF file set mismatch")
    for relative, path in actual_files.items():
        size, digest = expected[relative]
        if path.stat().st_size != size or _sha256_file(path) != digest:
            raise ValueError(f"RL provenance origin-HF content mismatch: {relative}")
    manifest_sha = hashlib.sha256("".join(sorted(manifest_rows)).encode()).hexdigest()
    if (
        recorded.get("manifest_sha256") != manifest_sha
        or int(recorded.get("file_count", -1)) != len(expected)
        or int(recorded.get("total_bytes", -1))
        != sum(size for size, _ in expected.values())
    ):
        raise ValueError("RL provenance origin-HF manifest mismatch")


def _validate_rl_run_provenance(filter_setting: str) -> RLRunProvenance:
    """Require the immutable RL identity and every append-only launch record."""

    setting = _normalize_filter(filter_setting)
    run_name = FILTER_INPUTS[setting]["run_name"]
    run_root = _require_under(RAW_RL_ROOT / run_name, RAW_RL_ROOT, label="RL run root")
    root_path = run_root / "run_provenance.json"
    if not root_path.is_file():
        raise FileNotFoundError(
            f"RL1 run provenance is required before Exp4: {root_path}"
        )
    root = json.loads(root_path.read_text())
    identity = root.get("identity")
    if int(root.get("schema_version", -1)) != 1 or not isinstance(identity, Mapping):
        raise ValueError("invalid RL run provenance root schema")
    identity_sha = _provenance_identity_sha256(identity)
    if root.get("identity_sha256") != identity_sha:
        raise ValueError("RL run provenance identity hash mismatch")
    run = identity.get("run")
    profile = identity.get("policy_update_profile")
    semantics = identity.get("fixed_rl_semantics")
    balanced = identity.get("balanced_data")
    sources = identity.get("sources")
    runtime = identity.get("runtime")
    required_mappings = (run, profile, semantics, balanced, sources, runtime)
    if not all(isinstance(value, Mapping) for value in required_mappings):
        raise ValueError("RL run provenance identity is incomplete")
    expected_run = {
        "app_name": "chess-interleave-rl",
        "run_name": run_name,
        "model_id": "interleave_47m_qwen3",
        "num_rollout": MAX_RL_STEP,
        "dynamic_filter": setting == "D",
        "rollout_seed": 42,
        "save_interval": 40,
        "eval_interval": 0,
        "canary": False,
    }
    if dict(run) != expected_run:
        raise ValueError("RL run provenance run identity mismatch")
    expected_profile = {
        "name": "small-model-h200",
        "gradient_checkpointing": False,
        "train_backend": "fsdp",
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 8,
        "gpu_type": "H200",
    }
    if any(profile.get(key) != value for key, value in expected_profile.items()):
        raise ValueError("RL run provenance policy profile mismatch")
    if profile.get("max_tokens_per_gpu") not in {65_536, 131_072}:
        raise ValueError("RL run provenance token budget is unverified")
    expected_semantics = {
        "rollout_batch_size": 256,
        "samples_per_prompt": 8,
        "global_batch_size": 2_048,
        "policy_loss_agg_mode": "token-mean",
        "advantage_estimator": "grpo",
        "cispo": False,
        "optimizer": "adamw",
        "lr": 1e-5,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
        "weight_decay": 0.01,
        "kl_loss_coef": 0.001,
        "rollout_max_prompt_len": 512,
        "rollout_max_response_len": 2_560,
        "rollout_max_context_len": 3_072,
    }
    if any(semantics.get(key) != value for key, value in expected_semantics.items()):
        raise ValueError("RL run provenance fixed semantics mismatch")
    if balanced.get("sha256") != BALANCED_DATA_SHA256:
        raise ValueError("RL run provenance balanced-data hash mismatch")
    if not isinstance(sources.get("chess_rl_miles"), Mapping) or not isinstance(
        sources.get("miles"), Mapping
    ):
        raise ValueError("RL run provenance source identities are missing")
    for source_name in ("chess_rl_miles", "miles"):
        if not _is_sha256(sources[source_name].get("manifest_sha256")):
            raise ValueError(f"RL run provenance source hash is invalid: {source_name}")
    if not runtime.get("image") or not isinstance(runtime.get("packages"), Mapping):
        raise ValueError("RL run provenance runtime identity is incomplete")
    origin = identity.get("origin_hf")
    if not isinstance(origin, Mapping):
        raise ValueError("RL run provenance has no origin-HF identity")
    _validate_p1_pretrain_contract()
    _validate_origin_hf_identity(origin, P1_CHECKPOINT)

    initial_command = root.get("initial_command")
    initial_command_sha = root.get("initial_command_sha256")
    if (
        not isinstance(initial_command, list)
        or not _is_sha256(initial_command_sha)
        or _command_sha256(initial_command) != initial_command_sha
    ):
        raise ValueError("RL run provenance initial command mismatch")
    launch_dir = run_root / "provenance"
    launch_paths = tuple(sorted(launch_dir.glob("launch_*.json")))
    if not launch_paths:
        raise FileNotFoundError(
            f"RL1 has no append-only launch provenance: {launch_dir}"
        )
    launch_records: list[Mapping[str, Any]] = []
    launch_contracts: list[dict[str, Any]] = []
    saw_initial = False
    for path in launch_paths:
        value = json.loads(path.read_text())
        command = value.get("command")
        command_sha = value.get("command_sha256")
        if (
            int(value.get("schema_version", -1)) != 1
            or value.get("identity_sha256") != identity_sha
            or not isinstance(command, list)
            or not _is_sha256(command_sha)
            or _command_sha256(command) != command_sha
            or path.name != f"launch_{command_sha[:16]}.json"
        ):
            raise ValueError(f"invalid RL launch provenance: {path}")
        if command_sha == initial_command_sha:
            saw_initial = True
        file_sha = _sha256_file(path)
        launch_records.append(value)
        launch_contracts.append(
            {
                "path": path.name,
                "sha256": file_sha,
                "command_sha256": command_sha,
            }
        )
    if not saw_initial:
        raise ValueError("RL initial command has no launch provenance record")
    bundle_contract = {
        "run_name": run_name,
        "identity_sha256": identity_sha,
        "root_manifest_sha256": _sha256_file(root_path),
        "launch_manifests": launch_contracts,
    }
    return RLRunProvenance(
        root_manifest_path=root_path,
        launch_manifest_paths=launch_paths,
        root_manifest_sha256=str(bundle_contract["root_manifest_sha256"]),
        launch_manifests=tuple(launch_records),
        identity_sha256=identity_sha,
        bundle_sha256=_content_fingerprint(
            "exp4-rl-run-provenance-bundle", bundle_contract
        ),
        bundle_contract=bundle_contract,
    )


def _copy_rl_run_provenance(
    provenance: RLRunProvenance,
    destination: Path,
) -> dict[str, Any]:
    """Copy the exact authenticated RL documents into the replay artifact."""

    destination.mkdir(parents=True, exist_ok=True)
    root_copy = destination / "run_provenance.json"
    copies: list[dict[str, Any]] = []
    source_pairs = [
        (
            provenance.root_manifest_path,
            root_copy,
            provenance.root_manifest_sha256,
        )
    ]
    launch_copy_dir = destination / "provenance"
    launch_hashes = {
        str(value["path"]): str(value["sha256"])
        for value in provenance.bundle_contract["launch_manifests"]
    }
    source_pairs.extend(
        (source, launch_copy_dir / source.name, launch_hashes[source.name])
        for source in provenance.launch_manifest_paths
    )
    for source, target, expected_sha in source_pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
        if _sha256_file(source) != expected_sha:
            raise RuntimeError(f"RL provenance mutated after authentication: {source}")
        if target.exists():
            if not target.is_file() or _sha256_file(target) != expected_sha:
                raise ValueError(f"copied RL provenance differs from source: {target}")
        else:
            shutil.copyfile(source, target)
            if _sha256_file(target) != expected_sha:
                raise RuntimeError(f"RL provenance copy failed: {target}")
        copies.append(
            {
                "path": str(target.relative_to(destination.parent)),
                "sha256": expected_sha,
            }
        )
    return {
        "identity_sha256": provenance.identity_sha256,
        "bundle_sha256": provenance.bundle_sha256,
        "bundle_contract": dict(provenance.bundle_contract),
        "copied_files": copies,
    }


def _assert_extraction_inputs_unchanged(
    *,
    filter_setting: str,
    source: Path,
    inventory_sha256: str,
    rl_provenance: RLRunProvenance,
    teacher_sha256: str,
) -> None:
    """Close the validation/use window before publishing a replay marker."""

    current_inventory = _rollout_inventory(source)
    if current_inventory["inventory_sha256"] != inventory_sha256:
        raise RuntimeError("RL rollout inputs mutated during positive extraction")
    current_provenance = _validate_rl_run_provenance(filter_setting)
    if (
        current_provenance.identity_sha256 != rl_provenance.identity_sha256
        or current_provenance.bundle_sha256 != rl_provenance.bundle_sha256
    ):
        raise RuntimeError("RL provenance mutated during positive extraction")
    if _checkpoint_fingerprint(_teacher_checkpoint(filter_setting)) != teacher_sha256:
        raise RuntimeError("RL teacher checkpoint mutated during positive extraction")


def _replay_contract(
    *,
    filter_setting: str,
    run_name: str,
    policy_checkpoint: str,
    policy_checkpoint_sha256: str,
    rollout_inventory_sha256: str,
    rl_run_provenance_identity_sha256: str,
    rl_run_provenance_bundle_sha256: str,
    source_tree_sha256: str,
) -> dict[str, Any]:
    hashes = (
        policy_checkpoint_sha256,
        rollout_inventory_sha256,
        rl_run_provenance_identity_sha256,
        rl_run_provenance_bundle_sha256,
        source_tree_sha256,
    )
    if not all(_is_sha256(value) for value in hashes):
        raise ValueError("positive replay contract hashes must be SHA-256")
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "exp4_version": EXP4_VERSION,
        "filter_setting": _normalize_filter(filter_setting),
        "run_name": _safe_component(run_name, name="run_name"),
        "policy_checkpoint": policy_checkpoint,
        "policy_checkpoint_sha256": policy_checkpoint_sha256,
        "rollout_inventory_sha256": rollout_inventory_sha256,
        "rl_run_provenance_identity_sha256": (rl_run_provenance_identity_sha256),
        "rl_run_provenance_bundle_sha256": (rl_run_provenance_bundle_sha256),
        "extraction_seed": TRANSFER_SEED,
        "max_rl_step": MAX_RL_STEP,
        "response_limit": 2560,
        "context_limit": SEQUENCE_LENGTH,
        "vocab_size": VOCAB_SIZE,
        "upstream_pretrain_source_tree_sha256": (UPSTREAM_PRETRAIN_SOURCE_TREE_SHA256),
        "source_tree_sha256": source_tree_sha256,
    }


def _method_contract(
    *,
    method: str,
    filter_setting: str,
    replay_sha256: str,
    replay_manifest_sha256: str,
    replay_artifact_sha256: str,
    p1_checkpoint_sha256: str | None,
    teacher_checkpoint_sha256: str | None,
    p2_manifest_sha256: str,
    source_tree_sha256: str,
) -> dict[str, Any]:
    normalized_method = _normalize_method(method)
    setting = _normalize_filter(filter_setting)
    values = (
        replay_sha256,
        replay_manifest_sha256,
        replay_artifact_sha256,
        p2_manifest_sha256,
        source_tree_sha256,
    )
    if not all(_is_sha256(value) for value in values):
        raise ValueError("method contract hashes must be full SHA-256 values")
    if normalized_method == "scratch-replay":
        if p1_checkpoint_sha256 is not None:
            raise ValueError("scratch replay must not bind/load P1 weights")
        if teacher_checkpoint_sha256 is not None:
            raise ValueError("scratch replay must not bind/load teacher weights")
    else:
        if not _is_sha256(p1_checkpoint_sha256):
            raise ValueError("positive transfer requires the P1 content hash")
        if normalized_method == "soft-kl":
            if not _is_sha256(teacher_checkpoint_sha256):
                raise ValueError("soft KL requires the teacher content hash")
        elif teacher_checkpoint_sha256 is not None:
            raise ValueError("hard SFT must not load/bind a teacher")
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "exp4_version": EXP4_VERSION,
        "method": normalized_method,
        "filter_setting": setting,
        "replay_sha256": replay_sha256,
        "replay_manifest_sha256": replay_manifest_sha256,
        "replay_artifact_sha256": replay_artifact_sha256,
        "p1_checkpoint_sha256": p1_checkpoint_sha256,
        "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
        "p2_manifest_sha256": p2_manifest_sha256,
        "topology": {
            "gpu_type": GPU_TYPE,
            "world_size": NUM_GPUS,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        },
        "runtime_backend": {
            "attention": PRODUCTION_ATTENTION_BACKEND,
            "flash_attention_config_version": PINNED_FLASH_ATTENTION_VERSION,
            "flash_attention_runtime_version": None,
            "torch_compile": PRODUCTION_TORCH_COMPILE_MODE,
        },
        "upstream_pretrain_contract": {
            "source_tree_sha256": UPSTREAM_PRETRAIN_SOURCE_TREE_SHA256,
            "attention_backend": EXP4_V1_ATTENTION_BACKEND,
            "torch_compile_mode": EXP4_V1_TORCH_COMPILE_MODE,
            "flash_attention_config_version": EXP4_V1_FLASH_ATTENTION_VERSION,
            "flash_attention_runtime_version": None,
        },
        "runtime_packages": RUNTIME_PACKAGE_CONTRACT,
        "transfer": (
            None
            if normalized_method == "scratch-replay"
            else {
                "contract_provenance": TRANSFER_CONTRACT_PROVENANCE,
                "learning_rate": TRANSFER_LR,
                "epochs": TRANSFER_EPOCHS,
                "seed": TRANSFER_SEED,
                "weight_decay": 0.1,
                "adam_betas": [0.9, 0.95],
                "adam_eps": 1e-8,
                "scheduler": "constant",
                "temperature": 1.0,
            }
        ),
        "p2": {
            "steps": P2_STEPS,
            "peak_lr": 1e-3,
            "eta_min": 1e-5,
            "warmup_ratio": 0.05,
        },
        "scratch": (
            {
                "shuffle_seed": SCRATCH_SHUFFLE_SEED,
                "model_init_seed": MODEL_INIT_SEED,
            }
            if normalized_method == "scratch-replay"
            else None
        ),
        "source_tree_sha256": source_tree_sha256,
    }


@dataclass(frozen=True)
class ReplayArtifact:
    replay_path: Path
    manifest_path: Path
    artifact_path: Path
    replay_sha256: str
    manifest_sha256: str
    artifact_sha256: str
    rows: int
    filter_setting: str


@dataclass(frozen=True)
class MethodPlan:
    method: str
    filter_setting: str
    fingerprint: str
    root: Path
    contract: Mapping[str, Any]
    replay: ReplayArtifact
    p1_checkpoint: Path | None
    teacher_checkpoint: Path | None
    p2_manifest_sha256: str

    @property
    def transfer_root(self) -> Path:
        return self.root / "transfer"

    @property
    def transfer_final(self) -> Path:
        return self.transfer_root / "final"

    @property
    def final_training_root(self) -> Path:
        name = "scratch_p2_replay" if self.method == "scratch-replay" else "p2"
        return self.root / name

    @property
    def final_hf(self) -> Path:
        return self.final_training_root / "final"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_copied_rl_provenance(
    replay_root: Path,
    artifact: Mapping[str, Any],
    artifact_contract: Mapping[str, Any],
) -> None:
    provenance = artifact.get("rl_run_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("positive replay lacks copied RL run provenance")
    bundle = provenance.get("bundle_contract")
    copies = provenance.get("copied_files")
    if not isinstance(bundle, Mapping) or not isinstance(copies, list):
        raise ValueError("copied RL run provenance is incomplete")
    if (
        provenance.get("identity_sha256")
        != artifact_contract.get("rl_run_provenance_identity_sha256")
        or provenance.get("bundle_sha256")
        != artifact_contract.get("rl_run_provenance_bundle_sha256")
        or provenance.get("bundle_sha256")
        != _content_fingerprint("exp4-rl-run-provenance-bundle", dict(bundle))
    ):
        raise ValueError("copied RL run provenance identity mismatch")
    actual_copies: dict[str, str] = {}
    for item in copies:
        if not isinstance(item, Mapping):
            raise ValueError("invalid copied RL provenance entry")
        relative = str(item.get("path", ""))
        digest = item.get("sha256")
        path = (replay_root / relative).resolve()
        if (
            not relative
            or ".." in Path(relative).parts
            or not path.is_relative_to(replay_root.resolve())
            or not path.is_file()
            or not _is_sha256(digest)
            or _sha256_file(path) != digest
            or relative in actual_copies
        ):
            raise ValueError("copied RL provenance file mismatch")
        actual_copies[relative] = str(digest)
    expected_copies = {
        "rl_run_provenance/run_provenance.json": str(bundle.get("root_manifest_sha256"))
    }
    launch_contracts = bundle.get("launch_manifests")
    if not isinstance(launch_contracts, list):
        raise ValueError("copied RL launch provenance list is missing")
    for item in launch_contracts:
        if not isinstance(item, Mapping):
            raise ValueError("invalid copied RL launch contract")
        name = str(item.get("path", ""))
        digest = item.get("sha256")
        if not re.fullmatch(r"launch_[0-9a-f]{16}\.json", name) or not _is_sha256(
            digest
        ):
            raise ValueError("invalid copied RL launch filename/hash")
        expected_copies[f"rl_run_provenance/provenance/{name}"] = str(digest)
    if actual_copies != expected_copies:
        raise ValueError("copied RL provenance file set mismatch")


def _validate_replay_artifact(
    replay_path: Path,
    replay_manifest_path: Path,
    *,
    filter_setting: str,
) -> ReplayArtifact:
    replay_path = _require_under(replay_path, EXP4_ROOT, label="positive replay")
    replay_manifest_path = _require_under(
        replay_manifest_path, EXP4_ROOT, label="positive replay manifest"
    )
    if replay_path.parent != replay_manifest_path.parent:
        raise ValueError("replay and replay manifest must share one artifact dir")
    artifact_path = replay_path.parent / "artifact_manifest.json"
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    manifest = json.loads(replay_manifest_path.read_text())
    artifact = json.loads(artifact_path.read_text())
    replay_sha = _sha256_file(replay_path)
    manifest_sha = _sha256_file(replay_manifest_path)
    artifact_sha = _sha256_file(artifact_path)
    output = manifest.get("output")
    config = manifest.get("config")
    if (
        manifest.get("kind") != "exp4_positive_rollout_replay"
        or not isinstance(output, Mapping)
        or not isinstance(config, Mapping)
        or output.get("sha256") != replay_sha
        or Path(str(output.get("path"))).resolve() != replay_path
        or _normalize_filter(str(config.get("filter_setting")))
        != _normalize_filter(filter_setting)
    ):
        raise ValueError("positive replay manifest contract mismatch")
    rows = int(output.get("rows", 0))
    if rows <= 0:
        raise ValueError("positive replay artifact is empty")
    artifact_contract = artifact.get("contract")
    if (
        artifact.get("kind") != "exp4_positive_replay_artifact"
        or artifact.get("state") != "complete"
        or not isinstance(artifact_contract, Mapping)
        or artifact.get("fingerprint")
        != _content_fingerprint("exp4-positive-replay", dict(artifact_contract))
        or artifact.get("replay_sha256") != replay_sha
        or artifact.get("replay_manifest_sha256") != manifest_sha
        or artifact_contract.get("filter_setting") != _normalize_filter(filter_setting)
        or artifact_contract.get("upstream_pretrain_source_tree_sha256")
        != UPSTREAM_PRETRAIN_SOURCE_TREE_SHA256
    ):
        raise ValueError("positive replay artifact manifest mismatch")
    _validate_copied_rl_provenance(
        replay_path.parent,
        artifact,
        artifact_contract,
    )
    return ReplayArtifact(
        replay_path=replay_path,
        manifest_path=replay_manifest_path,
        artifact_path=artifact_path,
        replay_sha256=replay_sha,
        manifest_sha256=manifest_sha,
        artifact_sha256=artifact_sha,
        rows=rows,
        filter_setting=_normalize_filter(filter_setting),
    )


def _load_p2_manifest_hash() -> str:
    if not MANIFEST_SET.is_file() or not P2_MANIFEST.is_file():
        raise FileNotFoundError("mixed-data manifests are missing; run data-prep first")
    manifest_set = json.loads(MANIFEST_SET.read_text())
    entries = manifest_set.get("manifests")
    if not isinstance(entries, Mapping) or not isinstance(entries.get("p2"), Mapping):
        raise ValueError("manifest_set has no P2 entry")
    expected = entries["p2"].get("sha256")
    actual = _sha256_file(P2_MANIFEST)
    if expected != actual:
        raise ValueError(f"P2 manifest hash mismatch: {actual} != {expected}")
    metadata = json.loads(P2_MANIFEST.read_text())
    required = {
        "leg": "p2",
        "world_size": NUM_GPUS,
        "local_batch_size": LOCAL_BATCH_SIZE,
        "total_steps": P2_STEPS,
        "sequence_length": SEQUENCE_LENGTH,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in required.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"P2 manifest settings mismatch: {mismatches}")
    return actual


def _build_method_plan(
    *,
    method: str,
    filter_setting: str,
    replay_path: Path,
    replay_manifest_path: Path,
) -> MethodPlan:
    method = _normalize_method(method)
    setting = _normalize_filter(filter_setting)
    replay = _validate_replay_artifact(
        replay_path, replay_manifest_path, filter_setting=setting
    )
    p2_hash = _load_p2_manifest_hash()
    p1 = None if method == "scratch-replay" else P1_CHECKPOINT
    teacher = _teacher_checkpoint(setting) if method == "soft-kl" else None
    if p1 is not None:
        _validate_p1_pretrain_contract(p1)
    p1_hash = _checkpoint_fingerprint(p1) if p1 is not None else None
    teacher_hash = _checkpoint_fingerprint(teacher) if teacher is not None else None
    contract = _method_contract(
        method=method,
        filter_setting=setting,
        replay_sha256=replay.replay_sha256,
        replay_manifest_sha256=replay.manifest_sha256,
        replay_artifact_sha256=replay.artifact_sha256,
        p1_checkpoint_sha256=p1_hash,
        teacher_checkpoint_sha256=teacher_hash,
        p2_manifest_sha256=p2_hash,
        source_tree_sha256=SOURCE_TREE_SHA256,
    )
    fingerprint = _content_fingerprint("exp4-method-plan", contract)
    root = EXP4_ROOT / setting.lower() / method / fingerprint
    return MethodPlan(
        method=method,
        filter_setting=setting,
        fingerprint=fingerprint,
        root=root,
        contract=contract,
        replay=replay,
        p1_checkpoint=p1,
        teacher_checkpoint=teacher,
        p2_manifest_sha256=p2_hash,
    )


def _ensure_plan_manifest(plan: MethodPlan) -> None:
    path = plan.root / "plan.json"
    expected = {
        "schema_version": EXP4_SCHEMA_VERSION,
        "kind": "exp4_method_plan",
        "state": "immutable",
        "fingerprint": plan.fingerprint,
        "contract": dict(plan.contract),
        "paths": {
            "root": str(plan.root),
            "replay": str(plan.replay.replay_path),
            "replay_manifest": str(plan.replay.manifest_path),
            "final_hf": str(plan.final_hf),
        },
    }
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(f"existing immutable plan differs: {path}")
    else:
        _atomic_json(path, expected)


def _hf_is_complete(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        _checkpoint_files(path)
    except (FileNotFoundError, NotADirectoryError, ValueError):
        return False
    return True


def _transfer_hf_is_complete(path: Path) -> bool:
    """Validate an intermediate transfer export without calling it a P2 endpoint."""

    if not path.is_dir():
        return False
    required = (
        "config.json",
        "tokenizer.py",
        "vocab.json",
        "positive_transfer_state.json",
    )
    if any(not (path / name).is_file() for name in required):
        return False
    if not list(path.glob("model*.safetensors")):
        return False
    try:
        config = json.loads((path / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return all(
        config.get(key) == expected for key, expected in EXPECTED_MODEL_CONFIG.items()
    )


def _transfer_command(plan: MethodPlan) -> list[str]:
    if plan.method not in {"hard-sft", "soft-kl"}:
        raise ValueError("transfer command is only for hard-sft/soft-kl")
    assert plan.p1_checkpoint is not None
    mode = "hard_sft" if plan.method == "hard-sft" else "soft_kl"
    command = [
        "accelerate",
        "launch",
        "--multi_gpu",
        "--num_processes",
        str(NUM_GPUS),
        "--mixed_precision",
        "bf16",
        "--main_process_port",
        "29671",
        TRANSFER_CLI,
        "--mode",
        mode,
        "--student-checkpoint",
        str(plan.p1_checkpoint),
        "--replay",
        str(plan.replay.replay_path),
        "--replay-manifest",
        str(plan.replay.manifest_path),
        "--output-dir",
        str(plan.transfer_final),
        "--learning-rate",
        str(TRANSFER_LR),
        "--local-batch-size",
        str(LOCAL_BATCH_SIZE),
        "--epochs",
        str(TRANSFER_EPOCHS),
        "--seed",
        str(TRANSFER_SEED),
        "--weight-decay",
        "0.1",
        "--temperature",
        "1.0",
        "--num-workers",
        "4",
        "--save-interval",
        str(SAVE_INTERVAL),
        "--checkpoint-dir",
        str(plan.transfer_root / "latest"),
        "--run-fingerprint",
        plan.fingerprint,
        "--attn-implementation",
        PRODUCTION_ATTENTION_BACKEND,
        "--flash-attention-version",
        PINNED_FLASH_ATTENTION_VERSION,
    ]
    if plan.teacher_checkpoint is not None:
        command.extend(["--teacher-checkpoint", str(plan.teacher_checkpoint)])
    resume_marker = plan.transfer_root / "latest/positive_transfer_resume.json"
    if resume_marker.is_file():
        command.extend(["--resume", str(plan.transfer_root / "latest")])
    return command


def _p2_overrides(
    plan: MethodPlan,
    *,
    output_root: Path,
    manifest_path: Path,
    manifest_hash: str,
    total_steps: int,
    floor_tail_steps: int,
) -> list[str]:
    run_name = (
        f"exp4-{plan.filter_setting.lower()}-{plan.method}-{plan.fingerprint[:12]}"
    )
    return [
        f"training.output_dir={output_root}",
        f"training.run_name={run_name}",
        f"training.seed={MODEL_INIT_SEED}",
        f"training.local_batch_size={LOCAL_BATCH_SIZE}",
        f"training.gradient_accumulation_steps={GRADIENT_ACCUMULATION_STEPS}",
        f"training.total_steps={total_steps}",
        f"training.arc_steps=[{P2_STEPS}]",
        f"training.floor_tail_steps={floor_tail_steps}",
        f"model.attn_implementation={PRODUCTION_ATTENTION_BACKEND}",
        (f"model.flash_attention_version={PINNED_FLASH_ATTENTION_VERSION}"),
        f"training.torch_compile={PRODUCTION_TORCH_COMPILE_MODE}",
        "training.reset_optimizer_between_arcs=true",
        "training.scheduler.warmup_ratio=0.05",
        "training.scheduler.eta_min=1e-5",
        "training.optimizer.lr=1e-3",
        "training.optimizer.weight_decay=0.1",
        "training.optimizer.betas=[0.9,0.95]",
        f"training.save_interval={SAVE_INTERVAL}",
        f"data.source_root={SOURCE_ROOT}",
        f"data.source_manifest_path={SOURCE_MANIFEST}",
        f"data.selection_manifest_path={SELECTION_MANIFEST}",
        f"data.sft_cache_dir={SFT_CACHE_DIR}",
        f"data.leg_manifest_path={manifest_path}",
        f"data.expected_manifest_hash={manifest_hash}",
        f"logging.project={WANDB_PROJECT}",
        f"logging.entity={WANDB_ENTITY}",
        f"provenance.experiment_version={EXPERIMENT_VERSION}",
        f"provenance.exp4_version={EXP4_VERSION}",
        f"provenance.exp4_method={plan.method}",
        f"provenance.exp4_filter_setting={plan.filter_setting}",
        f"provenance.exp4_plan_sha256={plan.fingerprint}",
        f"provenance.attention_backend={PRODUCTION_ATTENTION_BACKEND}",
        (f"provenance.flash_attention_version={PINNED_FLASH_ATTENTION_VERSION}"),
        (f"provenance.torch_compile_mode={PRODUCTION_TORCH_COMPILE_MODE}"),
        f"provenance.source_tree_sha256={SOURCE_TREE_SHA256}",
        (
            "provenance.upstream_pretrain_source_tree_sha256="
            f"{UPSTREAM_PRETRAIN_SOURCE_TREE_SHA256}"
        ),
    ]


def _interleaved_command(
    plan: MethodPlan,
    *,
    output_root: Path,
    manifest_path: Path,
    manifest_hash: str,
    total_steps: int,
    floor_tail_steps: int,
    weights_only: Path | None,
) -> list[str]:
    command = [
        "accelerate",
        "launch",
        "--multi_gpu",
        "--num_processes",
        str(NUM_GPUS),
        "--mixed_precision",
        "bf16",
        "--main_process_port",
        "29681",
        INTERLEAVED_CLI,
        "--config",
        BASE_CONFIG,
        "--override",
        *_p2_overrides(
            plan,
            output_root=output_root,
            manifest_path=manifest_path,
            manifest_hash=manifest_hash,
            total_steps=total_steps,
            floor_tail_steps=floor_tail_steps,
        ),
    ]
    resume = output_root / "latest/trainer_state.json"
    if resume.is_file():
        command.extend(["--resume", str(output_root / "latest")])
    elif weights_only is not None:
        command.extend(["--weights-only", str(weights_only)])
    return command


def _run_subprocess(command: Sequence[str]) -> None:
    print("[exp4] " + " ".join(command), flush=True)
    result = subprocess.run(
        list(command),
        cwd="/root/chess",
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
    )
    # Publish the most recent complete resume marker even when the child
    # returns a failure; a retry is then bound to the same content plan.
    checkpoint_volume.commit()
    if result.returncode:
        raise RuntimeError(f"Exp4 child exited with {result.returncode}")


def _validate_transfer_final(plan: MethodPlan) -> None:
    if not _transfer_hf_is_complete(plan.transfer_final):
        raise RuntimeError(f"incomplete transfer output: {plan.transfer_final}")
    state = json.loads(
        (plan.transfer_final / "positive_transfer_state.json").read_text()
    )
    expected_steps = (plan.replay.rows + GLOBAL_BATCH_SIZE - 1) // GLOBAL_BATCH_SIZE
    expected_teacher = plan.replay.rows if plan.method == "soft-kl" else 0
    expected = {
        "completed_steps": expected_steps,
        "processed_positive_examples": plan.replay.rows,
        "teacher_forward_examples": expected_teacher,
    }
    mismatches = {
        key: (state.get(key), value)
        for key, value in expected.items()
        if state.get(key) != value
    }
    contract = state.get("contract")
    if (
        mismatches
        or not isinstance(contract, Mapping)
        or contract.get("run_fingerprint") != plan.fingerprint
    ):
        raise ValueError(f"positive transfer completion mismatch: {mismatches}")


def _write_completion(plan: MethodPlan) -> dict[str, Any]:
    final_sha = _checkpoint_fingerprint(plan.final_hf)
    payload = {
        "schema_version": EXP4_SCHEMA_VERSION,
        "kind": "exp4_method_complete",
        "state": "complete",
        "fingerprint": plan.fingerprint,
        "method": plan.method,
        "filter_setting": plan.filter_setting,
        "final_hf": str(plan.final_hf),
        "final_hf_sha256": final_sha,
        "replay_rows": plan.replay.rows,
    }
    path = plan.root / "complete.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError(f"existing completion marker differs: {path}")
    if not path.exists():
        _atomic_json(path, payload)
    checkpoint_volume.commit()
    return payload


def _validate_interleaved_final(
    plan: MethodPlan,
    *,
    expected_manifest_hash: str,
    expected_steps: int,
    expected_floor_tail_steps: int,
) -> None:
    """Authenticate the clean HF endpoint against its exact data cursor."""

    if not _hf_is_complete(plan.final_hf):
        raise RuntimeError(f"incomplete final HF output: {plan.final_hf}")
    state_path = plan.final_hf / "interleaved_training_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    state = json.loads(state_path.read_text())
    expected = {
        "global_step": expected_steps,
        "manifest_hash": expected_manifest_hash,
        "manifest_cursor": expected_steps,
        "arc_steps": [P2_STEPS],
        "floor_tail_steps": expected_floor_tail_steps,
        "local_batch_size": LOCAL_BATCH_SIZE,
        "world_size": NUM_GPUS,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    }
    mismatches = {
        key: (state.get(key), value)
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Exp4 interleaved endpoint state mismatch: {mismatches}")


repo_dir = _LOCAL_REPO_DIR
SOURCE_TREE_SHA256 = _effective_source_digest(
    _source_tree_digest(repo_dir),
    os.environ.get("CHESS_EXP4_SOURCE_TREE_SHA256"),
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "curl", "git")
    .pip_install(
        "ninja==1.13.0",
        "packaging==25.0",
        "setuptools==80.9.0",
        "wheel==0.45.1",
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
            "CHESS_EXP4_SOURCE_TREE_SHA256": SOURCE_TREE_SHA256,
            "CHESS_EXP4_ATTENTION_BACKEND": PRODUCTION_ATTENTION_BACKEND,
            "CHESS_EXP4_FLASH_ATTENTION_VERSION": (PINNED_FLASH_ATTENTION_VERSION),
            "CHESS_EXP4_TORCH_COMPILE_MODE": (PRODUCTION_TORCH_COMPILE_MODE),
        }
    )
    .add_local_dir(str(repo_dir / "scripts"), remote_path="/root/chess/scripts")
    .add_local_dir(str(repo_dir / "training"), remote_path="/root/chess/training")
    .add_local_dir(str(repo_dir / "config"), remote_path="/root/chess/config")
    .add_local_dir(str(repo_dir / "llm_tokens"), remote_path="/root/chess/llm_tokens")
)

data_volume = modal.Volume.from_name(
    "rl-reasoning-training-data", create_if_missing=False
)
checkpoint_volume = modal.Volume.from_name(
    "rl-reasoning-checkpoints", create_if_missing=False
)
rl_volume = modal.Volume.from_name(
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
        "/rl-checkpoints": rl_volume,
    },
)


@app.function(
    cpu=16.0,
    memory=64 * 1024,
    timeout=60 * 60 * 12,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
)
def extract_positive(filter_setting: str) -> dict[str, Any]:
    """Create or authenticate one content-addressed U/D positive corpus."""

    from training.positive_replay import ExtractionConfig, extract_positive_replay

    setting = _normalize_filter(filter_setting)
    data_volume.reload()
    checkpoint_volume.reload()
    rl_volume.reload()
    source = _rollout_source(setting)
    inventory = _rollout_inventory(source)
    rl_provenance = _validate_rl_run_provenance(setting)
    teacher = _teacher_checkpoint(setting)
    teacher_sha = _checkpoint_fingerprint(teacher)
    run_name = FILTER_INPUTS[setting]["run_name"]
    contract = _replay_contract(
        filter_setting=setting,
        run_name=run_name,
        policy_checkpoint=str(teacher),
        policy_checkpoint_sha256=teacher_sha,
        rollout_inventory_sha256=str(inventory["inventory_sha256"]),
        rl_run_provenance_identity_sha256=(rl_provenance.identity_sha256),
        rl_run_provenance_bundle_sha256=rl_provenance.bundle_sha256,
        source_tree_sha256=SOURCE_TREE_SHA256,
    )
    fingerprint = _content_fingerprint("exp4-positive-replay", contract)
    root = EXP4_ROOT / setting.lower() / "positive-replay" / fingerprint
    replay = root / "positive_replay.jsonl"
    manifest = root / "positive_replay.manifest.json"
    artifact = root / "artifact_manifest.json"
    if artifact.is_file():
        validated = _validate_replay_artifact(replay, manifest, filter_setting=setting)
        if json.loads(artifact.read_text()).get("contract") != contract:
            raise ValueError("existing replay artifact contract differs")
        return {
            "state": "complete",
            "fingerprint": fingerprint,
            "replay": str(validated.replay_path),
            "manifest": str(validated.manifest_path),
            "rows": validated.rows,
        }
    overwrite_partial = False
    if root.exists():
        unexpected = {
            path.name
            for path in root.iterdir()
            if path.name
            not in {
                replay.name,
                manifest.name,
                "rl_run_provenance",
            }
        }
        if unexpected:
            raise FileExistsError(
                f"unrecognized partial replay artifacts in {root}: {sorted(unexpected)}"
            )
        # A retry may have failed after atomically publishing the replay JSONL
        # but before the final artifact marker. Regenerate only these known
        # deterministic files from the same authenticated content contract.
        overwrite_partial = replay.exists() or manifest.exists()
    root.mkdir(parents=True, exist_ok=True)
    payload = extract_positive_replay(
        [source],
        output_path=replay,
        manifest_path=manifest,
        config=ExtractionConfig(
            run_id=run_name,
            policy_checkpoint=str(teacher),
            filter_setting=setting,
            extraction_seed=TRANSFER_SEED,
            response_limit=2560,
            context_limit=SEQUENCE_LENGTH,
            vocab_size=VOCAB_SIZE,
            max_rl_step=MAX_RL_STEP,
            require_all_attempts_scope=True,
        ),
        overwrite=overwrite_partial,
    )
    _assert_extraction_inputs_unchanged(
        filter_setting=setting,
        source=source,
        inventory_sha256=str(inventory["inventory_sha256"]),
        rl_provenance=rl_provenance,
        teacher_sha256=teacher_sha,
    )
    copied_rl_provenance = _copy_rl_run_provenance(
        rl_provenance,
        root / "rl_run_provenance",
    )
    _assert_extraction_inputs_unchanged(
        filter_setting=setting,
        source=source,
        inventory_sha256=str(inventory["inventory_sha256"]),
        rl_provenance=rl_provenance,
        teacher_sha256=teacher_sha,
    )
    artifact_payload = {
        "schema_version": EXP4_SCHEMA_VERSION,
        "kind": "exp4_positive_replay_artifact",
        "state": "complete",
        "fingerprint": fingerprint,
        "contract": contract,
        "rollout_inventory": inventory,
        "rl_run_provenance": copied_rl_provenance,
        "replay_sha256": payload["output"]["sha256"],
        "replay_manifest_sha256": _sha256_file(manifest),
        "rows": payload["output"]["rows"],
    }
    _atomic_json(artifact, artifact_payload)
    validated = _validate_replay_artifact(replay, manifest, filter_setting=setting)
    checkpoint_volume.commit()
    return {
        "state": "complete",
        "fingerprint": fingerprint,
        "replay": str(validated.replay_path),
        "manifest": str(validated.manifest_path),
        "rows": validated.rows,
    }


@app.function(
    cpu=16.0,
    memory=64 * 1024,
    timeout=60 * 60 * 4,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
)
def validate_method(
    method: str,
    filter_setting: str,
    replay_path: str,
    replay_manifest_path: str,
) -> dict[str, Any]:
    """CPU-only preflight that returns the exact content-addressed plan."""

    data_volume.reload()
    checkpoint_volume.reload()
    plan = _build_method_plan(
        method=method,
        filter_setting=filter_setting,
        replay_path=Path(replay_path),
        replay_manifest_path=Path(replay_manifest_path),
    )
    return {
        "state": "ready",
        "method": plan.method,
        "filter_setting": plan.filter_setting,
        "fingerprint": plan.fingerprint,
        "root": str(plan.root),
        "final_hf": str(plan.final_hf),
        "replay_rows": plan.replay.rows,
        "contract": dict(plan.contract),
    }


@app.function(
    gpu=f"{GPU_TYPE}:{NUM_GPUS}",
    cpu=32.0,
    memory=128 * 1024,
    timeout=60 * 60 * 48,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
    max_containers=6,
)
def train_method(
    method: str,
    filter_setting: str,
    replay_path: str,
    replay_manifest_path: str,
) -> dict[str, Any]:
    """Run one immutable hard/soft/scratch Exp4 pre-RL endpoint."""

    data_volume.reload()
    checkpoint_volume.reload()
    plan = _build_method_plan(
        method=method,
        filter_setting=filter_setting,
        replay_path=Path(replay_path),
        replay_manifest_path=Path(replay_manifest_path),
    )
    complete_path = plan.root / "complete.json"
    if complete_path.is_file():
        existing = json.loads(complete_path.read_text())
        expected_existing = {
            "schema_version": EXP4_SCHEMA_VERSION,
            "kind": "exp4_method_complete",
            "state": "complete",
            "fingerprint": plan.fingerprint,
            "method": plan.method,
            "filter_setting": plan.filter_setting,
            "final_hf": str(plan.final_hf),
            "final_hf_sha256": _checkpoint_fingerprint(plan.final_hf),
            "replay_rows": plan.replay.rows,
        }
        if existing != expected_existing:
            raise ValueError("existing Exp4 completion marker differs")
        return existing

    _ensure_plan_manifest(plan)
    checkpoint_volume.commit()
    if plan.method in {"hard-sft", "soft-kl"}:
        if not _transfer_hf_is_complete(plan.transfer_final):
            _run_subprocess(_transfer_command(plan))
        _validate_transfer_final(plan)
        if not _hf_is_complete(plan.final_hf):
            _run_subprocess(
                _interleaved_command(
                    plan,
                    output_root=plan.final_training_root,
                    manifest_path=P2_MANIFEST,
                    manifest_hash=plan.p2_manifest_sha256,
                    total_steps=P2_STEPS,
                    floor_tail_steps=0,
                    weights_only=plan.transfer_final,
                )
            )
        _validate_interleaved_final(
            plan,
            expected_manifest_hash=plan.p2_manifest_sha256,
            expected_steps=P2_STEPS,
            expected_floor_tail_steps=0,
        )
    else:
        from training.scratch_replay import build_scratch_replay_manifest

        scratch = build_scratch_replay_manifest(
            p2_manifest_path=P2_MANIFEST,
            replay_path=plan.replay.replay_path,
            replay_manifest_path=plan.replay.manifest_path,
            output_dir=plan.root / "scratch_manifest",
            shuffle_seed=SCRATCH_SHUFFLE_SEED,
            model_init_seed=MODEL_INIT_SEED,
            validate_all_replay_rows=True,
        )
        checkpoint_volume.commit()
        if not _hf_is_complete(plan.final_hf):
            _run_subprocess(
                _interleaved_command(
                    plan,
                    output_root=plan.final_training_root,
                    manifest_path=scratch.metadata_path,
                    manifest_hash=_sha256_file(scratch.metadata_path),
                    total_steps=scratch.total_steps,
                    floor_tail_steps=scratch.floor_tail_steps,
                    weights_only=None,
                )
            )
        _validate_interleaved_final(
            plan,
            expected_manifest_hash=_sha256_file(scratch.metadata_path),
            expected_steps=scratch.total_steps,
            expected_floor_tail_steps=scratch.floor_tail_steps,
        )
    return _write_completion(plan)


def _dry_run(
    *,
    action: str,
    method: str,
    filter_setting: str,
    replay_path: str,
    replay_manifest_path: str,
) -> dict[str, Any]:
    setting = _normalize_filter(filter_setting)
    normalized_action = action.strip().lower()
    if normalized_action == "extract":
        if method or replay_path or replay_manifest_path:
            raise ValueError("extract accepts only filter_setting")
        return {
            "action": "extract",
            "filter_setting": setting,
            "rl_run_name": FILTER_INPUTS[setting]["run_name"],
            "source": str(_rollout_source(setting)),
            "policy_checkpoint": str(_teacher_checkpoint(setting)),
            "output_prefix": str(EXP4_ROOT / setting.lower() / "positive-replay"),
            "required_rollout_jsonl_summary_pairs": EXPECTED_ROLLOUT_FILES,
            "runtime_backend": {
                "attention": PRODUCTION_ATTENTION_BACKEND,
                "flash_attention_config_version": PINNED_FLASH_ATTENTION_VERSION,
                "flash_attention_runtime_version": None,
                "torch_compile": PRODUCTION_TORCH_COMPILE_MODE,
            },
            "upstream_pretrain_source_tree_sha256": (
                UPSTREAM_PRETRAIN_SOURCE_TREE_SHA256
            ),
            "note": "The exact output suffix is the remote content SHA-256.",
        }
    if normalized_action not in {"validate", "train"}:
        raise ValueError("action must be extract, validate, or train")
    normalized_method = _normalize_method(method)
    if not replay_path or not replay_manifest_path:
        raise ValueError(f"{normalized_action} requires replay paths")
    return {
        "action": normalized_action,
        "method": normalized_method,
        "filter_setting": setting,
        "replay": replay_path,
        "replay_manifest": replay_manifest_path,
        "gpus": None if normalized_action == "validate" else f"{GPU_TYPE}:{NUM_GPUS}",
        "local_batch_size": LOCAL_BATCH_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "runtime_backend": {
            "attention": PRODUCTION_ATTENTION_BACKEND,
            "flash_attention_config_version": PINNED_FLASH_ATTENTION_VERSION,
            "flash_attention_runtime_version": None,
            "torch_compile": PRODUCTION_TORCH_COMPILE_MODE,
        },
        "upstream_pretrain_source_tree_sha256": (UPSTREAM_PRETRAIN_SOURCE_TREE_SHA256),
        "positive_transfer_contract": (
            None
            if normalized_method == "scratch-replay"
            else {
                "epochs": TRANSFER_EPOCHS,
                "learning_rate": TRANSFER_LR,
                "seed": TRANSFER_SEED,
                "provenance": TRANSFER_CONTRACT_PROVENANCE,
            }
        ),
        "output_prefix": str(EXP4_ROOT / setting.lower() / normalized_method),
        "note": (
            "Remote validation authenticates every input and appends the "
            "full content SHA-256 output suffix."
        ),
    }


@app.local_entrypoint()
def main(
    action: str = "extract",
    method: str = "",
    filter_setting: str = "",
    replay_path: str = "",
    replay_manifest_path: str = "",
    dry_run: bool = False,
    wait: bool = False,
) -> None:
    normalized_action = action.strip().lower()
    setting = _normalize_filter(filter_setting)
    if dry_run:
        print(
            json.dumps(
                _dry_run(
                    action=normalized_action,
                    method=method,
                    filter_setting=setting,
                    replay_path=replay_path,
                    replay_manifest_path=replay_manifest_path,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if normalized_action == "extract":
        if method or replay_path or replay_manifest_path:
            raise ValueError("extract accepts only --filter-setting")
        handle = extract_positive.spawn(filter_setting=setting)
    elif normalized_action in {"validate", "train"}:
        if not method or not replay_path or not replay_manifest_path:
            raise ValueError(
                f"{normalized_action} requires --method, --replay-path, "
                "and --replay-manifest-path"
            )
        function = validate_method if normalized_action == "validate" else train_method
        handle = function.spawn(
            method=_normalize_method(method),
            filter_setting=setting,
            replay_path=replay_path,
            replay_manifest_path=replay_manifest_path,
        )
    else:
        raise ValueError("action must be extract, validate, or train")
    print(f"SPAWNED {normalized_action}: {handle.object_id}")
    if wait:
        print(json.dumps(handle.get(), indent=2, sort_keys=True))
