"""SFT-injection ablation: PT {5B,10B} x SFT {none,half,full} x mixed(w=1.0)/staged.

Wave-1 arms (all independent, launched in parallel):
  A1 5B+half  A2 5B+full  A3 10B+half  A4 10B+full   (mixed, sft_loss_weight=1.0)
  PT-5B, PT-10B (pure PT, cosine floor 1e-4 -> staged SFT arms B1-B4 follow)

Actions:
  modal run ... --action prep                 # build all selections+manifests (CPU)
  modal run ... --action canary --arm A1      # 1-update production-topology canary
  modal run --detach ... --action train --arm A1
  modal run --detach ... --action launch-wave1   # canary-gated, spawns all 6

Uses the existing verified primitives from training.interleaved_data and the
authenticated clean SFT cache (tensors proven == clean HF repo).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import modal

# --- identities -------------------------------------------------------------
EXPERIMENT_VERSION = "sft_injection_ablation_v1_20260801"
APP_NAME = "chess-sft-injection-ablation"

SEQUENCE_LENGTH = 3072
WORLD_SIZE = 8
LOCAL_BATCH_SIZE = 21
GLOBAL_BATCH_SIZE = WORLD_SIZE * LOCAL_BATCH_SIZE
GPU_TYPE = "H200"

SOURCE_REPO = "chess-pre-to-post/pretrain_v1_20b"
SOURCE_REVISION = "07dd1b7090ca5f0fb05ef624c26b20bff19483c8"
SOURCE_DIR = Path("/data/pretrain_v1_20b")

# clean SFT source (verified: 0 <verify>, 0 <unk>, 77,717 rows)
SFT_REPO = "Pre-to-Post-2/200M_SFT_dataset"
SFT_REVISION = "fd343bd28f6a40fc3dab4dcfb6e74c11b7a20b90"
SFT_ROWS = 77_717
SFT_SPLIT_SEED = 42          # same seed-42 split as all prior experiments
SFT_HALF_ROWS = 38_858       # P1 half
SFT_TARGETS_HALF = 26_289_598
SFT_TARGETS_FULL = 52_482_753

# authenticated existing artifacts (reused, never mutated)
V2R1_ROOT = Path("/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate")
SFT_CACHE_DIR = V2R1_ROOT / "sft_cache"
SFT_CACHE_HASH = (
    "d82378522d43d5db3e8333588c24b1f864bb9e8ecd46303e1d2cd2e31d31df98"
)
SOURCE_MANIFEST_TEMPLATE = V2R1_ROOT / "source_manifest.json"
SOURCE_MANIFEST_FILE_SHA256 = (
    "7f144d2329628759f2529540bfb9b10692e374d0c8b1933ec43c7c634b979253"
)

ARTIFACT_ROOT = Path(f"/data/{EXPERIMENT_VERSION}")
MANIFEST_SET_PATH = ARTIFACT_ROOT / "manifest_set.json"
MANIFEST_SET_A2R = ARTIFACT_ROOT / "manifest_set_a2r.json"
CHECKPOINT_ROOT = Path(
    f"/checkpoints/interleave_50m/pretrain/{EXPERIMENT_VERSION}"
)

BASE_CONFIG = "config/configs/interleaved_50m/base_3072.yaml"
TRAIN_CLI = "scripts/train/train_interleaved_hf.py"
WANDB_ENTITY = "jingyanshen-new-york-university"
WANDB_PROJECT = "chess-sft-injection-ablation"

PT_5B = 5_000_000_000
PT_10B = 10_000_000_000


def _steps(pt_records: int, sft_rows: int) -> int:
    total = pt_records + sft_rows
    return (total + (-total % GLOBAL_BATCH_SIZE)) // GLOBAL_BATCH_SIZE


@dataclass(frozen=True)
class Arm:
    key: str
    pt_tokens: int
    sft_mode: str            # "none" | "half" | "full"
    eta_min: float           # cosine floor: A arms 1e-5, PT arms 1e-4
    shuffle_seed: int

    @property
    def pt_records(self) -> int:
        return math.ceil(self.pt_tokens / SEQUENCE_LENGTH)

    @property
    def sft_rows(self) -> int:
        return {"none": 0, "half": SFT_HALF_ROWS, "full": SFT_ROWS}[self.sft_mode]

    @property
    def sft_targets(self) -> int:
        return {"none": 0, "half": SFT_TARGETS_HALF, "full": SFT_TARGETS_FULL}[
            self.sft_mode
        ]

    @property
    def total_steps(self) -> int:
        return _steps(self.pt_records, self.sft_rows)

    @property
    def slug(self) -> str:
        return self.key.lower().replace("-", "_")


ARMS: dict[str, Arm] = {
    "A1": Arm("A1", PT_5B, "half", 1e-5, 1042),
    "A2": Arm("A2", PT_5B, "full", 1e-5, 2042),
    "A3": Arm("A3", PT_10B, "half", 1e-5, 3042),
    "A4": Arm("A4", PT_10B, "full", 1e-5, 4042),
    "PT-5B": Arm("PT-5B", PT_5B, "none", 1e-4, 5042),
    "PT-10B": Arm("PT-10B", PT_10B, "none", 1e-4, 6042),
    # reseed replication of A2: only the data shuffle differs (2042 -> 20421)
    "A2R": Arm("A2R", PT_5B, "full", 1e-5, 20421),
}
EXPECTED_STEPS = {
    "A1": 9_920, "A2": 10_151, "A3": 19_608, "A4": 19_839,
    "PT-5B": 9_689, "PT-10B": 19_377, "A2R": 10_151,
}
MANIFEST_SET_A2R_PATH = None  # set below ARTIFACT_ROOT definition
for _k, _a in ARMS.items():
    if _a.total_steps != EXPECTED_STEPS[_k]:
        raise RuntimeError(f"{_k} step accounting drifted: {_a.total_steps}")

REPO_DIR = Path(__file__).resolve().parent.parent


def _source_tree_sha256() -> str:
    paths: list[Path] = []
    for relative in ("config", "llm_tokens", "scripts", "training"):
        root = REPO_DIR / relative
        if root.is_dir():
            paths.extend(
                p for p in root.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
                and p.suffix not in {".pyc", ".pyo"}
            )
    paths.append(Path(__file__).resolve())
    digest = hashlib.sha256()
    for p in sorted(set(paths), key=lambda x: str(x.relative_to(REPO_DIR))):
        digest.update(str(p.relative_to(REPO_DIR)).encode())
        digest.update(b"\0")
        digest.update(p.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


SOURCE_TREE_SHA256 = os.environ.get(
    "SFT_ABLATION_SOURCE_TREE_SHA256", ""
).strip() or _source_tree_sha256()

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("curl", "git")
    .pip_install(
        "torch==2.9.0", "accelerate==1.10.1", "transformers==4.57.0",
        "datasets==4.2.0", "huggingface-hub==0.35.3", "numpy==2.2.6",
        "safetensors==0.6.2", "pyarrow>=17.0.0", "pandas>=2.0.0",
        "pyyaml>=6.0", "omegaconf>=2.3.0", "wandb>=0.19.0", "einops>=0.7.0",
        "tokenizers==0.22.1", "tqdm>=4.66.0", "chess>=1.11.0",
        "sentencepiece>=0.2.0",
    )
    .env({
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/root/chess",
        "WANDB_ENTITY": WANDB_ENTITY,
        "SFT_ABLATION_SOURCE_TREE_SHA256": SOURCE_TREE_SHA256,
    })
    .add_local_dir(str(REPO_DIR / "scripts"), remote_path="/root/chess/scripts")
    .add_local_dir(str(REPO_DIR / "training"), remote_path="/root/chess/training")
    .add_local_dir(str(REPO_DIR / "config"), remote_path="/root/chess/config")
    .add_local_dir(str(REPO_DIR / "llm_tokens"), remote_path="/root/chess/llm_tokens")
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
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
)


# --- helpers ---------------------------------------------------------------
def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def _sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    tmp.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _validate_inputs() -> None:
    if _sha256_file(SOURCE_MANIFEST_TEMPLATE) != SOURCE_MANIFEST_FILE_SHA256:
        raise RuntimeError("Source manifest template drifted")
    metadata = _load_json(SFT_CACHE_DIR / "metadata.json")
    if metadata.get("cache_hash") != SFT_CACHE_HASH:
        raise RuntimeError("SFT cache hash drifted")
    if int(metadata.get("num_rows", -1)) != SFT_ROWS:
        raise RuntimeError("SFT cache row count drifted")
    if int(metadata.get("supervised_targets", -1)) != SFT_TARGETS_FULL:
        raise RuntimeError("SFT cache supervised-target count drifted")


# --- data preparation (CPU) -------------------------------------------------
def _prepare_impl() -> dict[str, Any]:
    import numpy as np
    from training.interleaved_data import (
        PretrainSelection,
        SFTCache,
        _sft_supervised_targets_per_row,
        _shuffled_leg_order,
        _write_leg_manifest,
        build_pretrain_selection,
    )
    import shutil

    _validate_inputs()
    if MANIFEST_SET_PATH.is_file():
        return _load_json(MANIFEST_SET_PATH)

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    source_manifest_path = ARTIFACT_ROOT / "source_manifest.json"
    if not source_manifest_path.is_file():
        shutil.copyfile(SOURCE_MANIFEST_TEMPLATE, source_manifest_path)
    source_manifest_hash = _load_json(source_manifest_path)["manifest_hash"]

    # one deterministic selection per PT budget (seed 42)
    selections: dict[int, Any] = {}
    for tokens, name in ((PT_5B, "5b"), (PT_10B, "10b")):
        path = ARTIFACT_ROOT / f"pretrain_selection_{name}.json"
        if not path.is_file():
            build_pretrain_selection(
                source_manifest_path=source_manifest_path,
                output_path=path,
                target_tokens=tokens,
                seed=42,
            )
        selection = PretrainSelection.load(path)
        if selection.target_tokens != tokens:
            raise RuntimeError(f"{name} selection target drifted")
        selections[tokens] = (path, selection)

    # SFT row splits (identical seed-42 permutation as prior experiments)
    cache = SFTCache.load(SFT_CACHE_DIR, verify_large_files=False)
    per_row = _sft_supervised_targets_per_row(cache)
    perm = np.random.Generator(
        np.random.PCG64(SFT_SPLIT_SEED)
    ).permutation(cache.num_rows)
    half = np.sort(perm[: cache.num_rows // 2])
    full = np.arange(cache.num_rows, dtype="<i8")
    if int(per_row[half].sum()) != SFT_TARGETS_HALF:
        raise RuntimeError("Half-split supervised targets drifted")
    if int(per_row.sum()) != SFT_TARGETS_FULL:
        raise RuntimeError("Full supervised targets drifted")
    sft_indices = {"none": np.empty(0, dtype="<i8"), "half": half, "full": full}

    manifests: dict[str, Any] = {}
    for arm in ARMS.values():
        selection_path, selection = selections[arm.pt_tokens]
        order, padding = _shuffled_leg_order(
            pretrain_records=arm.pt_records,
            sft_indices=sft_indices[arm.sft_mode],
            global_batch_size=GLOBAL_BATCH_SIZE,
            seed=arm.shuffle_seed,
        )
        manifest = _write_leg_manifest(
            ARTIFACT_ROOT / arm.slug,
            leg=arm.slug,
            order=order,
            target_start=0,
            target_count=arm.pt_tokens,
            sequence_length=SEQUENCE_LENGTH,
            pretrain_records=arm.pt_records,
            sft_records=arm.sft_rows,
            sft_supervised_targets=arm.sft_targets,
            padding_records=padding,
            world_size=WORLD_SIZE,
            local_batch_size=LOCAL_BATCH_SIZE,
            total_steps=arm.total_steps,
            source_manifest_hash=source_manifest_hash,
            selection_hash=selection.selection_hash,
            sft_cache_hash=SFT_CACHE_HASH,
            shuffle_seed=arm.shuffle_seed,
        )
        if manifest.physical_steps != arm.total_steps:
            raise RuntimeError(
                f"{arm.key} manifest steps {manifest.physical_steps} != "
                f"{arm.total_steps}"
            )
        manifests[arm.key] = {
            "metadata_path": str(manifest.metadata_path),
            "sha256": _sha256_file(manifest.metadata_path),
            "order_sha256": manifest.order_sha256,
            "selection_path": str(selection_path),
            "pt_tokens": arm.pt_tokens,
            "sft_mode": arm.sft_mode,
            "sft_records": arm.sft_rows,
            "sft_supervised_targets": arm.sft_targets,
            "total_steps": arm.total_steps,
            "eta_min": arm.eta_min,
            "sft_loss_weight": 1.0,
        }

    payload: dict[str, Any] = {
        "schema": "sft-injection-ablation-manifest-set-v1",
        "experiment_version": EXPERIMENT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest_hash": source_manifest_hash,
        "sft_cache_hash": SFT_CACHE_HASH,
        "sft_repo": SFT_REPO,
        "sft_revision": SFT_REVISION,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "manifests": manifests,
    }
    payload["manifest_set_hash"] = hashlib.sha256(
        _canonical_json(payload)
    ).hexdigest()
    _atomic_json(MANIFEST_SET_PATH, payload)
    data_volume.commit()
    return payload


def _prepare_a2r_impl() -> dict[str, Any]:
    import numpy as np
    from training.interleaved_data import (
        PretrainSelection, SFTCache, _sft_supervised_targets_per_row,
        _shuffled_leg_order, _write_leg_manifest,
    )
    _validate_inputs()
    if MANIFEST_SET_A2R.is_file():
        return _load_json(MANIFEST_SET_A2R)
    arm = ARMS["A2R"]
    source_manifest_hash = _load_json(
        ARTIFACT_ROOT / "source_manifest.json")["manifest_hash"]
    selection_path = ARTIFACT_ROOT / "pretrain_selection_5b.json"
    selection = PretrainSelection.load(selection_path)
    cache = SFTCache.load(SFT_CACHE_DIR, verify_large_files=False)
    per_row = _sft_supervised_targets_per_row(cache)
    full = np.arange(cache.num_rows, dtype="<i8")
    if int(per_row.sum()) != SFT_TARGETS_FULL:
        raise RuntimeError("Full supervised targets drifted")
    order, padding = _shuffled_leg_order(
        pretrain_records=arm.pt_records, sft_indices=full,
        global_batch_size=GLOBAL_BATCH_SIZE, seed=arm.shuffle_seed,
    )
    manifest = _write_leg_manifest(
        ARTIFACT_ROOT / arm.slug, leg=arm.slug, order=order,
        target_start=0, target_count=arm.pt_tokens,
        sequence_length=SEQUENCE_LENGTH, pretrain_records=arm.pt_records,
        sft_records=arm.sft_rows, sft_supervised_targets=arm.sft_targets,
        padding_records=padding, world_size=WORLD_SIZE,
        local_batch_size=LOCAL_BATCH_SIZE, total_steps=arm.total_steps,
        source_manifest_hash=source_manifest_hash,
        selection_hash=selection.selection_hash,
        sft_cache_hash=SFT_CACHE_HASH, shuffle_seed=arm.shuffle_seed,
    )
    payload = {
        "schema": "sft-injection-ablation-a2r-manifest-set-v1",
        "experiment_version": EXPERIMENT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifests": {"A2R": {
            "metadata_path": str(manifest.metadata_path),
            "sha256": _sha256_file(manifest.metadata_path),
            "order_sha256": manifest.order_sha256,
            "selection_path": str(selection_path),
            "pt_tokens": arm.pt_tokens, "sft_mode": arm.sft_mode,
            "sft_records": arm.sft_rows,
            "sft_supervised_targets": arm.sft_targets,
            "total_steps": arm.total_steps, "eta_min": arm.eta_min,
            "sft_loss_weight": 1.0, "shuffle_seed": arm.shuffle_seed,
        }},
    }
    payload["manifest_set_hash"] = hashlib.sha256(
        _canonical_json(payload)).hexdigest()
    _atomic_json(MANIFEST_SET_A2R, payload)
    data_volume.commit()
    return payload


@app.function(cpu=16.0, memory=64 * 1024, timeout=60 * 60)
def prepare_a2r() -> str:
    payload = _prepare_a2r_impl()
    return json.dumps({k: v["sha256"] for k, v in payload["manifests"].items()})


@app.function(cpu=16.0, memory=64 * 1024, timeout=60 * 60 * 4)
def prepare_data() -> str:
    payload = _prepare_impl()
    return json.dumps(
        {k: v["sha256"] for k, v in payload["manifests"].items()}, indent=2
    )


# --- training ---------------------------------------------------------------
def _canary_gate_path(arm: Arm) -> Path:
    return ARTIFACT_ROOT / arm.slug / "canary_gate.json"


def _overrides(arm: Arm, manifest: Mapping[str, Any], canary: bool) -> list[str]:
    output_dir = str(
        CHECKPOINT_ROOT / (arm.slug + ("_canary" if canary else ""))
    )
    run_name = f"ablation-{arm.slug}-{EXPERIMENT_VERSION}"
    if canary:
        run_name += "-canary"
    values = [
        f"training.output_dir={output_dir}",
        f"training.run_name={run_name}",
        "training.seed=42",
        f"training.local_batch_size={LOCAL_BATCH_SIZE}",
        "training.gradient_accumulation_steps=1",
        f"training.total_steps={arm.total_steps}",
        f"training.arc_steps=[{arm.total_steps}]",
        "training.reset_optimizer_between_arcs=true",
        "training.mixed_precision=bf16",
        "training.sft_loss_weight=1.0",
        "training.optimizer.lr=1e-3",
        "training.optimizer.weight_decay=0.1",
        "training.optimizer.betas=[0.9,0.95]",
        "training.scheduler.warmup_ratio=0.05",
        f"training.scheduler.eta_min={arm.eta_min}",
        "training.torch_compile=none",
        "model.attn_implementation=sdpa",
        "model.flash_attention_version=2.8.3",
        f"data.source_root={SOURCE_DIR}",
        f"data.source_manifest_path={ARTIFACT_ROOT / 'source_manifest.json'}",
        f"data.selection_manifest_path={manifest['selection_path']}",
        f"data.sft_cache_dir={SFT_CACHE_DIR}",
        f"data.leg_manifest_path={manifest['metadata_path']}",
        f"data.expected_manifest_hash={manifest['sha256']}",
        "data.num_workers=8",
        "training.num_workers=8",
        "training.persistent_workers=true",
        "training.save_interval=200",
        "training.log_interval=10",
        "logging.backend=none",
        f"logging.project={WANDB_PROJECT}",
        f"logging.entity={WANDB_ENTITY}",
        f"provenance.experiment_version={EXPERIMENT_VERSION}",
        f"provenance.data_artifact_version={EXPERIMENT_VERSION}",
        f"provenance.source_tree_sha256={SOURCE_TREE_SHA256}",
        f"provenance.source_repo={SOURCE_REPO}",
        f"provenance.source_revision={SOURCE_REVISION}",
        f"provenance.sft_repo={SFT_REPO}",
        f"provenance.sft_revision={SFT_REVISION}",
        "provenance.sft_loss_weight=1.0",
        f"provenance.arm={arm.key}",
        "provenance.sft_response_normalization="
        "strip-numeric-verify-score-pairs-normalize-whitespace-v1",
        "provenance.sft_supervised_unk_policy=reject-supervised-unk-v1",
    ]
    if canary:
        values.extend([
            "training.max_steps=1",
            "training.persistent_workers=false",
            "training.save_interval=1",
            "training.log_interval=1",
            "data.num_workers=0",
            "training.num_workers=0",
        ])
    return values


def _validate_manifest_entry(arm: Arm) -> dict[str, Any]:
    payload = _load_json(
        MANIFEST_SET_A2R if arm.key == "A2R" else MANIFEST_SET_PATH
    )
    body = {k: v for k, v in payload.items() if k != "manifest_set_hash"}
    if payload.get("manifest_set_hash") != hashlib.sha256(
        _canonical_json(body)
    ).hexdigest():
        raise RuntimeError("Manifest-set self hash drifted")
    entry = payload["manifests"].get(arm.key)
    if not isinstance(entry, Mapping):
        raise RuntimeError(f"Missing manifest entry for {arm.key}")
    actual = _sha256_file(Path(entry["metadata_path"]))
    if actual != entry["sha256"]:
        raise RuntimeError(f"{arm.key} leg manifest drifted on disk")
    if int(entry["total_steps"]) != arm.total_steps:
        raise RuntimeError(f"{arm.key} manifest step count drifted")
    return dict(entry)


def _run_training(arm: Arm, canary: bool) -> str:
    data_volume.reload()
    checkpoint_volume.reload()
    _validate_inputs()
    manifest = _validate_manifest_entry(arm)
    if not canary:
        gate_path = _canary_gate_path(arm)
        if not gate_path.is_file():
            raise RuntimeError(f"{arm.key} canary gate missing — run canary first")
        gate = _load_json(gate_path)
        recorded = gate.pop("gate_sha256", None)
        if recorded != hashlib.sha256(_canonical_json(gate)).hexdigest():
            raise RuntimeError(f"{arm.key} canary gate self-hash drifted")
        if gate.get("decision") != "pass" or gate.get("manifest_sha256") != manifest["sha256"]:
            raise RuntimeError(f"{arm.key} canary gate invalid for this manifest")

    output_dir = Path(
        str(CHECKPOINT_ROOT / (arm.slug + ("_canary" if canary else "")))
    )
    resume = None
    latest_state = output_dir / "latest" / "trainer_state.json"
    if latest_state.is_file():
        state = _load_json(latest_state)
        if state.get("manifest_hash") != manifest["sha256"]:
            raise RuntimeError(f"{arm.key} resume state does not match manifest")
        resume = str(output_dir / "latest")
    final_state = output_dir / "final" / "interleaved_training_state.json"
    if final_state.is_file():
        return str(output_dir)

    command = ["accelerate", "launch"]
    if WORLD_SIZE > 1:
        command.append("--multi_gpu")
    command.extend([
        "--num_processes", str(WORLD_SIZE),
        "--mixed_precision", "bf16",
        "--main_process_port", "29731" if canary else "29741",
        TRAIN_CLI, "--config", BASE_CONFIG,
        "--override", *_overrides(arm, manifest, canary),
    ])
    if resume:
        command.extend(["--resume", resume])
    print(f"[ablation:{arm.key}] " + " ".join(command), flush=True)
    result = subprocess.run(
        command, cwd="/root/chess", stdout=sys.stdout, stderr=sys.stderr,
        check=False,
    )
    checkpoint_volume.commit()
    if result.returncode != 0:
        raise RuntimeError(f"{arm.key} failed with exit code {result.returncode}")
    if canary:
        gate = {
            "schema": "sft-injection-ablation-canary-gate-v1",
            "arm": arm.key,
            "decision": "pass",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": manifest["sha256"],
            "source_tree_sha256": SOURCE_TREE_SHA256,
        }
        gate["gate_sha256"] = hashlib.sha256(_canonical_json(gate)).hexdigest()
        _atomic_json(_canary_gate_path(arm), gate)
        data_volume.commit()
    elif not final_state.is_file():
        raise RuntimeError(f"{arm.key} returned without a final HF export")
    return str(output_dir)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}",
    cpu=32.0,
    memory=128 * 1024,
    timeout=60 * 60 * 2,
    retries=0,
)
def train_canary(arm_key: str) -> str:
    return _run_training(ARMS[arm_key], canary=True)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}",
    cpu=32.0,
    memory=128 * 1024,
    timeout=60 * 60 * 24,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
    max_containers=6,
)
def train_production(arm_key: str) -> str:
    return _run_training(ARMS[arm_key], canary=False)


# --- wave 2: staged SFT stages (B1-B4) ---------------------------------------
@dataclass(frozen=True)
class StageArm:
    key: str            # B1..B4
    base_key: str       # PT-5B | PT-10B
    sft_mode: str       # half | full
    shuffle_seed: int
    epochs: int = 1

    @property
    def slug(self) -> str:
        return self.key.lower()

    @property
    def sft_rows(self) -> int:
        return SFT_HALF_ROWS if self.sft_mode == "half" else SFT_ROWS

    @property
    def sft_targets(self) -> int:
        return SFT_TARGETS_HALF if self.sft_mode == "half" else SFT_TARGETS_FULL

    @property
    def total_steps(self) -> int:
        return _steps(0, self.sft_rows * self.epochs)


WAVE2_ARMS: dict[str, StageArm] = {
    "B1": StageArm("B1", "PT-5B", "half", 7042),
    "B2": StageArm("B2", "PT-5B", "full", 7043),
    "B3": StageArm("B3", "PT-10B", "half", 8042),
    "B4": StageArm("B4", "PT-10B", "full", 8043),
}
WAVE2_EXPECTED_STEPS = {"B1": 232, "B2": 463, "B3": 232, "B4": 463}
for _k, _b in WAVE2_ARMS.items():
    if _b.total_steps != WAVE2_EXPECTED_STEPS[_k]:
        raise RuntimeError(f"{_k} stage step accounting drifted: {_b.total_steps}")

MANIFEST_SET_W2_PATH = ARTIFACT_ROOT / "manifest_set_wave2_r3.json"

# Historical staged-SFT recipe (verified against chess-pre-to-post/
# sft_trajectory_no_labels uploads): 3 epochs, lr 3e-4 cosine->1e-5, warmup 50.
WAVE2H_ARMS: dict[str, StageArm] = {
    "B1H": StageArm("B1H", "PT-5B", "half", 7142, 3),
    "B2H": StageArm("B2H", "PT-5B", "full", 7143, 3),
    "B3H": StageArm("B3H", "PT-10B", "half", 8142, 3),
    "B4H": StageArm("B4H", "PT-10B", "full", 8143, 3),
}
WAVE2H_EXPECTED = {"B1H": 694, "B2H": 1388, "B3H": 694, "B4H": 1388}
for _k, _b in WAVE2H_ARMS.items():
    if _b.total_steps != WAVE2H_EXPECTED[_k]:
        raise RuntimeError(f"{_k} hist step accounting drifted: {_b.total_steps}")
MANIFEST_SET_W2H_PATH = ARTIFACT_ROOT / "manifest_set_wave2_hist.json"
HIST_LR = 3e-4
HIST_WARMUP_STEPS = 50


def _prepare_wave2_impl() -> dict[str, Any]:
    import numpy as np
    from training.interleaved_data import (
        SFTCache,
        _sft_supervised_targets_per_row,
        _shuffled_leg_order,
        _write_leg_manifest,
    )

    _validate_inputs()
    if MANIFEST_SET_W2_PATH.is_file():
        return _load_json(MANIFEST_SET_W2_PATH)

    source_manifest_hash = _load_json(
        ARTIFACT_ROOT / "source_manifest.json"
    )["manifest_hash"]
    from training.interleaved_data import PretrainSelection
    selection = PretrainSelection.load(
        ARTIFACT_ROOT / "pretrain_selection_5b.json"
    )
    cache = SFTCache.load(SFT_CACHE_DIR, verify_large_files=False)
    per_row = _sft_supervised_targets_per_row(cache)
    perm = np.random.Generator(
        np.random.PCG64(SFT_SPLIT_SEED)
    ).permutation(cache.num_rows)
    half = np.sort(perm[: cache.num_rows // 2])
    full = np.arange(cache.num_rows, dtype="<i8")
    if int(per_row[half].sum()) != SFT_TARGETS_HALF:
        raise RuntimeError("Half-split supervised targets drifted")

    manifests: dict[str, Any] = {}
    for mode, indices, seed in (("half", half, 9042), ("full", full, 9043)):
        rows = SFT_HALF_ROWS if mode == "half" else SFT_ROWS
        targets = SFT_TARGETS_HALF if mode == "half" else SFT_TARGETS_FULL
        order, padding = _shuffled_leg_order(
            pretrain_records=0,
            sft_indices=indices,
            global_batch_size=GLOBAL_BATCH_SIZE,
            seed=seed,
        )
        manifest = _write_leg_manifest(
            ARTIFACT_ROOT / f"sft_stage_{mode}_r3",
            leg=f"sft_stage_{mode}_r3",
            order=order,
            target_start=0,
            target_count=SEQUENCE_LENGTH,
            sequence_length=SEQUENCE_LENGTH,
            pretrain_records=0,
            sft_records=rows,
            sft_supervised_targets=targets,
            padding_records=padding,
            world_size=WORLD_SIZE,
            local_batch_size=LOCAL_BATCH_SIZE,
            total_steps=_steps(0, rows),
            source_manifest_hash=source_manifest_hash,
            selection_hash=selection.selection_hash,
            sft_cache_hash=SFT_CACHE_HASH,
            shuffle_seed=seed,
        )
        manifests[mode] = {
            "metadata_path": str(manifest.metadata_path),
            "sha256": _sha256_file(manifest.metadata_path),
            "order_sha256": manifest.order_sha256,
            "sft_records": rows,
            "sft_supervised_targets": targets,
            "total_steps": _steps(0, rows),
        }

    payload: dict[str, Any] = {
        "schema": "sft-injection-ablation-wave2-manifest-set-v1",
        "experiment_version": EXPERIMENT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sft_cache_hash": SFT_CACHE_HASH,
        "manifests": manifests,
    }
    payload["manifest_set_hash"] = hashlib.sha256(
        _canonical_json(payload)
    ).hexdigest()
    _atomic_json(MANIFEST_SET_W2_PATH, payload)
    data_volume.commit()
    return payload


def _prepare_wave2h_impl() -> dict[str, Any]:
    import numpy as np
    from training.interleaved_data import (
        PretrainSelection,
        SFTCache,
        _sft_supervised_targets_per_row,
        _write_leg_manifest,
    )

    _validate_inputs()
    if MANIFEST_SET_W2H_PATH.is_file():
        return _load_json(MANIFEST_SET_W2H_PATH)

    source_manifest_hash = _load_json(
        ARTIFACT_ROOT / "source_manifest.json"
    )["manifest_hash"]
    selection = PretrainSelection.load(
        ARTIFACT_ROOT / "pretrain_selection_5b.json"
    )
    cache = SFTCache.load(SFT_CACHE_DIR, verify_large_files=False)
    per_row = _sft_supervised_targets_per_row(cache)
    perm = np.random.Generator(
        np.random.PCG64(SFT_SPLIT_SEED)
    ).permutation(cache.num_rows)
    half = np.sort(perm[: cache.num_rows // 2])
    full = np.arange(cache.num_rows, dtype="<i8")

    manifests: dict[str, Any] = {}
    for mode, indices, seed in (("half", half, 9142), ("full", full, 9143)):
        rows = len(indices)
        targets = int(per_row[indices].sum())
        epochs = 3
        parts = []
        for e in range(epochs):
            rng = np.random.Generator(np.random.PCG64(seed + e))
            parts.append(-(rng.permutation(indices).astype("<i8") + 1))
        order = np.concatenate(parts)
        padding = (-len(order)) % GLOBAL_BATCH_SIZE
        if padding:
            from training.interleaved_data import PAD_RECORD
            order = np.concatenate(
                (order, np.full(padding, PAD_RECORD, dtype="<i8"))
            )
        manifest = _write_leg_manifest(
            ARTIFACT_ROOT / f"sft_stage_{mode}_hist",
            leg=f"sft_stage_{mode}_hist",
            order=order,
            target_start=0,
            target_count=SEQUENCE_LENGTH,
            sequence_length=SEQUENCE_LENGTH,
            pretrain_records=0,
            sft_records=rows * epochs,
            sft_supervised_targets=targets * epochs,
            padding_records=padding,
            world_size=WORLD_SIZE,
            local_batch_size=LOCAL_BATCH_SIZE,
            total_steps=_steps(0, rows * epochs),
            source_manifest_hash=source_manifest_hash,
            selection_hash=selection.selection_hash,
            sft_cache_hash=SFT_CACHE_HASH,
            shuffle_seed=seed,
        )
        manifests[mode] = {
            "metadata_path": str(manifest.metadata_path),
            "sha256": _sha256_file(manifest.metadata_path),
            "order_sha256": manifest.order_sha256,
            "sft_records": rows * epochs,
            "sft_supervised_targets": targets * epochs,
            "total_steps": _steps(0, rows * epochs),
            "epochs": epochs,
        }

    payload: dict[str, Any] = {
        "schema": "sft-injection-ablation-wave2-hist-manifest-set-v1",
        "experiment_version": EXPERIMENT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sft_cache_hash": SFT_CACHE_HASH,
        "recipe": {"epochs": 3, "lr": HIST_LR, "warmup_steps": HIST_WARMUP_STEPS,
                   "eta_min": 1e-5},
        "manifests": manifests,
    }
    payload["manifest_set_hash"] = hashlib.sha256(
        _canonical_json(payload)
    ).hexdigest()
    _atomic_json(MANIFEST_SET_W2H_PATH, payload)
    data_volume.commit()
    return payload


@app.function(cpu=8.0, memory=32 * 1024, timeout=60 * 60)
def prepare_wave2h() -> str:
    payload = _prepare_wave2h_impl()
    return json.dumps(
        {k: v["sha256"] for k, v in payload["manifests"].items()}, indent=2
    )


@app.function(cpu=8.0, memory=32 * 1024, timeout=60 * 60)
def prepare_wave2() -> str:
    payload = _prepare_wave2_impl()
    return json.dumps(
        {k: v["sha256"] for k, v in payload["manifests"].items()}, indent=2
    )


def _run_wave2(arm_key: str, canary: bool, hist: bool = False) -> str:
    arm = (WAVE2H_ARMS if hist else WAVE2_ARMS)[arm_key]
    data_volume.reload()
    checkpoint_volume.reload()
    _validate_inputs()
    payload = _load_json(MANIFEST_SET_W2H_PATH if hist else MANIFEST_SET_W2_PATH)
    body = {k: v for k, v in payload.items() if k != "manifest_set_hash"}
    if payload.get("manifest_set_hash") != hashlib.sha256(
        _canonical_json(body)
    ).hexdigest():
        raise RuntimeError("Wave-2 manifest-set self hash drifted")
    manifest = payload["manifests"][arm.sft_mode]
    if _sha256_file(Path(manifest["metadata_path"])) != manifest["sha256"]:
        raise RuntimeError(f"{arm.key} stage manifest drifted on disk")

    base = ARMS[arm.base_key]
    base_final = CHECKPOINT_ROOT / base.slug / "final"
    base_state = base_final / "interleaved_training_state.json"
    if not base_state.is_file():
        raise RuntimeError(f"{arm.key} base {arm.base_key} final missing")
    state = _load_json(base_state)
    if int(state.get("global_step", -1)) != base.total_steps:
        raise RuntimeError(
            f"{arm.key} base endpoint step {state.get('global_step')} != "
            f"{base.total_steps}"
        )

    output_dir = Path(
        str(CHECKPOINT_ROOT / (arm.slug + ("_canary" if canary else "")))
    )
    if (output_dir / "final" / "interleaved_training_state.json").is_file():
        return str(output_dir)
    resume = None
    latest_state = output_dir / "latest" / "trainer_state.json"
    if latest_state.is_file():
        if _load_json(latest_state).get("manifest_hash") != manifest["sha256"]:
            raise RuntimeError(f"{arm.key} resume state mismatch")
        resume = str(output_dir / "latest")

    run_name = f"ablation-{arm.slug}-{EXPERIMENT_VERSION}"
    if canary:
        run_name += "-canary"
    overrides = [
        f"training.output_dir={output_dir}",
        f"training.run_name={run_name}",
        "training.seed=42",
        f"training.local_batch_size={LOCAL_BATCH_SIZE}",
        "training.gradient_accumulation_steps=1",
        f"training.total_steps={arm.total_steps}",
        f"training.arc_steps=[{arm.total_steps}]",
        "training.reset_optimizer_between_arcs=true",
        "training.mixed_precision=bf16",
        "training.sft_loss_weight=1.0",
        # hist recipe: 3 epochs @ 3e-4 (verified vs sft_trajectory_no_labels);
        # default staged handoff: 1 epoch @ 1e-4 from the PT base floor
        f"training.optimizer.lr={HIST_LR if hist else 1e-4}",
        "training.optimizer.weight_decay=0.1",
        "training.optimizer.betas=[0.9,0.95]",
        f"training.scheduler.warmup_ratio={HIST_WARMUP_STEPS / arm.total_steps if hist else 0.0}",
        "training.scheduler.eta_min=1e-5",
        "training.torch_compile=none",
        "model.attn_implementation=sdpa",
        "model.flash_attention_version=2.8.3",
        f"data.source_root={SOURCE_DIR}",
        f"data.source_manifest_path={ARTIFACT_ROOT / 'source_manifest.json'}",
        f"data.selection_manifest_path={ARTIFACT_ROOT / 'pretrain_selection_5b.json'}",
        f"data.sft_cache_dir={SFT_CACHE_DIR}",
        f"data.leg_manifest_path={manifest['metadata_path']}",
        f"data.expected_manifest_hash={manifest['sha256']}",
        "data.num_workers=4",
        "training.num_workers=4",
        "training.persistent_workers=false",
        "training.save_interval=100",
        "training.log_interval=10",
        "logging.backend=none",
        f"logging.project={WANDB_PROJECT}",
        f"logging.entity={WANDB_ENTITY}",
        f"provenance.experiment_version={EXPERIMENT_VERSION}",
        f"provenance.arm={arm.key}",
        f"provenance.base_arm={arm.base_key}",
        f"provenance.sft_repo={SFT_REPO}",
        f"provenance.sft_revision={SFT_REVISION}",
        "provenance.sft_loss_weight=1.0",
        f"provenance.source_tree_sha256={SOURCE_TREE_SHA256}",
    ]
    if canary:
        overrides.extend([
            "training.max_steps=1",
            "training.save_interval=1",
            "training.log_interval=1",
            "data.num_workers=0",
            "training.num_workers=0",
        ])
    command = [
        "accelerate", "launch", "--multi_gpu",
        "--num_processes", str(WORLD_SIZE),
        "--mixed_precision", "bf16",
        "--main_process_port", "29751" if canary else "29761",
        TRAIN_CLI, "--config", BASE_CONFIG,
        "--override", *overrides,
    ]
    if resume:
        command.extend(["--resume", resume])
    else:
        command.extend(["--weights-only", str(base_final)])
    print(f"[ablation:{arm.key}] " + " ".join(command), flush=True)
    result = subprocess.run(
        command, cwd="/root/chess", stdout=sys.stdout, stderr=sys.stderr,
        check=False,
    )
    checkpoint_volume.commit()
    if result.returncode != 0:
        raise RuntimeError(f"{arm.key} failed with exit code {result.returncode}")
    if not canary and not (
        output_dir / "final" / "interleaved_training_state.json"
    ).is_file():
        raise RuntimeError(f"{arm.key} returned without a final HF export")
    return str(output_dir)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}", cpu=32.0, memory=128 * 1024,
    timeout=60 * 60, retries=0,
)
def train_wave2_canary(arm_key: str) -> str:
    return _run_wave2(arm_key, canary=True)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}", cpu=32.0, memory=128 * 1024,
    timeout=60 * 60, retries=0,
)
def train_wave2h_canary(arm_key: str) -> str:
    return _run_wave2(arm_key, canary=True, hist=True)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}", cpu=32.0, memory=128 * 1024,
    timeout=60 * 60 * 8,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
    max_containers=4,
)
def train_wave2h(arm_key: str) -> str:
    return _run_wave2(arm_key, canary=False, hist=True)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}", cpu=32.0, memory=128 * 1024,
    timeout=60 * 60 * 6,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
    max_containers=4,
)
def train_wave2(arm_key: str) -> str:
    return _run_wave2(arm_key, canary=False)


# --- E2W1: original clean exp2 composite manifest at weight 1.0 ---------------
# Trust-resolving run: original pipeline's authenticated manifests (v2r1 clean),
# leg-wise P1||P2 full-SFT stream, one 19,840-step cosine, sft_loss_weight=1.0.
# None of this experiment's manifest-building code is involved.
E2W1 = {
    "leg_manifest_path": str(V2R1_ROOT / "legs/exp2/metadata.json"),
    "manifest_sha256": "30b2ccdd98bb5b1180bfe85354a66bf4271eef621b4842d96e5b68f0b337f64d",
    "selection_path": str(V2R1_ROOT / "pretrain_selection.json"),
    "total_steps": 19_840,
    "dir_name": "e2w1",
    "arm": "E2W1",
    "manifest_origin": "v2r1_original_exp2_composite",
    "ports": ("29771", "29781"),
}

# P1W1: the v2r1 P1 leg (first 5B PT tokens + first SFT half) at weight 1.0.
# This is the shared pretraining root for Experiment 1 and Experiment 3.
P1W1 = {
    "leg_manifest_path": str(V2R1_ROOT / "legs/p1/metadata.json"),
    "manifest_sha256": "b3a67af83912a6f82290b23ff7463b22e9cb9cad6403e9d2a54c783d588a55ba",
    "selection_path": str(V2R1_ROOT / "pretrain_selection.json"),
    "total_steps": 9_920,
    "dir_name": "p1w1",
    "arm": "P1W1",
    "manifest_origin": "v2r1_original_p1_leg",
    "ports": ("29772", "29782"),
}

# E3P2: the v2r1 P2 leg (second 5B PT tokens + second SFT half) at weight 1.0,
# initialized from the P1W1 endpoint weights with a fresh optimizer and a fresh
# 9,920-step cosine. This is the Experiment 3 (two-cosine, no-RL) control.
E3P2 = {
    "leg_manifest_path": str(V2R1_ROOT / "legs/p2/metadata.json"),
    "manifest_sha256": "2536c129a5bbd04c082533b9a4ffed2d318723ea8ac3dec6b85583f217691eed",
    "selection_path": str(V2R1_ROOT / "pretrain_selection.json"),
    "total_steps": 9_920,
    "dir_name": "e3p2",
    "arm": "E3P2",
    "manifest_origin": "v2r1_original_p2_leg",
    "ports": ("29773", "29783"),
    "weights_only": str(CHECKPOINT_ROOT / "p1w1" / "final"),
}

# E1 midpoint P2 stages: same v2r1 P2 leg, weights initialized from the
# RL-1500 endpoint (converted HF export) of the corresponding first RL leg.
RL_HF_ROOT = Path("/checkpoints/interleave_50m/rl_hf")
E1UP2 = {
    "leg_manifest_path": str(V2R1_ROOT / "legs/p2/metadata.json"),
    "manifest_sha256": "2536c129a5bbd04c082533b9a4ffed2d318723ea8ac3dec6b85583f217691eed",
    "selection_path": str(V2R1_ROOT / "pretrain_selection.json"),
    "total_steps": 9_920,
    "dir_name": "e1up2",
    "arm": "E1UP2",
    "manifest_origin": "v2r1_original_p2_leg",
    "ports": ("29774", "29784"),
    "weights_only": str(RL_HF_ROOT / "e1-u-rl1500-s1500"),
}
# Second-stage pretraining from the lr-1e-4 RL endpoint (weights moved 3.97%):
# the strongest wash-out test — same P2 leg as all other midpoint handoffs.
LR4P2 = {
    "leg_manifest_path": str(V2R1_ROOT / "legs/p2/metadata.json"),
    "manifest_sha256": "2536c129a5bbd04c082533b9a4ffed2d318723ea8ac3dec6b85583f217691eed",
    "selection_path": str(V2R1_ROOT / "pretrain_selection.json"),
    "total_steps": 9_920,
    "dir_name": "lr4p2",
    "arm": "LR4P2",
    "manifest_origin": "v2r1_original_p2_leg",
    "ports": ("29776", "29786"),
    "weights_only": str(RL_HF_ROOT / "p1w1-band-lr1e4-rl1500-s1500"),
}

E1DP2 = {
    "leg_manifest_path": str(V2R1_ROOT / "legs/p2/metadata.json"),
    "manifest_sha256": "2536c129a5bbd04c082533b9a4ffed2d318723ea8ac3dec6b85583f217691eed",
    "selection_path": str(V2R1_ROOT / "pretrain_selection.json"),
    "total_steps": 9_920,
    "dir_name": "e1dp2",
    "arm": "E1DP2",
    "manifest_origin": "v2r1_original_p2_leg",
    "ports": ("29775", "29785"),
    "weights_only": str(RL_HF_ROOT / "e1-d-rl1500-s1500"),
}


def _run_v2r1_leg(cfg: dict, canary: bool) -> str:
    data_volume.reload()
    checkpoint_volume.reload()
    _validate_inputs()
    if _sha256_file(Path(cfg["leg_manifest_path"])) != cfg["manifest_sha256"]:
        raise RuntimeError(f"{cfg['arm']} leg manifest drifted")
    output_dir = Path(
        str(CHECKPOINT_ROOT / (cfg["dir_name"] + ("_canary" if canary else "")))
    )
    if (output_dir / "final" / "interleaved_training_state.json").is_file():
        return str(output_dir)
    resume = None
    latest_state = output_dir / "latest" / "trainer_state.json"
    if latest_state.is_file():
        if _load_json(latest_state).get("manifest_hash") != cfg["manifest_sha256"]:
            raise RuntimeError(f"{cfg['arm']} resume state mismatch")
        resume = str(output_dir / "latest")
    run_name = (
        f"ablation-{cfg['dir_name']}-{EXPERIMENT_VERSION}"
        + ("-canary" if canary else "")
    )
    total = cfg["total_steps"]
    overrides = [
        f"training.output_dir={output_dir}",
        f"training.run_name={run_name}",
        "training.seed=42",
        f"training.local_batch_size={LOCAL_BATCH_SIZE}",
        "training.gradient_accumulation_steps=1",
        f"training.total_steps={total}",
        f"training.arc_steps=[{total}]",
        "training.reset_optimizer_between_arcs=true",
        "training.mixed_precision=bf16",
        "training.sft_loss_weight=1.0",
        "training.optimizer.lr=1e-3",
        "training.optimizer.weight_decay=0.1",
        "training.optimizer.betas=[0.9,0.95]",
        "training.scheduler.warmup_ratio=0.05",
        "training.scheduler.eta_min=1e-5",
        "training.torch_compile=none",
        "model.attn_implementation=sdpa",
        "model.flash_attention_version=2.8.3",
        f"data.source_root={SOURCE_DIR}",
        f"data.source_manifest_path={V2R1_ROOT / 'source_manifest.json'}",
        f"data.selection_manifest_path={cfg['selection_path']}",
        f"data.sft_cache_dir={cfg.get('sft_cache_dir', SFT_CACHE_DIR)}",
        f"data.leg_manifest_path={cfg['leg_manifest_path']}",
        f"data.expected_manifest_hash={cfg['manifest_sha256']}",
        "data.num_workers=8",
        "training.num_workers=8",
        "training.persistent_workers=true",
        "training.save_interval=200",
        "training.log_interval=10",
        "logging.backend=none",
        f"logging.project={WANDB_PROJECT}",
        f"logging.entity={WANDB_ENTITY}",
        f"provenance.experiment_version={EXPERIMENT_VERSION}",
        f"provenance.arm={cfg['arm']}",
        f"provenance.manifest_origin={cfg['manifest_origin']}",
        f"provenance.sft_repo={SFT_REPO}",
        f"provenance.sft_revision={SFT_REVISION}",
        "provenance.sft_loss_weight=1.0",
        f"provenance.source_tree_sha256={SOURCE_TREE_SHA256}",
    ]
    if canary:
        overrides.extend([
            "training.max_steps=1", "training.save_interval=1",
            "training.log_interval=1", "data.num_workers=0",
            "training.num_workers=0", "training.persistent_workers=false",
        ])
    command = [
        "accelerate", "launch", "--multi_gpu",
        "--num_processes", str(WORLD_SIZE),
        "--mixed_precision", "bf16",
        "--main_process_port", cfg["ports"][0] if canary else cfg["ports"][1],
        TRAIN_CLI, "--config", BASE_CONFIG,
        "--override", *overrides,
    ]
    if resume:
        command.extend(["--resume", resume])
    elif cfg.get("weights_only"):
        weights_dir = Path(cfg["weights_only"])
        if not (weights_dir / "interleaved_training_state.json").is_file():
            raise RuntimeError(
                f"{cfg['arm']} init checkpoint missing or incomplete: {weights_dir}"
            )
        command.extend(["--weights-only", str(weights_dir)])
    print(f"[ablation:{cfg['arm']}] " + " ".join(command), flush=True)
    result = subprocess.run(
        command, cwd="/root/chess", stdout=sys.stdout, stderr=sys.stderr,
        check=False,
    )
    checkpoint_volume.commit()
    if result.returncode != 0:
        raise RuntimeError(f"{cfg['arm']} failed with exit code {result.returncode}")
    return str(output_dir)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}", cpu=32.0, memory=128 * 1024,
    timeout=60 * 60, retries=0,
)
def train_e2w1_canary() -> str:
    return _run_v2r1_leg(E2W1, canary=True)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}", cpu=32.0, memory=128 * 1024,
    timeout=60 * 60 * 24,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
)
def train_e2w1() -> str:
    return _run_v2r1_leg(E2W1, canary=False)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}", cpu=32.0, memory=128 * 1024,
    timeout=60 * 60, retries=0,
)
def train_p1w1_canary() -> str:
    return _run_v2r1_leg(P1W1, canary=True)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}", cpu=32.0, memory=128 * 1024,
    timeout=60 * 60 * 24,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
)
def train_p1w1() -> str:
    return _run_v2r1_leg(P1W1, canary=False)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}", cpu=32.0, memory=128 * 1024,
    timeout=60 * 60, retries=0,
)
def train_e3p2_canary() -> str:
    return _run_v2r1_leg(E3P2, canary=True)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}", cpu=32.0, memory=128 * 1024,
    timeout=60 * 60 * 24,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
)
def train_e3p2() -> str:
    return _run_v2r1_leg(E3P2, canary=False)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}", cpu=32.0, memory=128 * 1024,
    timeout=60 * 60, retries=0,
)
def train_e1up2_canary() -> str:
    return _run_v2r1_leg(E1UP2, canary=True)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}", cpu=32.0, memory=128 * 1024,
    timeout=60 * 60 * 24,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
)
def train_e1up2() -> str:
    return _run_v2r1_leg(E1UP2, canary=False)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}", cpu=32.0, memory=128 * 1024,
    timeout=60 * 60, retries=0,
)
def train_e1dp2_canary() -> str:
    return _run_v2r1_leg(E1DP2, canary=True)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}", cpu=32.0, memory=128 * 1024,
    timeout=60 * 60 * 24,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
)
def train_e1dp2() -> str:
    return _run_v2r1_leg(E1DP2, canary=False)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}", cpu=32.0, memory=128 * 1024,
    timeout=60 * 60, retries=0,
)
def train_lr4p2_canary() -> str:
    return _run_v2r1_leg(LR4P2, canary=True)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}", cpu=32.0, memory=128 * 1024,
    timeout=60 * 60 * 24,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
)
def train_lr4p2() -> str:
    return _run_v2r1_leg(LR4P2, canary=False)



# --- trace-transfer leg: RL traces as data in the second PT leg --------------
TRACE_ROOT = ARTIFACT_ROOT / "trace_transfer"
TRACE_SHUFFLE_SEED = 20260807


def _trace_suffix(k) -> str:
    s = str(k)
    if s == "1":
        return ""
    if s.isdigit():
        return f"_k{s}"
    return f"_{s}"


def _trace_seed_offset(k) -> int:
    s = str(k)
    return int(s) - 1 if s.isdigit() else 1000


@app.function(cpu=16.0, memory=192 * 1024, timeout=3600 * 6)
def prepare_trace_leg(k: str = "1") -> str:
    import numpy as np
    import pandas as pd
    from omegaconf import OmegaConf

    import sys
    sys.path.insert(0, "/root/chess")
    from llm_tokens.chess.tokenizer_factory import init_tokenizer
    from training.interleaved_data import (
        PAD_RECORD,
        SFTCache,
        _sft_supervised_targets_per_row,
        _write_leg_manifest,
        tokenize_masked_sft_row,
    )

    data_volume.reload()
    suffix = _trace_suffix(k)
    set_path = TRACE_ROOT / f"manifest_set{suffix}.json"
    if set_path.is_file():
        return json.dumps(_load_json(set_path))

    # tokenizer identical to the trainer's
    base_cfg = OmegaConf.load(f"/root/chess/{BASE_CONFIG}")
    tokenizer = init_tokenizer(
        name=str(base_cfg.tokenizer.get("name", "LanTokenizerSFT")),
        config=base_cfg.tokenizer,
    )
    if len(tokenizer.get_vocab()) != 85:
        raise RuntimeError("tokenizer vocab drifted")
    vocab = tokenizer.get_vocab()
    t_end_id = vocab["</T>"]
    call_env_id = vocab["<call_env>"]

    # v2r1 P2 leg: PT records are local codes 0..N-1; SFT negatives are the
    # second-half global cache rows
    p2_meta = _load_json(V2R1_ROOT / "legs/p2/metadata.json")
    p2_order = np.load(V2R1_ROOT / "legs/p2/order.npy", allow_pickle=False)
    p2_pt = int(p2_meta["pretrain_records"])
    neg = p2_order[(p2_order < 0) & (p2_order != PAD_RECORD)]
    p2half_rows = np.sort(-neg - 1)
    if len(p2half_rows) != int(p2_meta["sft_records"]):
        raise RuntimeError("P2-half SFT row extraction drifted")

    cache = SFTCache.load(SFT_CACHE_DIR, verify_large_files=False)
    per_row = _sft_supervised_targets_per_row(cache)
    src_offsets = np.load(cache.directory / "offsets.npy", allow_pickle=False)
    src_inputs = np.memmap(cache.directory / "input_ids.i32", mode="r",
                           dtype="<i4", shape=(cache.total_positions,))
    src_labels = np.memmap(cache.directory / "labels.i32", mode="r",
                           dtype="<i4", shape=(cache.total_positions,))

    import pyarrow.parquet as pq
    trace_file = pq.ParquetFile(TRACE_ROOT / f"traces{suffix}.parquet")

    TRACE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_dir = TRACE_ROOT / f"sft_cache_combined{suffix}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    offsets = [0]
    supervised = []
    n_dropped = 0
    with open(cache_dir / "input_ids.i32", "wb") as fi, \
            open(cache_dir / "labels.i32", "wb") as fl:
        # 1) copy the P2-half SFT rows verbatim
        for r in p2half_rows:
            a, b = int(src_offsets[r]), int(src_offsets[r + 1])
            fi.write(src_inputs[a:b].tobytes())
            fl.write(src_labels[a:b].tobytes())
            offsets.append(offsets[-1] + (b - a))
            supervised.append(int(per_row[r]))
        # 2) tokenize + append trace rows with identical masking semantics
        def _trace_rows():
            for batch in trace_file.iter_batches(batch_size=20_000,
                                                 columns=["pgn", "resp"]):
                pgns = batch.column("pgn").to_pylist()
                resps = batch.column("resp").to_pylist()
                yield from zip(pgns, resps)

        for pgn_value, resp_value in _trace_rows():
            try:
                input_ids, labels = tokenize_masked_sft_row(
                    {"pgn": pgn_value, "resp": resp_value},
                    tokenizer,
                    cot_field="resp",
                    prompt_field="pgn",
                    sequence_length=SEQUENCE_LENGTH,
                )
            except ValueError:
                n_dropped += 1
                continue
            sup_tend = int(np.count_nonzero(labels == t_end_id))
            sup_call = int(np.count_nonzero(labels == call_env_id))
            if sup_tend != 1 or sup_call < 1:
                n_dropped += 1
                continue
            fi.write(input_ids.astype("<i4").tobytes())
            fl.write(labels.astype("<i4").tobytes())
            offsets.append(offsets[-1] + len(input_ids))
            supervised.append(int(np.count_nonzero(labels != -100)))

    offsets_arr = np.asarray(offsets, dtype="<i8")
    np.save(cache_dir / "offsets.npy", offsets_arr)
    n_rows = len(offsets_arr) - 1
    n_trace_rows = n_rows - len(p2half_rows)
    total_positions = int(offsets_arr[-1])

    meta = {
        "schema": "interleaved-sft-cache-v1",
        "schema_version": 1,
        "num_rows": n_rows,
        "total_positions": total_positions,
        "supervised_targets": int(sum(supervised)),
        "sequence_length": SEQUENCE_LENGTH,
        "cot_field": "combined:p2half.cot_format_no_labels+rl_trace.resp",
        "prompt_field": "pgn",
        "input_ids_sha256": _sha256_file(cache_dir / "input_ids.i32"),
        "labels_sha256": _sha256_file(cache_dir / "labels.i32"),
        "offsets_sha256": _sha256_file(cache_dir / "offsets.npy"),
        "dtype": "<i4",
        "masking": "multiturn-prompt-and-env-v1",
        "provenance": {
            "p2half_source_cache_hash": SFT_CACHE_HASH,
            "p2half_rows": int(len(p2half_rows)),
            "trace_rows": int(n_trace_rows),
            "trace_rows_dropped": int(n_dropped),
            "trace_generator": "rl_hf/p1w1-band-lr1e4-rl1500-s1500",
            "traces_variant": str(k),
            "traces_parquet_sha256": _sha256_file(
                TRACE_ROOT / f"traces{suffix}.parquet"
            ),
        },
    }
    meta["cache_hash"] = hashlib.sha256(_canonical_json(meta)).hexdigest()
    _atomic_json(cache_dir / "metadata.json", meta)
    combined = SFTCache.load(cache_dir, verify_large_files=True)

    order = np.concatenate((
        np.arange(p2_pt, dtype="<i8"),
        -(np.arange(n_rows, dtype="<i8") + 1),
    ))
    np.random.Generator(
        np.random.PCG64(TRACE_SHUFFLE_SEED + _trace_seed_offset(k))
    ).shuffle(order)
    padding = (-len(order)) % GLOBAL_BATCH_SIZE
    if padding:
        order = np.concatenate(
            (order, np.full(padding, PAD_RECORD, dtype="<i8"))
        )
    total_steps = len(order) // GLOBAL_BATCH_SIZE
    manifest = _write_leg_manifest(
        TRACE_ROOT / f"leg{suffix}",
        leg=f"trace_transfer_p2{suffix}",
        order=order,
        target_start=int(p2_meta["target_start"]),
        target_count=int(p2_meta["target_count"]),
        sequence_length=SEQUENCE_LENGTH,
        pretrain_records=p2_pt,
        sft_records=n_rows,
        sft_supervised_targets=int(sum(supervised)),
        padding_records=int(padding),
        world_size=WORLD_SIZE,
        local_batch_size=LOCAL_BATCH_SIZE,
        total_steps=total_steps,
        source_manifest_hash=p2_meta["source_manifest_hash"],
        selection_hash=p2_meta["selection_hash"],
        sft_cache_hash=combined.cache_hash,
        shuffle_seed=TRACE_SHUFFLE_SEED + _trace_seed_offset(k),
    )
    payload = {
        "schema": "trace-transfer-manifest-set-v1",
        "traces_variant": str(k),
        "experiment_version": EXPERIMENT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "metadata_path": str(manifest.metadata_path),
            "sha256": _sha256_file(manifest.metadata_path),
            "order_sha256": manifest.order_sha256,
            "sft_cache_dir": str(cache_dir),
            "sft_cache_hash": combined.cache_hash,
            "pretrain_records": p2_pt,
            "p2half_rows": int(len(p2half_rows)),
            "trace_rows": int(n_trace_rows),
            "trace_rows_dropped": int(n_dropped),
            "total_steps": total_steps,
        },
    }
    payload["set_hash"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    _atomic_json(set_path, payload)
    data_volume.commit()
    return json.dumps(payload["manifest"], indent=2)


def _tracep2_cfg(k="1") -> dict[str, Any]:
    suffix = _trace_suffix(k)
    payload = _load_json(TRACE_ROOT / f"manifest_set{suffix}.json")
    body = {k: v for k, v in payload.items() if k != "set_hash"}
    if payload.get("set_hash") != hashlib.sha256(
        _canonical_json(body)
    ).hexdigest():
        raise RuntimeError("trace manifest set self-hash drifted")
    m = payload["manifest"]
    if _sha256_file(Path(m["metadata_path"])) != m["sha256"]:
        raise RuntimeError("trace leg manifest drifted on disk")
    s = str(k)
    name = "tracep2" if s == "1" else (f"tracep2k{s}" if s.isdigit() else f"tracep2{s}")
    return {
        "leg_manifest_path": m["metadata_path"],
        "manifest_sha256": m["sha256"],
        "selection_path": str(V2R1_ROOT / "pretrain_selection.json"),
        "total_steps": int(m["total_steps"]),
        "dir_name": name,
        "arm": name.upper(),
        "manifest_origin": f"trace_transfer_p2_combined{_trace_suffix(k)}",
        "ports": (str(29700 + (int(s) if s.isdigit() else 25)),
                  str(29730 + (int(s) if s.isdigit() else 25))),
        "weights_only": str(CHECKPOINT_ROOT / "p1w1" / "final"),
        "sft_cache_dir": m["sft_cache_dir"],
    }


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}", cpu=32.0, memory=128 * 1024,
    timeout=60 * 60, retries=0,
)
def train_tracep2_canary(k: str = "1") -> str:
    return _run_v2r1_leg(_tracep2_cfg(k), canary=True)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}", cpu=32.0, memory=128 * 1024,
    timeout=60 * 60 * 24,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
)
def train_tracep2(k: str = "1") -> str:
    return _run_v2r1_leg(_tracep2_cfg(k), canary=False)


# --- loop legs for the 10B models -------------------------------------------
# Fresh 5B tokens from corpus shards untouched by the frozen 10B selection and
# by the held-out-loss evaluator, mixed with the P2 SFT half and each model's
# own harvested RL rollout traces. Init from the plain 10B checkpoints (the RL
# round contributes data only, matching the 5B trace-transfer recipe).
LOOP_SELECTION_PATH = TRACE_ROOT / "pretrain_selection_fresh5b.json"
LOOP_SELECTION_SEED = 20260808
LOOP_HELDOUT_SHARDS = frozenset(
    (41859, 35917, 27837, 18175, 44567, 44931)
)  # pt_heldout_shards of the eval-loss evaluator (seed 20260804)
LOOP_ARMS: dict[str, dict[str, Any]] = {
    "e2w1": {
        "suffix": "e2w1roll",
        "init": "e2w1",
        "dir_name": "e2w1loop",
        "rl_run": "e2w1band-lr1e4-rl1500",
        "ports": ("29760", "29790"),
        "seed_offset": 2000,
    },
    "e3p2": {
        "suffix": "e3p2roll",
        "init": "e3p2",
        "dir_name": "e3p2loop",
        "rl_run": "e3p2band-lr1e4-rl1500",
        "ports": ("29761", "29791"),
        "seed_offset": 3000,
    },
}


@app.function(cpu=8.0, memory=32 * 1024, timeout=3600)
def prepare_fresh_selection() -> str:
    import random

    import sys
    sys.path.insert(0, "/root/chess")
    from training.interleaved_data import (
        SCHEMA_VERSION,
        PretrainSelection,
        SourceShardManifest,
        _atomic_json,
        _hash_dict,
    )

    data_volume.reload()
    if LOOP_SELECTION_PATH.is_file():
        sel = PretrainSelection.load(LOOP_SELECTION_PATH)
        return json.dumps({"selection_hash": sel.selection_hash,
                           "target_tokens": sel.target_tokens,
                           "spans": len(sel.spans), "existing": True})

    source = SourceShardManifest.load(SOURCE_MANIFEST_TEMPLATE)
    frozen = PretrainSelection.load(V2R1_ROOT / "pretrain_selection.json")
    if frozen.source_manifest_hash != source.manifest_hash:
        raise RuntimeError("frozen selection/source manifest mismatch")
    used = {span.shard_number for span in frozen.spans}
    excluded = used | set(LOOP_HELDOUT_SHARDS)
    allowed = [s for s in source.shards if s.shard_number not in excluded]
    available = sum(s.num_tokens for s in allowed)
    required = PT_5B + 1
    if available < required:
        raise RuntimeError(
            f"only {available:,} unused tokens available, need {required:,}"
        )

    rng = random.Random(LOOP_SELECTION_SEED)
    order = list(range(len(allowed)))
    rng.shuffle(order)
    remaining = required
    spans = []
    for ordinal in order:
        shard = allowed[ordinal]
        take = min(remaining, shard.num_tokens)
        if take == shard.num_tokens:
            start = 0
        else:
            start = rng.randrange(0, shard.num_tokens - take + 1)
        spans.append({
            "shard_number": shard.shard_number,
            "relative_path": shard.relative_path,
            "start": start,
            "stop": start + take,
        })
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise AssertionError(f"selection left {remaining} tokens")

    value: dict[str, Any] = {
        "schema": "interleaved-pretrain-selection-v1",
        "schema_version": SCHEMA_VERSION,
        "algorithm": "python-random-shard-permutation-v1-unused-shards",
        "source_manifest_hash": source.manifest_hash,
        "target_tokens": PT_5B,
        "source_tokens": required,
        "seed": LOOP_SELECTION_SEED,
        "excluded_used_shards": len(used),
        "excluded_heldout_shards": sorted(LOOP_HELDOUT_SHARDS),
        "spans": spans,
    }
    value["selection_hash"] = _hash_dict(value, "selection_hash")
    _atomic_json(LOOP_SELECTION_PATH, value)
    sel = PretrainSelection.load(LOOP_SELECTION_PATH)
    data_volume.commit()
    return json.dumps({"selection_hash": sel.selection_hash,
                       "target_tokens": sel.target_tokens,
                       "spans": len(sel.spans),
                       "allowed_shards": len(allowed),
                       "available_tokens": available})


@app.function(cpu=16.0, memory=192 * 1024, timeout=3600 * 6)
def prepare_loop_leg(arm_key: str) -> str:
    import numpy as np
    from omegaconf import OmegaConf

    import sys
    sys.path.insert(0, "/root/chess")
    from llm_tokens.chess.tokenizer_factory import init_tokenizer
    from training.interleaved_data import (
        PAD_RECORD,
        PretrainSelection,
        SFTCache,
        _sft_supervised_targets_per_row,
        _write_leg_manifest,
        tokenize_masked_sft_row,
    )

    data_volume.reload()
    arm = LOOP_ARMS[arm_key]
    suffix = arm["suffix"]
    set_path = TRACE_ROOT / f"manifest_set_loop_{arm_key}.json"
    if set_path.is_file():
        return json.dumps(_load_json(set_path))

    selection = PretrainSelection.load(LOOP_SELECTION_PATH)
    if selection.target_tokens != PT_5B:
        raise RuntimeError("fresh selection token count drifted")
    pt_records = math.ceil(PT_5B / SEQUENCE_LENGTH)

    base_cfg = OmegaConf.load(f"/root/chess/{BASE_CONFIG}")
    tokenizer = init_tokenizer(
        name=str(base_cfg.tokenizer.get("name", "LanTokenizerSFT")),
        config=base_cfg.tokenizer,
    )
    if len(tokenizer.get_vocab()) != 85:
        raise RuntimeError("tokenizer vocab drifted")
    vocab = tokenizer.get_vocab()
    t_end_id = vocab["</T>"]
    call_env_id = vocab["<call_env>"]

    p2_meta = _load_json(V2R1_ROOT / "legs/p2/metadata.json")
    if p2_meta["source_manifest_hash"] != selection.source_manifest_hash:
        raise RuntimeError("fresh selection built from a different source manifest")
    p2_order = np.load(V2R1_ROOT / "legs/p2/order.npy", allow_pickle=False)
    neg = p2_order[(p2_order < 0) & (p2_order != PAD_RECORD)]
    p2half_rows = np.sort(-neg - 1)
    if len(p2half_rows) != int(p2_meta["sft_records"]):
        raise RuntimeError("P2-half SFT row extraction drifted")

    cache = SFTCache.load(SFT_CACHE_DIR, verify_large_files=False)
    per_row = _sft_supervised_targets_per_row(cache)
    src_offsets = np.load(cache.directory / "offsets.npy", allow_pickle=False)
    src_inputs = np.memmap(cache.directory / "input_ids.i32", mode="r",
                           dtype="<i4", shape=(cache.total_positions,))
    src_labels = np.memmap(cache.directory / "labels.i32", mode="r",
                           dtype="<i4", shape=(cache.total_positions,))

    import pyarrow.parquet as pq
    traces_path = TRACE_ROOT / f"traces_{suffix}.parquet"
    trace_file = pq.ParquetFile(traces_path)

    cache_dir = TRACE_ROOT / f"sft_cache_combined_loop_{arm_key}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    offsets = [0]
    supervised = []
    n_dropped = 0
    with open(cache_dir / "input_ids.i32", "wb") as fi, \
            open(cache_dir / "labels.i32", "wb") as fl:
        for r in p2half_rows:
            a, b = int(src_offsets[r]), int(src_offsets[r + 1])
            fi.write(src_inputs[a:b].tobytes())
            fl.write(src_labels[a:b].tobytes())
            offsets.append(offsets[-1] + (b - a))
            supervised.append(int(per_row[r]))

        def _trace_rows():
            for batch in trace_file.iter_batches(batch_size=20_000,
                                                 columns=["pgn", "resp"]):
                pgns = batch.column("pgn").to_pylist()
                resps = batch.column("resp").to_pylist()
                yield from zip(pgns, resps)

        for pgn_value, resp_value in _trace_rows():
            try:
                input_ids, labels = tokenize_masked_sft_row(
                    {"pgn": pgn_value, "resp": resp_value},
                    tokenizer,
                    cot_field="resp",
                    prompt_field="pgn",
                    sequence_length=SEQUENCE_LENGTH,
                )
            except ValueError:
                n_dropped += 1
                continue
            sup_tend = int(np.count_nonzero(labels == t_end_id))
            sup_call = int(np.count_nonzero(labels == call_env_id))
            if sup_tend != 1 or sup_call < 1:
                n_dropped += 1
                continue
            fi.write(input_ids.astype("<i4").tobytes())
            fl.write(labels.astype("<i4").tobytes())
            offsets.append(offsets[-1] + len(input_ids))
            supervised.append(int(np.count_nonzero(labels != -100)))

    offsets_arr = np.asarray(offsets, dtype="<i8")
    np.save(cache_dir / "offsets.npy", offsets_arr)
    n_rows = len(offsets_arr) - 1
    n_trace_rows = n_rows - len(p2half_rows)
    total_positions = int(offsets_arr[-1])

    meta = {
        "schema": "interleaved-sft-cache-v1",
        "schema_version": 1,
        "num_rows": n_rows,
        "total_positions": total_positions,
        "supervised_targets": int(sum(supervised)),
        "sequence_length": SEQUENCE_LENGTH,
        "cot_field": "combined:p2half.cot_format_no_labels+rl_trace.resp",
        "prompt_field": "pgn",
        "input_ids_sha256": _sha256_file(cache_dir / "input_ids.i32"),
        "labels_sha256": _sha256_file(cache_dir / "labels.i32"),
        "offsets_sha256": _sha256_file(cache_dir / "offsets.npy"),
        "dtype": "<i4",
        "masking": "multiturn-prompt-and-env-v1",
        "provenance": {
            "p2half_source_cache_hash": SFT_CACHE_HASH,
            "p2half_rows": int(len(p2half_rows)),
            "trace_rows": int(n_trace_rows),
            "trace_rows_dropped": int(n_dropped),
            "trace_generator": (
                f"training rollouts of {arm['rl_run']} (no extra generation)"
            ),
            "traces_variant": f"loop_{arm_key}",
            "traces_parquet_sha256": _sha256_file(traces_path),
        },
    }
    meta["cache_hash"] = hashlib.sha256(_canonical_json(meta)).hexdigest()
    _atomic_json(cache_dir / "metadata.json", meta)
    combined = SFTCache.load(cache_dir, verify_large_files=True)

    order = np.concatenate((
        np.arange(pt_records, dtype="<i8"),
        -(np.arange(n_rows, dtype="<i8") + 1),
    ))
    np.random.Generator(
        np.random.PCG64(TRACE_SHUFFLE_SEED + arm["seed_offset"])
    ).shuffle(order)
    padding = (-len(order)) % GLOBAL_BATCH_SIZE
    if padding:
        order = np.concatenate(
            (order, np.full(padding, PAD_RECORD, dtype="<i8"))
        )
    total_steps = len(order) // GLOBAL_BATCH_SIZE
    manifest = _write_leg_manifest(
        TRACE_ROOT / f"leg_loop_{arm_key}",
        leg=f"loop_fresh5b_{arm_key}",
        order=order,
        target_start=0,
        target_count=PT_5B,
        sequence_length=SEQUENCE_LENGTH,
        pretrain_records=pt_records,
        sft_records=n_rows,
        sft_supervised_targets=int(sum(supervised)),
        padding_records=int(padding),
        world_size=WORLD_SIZE,
        local_batch_size=LOCAL_BATCH_SIZE,
        total_steps=total_steps,
        source_manifest_hash=selection.source_manifest_hash,
        selection_hash=selection.selection_hash,
        sft_cache_hash=combined.cache_hash,
        shuffle_seed=TRACE_SHUFFLE_SEED + arm["seed_offset"],
    )
    payload = {
        "schema": "loop-leg-manifest-set-v1",
        "arm": arm_key,
        "experiment_version": EXPERIMENT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "metadata_path": str(manifest.metadata_path),
            "sha256": _sha256_file(manifest.metadata_path),
            "order_sha256": manifest.order_sha256,
            "sft_cache_dir": str(cache_dir),
            "sft_cache_hash": combined.cache_hash,
            "pretrain_records": pt_records,
            "p2half_rows": int(len(p2half_rows)),
            "trace_rows": int(n_trace_rows),
            "trace_rows_dropped": int(n_dropped),
            "total_steps": total_steps,
        },
    }
    payload["set_hash"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    _atomic_json(set_path, payload)
    data_volume.commit()
    return json.dumps(payload["manifest"], indent=2)


def _loop_cfg(arm_key: str) -> dict[str, Any]:
    arm = LOOP_ARMS[arm_key]
    payload = _load_json(TRACE_ROOT / f"manifest_set_loop_{arm_key}.json")
    body = {k: v for k, v in payload.items() if k != "set_hash"}
    if payload.get("set_hash") != hashlib.sha256(
        _canonical_json(body)
    ).hexdigest():
        raise RuntimeError("loop manifest set self-hash drifted")
    m = payload["manifest"]
    if _sha256_file(Path(m["metadata_path"])) != m["sha256"]:
        raise RuntimeError("loop leg manifest drifted on disk")
    return {
        "leg_manifest_path": m["metadata_path"],
        "manifest_sha256": m["sha256"],
        "selection_path": str(LOOP_SELECTION_PATH),
        "total_steps": int(m["total_steps"]),
        "dir_name": arm["dir_name"],
        "arm": arm["dir_name"].upper(),
        "manifest_origin": f"loop_fresh5b_{arm_key}",
        "ports": arm["ports"],
        "weights_only": str(CHECKPOINT_ROOT / arm["init"] / "final"),
        "sft_cache_dir": m["sft_cache_dir"],
    }


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}", cpu=32.0, memory=128 * 1024,
    timeout=60 * 60, retries=0,
)
def train_loop_canary(arm_key: str) -> str:
    return _run_v2r1_leg(_loop_cfg(arm_key), canary=True)


@app.function(
    gpu=f"{GPU_TYPE}:{WORLD_SIZE}", cpu=32.0, memory=128 * 1024,
    timeout=60 * 60 * 24,
    retries=modal.Retries(initial_delay=10.0, max_retries=2),
)
def train_loop(arm_key: str) -> str:
    return _run_v2r1_leg(_loop_cfg(arm_key), canary=False)


# --- entrypoint --------------------------------------------------------------
@app.local_entrypoint()
def main(action: str = "prep", arm: str = "", dry_run: bool = False) -> None:
    action = action.strip().lower()
    if dry_run:
        print(json.dumps({
            "action": action,
            "experiment_version": EXPERIMENT_VERSION,
            "arms": {
                k: {
                    "pt_tokens": a.pt_tokens, "sft_mode": a.sft_mode,
                    "steps": a.total_steps, "eta_min": a.eta_min,
                    "sft_loss_weight": 1.0,
                } for k, a in ARMS.items()
            },
            "source_tree_sha256": SOURCE_TREE_SHA256,
        }, indent=2))
        return
    if action == "prep":
        print(prepare_data.remote())
    elif action == "prep-a2r":
        print(prepare_a2r.remote())
    elif action == "canary-e2w1":
        print(train_e2w1_canary.remote())
    elif action == "train-e2w1":
        handle = train_e2w1.spawn()
        print(json.dumps({"arm": "E2W1", "function_call_id": handle.object_id}))
    elif action == "canary-p1w1":
        print(train_p1w1_canary.remote())
    elif action == "train-p1w1":
        handle = train_p1w1.spawn()
        print(json.dumps({"arm": "P1W1", "function_call_id": handle.object_id}))
    elif action == "canary-e3p2":
        print(train_e3p2_canary.remote())
    elif action == "train-e3p2":
        handle = train_e3p2.spawn()
        print(json.dumps({"arm": "E3P2", "function_call_id": handle.object_id}))
    elif action == "canary-e1up2":
        print(train_e1up2_canary.remote())
    elif action == "train-e1up2":
        handle = train_e1up2.spawn()
        print(json.dumps({"arm": "E1UP2", "function_call_id": handle.object_id}))
    elif action == "prep-trace":
        ks = [x.strip() for x in (arm or "1").split(",")]
        calls = {kk: prepare_trace_leg.spawn(kk) for kk in ks}
        for kk, c in calls.items():
            print(f"k={kk}: {c.get()}")
    elif action == "canary-tracep2":
        ks = [x.strip() for x in (arm or "1").split(",")]
        calls = {kk: train_tracep2_canary.spawn(kk) for kk in ks}
        for kk, c in calls.items():
            print(f"k={kk} canary: {c.get()}")
    elif action == "train-tracep2":
        ks = [x.strip() for x in (arm or "1").split(",")]
        for kk in ks:
            handle = train_tracep2.spawn(kk)
            print(json.dumps({"arm": f"tracep2 k={kk}",
                              "function_call_id": handle.object_id}))
    elif action == "prep-loop-selection":
        print(prepare_fresh_selection.remote())
    elif action == "prep-loop":
        arms = [x.strip() for x in (arm or "e2w1,e3p2").split(",")]
        calls = {a: prepare_loop_leg.spawn(a) for a in arms}
        for a, c in calls.items():
            print(f"loop {a}: {c.get()}")
    elif action == "canary-loop":
        arms = [x.strip() for x in (arm or "e2w1,e3p2").split(",")]
        calls = {a: train_loop_canary.spawn(a) for a in arms}
        for a, c in calls.items():
            print(f"loop {a} canary: {c.get()}")
    elif action == "train-loop":
        arms = [x.strip() for x in (arm or "e2w1,e3p2").split(",")]
        for a in arms:
            handle = train_loop.spawn(a)
            print(json.dumps({"arm": f"loop {a}",
                              "function_call_id": handle.object_id}))
    elif action == "canary-lr4p2":
        print(train_lr4p2_canary.remote())
    elif action == "train-lr4p2":
        handle = train_lr4p2.spawn()
        print(json.dumps({"arm": "LR4P2", "function_call_id": handle.object_id}))
    elif action == "canary-e1dp2":
        print(train_e1dp2_canary.remote())
    elif action == "train-e1dp2":
        handle = train_e1dp2.spawn()
        print(json.dumps({"arm": "E1DP2", "function_call_id": handle.object_id}))
    elif action == "canary":
        keys = [arm] if arm else list(ARMS)
        calls = {k: train_canary.spawn(k) for k in keys}
        print(json.dumps({k: c.object_id for k, c in calls.items()}, indent=2))
        for k, c in calls.items():
            print(f"canary {k}: {c.get()}")
    elif action == "train":
        if not arm:
            raise ValueError("--arm required for train")
        handle = train_production.spawn(arm)
        print(json.dumps({"arm": arm, "function_call_id": handle.object_id}))
    elif action == "launch-wave1":
        calls = {k: train_production.spawn(k) for k in ARMS}
        print(json.dumps(
            {k: c.object_id for k, c in calls.items()}, indent=2
        ))
    elif action == "prep-wave2":
        print(prepare_wave2.remote())
    elif action == "canary-wave2":
        keys = [arm] if arm else list(WAVE2_ARMS)
        calls = {k: train_wave2_canary.spawn(k) for k in keys}
        for k, c in calls.items():
            print(f"wave2 canary {k}: {c.get()}")
    elif action == "launch-wave2":
        calls = {k: train_wave2.spawn(k) for k in WAVE2_ARMS}
        print(json.dumps(
            {k: c.object_id for k, c in calls.items()}, indent=2
        ))
    elif action == "prep-wave2h":
        print(prepare_wave2h.remote())
    elif action == "canary-wave2h":
        keys = [arm] if arm else list(WAVE2H_ARMS)
        calls = {k: train_wave2h_canary.spawn(k) for k in keys}
        for k, c in calls.items():
            print(f"wave2h canary {k}: {c.get()}")
    elif action == "launch-wave2h":
        calls = {k: train_wave2h.spawn(k) for k in WAVE2H_ARMS}
        print(json.dumps(
            {k: c.object_id for k, c in calls.items()}, indent=2
        ))
    else:
        raise ValueError(f"Unknown action: {action}")
