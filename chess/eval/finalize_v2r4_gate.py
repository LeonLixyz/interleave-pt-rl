"""Read-only, fail-closed finalizer for the amended v2r4a production gate.

The finalizer performs no Modal spawn, write, commit, or delete operation.
It authenticates the three exact-once launch ledgers, the external runtime
contract and prompt manifest, all completed FunctionCall return values, and
the immutable rollout artifacts.  Only after every input passes does it call
the frozen :mod:`Eval.v2r4_gate_analysis` analyzer.

The optional local report write is immutable: a temporary file is fsynced and
hard-linked into place, so an existing output can never be overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from Eval.v2r4_gate_analysis import (
    CONTRACT_VERSION as ANALYZER_CONTRACT_VERSION,
    SCHEMA as ANALYZER_REPORT_SCHEMA,
    BATCH_LABELS,
    CANDIDATE_STEPS,
    ROWS_PER_CELL,
    analyze_grid,
    audit_cell,
    content_hash,
)
from chess_reasoning.training import v2r4_p2_sft_eval as p2_contract


REPORT_SCHEMA = "interleaved-v2r4a-production-gate-finalized-report-v1"
CONTRACT_SCHEMA = "interleaved-v2r4a-production-gate-runtime-contract-v1"
CONTRACT_VERSION = "v2r4a_production_gate_20260730"
CONTRACT_SHA256 = (
    "3e201cfd9094815cf72a63058d4225b334e3e5cd77fc0d79fbe6379d48778c9d"
)
CONTRACT_FILE_SHA256 = (
    "04945f691aa55e3aaa860851b33b49aba9eea789533e5f86f3e0d2345bbc1c38"
)
GATE_LEDGER_SCHEMA = "interleaved-v2r4-gate-launch-ledger-v1"
GATE_LEDGER_SHA256 = (
    "a09a1a10a651417b3b91294d757a03ce1236e9ede170ad32ff3f2bb193f2f648"
)
GATE_LEDGER_FILE_SHA256 = (
    "065163efa7fa70249ebe0b99889e326ed0f68dad591dec383341027169c720cf"
)
GATE_SUCCESS_SCHEMA = "interleaved-v2r4-gate-cell-success-v1"
ENDPOINT_LEDGER_SCHEMA = "interleaved-v2r4-endpoint-launch-ledger-v1"
ENDPOINT_LEDGER_SHA256 = (
    "e150ec1f66670301a789f736a1522ae96e8abf490d94cd43ab2d98978ffad032"
)
ENDPOINT_LEDGER_FILE_SHA256 = (
    "3351fc7298b189dad9ef6e865224ea27df9d924a477cac4dc270fe526ebdb631"
)
ENDPOINT_RESULT_SCHEMA = "interleaved-endpoint-result-v1"
P2_LEDGER_SCHEMA = "interleaved-v2r4-p2-sft-launch-ledger-v2"
P2_LEDGER_SHA256 = (
    "3440c2429be81d4bd3157dacd3ea3b81d0d9bb8bec9e13032b8b5e019baa77dc"
)
P2_LEDGER_FILE_SHA256 = (
    "f3755c9c01719d04db35b2cb435cc933c1c7fc00ddc45a11b995e0389262be30"
)
P2_OUTPUT_SCHEMA = "interleaved-v2r4-p2-sft-modal-result-v2"

ANALYZER_SHA256 = (
    "16c9dfd5cc421b196344b773998629cd707f4fadaa405bdba9efaf806d876e6f"
)
P2_CONTRACT_SOURCE_SHA256 = (
    "d2130127ac50fc35644472263f22207f71a14796c6281669427814bdb57a9bf3"
)
PLAN_SHA256 = (
    "bff6abc2131b9ad2bc5e43138838793e89204aefe11a944ee68d248cc3a58a4e"
)
BASE_PLAN_SHA256 = (
    "ec01e2639b532081f5fb928f7996b3b7a40d71377e1cd74ac2f45fa73864f229"
)
ENDPOINT_EVALUATOR_SHA256 = (
    "80caa51691611ad89a2496e2ca89f1c4039777d1d1a9b663fc550a26cff585f0"
)
P2_EVALUATOR_SHA256 = (
    "65551135a0eb3eac5e1b65447a499c5849c154ec75f16db762903198eb2bf920"
)
P2_RUNTIME_CONTRACT_SHA256 = (
    "b4a69fc7c15df54b0ac8d4b89ba6d2fe9ec169568becfedf2615e47260a7bf3d"
)
P2_SELECTION_SHA256 = (
    "99d20a1ee7dad9ab88ab5de2dfe0df50cc9d9e076636cf41252fbb1db2ea371e"
)
P2_CACHE_SHAPE_SHA256 = (
    "6b8b068a1d02480d9c0a9933c19a534bb64eb15fe16e9ae7a313f4ea66c4d5c5"
)
PT_HOLDOUT_SHA256 = (
    "c6f1ed19085c43987775e2013c3dd9a687b04138ec199dc583c1b382a0b4df02"
)
PT_RECORDS = 4_096
PT_TARGET_TOKENS = 12_582_912
CHESS_ROWS = 23_680
ENDPOINT_NAMESPACE = "endpoint_v2r1_weighted_clean"
ENDPOINT_EXPERIMENT_VERSION = (
    "mix10b_sft90k_3072_v2r1_weighted_clean_20260730"
)
ENDPOINT_FINGERPRINTS = {
    "losses": "7812f59ccbfe7162669111b8abf44e21a4a010264a7258564428ef0d5c1d0cf4",
    "chess": "206c3c1ad7671f3bcf139004b2aa406a1231e35d165d9ae01b257c22f7e1bf69",
}

DATA_VOLUME = "chess-rl-miles-data"
CHECKPOINT_VOLUME = "chess-rl-miles-checkpoints"
RESULTS_VOLUME = "chess-rl-eval-results-r6"
DEFAULT_CONTRACT_PATH = (
    "50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/"
    "v2r4a_production_gate_20260730/runtime_contract.json"
)
DEFAULT_GATE_LEDGER = "INTERLEAVED_V2R4A_GATE_LAUNCH_LEDGER.json"
DEFAULT_ENDPOINT_LEDGER = "INTERLEAVED_V2R4_ENDPOINT_LAUNCH_LEDGER.json"
DEFAULT_P2_LEDGER = "INTERLEAVED_V2R4_P2_SFT_V2_LAUNCH_LEDGER.json"
DEFAULT_QUARANTINE_LEDGER = "INTERLEAVED_V2R4_GATE_LAUNCH_LEDGER.json"
DEFAULT_OUTPUT = "INTERLEAVED_V2R4A_PRODUCTION_GATE_REPORT.json"
RAW_VOLUME_ROOT = PurePosixPath("chess-rl-miles-interleave")
CHECKPOINT_MOUNT = PurePosixPath("/rl-checkpoints")

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FUNCTION_CALL_RE = re.compile(r"fc-[A-Za-z0-9]+")
_EXPECTED_RECURSIVE_HF = {
    6_000: "5df40e4794193a490297e19837ea5d8ec49326329ab405e58234b67519862425",
    8_000: "d17a709df6debd483932e3e38214a91a1ec1f62814dd73dd6cad1f51a9b6070e",
    9_920: "d0c013bf51c17691ef9bdf5e5d65561912471ef949a161f80b4aa818da96c4fd",
}
_EXPECTED_P2_CANDIDATE_IDS = {
    6_000: "v2r4-w190-step6000",
    8_000: "v2r4-w190-step8000",
    9_920: "v2r4-w190-step9920",
}
_QUARANTINE_LEDGER_SHA256 = (
    "8367979ea65f37d5bcf921cda3c3bbf465e39cc15bf20107b23ea68d9d3b980b"
)
_QUARANTINE_LEDGER_FILE_SHA256 = (
    "632ea875c1403f9681c973f8074094f0d206ec6ffa67787f827756d69976608e"
)
_QUARANTINE_CONTRACT_SHA256 = (
    "3127bdd6dfca62b34813e3fe938300d5d44c8d7ac253bf4a65836f4b2fc1ffd3"
)
_QUARANTINE_CONTRACT_FILE_SHA256 = (
    "9bc355fb7ac89dda15cbf4d0c1a4767a3ac5e314e6c800c398f3e9062de02f29"
)
_QUARANTINE_VERSION = "v2r4_production_gate_20260730"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    return str(value)


def _require_call_id(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _FUNCTION_CALL_RE.fullmatch(value) is not None,
        f"{label} is not a FunctionCall ID",
    )
    return str(value)


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    _require(
        set(value) == expected,
        f"{label} fields drifted: {sorted(value)} != {sorted(expected)}",
    )


def _require_self_hash(
    value: Mapping[str, Any], field: str, label: str
) -> str:
    observed = _require_sha256(value.get(field), f"{label}.{field}")
    expected = content_hash(value, field)
    _require(observed == expected, f"{label} has an invalid {field}")
    return observed


def _finite_number(value: Any, label: str) -> float:
    _require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{label} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _exact_nonnegative_int(value: Any, label: str) -> int:
    _require(
        not isinstance(value, bool) and isinstance(value, int) and value >= 0,
        f"{label} must be a non-negative integer",
    )
    return int(value)


def read_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    return _require_mapping(value, label)


def read_json_file(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    return raw, read_json_bytes(raw, label)


def validate_local_sources(workspace: Path) -> dict[str, Any]:
    _require(
        ANALYZER_CONTRACT_VERSION == CONTRACT_VERSION
        and ANALYZER_REPORT_SCHEMA
        == "interleaved-v2r4a-production-gate-report-v1",
        "loaded analyzer contract identity drifted",
    )
    files = {
        "plan": (
            workspace / "INTERLEAVED_V2R4A_GATE_AMENDMENT.md",
            PLAN_SHA256,
        ),
        "base_plan": (
            workspace / "INTERLEAVED_V2R4_PRODUCTION_GATE_PLAN.md",
            BASE_PLAN_SHA256,
        ),
        "analyzer": (workspace / "Eval/v2r4_gate_analysis.py", ANALYZER_SHA256),
        "p2_contract": (
            workspace / "chess_reasoning/training/v2r4_p2_sft_eval.py",
            P2_CONTRACT_SOURCE_SHA256,
        ),
    }
    result: dict[str, Any] = {}
    for label, (path, expected) in files.items():
        _require(path.is_file(), f"missing frozen local {label}: {path}")
        observed = sha256_file(path)
        _require(observed == expected, f"frozen local {label} source drifted")
        result[label] = {
            "path": str(path),
            "sha256": observed,
            "bytes": path.stat().st_size,
        }
    return result


def validate_runtime_contract(raw: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = read_json_bytes(raw, "runtime contract")
    contract_sha = _require_self_hash(
        contract, "contract_sha256", "runtime contract"
    )
    _require(
        contract_sha == CONTRACT_SHA256,
        "runtime contract canonical digest differs from frozen v2r4a",
    )
    _require(contract.get("schema") == CONTRACT_SCHEMA, "contract schema drifted")
    _require(
        contract.get("version") == CONTRACT_VERSION,
        "contract version drifted",
    )
    plan = _require_mapping(contract.get("plan"), "contract.plan")
    _require(
        plan
        == {
            "path": "INTERLEAVED_V2R4A_GATE_AMENDMENT.md",
            "sha256": PLAN_SHA256,
        },
        "contract plan identity drifted",
    )
    evaluators = _require_mapping(
        contract.get("endpoint_evaluators"), "contract.endpoint_evaluators"
    )
    _require(
        evaluators
        == {
            "pt_b1_b5": ENDPOINT_EVALUATOR_SHA256,
            "p2_sft_at_p1": P2_EVALUATOR_SHA256,
        },
        "contract endpoint evaluator identities drifted",
    )
    cells = contract.get("cells")
    expected_cells = [
        {
            "candidate_step": step,
            "batch_label": batch,
            "run_name": f"v2r4a-gate-w190-s{step}-batch-{batch.lower()}",
        }
        for step in CANDIDATE_STEPS
        for batch in BATCH_LABELS
    ]
    _require(cells == expected_cells, "contract cell grid or order drifted")
    candidates = _require_mapping(contract.get("candidates"), "contract.candidates")
    _require(set(candidates) == {str(x) for x in CANDIDATE_STEPS}, "candidate grid drifted")
    batches = _require_mapping(
        contract.get("prompt_batches"), "contract.prompt_batches"
    )
    _require(set(batches) == set(BATCH_LABELS), "prompt batch grid drifted")
    for step in CANDIDATE_STEPS:
        candidate = _require_mapping(candidates[str(step)], f"candidate {step}")
        _require_sha256(
            candidate.get("endpoint_checkpoint_sha256"),
            f"candidate {step} endpoint identity",
        )
        _require_sha256(
            candidate.get("hf_directory_manifest_sha256"),
            f"candidate {step} HF manifest",
        )
        _require(
            candidate.get("original_p1_eligible") is (step == 9_920),
            f"candidate {step} eligibility drifted",
        )
    for batch in BATCH_LABELS:
        value = _require_mapping(batches[batch], f"prompt batch {batch}")
        _require(value.get("rows") == 1_024, f"batch {batch} row count drifted")
        _require_sha256(value.get("sha256"), f"batch {batch} byte hash")
        _require_sha256(
            value.get("prompt_set_sha256"), f"batch {batch} prompt set hash"
        )
        _exact_nonnegative_int(value.get("rollout_seed"), f"batch {batch} seed")
    semantics = _require_mapping(contract.get("semantics"), "contract.semantics")
    expected_semantics = {
        "automatic_retries": 0,
        "debug_rollout_only": True,
        "deterministic_inference": True,
        "dynamic_filter": False,
        "no_requeue": True,
        "no_wrap": True,
        "num_rollout": 4,
        "partial_rollout": False,
        "policy_updates": False,
        "rollout_batch_size": 256,
        "samples_per_prompt": 8,
        "total_prompt_groups": 1_024,
        "total_rows": ROWS_PER_CELL,
    }
    for key, expected in expected_semantics.items():
        _require(semantics.get(key) == expected, f"contract semantic {key} drifted")
    file_sha = sha256_bytes(raw)
    _require(
        file_sha == CONTRACT_FILE_SHA256,
        "runtime contract file digest differs from frozen v2r4a",
    )
    return contract, {
        "schema": contract["schema"],
        "version": contract["version"],
        "contract_sha256": contract_sha,
        "file_sha256": file_sha,
        "bytes": len(raw),
        "plan_sha256": PLAN_SHA256,
        "endpoint_evaluators": evaluators,
    }


def validate_prompt_manifest(
    raw: bytes, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _require_mapping(
        contract.get("prompt_manifest"), "contract.prompt_manifest"
    )
    file_sha = sha256_bytes(raw)
    _require(
        file_sha == binding.get("file_sha256"),
        "prompt manifest byte hash drifted",
    )
    manifest = read_json_bytes(raw, "prompt manifest")
    manifest_sha = _require_self_hash(
        manifest, "manifest_sha256", "prompt manifest"
    )
    _require(
        manifest_sha == binding.get("manifest_sha256"),
        "prompt manifest content hash drifted",
    )
    batches = _require_mapping(manifest.get("batches"), "prompt manifest batches")
    _require(set(batches) == set(BATCH_LABELS), "prompt manifest batch grid drifted")
    contract_batches = _require_mapping(
        contract.get("prompt_batches"), "contract.prompt_batches"
    )
    sets: dict[str, set[str]] = {}
    for label in BATCH_LABELS:
        value = _require_mapping(batches[label], f"prompt manifest batch {label}")
        expected = _require_mapping(
            contract_batches[label], f"contract prompt batch {label}"
        )
        for manifest_key, contract_key in (
            ("file_sha256", "sha256"),
            ("logical_path", "path"),
            ("prompt_set_sha256", "prompt_set_sha256"),
            ("epoch0_prompt_order_sha256", "epoch0_prompt_order_sha256"),
            ("rollout_seed", "rollout_seed"),
            ("rows", "rows"),
        ):
            _require(
                value.get(manifest_key) == expected.get(contract_key),
                f"prompt manifest batch {label} {manifest_key} drifted",
            )
        quarters = value.get("rollout_quarters")
        _require(
            isinstance(quarters, list) and len(quarters) == 4,
            f"batch {label} must contain four rollout quarters",
        )
        flattened: list[str] = []
        for rollout_id, quarter_value in enumerate(quarters):
            quarter = _require_mapping(
                quarter_value, f"batch {label} quarter {rollout_id}"
            )
            prompts = quarter.get("ordered_prompt_fingerprints")
            _require(
                quarter.get("rollout_id") == rollout_id
                and quarter.get("prompt_count") == 256
                and isinstance(prompts, list)
                and len(prompts) == 256
                and all(
                    isinstance(item, str)
                    and _SHA256_RE.fullmatch(item) is not None
                    for item in prompts
                ),
                f"batch {label} quarter {rollout_id} inventory drifted",
            )
            _require(
                quarter.get("prompt_order_sha256")
                == sha256_bytes(canonical_json(prompts)),
                f"batch {label} quarter {rollout_id} prompt hash drifted",
            )
            flattened.extend(prompts)
        _require(len(set(flattened)) == 1_024, f"batch {label} prompts are not unique")
        sets[label] = set(flattened)
    _require(not sets["A"].intersection(sets["B"]), "prompt batches overlap")
    return manifest, {
        "manifest_sha256": manifest_sha,
        "file_sha256": file_sha,
        "bytes": len(raw),
        "batch_prompt_set_sha256": {
            label: contract_batches[label]["prompt_set_sha256"]
            for label in BATCH_LABELS
        },
    }


def validate_gate_ledger(
    raw: bytes, contract: Mapping[str, Any], contract_file_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = read_json_bytes(raw, "gate launch ledger")
    ledger_sha = _require_self_hash(ledger, "ledger_sha256", "gate launch ledger")
    _require(
        ledger_sha == GATE_LEDGER_SHA256
        and sha256_bytes(raw) == GATE_LEDGER_FILE_SHA256,
        "gate launch ledger differs from the frozen v2r4a ledger",
    )
    _require(ledger.get("schema") == GATE_LEDGER_SCHEMA, "gate ledger schema drifted")
    _require(ledger.get("version") == CONTRACT_VERSION, "gate ledger version drifted")
    _require(ledger.get("state") == "launched_all", "gate ledger is not launched_all")
    _require(ledger.get("expected_call_count") == 6, "gate ledger count drifted")
    _require(
        ledger.get("contract_sha256") == contract.get("contract_sha256"),
        "gate ledger contract hash drifted",
    )
    preflight = _require_mapping(ledger.get("preflight"), "gate ledger preflight")
    _require(
        preflight.get("schema") == "interleaved-v2r4-gate-preflight-v1"
        and preflight.get("version") == CONTRACT_VERSION
        and preflight.get("contract_sha256") == contract.get("contract_sha256")
        and preflight.get("contract_file_sha256") == contract_file_sha256
        and preflight.get("all_six_canonical_roots_absent") is True,
        "gate preflight binding drifted",
    )
    expected_run_roots = [
        f"/rl-checkpoints/{RAW_VOLUME_ROOT.as_posix()}/{cell['run_name']}"
        for cell in contract["cells"]
    ]
    _require(
        preflight.get("run_roots") == expected_run_roots
        and preflight.get("ray_worker_environment")
        == {
            "artifact_root": (
                f"/rl-checkpoints/{RAW_VOLUME_ROOT.as_posix()}/"
                "v2r4a-ray-env-preflight"
            ),
            "gpu_allocated": False,
            "seed_mode": "sample-index",
        },
        "gate preflight Ray environment or root inventory drifted",
    )
    _require_call_id(ledger.get("preflight_call_id"), "gate preflight call")
    expected_cells = contract["cells"]
    calls = ledger.get("calls")
    _require(isinstance(calls, list) and len(calls) == 6, "gate call grid incomplete")
    call_ids: list[str] = []
    for observed, expected in zip(calls, expected_cells, strict=True):
        record = _require_mapping(observed, "gate call")
        for key in ("candidate_step", "batch_label", "run_name"):
            _require(record.get(key) == expected[key], f"gate call {key} drifted")
        call_ids.append(_require_call_id(record.get("function_call_id"), "gate call"))
    _require(len(set(call_ids)) == 6, "gate FunctionCall IDs are not unique")
    return ledger, {
        "ledger_sha256": ledger_sha,
        "file_sha256": sha256_bytes(raw),
        "bytes": len(raw),
        "preflight_call_id": ledger["preflight_call_id"],
        "call_ids": call_ids,
    }


def _volume_relative(path: str, label: str) -> str:
    candidate = PurePosixPath(path)
    _require(candidate.is_absolute(), f"{label} must be an absolute mount path")
    try:
        relative = candidate.relative_to(CHECKPOINT_MOUNT)
    except ValueError as exc:
        raise ValueError(f"{label} is outside {CHECKPOINT_MOUNT}") from exc
    _require(".." not in relative.parts, f"{label} contains parent traversal")
    return relative.as_posix()


def _prompt_fingerprint(row: Mapping[str, Any]) -> str:
    return sha256_bytes(
        canonical_json(
            {
                "input": str(row.get("input") or ""),
                "FEN": str(row.get("FEN") or ""),
                "PuzzleId": str(row.get("PuzzleId") or ""),
                "ground_truth": str(row.get("ground_truth") or ""),
            }
        )
    )


def _parse_jsonl(raw: bytes, label: str) -> list[dict[str, Any]]:
    _require(raw.endswith(b"\n"), f"{label} must end with a newline")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        _require(bool(line), f"{label}:{line_number} is blank")
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label}:{line_number} is invalid JSON") from exc
        rows.append(_require_mapping(value, f"{label}:{line_number}"))
    return rows


def validate_gate_cell(
    *,
    record: Mapping[str, Any],
    function_result: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_file_sha256: str,
    prompt_manifest: Mapping[str, Any],
    read_checkpoint_file: Callable[[str], bytes],
) -> tuple[dict[str, Any], dict[str, Any]]:
    step = int(record["candidate_step"])
    batch = str(record["batch_label"])
    run_name = str(record["run_name"])
    call_id = _require_call_id(record.get("function_call_id"), "gate call")
    expected_run = f"v2r4a-gate-w190-s{step}-batch-{batch.lower()}"
    _require(run_name == expected_run, "gate cell run name drifted")
    result = _require_mapping(function_result, f"gate result {call_id}")
    success_path = result.get("success_path")
    expected_success_path = (
        f"/rl-checkpoints/{RAW_VOLUME_ROOT.as_posix()}/{run_name}/"
        "_V2R4_GATE_SUCCESS.json"
    )
    _require(success_path == expected_success_path, "gate success path drifted")
    marker_relative = _volume_relative(str(success_path), "gate success path")
    marker_raw = read_checkpoint_file(marker_relative)
    marker = read_json_bytes(marker_raw, f"gate marker {run_name}")
    expected_marker = {key: value for key, value in result.items() if key != "success_path"}
    _require(marker == expected_marker, "gate marker differs from FunctionCall return")
    _require(marker.get("schema") == GATE_SUCCESS_SCHEMA, "gate success schema drifted")
    success_sha = _require_sha256(marker.get("success_sha256"), "gate success hash")
    success_core = {
        key: value for key, value in marker.items() if key != "success_sha256"
    }
    _require(
        success_sha == sha256_bytes(canonical_json(success_core)),
        "gate success self hash is invalid",
    )
    expected_batch = contract["prompt_batches"][batch]
    expected_cell = {
        "version": CONTRACT_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "run_name": run_name,
        "candidate_step": step,
        "batch_label": batch,
        "prompt_batch_sha256": expected_batch["sha256"],
        "prompt_set_sha256": expected_batch["prompt_set_sha256"],
        "rollout_seed": expected_batch["rollout_seed"],
        "shape_authenticated": True,
        "reward_metrics_inspected": False,
    }
    for key, expected in expected_cell.items():
        _require(marker.get(key) == expected, f"gate success {key} drifted")
    _require(
        marker.get("contract_file_sha256") == contract_file_sha256,
        "gate success contract file hash drifted",
    )

    provenance = _require_mapping(marker.get("provenance"), "gate provenance")
    root_path = provenance.get("root_manifest")
    launch_path = provenance.get("launch_manifest")
    _require(
        root_path
        == f"/rl-checkpoints/{RAW_VOLUME_ROOT.as_posix()}/{run_name}/run_provenance.json",
        "gate root provenance path drifted",
    )
    _require(
        isinstance(launch_path, str)
        and launch_path.startswith(
            f"/rl-checkpoints/{RAW_VOLUME_ROOT.as_posix()}/{run_name}/provenance/launch_"
        )
        and launch_path.endswith(".json"),
        "gate launch provenance path drifted",
    )
    root_raw = read_checkpoint_file(_volume_relative(root_path, "root provenance"))
    launch_raw = read_checkpoint_file(
        _volume_relative(str(launch_path), "launch provenance")
    )
    root_doc = read_json_bytes(root_raw, f"root provenance {run_name}")
    launch_doc = read_json_bytes(launch_raw, f"launch provenance {run_name}")
    identity = _require_mapping(root_doc.get("identity"), "run identity")
    identity_sha = sha256_bytes(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    )
    command = root_doc.get("initial_command")
    _require(isinstance(command, list) and all(isinstance(x, str) for x in command), "gate command drifted")
    command_sha = sha256_bytes(json.dumps(command, separators=(",", ":")).encode())
    _require(
        root_doc.get("identity_sha256") == identity_sha
        and root_doc.get("initial_command_sha256") == command_sha
        and launch_doc.get("identity_sha256") == identity_sha
        and launch_doc.get("command_sha256") == command_sha
        and launch_doc.get("command") == command
        and provenance.get("identity_sha256") == identity_sha
        and provenance.get("command_sha256") == command_sha,
        "gate provenance hashes drifted",
    )
    _require(
        identity.get("kind")
        == "chess_rl_miles_v2r4_production_gate_rollout"
        and identity.get("version") == CONTRACT_VERSION
        and identity.get("contract_sha256") == contract["contract_sha256"]
        and identity.get("contract_file_sha256") == contract_file_sha256
        and identity.get("authorized_cell")
        == {"candidate_step": step, "batch_label": batch, "run_name": run_name}
        and identity.get("semantics") == contract["semantics"]
        and identity.get("sources") == contract["sources"],
        "gate run identity drifted",
    )
    identity_candidate = _require_mapping(identity.get("candidate"), "identity candidate")
    expected_candidate = contract["candidates"][str(step)]
    _require(
        identity_candidate.get("step") == step
        and all(
            identity_candidate.get(key) == expected
            for key, expected in expected_candidate.items()
        )
        and _require_mapping(
            identity_candidate.get("directory_identity"),
            "identity candidate directory",
        ).get("manifest_sha256")
        == expected_candidate["hf_directory_manifest_sha256"],
        "gate candidate provenance drifted",
    )
    identity_batch = _require_mapping(identity.get("prompt_batch"), "identity prompt batch")
    _require(
        identity_batch.get("label") == batch
        and all(
            identity_batch.get(key) == expected
            for key, expected in expected_batch.items()
        )
        and identity_batch.get("manifest_sha256")
        == contract["prompt_manifest"]["manifest_sha256"]
        and identity_batch.get("manifest_file_sha256")
        == contract["prompt_manifest"]["file_sha256"],
        "gate prompt provenance drifted",
    )
    identity_runtime = _require_mapping(identity.get("runtime"), "identity runtime")
    _require(
        identity_runtime.get("image") == contract["runtime"]["miles_image"],
        "gate runtime image drifted",
    )

    records = marker.get("artifact_records")
    _require(
        isinstance(records, list) and len(records) == 4,
        "gate artifact record count drifted",
    )
    quarters = prompt_manifest["batches"][batch]["rollout_quarters"]
    all_rows: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = []
    for rollout_id, (artifact_value, quarter) in enumerate(
        zip(records, quarters, strict=True)
    ):
        artifact = _require_mapping(
            artifact_value, f"gate artifact {run_name}/{rollout_id}"
        )
        expected_path = (
            f"/rl-checkpoints/{RAW_VOLUME_ROOT.as_posix()}/{run_name}/"
            f"rollouts/training/rollout_{rollout_id}.jsonl"
        )
        _require(
            artifact.get("rollout_id") == rollout_id
            and artifact.get("path") == expected_path
            and artifact.get("rows") == 2_048,
            f"gate artifact {rollout_id} identity drifted",
        )
        raw = read_checkpoint_file(_volume_relative(expected_path, "rollout artifact"))
        observed_sha = sha256_bytes(raw)
        _require(
            artifact.get("sha256") == observed_sha
            and artifact.get("bytes") == len(raw),
            f"gate artifact {rollout_id} byte identity drifted",
        )
        rows = _parse_jsonl(raw, f"{run_name}/rollout_{rollout_id}.jsonl")
        _require(len(rows) == 2_048, f"rollout {rollout_id} row count drifted")
        prompts: list[str] = []
        for local_group in range(256):
            group_rows = rows[local_group * 8 : (local_group + 1) * 8]
            fingerprints = {_prompt_fingerprint(row) for row in group_rows}
            _require(
                len(fingerprints) == 1,
                f"rollout {rollout_id} sibling prompt identity drifted",
            )
            prompts.append(next(iter(fingerprints)))
            for row in group_rows:
                _require(
                    row.get("rollout_id") == rollout_id,
                    f"rollout {rollout_id} row rollout_id drifted",
                )
        _require(
            prompts == quarter["ordered_prompt_fingerprints"],
            f"rollout {rollout_id} prompt order drifted",
        )
        prompt_order_sha = sha256_bytes(canonical_json(prompts))
        _require(
            artifact.get("prompt_order_sha256") == prompt_order_sha
            and quarter.get("prompt_order_sha256") == prompt_order_sha,
            f"rollout {rollout_id} prompt order hash drifted",
        )
        all_rows.extend(rows)
        evidence_records.append(
            {
                "rollout_id": rollout_id,
                "volume_path": _volume_relative(expected_path, "rollout artifact"),
                "rows": len(rows),
                "bytes": len(raw),
                "sha256": observed_sha,
                "prompt_order_sha256": prompt_order_sha,
            }
        )
    cell = audit_cell(
        all_rows,
        candidate_step=step,
        batch_label=batch,
        rollout_seed=int(expected_batch["rollout_seed"]),
    )
    return cell, {
        "candidate_step": step,
        "batch_label": batch,
        "run_name": run_name,
        "function_call_id": call_id,
        "success_sha256": success_sha,
        "success_marker": {
            "volume_path": marker_relative,
            "file_sha256": sha256_bytes(marker_raw),
            "bytes": len(marker_raw),
        },
        "provenance": {
            "identity_sha256": identity_sha,
            "command_sha256": command_sha,
            "root_manifest_file_sha256": sha256_bytes(root_raw),
            "launch_manifest_file_sha256": sha256_bytes(launch_raw),
        },
        "artifacts": evidence_records,
    }


def validate_endpoint_ledger(
    raw: bytes, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = read_json_bytes(raw, "endpoint launch ledger")
    ledger_sha = _require_self_hash(
        ledger, "ledger_sha256", "endpoint launch ledger"
    )
    _require(
        ledger_sha == ENDPOINT_LEDGER_SHA256
        and sha256_bytes(raw) == ENDPOINT_LEDGER_FILE_SHA256,
        "endpoint launch ledger differs from the frozen exact six calls",
    )
    _require(
        ledger.get("schema") == ENDPOINT_LEDGER_SCHEMA,
        "endpoint ledger schema drifted",
    )
    _require(
        ledger.get("app_name") == "chess-interleave-endpoint-eval-v2r1",
        "endpoint ledger app identity drifted",
    )
    _require(ledger.get("state") == "launched_all", "endpoint ledger not launched_all")
    _require(ledger.get("expected_call_count") == 6, "endpoint ledger count drifted")
    _require(
        ledger.get("endpoint_eval_source_sha256")
        == contract["endpoint_evaluators"]["pt_b1_b5"],
        "endpoint ledger evaluator source drifted",
    )
    calls = ledger.get("calls")
    expected_grid = [
        (step, component)
        for step in CANDIDATE_STEPS
        for component in ("losses", "chess")
    ]
    _require(isinstance(calls, list) and len(calls) == 6, "endpoint call grid incomplete")
    ids: list[str] = []
    for value, (step, component) in zip(calls, expected_grid, strict=True):
        record = _require_mapping(value, "endpoint call")
        _require(
            record.get("step") == step
            and record.get("component") == component
            and record.get("endpoint_id") == f"v2r4-s{step}"
            and record.get("checkpoint_sha256")
            == contract["candidates"][str(step)]["endpoint_checkpoint_sha256"],
            f"endpoint call {step}/{component} identity drifted",
        )
        ids.append(_require_call_id(record.get("function_call_id"), "endpoint call"))
    _require(len(set(ids)) == 6, "endpoint FunctionCall IDs are not unique")
    return ledger, {
        "ledger_sha256": ledger_sha,
        "file_sha256": sha256_bytes(raw),
        "bytes": len(raw),
        "call_ids": ids,
        "evaluator_source_sha256": ledger["endpoint_eval_source_sha256"],
    }


def _validate_result_self_hash(
    value: Mapping[str, Any], field: str, label: str
) -> str:
    return _require_self_hash(value, field, label)


def validate_endpoint_result(
    record: Mapping[str, Any],
    function_result: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    step = int(record["step"])
    component = str(record["component"])
    call_id = _require_call_id(record.get("function_call_id"), "endpoint call")
    result = _require_mapping(function_result, f"endpoint result {call_id}")
    result_hash = _validate_result_self_hash(result, "result_hash", "endpoint result")
    checkpoint_sha = contract["candidates"][str(step)][
        "endpoint_checkpoint_sha256"
    ]
    _require(
        result.get("schema") == ENDPOINT_RESULT_SCHEMA
        and result.get("state") == "complete"
        and result.get("component") == component
        and result.get("namespace") == ENDPOINT_NAMESPACE
        and result.get("experiment_version") == ENDPOINT_EXPERIMENT_VERSION
        and result.get("endpoint_id") == f"v2r4-s{step}"
        and result.get("checkpoint_sha256") == checkpoint_sha,
        f"endpoint result {step}/{component} identity drifted",
    )
    _require(
        result.get("eval_fingerprint") == ENDPOINT_FINGERPRINTS[component],
        f"endpoint result {step}/{component} evaluator fingerprint drifted",
    )
    endpoint = _require_mapping(result.get("endpoint"), "endpoint result declaration")
    training_state = _require_mapping(
        endpoint.get("training_state"), "endpoint training state"
    )
    _require(
        endpoint.get("declared_checkpoint_sha256") == checkpoint_sha
        and training_state.get("snapshot_step") == step
        and training_state.get("p2_consumed") is False,
        f"endpoint result {step}/{component} checkpoint declaration drifted",
    )
    metrics = _require_mapping(result.get("metrics"), "endpoint metrics")
    if component == "losses":
        datasets = _require_mapping(result.get("datasets"), "loss datasets")
        pt = _require_mapping(datasets.get("pretraining"), "PT holdout")
        _require(
            pt.get("holdout_hash") == PT_HOLDOUT_SHA256
            and pt.get("records") == PT_RECORDS
            and pt.get("target_tokens") == PT_TARGET_TOKENS,
            "PT holdout identity drifted",
        )
        normalized = {
            "cross_entropy": _finite_number(
                metrics.get("heldout_pretrain_loss"), "PT cross entropy"
            ),
            "perplexity": _finite_number(
                metrics.get("heldout_pretrain_perplexity"), "PT perplexity"
            ),
            "token_accuracy": _finite_number(
                metrics.get("heldout_pretrain_token_accuracy"), "PT accuracy"
            ),
            "correct_tokens": _exact_nonnegative_int(
                metrics.get("heldout_pretrain_correct_tokens"), "PT correct tokens"
            ),
            "target_tokens": _exact_nonnegative_int(
                metrics.get("heldout_pretrain_target_tokens"), "PT targets"
            ),
        }
        _require(normalized["target_tokens"] == PT_TARGET_TOKENS, "PT denominator drifted")
        _require(
            normalized["correct_tokens"] <= normalized["target_tokens"],
            "PT correct-token count exceeds denominator",
        )
        _require(
            math.isclose(
                normalized["token_accuracy"],
                normalized["correct_tokens"] / normalized["target_tokens"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ),
            "PT accuracy is inconsistent with counts",
        )
        analyzer_component = "pt"
    else:
        _require(
            result.get("expected_rows") == CHESS_ROWS
            and result.get("actual_rows") == CHESS_ROWS,
            "B1-B5 row inventory drifted",
        )
        benchmarks = _require_mapping(metrics.get("benchmarks"), "chess benchmarks")
        _require(
            set(benchmarks) == {"B1", "B2", "B3", "B4", "B5"},
            "chess benchmark grid drifted",
        )
        normalized = {
            "avg_reward": _finite_number(metrics.get("avg_reward"), "chess avg reward"),
            "pass_at_1": _finite_number(metrics.get("pass_at_1"), "chess Pass@1"),
            "b3_avg": _finite_number(metrics.get("b3_avg"), "B3 average"),
            "b4_avg": _finite_number(metrics.get("b4_avg"), "B4 average"),
            "b3_b4_avg": _finite_number(metrics.get("b3_b4_avg"), "B3-B4 average"),
        }
        for benchmark in ("B1", "B2", "B3", "B4", "B5"):
            item = _require_mapping(benchmarks[benchmark], f"benchmark {benchmark}")
            normalized[f"{benchmark.lower()}_avg_reward"] = _finite_number(
                item.get("avg_reward"), f"{benchmark} avg reward"
            )
            normalized[f"{benchmark.lower()}_pass_at_1"] = _finite_number(
                item.get("pass_at_1"), f"{benchmark} Pass@1"
            )
        _require(
            math.isclose(
                normalized["b3_avg"],
                normalized["b3_avg_reward"],
                rel_tol=1e-15,
                abs_tol=1e-15,
            )
            and math.isclose(
                normalized["b4_avg"],
                normalized["b4_avg_reward"],
                rel_tol=1e-15,
                abs_tol=1e-15,
            )
            and math.isclose(
                normalized["b3_b4_avg"],
                (normalized["b3_avg"] + normalized["b4_avg"]) / 2.0,
                rel_tol=1e-15,
                abs_tol=1e-15,
            ),
            "B3-B4 aggregate metrics are inconsistent",
        )
        analyzer_component = "chess"
    evidence = {
        "candidate_step": step,
        "component": analyzer_component,
        "function_call_id": call_id,
        "result_hash": result_hash,
        "eval_fingerprint": result.get("eval_fingerprint"),
        "checkpoint_sha256": checkpoint_sha,
        "metrics": normalized,
    }
    return analyzer_component, {
        "state": "complete",
        "metrics": normalized,
    }, evidence


def _results_relative(path: str, label: str) -> str:
    candidate = PurePosixPath(path)
    _require(candidate.is_absolute(), f"{label} must be an absolute path")
    try:
        relative = candidate.relative_to(PurePosixPath("/results"))
    except ValueError as exc:
        raise ValueError(f"{label} is outside /results") from exc
    _require(".." not in relative.parts, f"{label} contains parent traversal")
    return relative.as_posix()


def _audit_jsonl_rows(raw: bytes, expected_rows: int, label: str) -> int:
    _require(raw.endswith(b"\n"), f"{label} must end with a newline")
    rows = 0
    for rows, line in enumerate(raw.splitlines(), start=1):
        _require(bool(line), f"{label}:{rows} is blank")
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label}:{rows} is invalid JSON") from exc
        _require(isinstance(value, Mapping), f"{label}:{rows} is not an object")
    _require(rows == expected_rows, f"{label} row count drifted")
    return rows


def authenticate_endpoint_artifacts(
    function_result: Mapping[str, Any],
    *,
    read_results_file: Callable[[str], bytes],
) -> dict[str, Any]:
    """Bind the persisted success file and, for chess, both raw artifacts."""

    result = _require_mapping(function_result, "endpoint artifact result")
    component = str(result.get("component"))
    _require(component in {"losses", "chess"}, "endpoint component drifted")
    endpoint_id = str(result.get("endpoint_id"))
    checkpoint_sha = _require_sha256(
        result.get("checkpoint_sha256"), "endpoint checkpoint"
    )
    fingerprint = _require_sha256(
        result.get("eval_fingerprint"), "endpoint evaluator fingerprint"
    )
    component_root = (
        f"{ENDPOINT_NAMESPACE}/{endpoint_id}/{checkpoint_sha}/"
        f"{component}_{fingerprint[:12]}"
    )
    success_path = f"{component_root}/_SUCCESS.json"
    success_raw = read_results_file(success_path)
    success = read_json_bytes(success_raw, f"endpoint success {endpoint_id}/{component}")
    _require(
        success == result,
        f"persisted endpoint success differs from {component} FunctionCall return",
    )
    evidence: dict[str, Any] = {
        "success_file": {
            "volume_path": success_path,
            "sha256": sha256_bytes(success_raw),
            "bytes": len(success_raw),
        }
    }
    if component == "losses":
        return evidence

    expected_generation_path = (
        f"{component_root}/output/eval/generations/0.jsonl"
    )
    expected_metrics_path = (
        f"{component_root}/output/eval/generations/metrics.json"
    )
    observed_generation_path = _results_relative(
        str(result.get("generations")), "chess generations"
    )
    observed_metrics_path = _results_relative(
        str(result.get("raw_metrics_path")), "chess raw metrics"
    )
    _require(
        observed_generation_path == expected_generation_path
        and observed_metrics_path == expected_metrics_path,
        "chess raw artifact paths drifted",
    )
    generations_raw = read_results_file(expected_generation_path)
    raw_metrics_bytes = read_results_file(expected_metrics_path)
    rows = _audit_jsonl_rows(
        generations_raw, CHESS_ROWS, f"{endpoint_id} raw generations"
    )
    raw_metrics = read_json_bytes(
        raw_metrics_bytes, f"{endpoint_id} raw chess metrics"
    )
    _require(bool(raw_metrics), "raw chess metrics are empty")
    for key, value in raw_metrics.items():
        _require(isinstance(key, str) and key, "raw chess metric key drifted")
        _finite_number(value, f"raw chess metric {key}")
    evidence["raw_chess_artifacts"] = {
        "generations": {
            "volume_path": expected_generation_path,
            "sha256": sha256_bytes(generations_raw),
            "bytes": len(generations_raw),
            "rows": rows,
        },
        "metrics": {
            "volume_path": expected_metrics_path,
            "sha256": sha256_bytes(raw_metrics_bytes),
            "bytes": len(raw_metrics_bytes),
            "metric_count": len(raw_metrics),
        },
    }
    return evidence


def validate_p2_ledger(
    raw: bytes, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = read_json_bytes(raw, "P2 launch ledger")
    ledger_sha = _require_self_hash(ledger, "ledger_hash", "P2 launch ledger")
    _require(
        ledger_sha == P2_LEDGER_SHA256
        and sha256_bytes(raw) == P2_LEDGER_FILE_SHA256,
        "P2 launch ledger differs from the frozen exact three calls",
    )
    _require(ledger.get("schema") == P2_LEDGER_SCHEMA, "P2 ledger schema drifted")
    _require(
        ledger.get("app_name") == "chess-interleave-v2r4-p2-sft-eval-v2"
        and ledger.get("function_name") == "evaluate_p2_sft_candidate",
        "P2 ledger app or function identity drifted",
    )
    _require(ledger.get("state") == "launched_all", "P2 ledger is not launched_all")
    _require(ledger.get("expected_call_count") == 3, "P2 ledger count drifted")
    _require(
        ledger.get("evaluator_source_sha256")
        == contract["endpoint_evaluators"]["p2_sft_at_p1"]
        == P2_EVALUATOR_SHA256,
        "P2 evaluator source drifted",
    )
    _require(
        ledger.get("runtime_contract_sha256") == P2_RUNTIME_CONTRACT_SHA256,
        "P2 runtime contract drifted",
    )
    for field, hash_field in (
        ("dependency_preflight", "dependency_preflight_hash"),
        ("preflight", "preflight_hash"),
    ):
        preflight = _require_mapping(ledger.get(field), f"P2 {field}")
        _require(preflight.get("state") == "complete", f"P2 {field} did not complete")
        _require_self_hash(preflight, hash_field, f"P2 {field}")
        _require(
            preflight.get("runtime_contract_sha256")
            == P2_RUNTIME_CONTRACT_SHA256,
            f"P2 {field} runtime contract drifted",
        )
    calls = ledger.get("calls")
    _require(isinstance(calls, list) and len(calls) == 3, "P2 call grid incomplete")
    ids: list[str] = []
    for value, step in zip(calls, CANDIDATE_STEPS, strict=True):
        record = _require_mapping(value, "P2 call")
        kwargs = _require_mapping(record.get("kwargs"), "P2 call kwargs")
        _require(
            record.get("app_name")
            == "chess-interleave-v2r4-p2-sft-eval-v2"
            and record.get("function_name") == "evaluate_p2_sft_candidate"
            and kwargs.get("candidate_step") == step
            and kwargs.get("expected_recursive_hf_identity")
            == _EXPECTED_RECURSIVE_HF[step]
            and kwargs.get("expected_endpoint_checkpoint_sha256")
            == contract["candidates"][str(step)]["endpoint_checkpoint_sha256"]
            and kwargs.get("runtime_contract_sha256")
            == P2_RUNTIME_CONTRACT_SHA256,
            f"P2 call {step} identity drifted",
        )
        expected_result = (
            "/results/v2r4_p2_sft_at_p1_20260730_v2/"
            f"{_EXPECTED_P2_CANDIDATE_IDS[step]}/"
            f"{_EXPECTED_RECURSIVE_HF[step]}/_SUCCESS.json"
        )
        _require(
            record.get("expected_result_path") == expected_result,
            f"P2 call {step} result path drifted",
        )
        ids.append(_require_call_id(record.get("function_call_id"), "P2 call"))
    _require(len(set(ids)) == 3, "P2 FunctionCall IDs are not unique")
    return ledger, {
        "ledger_sha256": ledger_sha,
        "file_sha256": sha256_bytes(raw),
        "bytes": len(raw),
        "call_ids": ids,
        "evaluator_source_sha256": ledger["evaluator_source_sha256"],
        "runtime_contract_sha256": ledger["runtime_contract_sha256"],
    }


def validate_p2_result(
    record: Mapping[str, Any],
    function_result: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    kwargs = _require_mapping(record.get("kwargs"), "P2 call kwargs")
    step = int(kwargs["candidate_step"])
    call_id = _require_call_id(record.get("function_call_id"), "P2 call")
    result = _require_mapping(function_result, f"P2 result {call_id}")
    output_hash = _require_self_hash(result, "output_hash", "P2 result")
    _require(
        result.get("schema") == P2_OUTPUT_SCHEMA
        and result.get("state") == "complete"
        and result.get("runtime_contract_sha256") == P2_RUNTIME_CONTRACT_SHA256,
        f"P2 result {step} envelope drifted",
    )
    source = _require_mapping(result.get("evaluator_source"), "P2 evaluator source")
    _require(
        source.get("bundle_sha256") == P2_EVALUATOR_SHA256,
        "P2 output evaluator source drifted",
    )
    selection = _require_mapping(result.get("selection"), "P2 selection")
    cache_shape = _require_mapping(result.get("cache_shape"), "P2 cache shape")
    _require_self_hash(selection, "selection_hash", "P2 selection")
    _require_self_hash(cache_shape, "cache_shape_hash", "P2 cache shape")
    _require(
        selection.get("selection_hash") == P2_SELECTION_SHA256
        and cache_shape.get("cache_shape_hash") == P2_CACHE_SHAPE_SHA256,
        "P2 selection or denominator identity drifted",
    )
    pure_result = _require_mapping(result.get("p2_sft_result"), "P2 pure result")
    p2_contract.validate_candidate_sft_result(
        pure_result,
        selection_manifest=selection,
        cache_shape=cache_shape,
    )
    candidate = _require_mapping(pure_result.get("candidate"), "P2 candidate")
    _require(
        candidate.get("checkpoint_step") == step
        and candidate.get("candidate_id") == _EXPECTED_P2_CANDIDATE_IDS[step]
        and candidate.get("checkpoint_sha256") == _EXPECTED_RECURSIVE_HF[step]
        and candidate.get("training_leg") == "p1"
        and candidate.get("has_consumed_p2") is False,
        f"P2 result {step} candidate drifted",
    )
    checkpoint = _require_mapping(result.get("checkpoint"), "P2 checkpoint")
    _require(
        checkpoint.get("recursive_hf_identity") == _EXPECTED_RECURSIVE_HF[step]
        and checkpoint.get("endpoint_checkpoint_sha256")
        == contract["candidates"][str(step)]["endpoint_checkpoint_sha256"]
        and checkpoint.get("directory_manifest_sha256")
        == contract["candidates"][str(step)]["hf_directory_manifest_sha256"],
        f"P2 result {step} checkpoint identity drifted",
    )
    bindings = _require_mapping(result.get("hash_bindings"), "P2 hash bindings")
    _require(
        bindings.get("selection_hash") == P2_SELECTION_SHA256
        and bindings.get("cache_shape_hash") == P2_CACHE_SHAPE_SHA256
        and bindings.get("candidate_result_hash") == pure_result["result_hash"]
        and bindings.get("checkpoint_recursive_hf_identity")
        == _EXPECTED_RECURSIVE_HF[step]
        and bindings.get("evaluator_source_sha256") == P2_EVALUATOR_SHA256
        and bindings.get("runtime_contract_sha256")
        == P2_RUNTIME_CONTRACT_SHA256,
        f"P2 result {step} hash bindings drifted",
    )
    metrics = _require_mapping(pure_result.get("metrics"), "P2 metrics")
    normalized = {
        "cross_entropy": _finite_number(
            metrics.get("masked_sft_unweighted_token_ce"), "P2 cross entropy"
        ),
        "perplexity": _finite_number(
            metrics.get("masked_sft_perplexity"), "P2 perplexity"
        ),
        "token_accuracy": _finite_number(
            metrics.get("masked_sft_token_accuracy"), "P2 accuracy"
        ),
    }
    aggregate = _require_mapping(pure_result.get("aggregate"), "P2 aggregate")
    normalized.update(
        {
            "negative_log_likelihood_sum": _finite_number(
                aggregate.get("negative_log_likelihood_sum"), "P2 NLL sum"
            ),
            "correct_supervised_tokens": _exact_nonnegative_int(
                aggregate.get("correct_supervised_tokens"), "P2 correct tokens"
            ),
            "supervised_targets": _exact_nonnegative_int(
                aggregate.get("supervised_targets"), "P2 supervised targets"
            ),
            "rows_evaluated": _exact_nonnegative_int(
                aggregate.get("rows_evaluated"), "P2 evaluated rows"
            ),
        }
    )
    return {"state": "complete", "metrics": normalized}, {
        "candidate_step": step,
        "component": "p2_sft",
        "function_call_id": call_id,
        "output_hash": output_hash,
        "candidate_result_hash": pure_result["result_hash"],
        "checkpoint_recursive_hf_identity": _EXPECTED_RECURSIVE_HF[step],
        "metrics": normalized,
    }


def authenticate_p2_success_file(
    record: Mapping[str, Any],
    function_result: Mapping[str, Any],
    *,
    read_results_file: Callable[[str], bytes],
) -> dict[str, Any]:
    expected = _results_relative(
        str(record.get("expected_result_path")), "P2 expected result path"
    )
    raw = read_results_file(expected)
    persisted = read_json_bytes(raw, "persisted P2 success")
    _require(
        persisted == dict(function_result),
        "persisted P2 success differs from FunctionCall return",
    )
    return {
        "volume_path": expected,
        "sha256": sha256_bytes(raw),
        "bytes": len(raw),
    }


def validate_quarantined_v2r4(
    raw: bytes,
    *,
    terminal_failure: Callable[[str], Mapping[str, Any]],
    list_checkpoint_entries: Callable[[str], Sequence[Mapping[str, Any]]],
    read_checkpoint_file: Callable[[str], bytes],
) -> dict[str, Any]:
    """Prove the superseded launch is terminal and contains zero outcomes."""

    _require(
        sha256_bytes(raw) == _QUARANTINE_LEDGER_FILE_SHA256,
        "quarantine ledger byte hash drifted",
    )
    ledger = read_json_bytes(raw, "quarantined v2r4 ledger")
    ledger_sha = _require_self_hash(
        ledger, "ledger_sha256", "quarantined v2r4 ledger"
    )
    _require(
        ledger_sha == _QUARANTINE_LEDGER_SHA256
        and ledger.get("schema") == GATE_LEDGER_SCHEMA
        and ledger.get("version") == _QUARANTINE_VERSION
        and ledger.get("state") == "launched_all"
        and ledger.get("expected_call_count") == 6
        and ledger.get("contract_sha256") == _QUARANTINE_CONTRACT_SHA256,
        "quarantined v2r4 ledger identity drifted",
    )
    preflight = _require_mapping(
        ledger.get("preflight"), "quarantined v2r4 preflight"
    )
    _require(
        preflight.get("contract_sha256") == _QUARANTINE_CONTRACT_SHA256
        and preflight.get("contract_file_sha256")
        == _QUARANTINE_CONTRACT_FILE_SHA256
        and preflight.get("all_six_canonical_roots_absent") is True,
        "quarantined v2r4 preflight drifted",
    )
    calls = ledger.get("calls")
    _require(
        isinstance(calls, list) and len(calls) == 6,
        "quarantined v2r4 call grid drifted",
    )
    expected_grid = [
        (step, batch)
        for step in CANDIDATE_STEPS
        for batch in BATCH_LABELS
    ]
    call_ids: list[str] = []
    roots: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for value, (step, batch) in zip(calls, expected_grid, strict=True):
        record = _require_mapping(value, "quarantined v2r4 call")
        run_name = f"v2r4-gate-w190-s{step}-batch-{batch.lower()}"
        call_id = _require_call_id(
            record.get("function_call_id"), "quarantined FunctionCall"
        )
        _require(
            record.get("candidate_step") == step
            and record.get("batch_label") == batch
            and record.get("run_name") == run_name,
            "quarantined v2r4 cell identity drifted",
        )
        failure = _require_mapping(
            terminal_failure(call_id), f"quarantined failure {call_id}"
        )
        _require(
            failure.get("type") == "RuntimeError"
            and failure.get("message")
            == f"v2r4 rollout gate failed for {run_name}: exit 1",
            f"quarantined FunctionCall {call_id} failure drifted",
        )
        call_ids.append(call_id)
        failures.append(
            {
                "candidate_step": step,
                "batch_label": batch,
                "function_call_id": call_id,
                **failure,
            }
        )

        root = f"{RAW_VOLUME_ROOT.as_posix()}/{run_name}"
        entries = [
            _require_mapping(item, f"quarantine entry {run_name}")
            for item in list_checkpoint_entries(root)
        ]
        by_relative: dict[str, dict[str, Any]] = {}
        for entry in entries:
            path = str(PurePosixPath(str(entry.get("path")))).lstrip("/")
            prefix = root.rstrip("/") + "/"
            _require(path.startswith(prefix), "quarantine listing escaped run root")
            relative = path.removeprefix(prefix)
            _require(relative and relative not in by_relative, "duplicate quarantine entry")
            by_relative[relative] = entry
        expected_directories = {
            "provenance",
            "rollouts",
            "rollouts/validation",
            "rollouts/training",
            "mlflow",
            "logs",
        }
        directories = {
            relative
            for relative, entry in by_relative.items()
            if entry.get("type") == "directory"
        }
        files = {
            relative
            for relative, entry in by_relative.items()
            if entry.get("type") == "file"
        }
        _require(
            directories == expected_directories,
            f"quarantined {run_name} directory inventory drifted",
        )
        launch_files = {
            item
            for item in files
            if re.fullmatch(r"provenance/launch_[0-9a-f]{16}\.json", item)
        }
        _require(
            files
            == {
                "_V2R4_GATE_INTENT.json",
                "run_provenance.json",
                *launch_files,
            }
            and len(launch_files) == 1,
            f"quarantined {run_name} file inventory exposes unexpected data",
        )
        launch_relative = next(iter(launch_files))
        file_bytes = {
            relative: read_checkpoint_file(f"{root}/{relative}")
            for relative in sorted(files)
        }
        intent = read_json_bytes(
            file_bytes["_V2R4_GATE_INTENT.json"], "quarantine intent"
        )
        provenance = read_json_bytes(
            file_bytes["run_provenance.json"], "quarantine provenance"
        )
        launch = read_json_bytes(
            file_bytes[launch_relative], "quarantine launch provenance"
        )
        _require(
            intent.get("schema") == "interleaved-v2r4-gate-cell-intent-v1"
            and intent.get("version") == _QUARANTINE_VERSION
            and intent.get("contract_sha256") == _QUARANTINE_CONTRACT_SHA256
            and intent.get("contract_file_sha256")
            == _QUARANTINE_CONTRACT_FILE_SHA256
            and intent.get("candidate_step") == step
            and intent.get("batch_label") == batch
            and intent.get("run_name") == run_name,
            f"quarantined {run_name} intent drifted",
        )
        identity = _require_mapping(
            provenance.get("identity"), "quarantine run identity"
        )
        identity_sha = sha256_bytes(
            json.dumps(
                identity,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        )
        command = provenance.get("initial_command")
        _require(
            isinstance(command, list) and all(isinstance(item, str) for item in command),
            "quarantine initial command drifted",
        )
        command_sha = sha256_bytes(
            json.dumps(command, separators=(",", ":")).encode()
        )
        _require(
            identity.get("version") == _QUARANTINE_VERSION
            and identity.get("contract_sha256") == _QUARANTINE_CONTRACT_SHA256
            and identity.get("contract_file_sha256")
            == _QUARANTINE_CONTRACT_FILE_SHA256
            and identity.get("authorized_cell")
            == {
                "candidate_step": step,
                "batch_label": batch,
                "run_name": run_name,
            }
            and provenance.get("identity_sha256") == identity_sha
            and provenance.get("initial_command_sha256") == command_sha
            and launch.get("identity_sha256") == identity_sha
            and launch.get("command_sha256") == command_sha
            and launch.get("command") == command,
            f"quarantined {run_name} provenance drifted",
        )
        roots.append(
            {
                "candidate_step": step,
                "batch_label": batch,
                "run_name": run_name,
                "zero_rollout_jsonls": True,
                "zero_success_markers": True,
                "zero_reward_or_outcome_artifacts": True,
                "directories": sorted(directories),
                "files": [
                    {
                        "path": relative,
                        "bytes": len(file_bytes[relative]),
                        "sha256": sha256_bytes(file_bytes[relative]),
                    }
                    for relative in sorted(files)
                ],
            }
        )
    _require(len(set(call_ids)) == 6, "quarantined FunctionCall IDs are not unique")
    return {
        "schema": "interleaved-v2r4-no-outcome-quarantine-evidence-v1",
        "state": "authenticated_terminal_failure_no_outcomes",
        "ledger": {
            "ledger_sha256": ledger_sha,
            "file_sha256": sha256_bytes(raw),
            "bytes": len(raw),
            "contract_sha256": _QUARANTINE_CONTRACT_SHA256,
            "contract_file_sha256": _QUARANTINE_CONTRACT_FILE_SHA256,
        },
        "terminal_failures": failures,
        "roots": roots,
        "aggregate": {
            "calls": 6,
            "terminal_failures": 6,
            "rollout_jsonl_files": 0,
            "success_markers": 0,
            "sampled_prompts": 0,
            "reward_or_outcome_artifacts": 0,
        },
    }


def build_final_report(
    *,
    contract_evidence: Mapping[str, Any],
    prompt_evidence: Mapping[str, Any],
    gate_ledger_evidence: Mapping[str, Any],
    gate_cell_evidence: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    endpoint_ledger_evidence: Mapping[str, Any],
    endpoint_evidence: Sequence[Mapping[str, Any]],
    p2_ledger_evidence: Mapping[str, Any],
    p2_evidence: Sequence[Mapping[str, Any]],
    quarantine_evidence: Mapping[str, Any],
    endpoints: Mapping[int, Mapping[str, Mapping[str, Any]]],
    source_evidence: Mapping[str, Any],
    finalizer_source: Mapping[str, Any],
) -> dict[str, Any]:
    analysis = analyze_grid(cells, endpoints)
    _require(
        analysis.get("schema") == ANALYZER_REPORT_SCHEMA
        and analysis.get("contract_version") == CONTRACT_VERSION,
        "analyzer emitted a non-v2r4a report",
    )
    core: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "schema_version": 1,
        "contract_version": CONTRACT_VERSION,
        "status": "complete",
        "inputs_authenticated": True,
        "sources": {
            **dict(source_evidence),
            "finalizer": dict(finalizer_source),
        },
        "runtime_contract": dict(contract_evidence),
        "prompt_manifest": dict(prompt_evidence),
        "rollout_gate": {
            "launch_ledger": dict(gate_ledger_evidence),
            "cells": [dict(value) for value in gate_cell_evidence],
        },
        "endpoint_evaluations": {
            "launch_ledger": dict(endpoint_ledger_evidence),
            "results": [dict(value) for value in endpoint_evidence],
        },
        "p2_sft_evaluations": {
            "launch_ledger": dict(p2_ledger_evidence),
            "results": [dict(value) for value in p2_evidence],
        },
        "superseded_v2r4_quarantine": dict(quarantine_evidence),
        "analysis": analysis,
        "analysis_report_sha256": analysis["report_sha256"],
    }
    return {**core, "report_sha256": content_hash(core, "report_sha256")}


class ModalReadOnlyBackend:
    """Minimal read-only Modal adapter, intentionally exposing no mutations."""

    def __init__(self, *, environment_name: str | None, call_timeout: float) -> None:
        import modal

        self._modal = modal
        self._environment_name = environment_name
        self._call_timeout = call_timeout
        self._volumes: dict[str, Any] = {}

    def _volume(self, name: str) -> Any:
        if name not in self._volumes:
            self._volumes[name] = self._modal.Volume.from_name(
                name,
                environment_name=self._environment_name,
                create_if_missing=False,
            )
        return self._volumes[name]

    def read_volume_file(self, volume_name: str, path: str) -> bytes:
        normalized = "/" + str(PurePosixPath(path)).lstrip("/")
        return b"".join(self._volume(volume_name).read_file(normalized))

    def list_volume_entries(
        self, volume_name: str, root: str
    ) -> list[dict[str, Any]]:
        normalized = "/" + str(PurePosixPath(root)).lstrip("/")
        entries: Sequence[Any] | None = None
        for attempt in range(5):
            try:
                entries = list(
                    self._volume(volume_name).iterdir(
                        normalized, recursive=True
                    )
                )
                break
            except Exception as exc:
                if (
                    type(exc).__name__ != "ResourceExhaustedError"
                    or attempt == 4
                ):
                    raise
                time.sleep(2**attempt)
        assert entries is not None
        result: list[dict[str, Any]] = []
        for entry in entries:
            entry_type = int(getattr(entry, "type", 0))
            _require(
                entry_type in {1, 2},
                f"Modal volume entry has unexpected type: {entry.path}",
            )
            result.append(
                {
                    "path": str(entry.path),
                    "type": "file" if entry_type == 1 else "directory",
                    "bytes": int(entry.size),
                }
            )
        return result

    def function_result(self, call_id: str) -> dict[str, Any]:
        _require_call_id(call_id, "FunctionCall")
        try:
            value = self._modal.FunctionCall.from_id(call_id).get(
                timeout=self._call_timeout
            )
        except Exception as exc:
            raise RuntimeError(
                f"FunctionCall {call_id} is not a successful terminal call: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return _require_mapping(value, f"FunctionCall {call_id} return")

    def terminal_failure(self, call_id: str) -> dict[str, Any]:
        _require_call_id(call_id, "FunctionCall")
        try:
            value = self._modal.FunctionCall.from_id(call_id).get(
                timeout=self._call_timeout
            )
        except Exception as exc:
            return {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        raise RuntimeError(
            f"quarantined FunctionCall {call_id} unexpectedly returned: "
            f"{type(value).__name__}"
        )


def collect_and_finalize(
    *,
    workspace: Path,
    gate_ledger_path: Path,
    endpoint_ledger_path: Path,
    p2_ledger_path: Path,
    quarantine_ledger_path: Path,
    backend: ModalReadOnlyBackend,
    data_volume: str = DATA_VOLUME,
    checkpoint_volume: str = CHECKPOINT_VOLUME,
    results_volume: str = RESULTS_VOLUME,
    contract_path: str = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    source_evidence = validate_local_sources(workspace)
    finalizer_path = Path(__file__).resolve()
    finalizer_source = {
        "path": str(finalizer_path),
        "sha256": sha256_file(finalizer_path),
        "bytes": finalizer_path.stat().st_size,
    }

    contract_raw = backend.read_volume_file(data_volume, contract_path)
    contract, contract_evidence = validate_runtime_contract(contract_raw)
    prompt_binding = _require_mapping(
        contract.get("prompt_manifest"), "contract prompt manifest"
    )
    prompt_path = str(prompt_binding["path"])
    _require(
        prompt_path.startswith("/data/"),
        "contract prompt manifest path is outside /data",
    )
    prompt_raw = backend.read_volume_file(data_volume, prompt_path.removeprefix("/data/"))
    prompt_manifest, prompt_evidence = validate_prompt_manifest(
        prompt_raw, contract
    )

    gate_raw, gate_ledger = read_json_file(
        gate_ledger_path, "gate launch ledger"
    )
    gate_ledger, gate_ledger_evidence = validate_gate_ledger(
        gate_raw, contract, contract_evidence["file_sha256"]
    )
    endpoint_raw, endpoint_ledger = read_json_file(
        endpoint_ledger_path, "endpoint launch ledger"
    )
    endpoint_ledger, endpoint_ledger_evidence = validate_endpoint_ledger(
        endpoint_raw, contract
    )
    p2_raw, p2_ledger = read_json_file(p2_ledger_path, "P2 launch ledger")
    p2_ledger, p2_ledger_evidence = validate_p2_ledger(p2_raw, contract)

    checkpoint_reader = lambda path: backend.read_volume_file(
        checkpoint_volume, path
    )
    results_reader = lambda path: backend.read_volume_file(results_volume, path)
    quarantine_raw, _ = read_json_file(
        quarantine_ledger_path, "quarantined v2r4 ledger"
    )
    quarantine_evidence = validate_quarantined_v2r4(
        quarantine_raw,
        terminal_failure=backend.terminal_failure,
        list_checkpoint_entries=lambda root: backend.list_volume_entries(
            checkpoint_volume, root
        ),
        read_checkpoint_file=checkpoint_reader,
    )
    cells: list[dict[str, Any]] = []
    gate_cell_evidence: list[dict[str, Any]] = []
    for record in gate_ledger["calls"]:
        call_id = str(record["function_call_id"])
        cell, evidence = validate_gate_cell(
            record=record,
            function_result=backend.function_result(call_id),
            contract=contract,
            contract_file_sha256=contract_evidence["file_sha256"],
            prompt_manifest=prompt_manifest,
            read_checkpoint_file=checkpoint_reader,
        )
        cells.append(cell)
        gate_cell_evidence.append(evidence)

    endpoints: dict[int, dict[str, dict[str, Any]]] = {
        step: {} for step in CANDIDATE_STEPS
    }
    endpoint_evidence: list[dict[str, Any]] = []
    for record in endpoint_ledger["calls"]:
        call_id = str(record["function_call_id"])
        function_result = backend.function_result(call_id)
        component, normalized, evidence = validate_endpoint_result(
            record, function_result, contract
        )
        evidence["persisted_artifacts"] = authenticate_endpoint_artifacts(
            function_result,
            read_results_file=results_reader,
        )
        endpoints[int(record["step"])][component] = normalized
        endpoint_evidence.append(evidence)
    p2_evidence: list[dict[str, Any]] = []
    for record in p2_ledger["calls"]:
        call_id = str(record["function_call_id"])
        function_result = backend.function_result(call_id)
        normalized, evidence = validate_p2_result(
            record, function_result, contract
        )
        evidence["persisted_success_file"] = authenticate_p2_success_file(
            record,
            function_result,
            read_results_file=results_reader,
        )
        step = int(record["kwargs"]["candidate_step"])
        endpoints[step]["p2_sft"] = normalized
        p2_evidence.append(evidence)

    return build_final_report(
        contract_evidence=contract_evidence,
        prompt_evidence=prompt_evidence,
        gate_ledger_evidence=gate_ledger_evidence,
        gate_cell_evidence=gate_cell_evidence,
        cells=cells,
        endpoint_ledger_evidence=endpoint_ledger_evidence,
        endpoint_evidence=endpoint_evidence,
        p2_ledger_evidence=p2_ledger_evidence,
        p2_evidence=p2_evidence,
        quarantine_evidence=quarantine_evidence,
        endpoints=endpoints,
        source_evidence=source_evidence,
        finalizer_source=finalizer_source,
    )


def write_immutable_report(path: Path, report: Mapping[str, Any]) -> None:
    _require(
        report.get("report_sha256") == content_hash(report, "report_sha256"),
        "refusing to write a report with an invalid self hash",
    )
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable report already exists: {path}")
    encoded = (
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--gate-ledger", type=Path, default=Path(DEFAULT_GATE_LEDGER))
    parser.add_argument(
        "--endpoint-ledger", type=Path, default=Path(DEFAULT_ENDPOINT_LEDGER)
    )
    parser.add_argument("--p2-ledger", type=Path, default=Path(DEFAULT_P2_LEDGER))
    parser.add_argument(
        "--quarantine-ledger",
        type=Path,
        default=Path(DEFAULT_QUARANTINE_LEDGER),
    )
    parser.add_argument("--contract-path", default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--data-volume", default=DATA_VOLUME)
    parser.add_argument("--checkpoint-volume", default=CHECKPOINT_VOLUME)
    parser.add_argument("--results-volume", default=RESULTS_VOLUME)
    parser.add_argument("--modal-environment", default=None)
    parser.add_argument("--call-timeout", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="authenticate and print the report digest without creating a file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace = args.workspace.expanduser().resolve()

    def workspace_path(value: Path) -> Path:
        return value if value.is_absolute() else workspace / value

    backend = ModalReadOnlyBackend(
        environment_name=args.modal_environment,
        call_timeout=args.call_timeout,
    )
    report = collect_and_finalize(
        workspace=workspace,
        gate_ledger_path=workspace_path(args.gate_ledger),
        endpoint_ledger_path=workspace_path(args.endpoint_ledger),
        p2_ledger_path=workspace_path(args.p2_ledger),
        quarantine_ledger_path=workspace_path(args.quarantine_ledger),
        backend=backend,
        data_volume=args.data_volume,
        checkpoint_volume=args.checkpoint_volume,
        results_volume=args.results_volume,
        contract_path=args.contract_path,
    )
    summary = {
        "state": "authenticated",
        "report_sha256": report["report_sha256"],
        "authorization": report["analysis"]["authorization"],
        "behavioral_selection": report["analysis"]["behavioral_selection"],
    }
    if args.check_only:
        summary["output"] = None
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    output = workspace_path(args.output)
    write_immutable_report(output, report)
    summary["output"] = str(output.resolve())
    summary["output_file_sha256"] = sha256_file(output.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
