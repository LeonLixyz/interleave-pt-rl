"""Read-only, fail-closed finalizer for the fresh-prompt v2r4d gate.

This module has no Modal mutation surface.  It authenticates the frozen
contract, prompt data, canary, exact-once ledgers, FunctionCall returns,
recursive rollout-root allowlists, endpoint evidence, and the complete v2r4a
quarantine. It deliberately fetches all six successful grid FunctionCall
returns before reading any grid JSONL.  Only after that barrier does it invoke
the frozen v2r4c analyzer and optionally create one immutable v2r4d report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from Eval import finalize_v2r4_gate as frozen_evidence
from Eval.v2r4c_gate_analysis import (
    BATCH_LABELS,
    CANDIDATE_STEPS,
    CONTRACT_VERSION as ANALYZER_CONTRACT_VERSION,
    ROWS_PER_CELL,
    SCHEMA as ANALYZER_REPORT_SCHEMA,
    analyze_grid,
    audit_cell,
    content_hash,
)

try:
    from chess_rl_miles import v2r4d_contract_binding as contract_binding
except ImportError:
    # Normal repository invocation has the package one directory below the
    # workspace.  Adding that exact local root does not import launcher code.
    _workspace_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_workspace_root / "chess-rl-miles"))
    from chess_rl_miles import v2r4d_contract_binding as contract_binding


REPORT_SCHEMA = "interleaved-v2r4d-production-gate-finalized-report-v1"
CONTRACT_SCHEMA = "interleaved-v2r4d-production-gate-runtime-contract-v1"
CONTRACT_VERSION = "v2r4d_production_gate_20260730"
CONTRACT_SHA256 = contract_binding.EXPECTED_CONTRACT_SHA256
CONTRACT_FILE_SHA256 = contract_binding.EXPECTED_CONTRACT_FILE_SHA256

CANARY_LEDGER_SCHEMA = "interleaved-v2r4d-canary-launch-ledger-v1"
GRID_LEDGER_SCHEMA = "interleaved-v2r4d-gate-launch-ledger-v1"
PREFLIGHT_SCHEMA = "interleaved-v2r4d-gate-preflight-v1"
CANARY_SUCCESS_SCHEMA = "interleaved-v2r4d-canary-success-v1"
GRID_SUCCESS_SCHEMA = "interleaved-v2r4d-gate-cell-success-v1"

PROMPT_MANIFEST_SCHEMA = "interleaved-v2r4c-fresh-prompt-batches-v1"
PROMPT_SOURCE_VERSION = "v2r4c_production_gate_20260730"
PROMPT_MANIFEST_SHA256 = (
    "83f4718b829b955cb000908c2ecbb9052883d14404114cbca4ecd42988659056"
)
PROMPT_MANIFEST_FILE_SHA256 = (
    "261a313c687bb328d3306301c5477705f6bd8f1b5334d8bccd657102ecfdce60"
)
PROMPT_MANIFEST_PATH = (
    "/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/"
    "v2r4c_production_gate_20260730/prompt_batches_manifest.json"
)
PROMPT_INTERSECTIONS = {
    "A_B": 0,
    "A_CANARY": 0,
    "A_EXCLUDED_PRIOR": 0,
    "B_CANARY": 0,
    "B_EXCLUDED_PRIOR": 0,
    "CANARY_EXCLUDED_PRIOR": 0,
}
PROMPT_BATCHES = {
    "A": {
        "path": (
            "/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/"
            "v2r4c_production_gate_20260730/batch_a.parquet"
        ),
        "sha256": (
            "c5f6f208f348b079ee476ecf99c4fad14bba4210ac31c850ed0ce9d801caab61"
        ),
        "bytes": 1_072_820,
        "rows": 1_024,
        "rollout_seed": 1_138_054_401,
        "prompt_set_sha256": (
            "b15170a3799027e0ac37af842ed9915eb78a682d2963820988f46edf5cb96e4f"
        ),
        "epoch0_prompt_order_sha256": (
            "245910389323f01312de5bbf5dfccb5fd435462b278d5cdcaf1c5dcd57db9414"
        ),
    },
    "B": {
        "path": (
            "/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/"
            "v2r4c_production_gate_20260730/batch_b.parquet"
        ),
        "sha256": (
            "230911d2fb7ddc7a331cfac5b3ae3ebd8149270fe138d6beca683742a3a6541f"
        ),
        "bytes": 1_066_692,
        "rows": 1_024,
        "rollout_seed": 893_756_028,
        "prompt_set_sha256": (
            "e9c536e23eb1d829b1afbe1841c4163384663dce0a4e27d4e6db24eb28ce5d40"
        ),
        "epoch0_prompt_order_sha256": (
            "f736b739d7d69dd7b36b2f7a85b5f3de82becb02087de1e818e3bf9c1b4f78f5"
        ),
    },
}
CANARY = {
    "run_name": "v2r4d-runtime-canary-s6000-seed13477620",
    "candidate_step": 6_000,
    "path": (
        "/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/"
        "v2r4c_production_gate_20260730/canary.parquet"
    ),
    "sha256": (
        "8c714172b9f9b90348673b92705f3c4ddbded404a8ced5c22e38be031e14accb"
    ),
    "bytes": 286_673,
    "rows": 256,
    "prompt_set_sha256": (
        "bac5f41fbed4e41ea511203078a9195c613d0eb04a682a7ea076db56158a8663"
    ),
    "epoch0_prompt_order_sha256": (
        "1ae1bdece961311cd3fe57c07d4f4a4eb05826d6a7e193c8d37585347f17b73b"
    ),
    "rollout_seed": 13_477_620,
    "num_rollout": 1,
}

ANALYZER_SHA256 = (
    "00ff347e4accb8927751bf02f4702df7777f7f621db199cc29ffa9dcc25addc5"
)
CORRECTED_ANALYZER_SHA256 = (
    "ad0071df90a50d99802b5998038539131dde8226890673a9798765c89a023933"
)
BASE_ANALYZER_SHA256 = (
    "16c9dfd5cc421b196344b773998629cd707f4fadaa405bdba9efaf806d876e6f"
)
FROZEN_EVIDENCE_SOURCE_SHA256 = (
    "711003060a730c5ba58dfdb1e20e6cd6b07c99a3f3b958e86efd896ae5fef2bb"
)
QUARANTINE_REPORT_SHA256 = (
    "576bf2fb346666f8b9da2d3df5563d9e29e65b60ce1a66c07b74bf6318428947"
)
QUARANTINE_REPORT_FILE_SHA256 = (
    "b800d3f8289cf1e4d2ef1320efb4185ab62761c576f3dd05b863d11c8d694970"
)
INCIDENT_REPORT_SHA256 = (
    "2a2eaeb867d73366b69377e6f965a925eff0c3db8bd8bae881ec4d035e3ec5f0"
)
INCIDENT_REPORT_FILE_SHA256 = (
    "8c12073b996486318c7df5b107669c4cb3d55952e32b2cb22a3ca68a855a3168"
)
PREDECESSOR_INCIDENT_BINDING = {
    "path": "INTERLEAVED_V2R4C_PREFLIGHT_INCIDENT_REPORT.json",
    "report_sha256": INCIDENT_REPORT_SHA256,
    "file_sha256": INCIDENT_REPORT_FILE_SHA256,
    "bytes": 4_508,
    "failed_version": "v2r4c_production_gate_20260730",
    "failed_contract_sha256": (
        "d80e77b2e80149342983a5d37ed90cc3c2b74e058c51a108a950435171198940"
    ),
    "failed_ledger_file_sha256": (
        "27e9a093cba15a31129a00f99049db0b6ca4e554079c17f25552880375342fde"
    ),
    "preflight_call_id": "fc-01KYT9E8019J10C6FST75VNVP8",
    "remote_call_children": 0,
    "gpu_roots_created": 0,
    "outcome_exposure": False,
    "prompt_artifact_reuse_authorized": True,
}
PREDECESSOR_ROOTS = (
    "chess-rl-miles-interleave/v2r4c-runtime-canary-s6000-seed13477620",
    "chess-rl-miles-interleave/v2r4c-gate-w190-s6000-batch-a",
    "chess-rl-miles-interleave/v2r4c-gate-w190-s6000-batch-b",
    "chess-rl-miles-interleave/v2r4c-gate-w190-s8000-batch-a",
    "chess-rl-miles-interleave/v2r4c-gate-w190-s8000-batch-b",
    "chess-rl-miles-interleave/v2r4c-gate-w190-s9920-batch-a",
    "chess-rl-miles-interleave/v2r4c-gate-w190-s9920-batch-b",
)

DATA_VOLUME = frozen_evidence.DATA_VOLUME
CHECKPOINT_VOLUME = frozen_evidence.CHECKPOINT_VOLUME
RESULTS_VOLUME = frozen_evidence.RESULTS_VOLUME
DEFAULT_CONTRACT_PATH = (
    "50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/"
    "v2r4d_production_gate_20260730/runtime_contract.json"
)
DEFAULT_INCIDENT_PATH = (
    "50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/"
    "v2r4d_production_gate_20260730/"
    "v2r4c_preflight_incident_report.json"
)
DEFAULT_IMPORT_PROBE_PATH = (
    "50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/"
    "v2r4d_production_gate_20260730/import_probe_report.json"
)
IMPORT_PROBE_LOGICAL_PATH = f"/data/{DEFAULT_IMPORT_PROBE_PATH}"
DEFAULT_CANARY_LEDGER = "INTERLEAVED_V2R4D_CANARY_LAUNCH_LEDGER.json"
DEFAULT_GRID_LEDGER = "INTERLEAVED_V2R4D_GATE_LAUNCH_LEDGER.json"
DEFAULT_ENDPOINT_LEDGER = frozen_evidence.DEFAULT_ENDPOINT_LEDGER
DEFAULT_P2_LEDGER = frozen_evidence.DEFAULT_P2_LEDGER
DEFAULT_QUARANTINE_REPORT = (
    "INTERLEAVED_V2R4A_TERMINAL_QUARANTINE_REPORT.json"
)
DEFAULT_OUTPUT = "INTERLEAVED_V2R4D_PRODUCTION_GATE_REPORT.json"
RAW_VOLUME_ROOT = PurePosixPath("chess-rl-miles-interleave")

ENDPOINT_EVALUATOR_SHA256 = frozen_evidence.ENDPOINT_EVALUATOR_SHA256
P2_EVALUATOR_SHA256 = frozen_evidence.P2_EVALUATOR_SHA256
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CALL_RE = re.compile(r"fc-[A-Za-z0-9_-]+")

canonical_json = frozen_evidence.canonical_json
sha256_bytes = frozen_evidence.sha256_bytes
sha256_file = frozen_evidence.sha256_file
read_json_bytes = frozen_evidence.read_json_bytes
read_json_file = frozen_evidence.read_json_file
_require = frozen_evidence._require
_require_mapping = frozen_evidence._require_mapping
_require_sha256 = frozen_evidence._require_sha256
_require_call_id = frozen_evidence._require_call_id
_require_self_hash = frozen_evidence._require_self_hash
validate_endpoint_ledger = frozen_evidence.validate_endpoint_ledger
validate_endpoint_result = frozen_evidence.validate_endpoint_result
authenticate_endpoint_artifacts = (
    frozen_evidence.authenticate_endpoint_artifacts
)
validate_p2_ledger = frozen_evidence.validate_p2_ledger
validate_p2_result = frozen_evidence.validate_p2_result
authenticate_p2_success_file = frozen_evidence.authenticate_p2_success_file
ModalReadOnlyBackend = frozen_evidence.ModalReadOnlyBackend
write_immutable_report = frozen_evidence.write_immutable_report


def _expected_cells() -> list[dict[str, Any]]:
    return [
        {
            "candidate_step": step,
            "batch_label": batch,
            "run_name": (
                f"v2r4d-gate-w190-s{step}-batch-{batch.lower()}"
            ),
        }
        for step in CANDIDATE_STEPS
        for batch in BATCH_LABELS
    ]


def _normalized_volume_path(path: Any) -> str:
    candidate = PurePosixPath(str(path))
    _require(".." not in candidate.parts, "volume path contains parent traversal")
    return candidate.as_posix().lstrip("/")


def _checkpoint_relative(path: Any, label: str) -> str:
    candidate = PurePosixPath(str(path))
    _require(candidate.is_absolute(), f"{label} must be absolute")
    try:
        relative = candidate.relative_to("/rl-checkpoints")
    except ValueError as exc:
        raise ValueError(f"{label} is outside /rl-checkpoints") from exc
    _require(".." not in relative.parts, f"{label} contains parent traversal")
    return relative.as_posix()


def _data_relative(path: Any, label: str) -> str:
    candidate = PurePosixPath(str(path))
    _require(candidate.is_absolute(), f"{label} must be absolute")
    try:
        relative = candidate.relative_to("/data")
    except ValueError as exc:
        raise ValueError(f"{label} is outside /data") from exc
    _require(".." not in relative.parts, f"{label} contains parent traversal")
    return relative.as_posix()


def _source_tree_identity(
    root: Path, *, excluded_relatives: Sequence[str] = ()
) -> dict[str, Any]:
    suffixes = {".py", ".pyi", ".toml", ".yaml", ".yml", ".json", ".sh"}
    filenames = {"Dockerfile", "Makefile", "uv.lock", "requirements.txt"}
    excluded_parts = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "wandb",
    }
    excluded = {Path(value).as_posix() for value in excluded_relatives}
    rows: list[str] = []
    total = 0
    for path in sorted(root.resolve().rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root.resolve())
        if (
            relative.as_posix() in excluded
            or any(part in excluded_parts for part in relative.parts)
            or (
                path.suffix.lower() not in suffixes
                and path.name not in filenames
            )
        ):
            continue
        size = path.stat().st_size
        rows.append(
            f"{relative.as_posix()}\t{size}\t{sha256_file(path)}\n"
        )
        total += size
    _require(bool(rows), f"source tree is empty: {root}")
    result: dict[str, Any] = {
        "file_count": len(rows),
        "total_bytes": total,
        "manifest_format": (
            "relative_path<TAB>bytes<TAB>sha256<NEWLINE>"
        ),
        "manifest_sha256": sha256_bytes("".join(rows).encode()),
    }
    if excluded:
        result["excluded_relatives"] = sorted(excluded)
    return result


def validate_local_sources(workspace: Path) -> dict[str, Any]:
    _require(
        ANALYZER_CONTRACT_VERSION == PROMPT_SOURCE_VERSION
        and ANALYZER_REPORT_SCHEMA
        == "interleaved-v2r4c-production-gate-report-v1",
        "loaded frozen v2r4c analyzer identity drifted",
    )
    fixed = {
        "analyzer": (
            workspace / "Eval/v2r4c_gate_analysis.py",
            ANALYZER_SHA256,
        ),
        "corrected_analyzer": (
            workspace / "Eval/v2r4b_gate_analysis.py",
            CORRECTED_ANALYZER_SHA256,
        ),
        "base_analyzer": (
            workspace / "Eval/v2r4_gate_analysis.py",
            BASE_ANALYZER_SHA256,
        ),
        "frozen_endpoint_p2_evidence_helpers": (
            workspace / "Eval/finalize_v2r4_gate.py",
            FROZEN_EVIDENCE_SOURCE_SHA256,
        ),
        "p2_contract": (
            workspace / "chess_reasoning/training/v2r4_p2_sft_eval.py",
            frozen_evidence.P2_CONTRACT_SOURCE_SHA256,
        ),
    }
    result: dict[str, Any] = {}
    for label, (path, expected) in fixed.items():
        _require(path.is_file(), f"missing frozen source: {path}")
        observed = sha256_file(path)
        _require(observed == expected, f"frozen {label} source drifted")
        result[label] = {
            "path": str(path),
            "sha256": observed,
            "bytes": path.stat().st_size,
        }
    plan = workspace / "INTERLEAVED_V2R4D_GATE_AMENDMENT.md"
    _require(plan.is_file(), "v2r4d amendment is missing")
    result["plan"] = {
        "path": str(plan),
        "sha256": sha256_file(plan),
        "bytes": plan.stat().st_size,
    }
    finalizer = workspace / "Eval/finalize_v2r4d_gate.py"
    finalizer_test = workspace / "Eval/tests/test_finalize_v2r4d_gate.py"
    _require(finalizer.is_file(), "v2r4d finalizer source is missing")
    _require(finalizer_test.is_file(), "v2r4d finalizer test source is missing")
    result["finalizer"] = {
        "path": str(finalizer),
        "sha256": sha256_file(finalizer),
        "bytes": finalizer.stat().st_size,
    }
    result["finalizer_test"] = {
        "path": str(finalizer_test),
        "sha256": sha256_file(finalizer_test),
        "bytes": finalizer_test.stat().st_size,
    }
    launcher = (
        workspace
        / "chess-rl-miles/chess_rl_miles/scripts/modal_v2r4d_gate.py"
    )
    _require(launcher.is_file(), "v2r4d launcher source is missing")
    result["launcher"] = {
        "path": str(launcher),
        "sha256": sha256_file(launcher),
        "bytes": launcher.stat().st_size,
    }
    result["source_manifests"] = {
        "chess_rl_miles": _source_tree_identity(
            workspace / "chess-rl-miles",
            excluded_relatives=(
                "chess_rl_miles/v2r4d_contract_binding.py",
            ),
        ),
        "miles": _source_tree_identity(workspace / "miles"),
    }
    return result


def validate_runtime_contract(
    raw: bytes, source_evidence: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(
        CONTRACT_SHA256 != "0" * 64
        and CONTRACT_FILE_SHA256 != "0" * 64,
        "v2r4d contract binding is still a placeholder",
    )
    contract = read_json_bytes(raw, "v2r4d runtime contract")
    contract_sha = _require_self_hash(
        contract, "contract_sha256", "v2r4d runtime contract"
    )
    _require(
        contract_sha == CONTRACT_SHA256
        and sha256_bytes(raw) == CONTRACT_FILE_SHA256,
        "v2r4d runtime contract differs from its binding",
    )
    _require(
        contract.get("schema") == CONTRACT_SCHEMA
        and contract.get("version") == CONTRACT_VERSION
        and contract.get("model_id") == "interleave_47m_qwen3"
        and contract.get("cells") == _expected_cells(),
        "v2r4d contract grid identity drifted",
    )
    candidates = _require_mapping(
        contract.get("candidates"), "contract candidates"
    )
    _require(
        set(candidates) == {str(step) for step in CANDIDATE_STEPS},
        "v2r4d candidate set drifted",
    )
    for step in CANDIDATE_STEPS:
        candidate = _require_mapping(
            candidates[str(step)], f"candidate {step}"
        )
        _require_sha256(
            candidate.get("endpoint_checkpoint_sha256"),
            f"candidate {step} endpoint hash",
        )
        _require_sha256(
            candidate.get("hf_directory_manifest_sha256"),
            f"candidate {step} directory hash",
        )
        _require(
            candidate.get("original_p1_eligible") is (step == 9_920),
            f"candidate {step} eligibility drifted",
        )
    import_probe = _require_mapping(
        contract.get("import_probe"), "contract import probe"
    )
    import_probe_result = _require_mapping(
        import_probe.get("result"), "contract import-probe result"
    )
    _require(
        set(import_probe)
        == {
            "report_path",
            "report_sha256",
            "report_file_sha256",
            "report_bytes",
            "app_id",
            "result",
        }
        and import_probe.get("report_path") == IMPORT_PROBE_LOGICAL_PATH
        and bool(
            _SHA256_RE.fullmatch(
                str(import_probe.get("report_sha256", ""))
            )
        )
        and bool(
            _SHA256_RE.fullmatch(
                str(import_probe.get("report_file_sha256", ""))
            )
        )
        and isinstance(import_probe.get("report_bytes"), int)
        and int(import_probe["report_bytes"]) > 0
        and bool(
            re.fullmatch(
                r"ap-[A-Za-z0-9_-]+",
                str(import_probe.get("app_id", "")),
            )
        )
        and import_probe_result
        == {
            "schema": "interleaved-v2r4d-import-probe-v1",
            "version": CONTRACT_VERSION,
            "launcher_path": "/root/modal_v2r4d_gate.py",
            "launcher_sha256": source_evidence["launcher"]["sha256"],
            "project_path": "/root/chess-rl-miles",
            "package_path": (
                "/root/chess-rl-miles/chess_rl_miles/"
                "scripts/modal_interleave.py"
            ),
            "project_on_sys_path": True,
            "volumes_mounted": False,
            "gpu_requested": False,
        },
        "contract import-probe evidence drifted",
    )
    batches = _require_mapping(
        contract.get("prompt_batches"), "contract prompt batches"
    )
    expected_batches = {
        label: {
            key: value
            for key, value in PROMPT_BATCHES[label].items()
            if key != "bytes"
        }
        for label in BATCH_LABELS
    }
    _require(batches == expected_batches, "contract prompt batches drifted")
    prompt_binding = _require_mapping(
        contract.get("prompt_manifest"), "contract prompt manifest"
    )
    _require(
        prompt_binding
        == {
            "path": PROMPT_MANIFEST_PATH,
            "source_schema": PROMPT_MANIFEST_SCHEMA,
            "source_version": PROMPT_SOURCE_VERSION,
            "manifest_sha256": PROMPT_MANIFEST_SHA256,
            "file_sha256": PROMPT_MANIFEST_FILE_SHA256,
            "reused_after_zero_outcome_preflight_failure": True,
            "intersections": PROMPT_INTERSECTIONS,
        },
        "contract prompt-manifest binding drifted",
    )
    _require(
        contract.get("predecessor_incident")
        == PREDECESSOR_INCIDENT_BINDING,
        "contract predecessor-incident binding drifted",
    )
    expected_canary = {
        key: value for key, value in CANARY.items() if key != "bytes"
    }
    _require(
        contract.get("canary") == expected_canary,
        "contract canary binding drifted",
    )
    endpoints = _require_mapping(
        contract.get("endpoint_evaluators"), "contract endpoint evaluators"
    )
    _require(
        endpoints
        == {
            "pt_b1_b5": ENDPOINT_EVALUATOR_SHA256,
            "p2_sft_at_p1": P2_EVALUATOR_SHA256,
        },
        "contract endpoint evaluator identities drifted",
    )
    plan = _require_mapping(contract.get("plan"), "contract plan")
    _require(
        plan
        == {
            "path": "INTERLEAVED_V2R4D_GATE_AMENDMENT.md",
            "sha256": source_evidence["plan"]["sha256"],
        },
        "contract plan identity drifted",
    )
    analysis = _require_mapping(contract.get("analysis"), "contract analysis")
    _require(
        analysis
        == {
            "path": "Eval/v2r4c_gate_analysis.py",
            "sha256": ANALYZER_SHA256,
            "corrected_dependency": {
                "path": "Eval/v2r4b_gate_analysis.py",
                "sha256": CORRECTED_ANALYZER_SHA256,
            },
            "base_dependency": {
                "path": "Eval/v2r4_gate_analysis.py",
                "sha256": BASE_ANALYZER_SHA256,
            },
        },
        "contract analyzer identity drifted",
    )
    finalizer = _require_mapping(
        contract.get("finalizer"), "contract finalizer"
    )
    _require(
        finalizer
        == {
            "path": "Eval/finalize_v2r4d_gate.py",
            "sha256": source_evidence["finalizer"]["sha256"],
            "reused_dependency": {
                "path": "Eval/finalize_v2r4_gate.py",
                "sha256": FROZEN_EVIDENCE_SOURCE_SHA256,
            },
            "test_source": {
                "path": "Eval/tests/test_finalize_v2r4d_gate.py",
                "sha256": source_evidence["finalizer_test"]["sha256"],
            },
        },
        "contract finalizer identity drifted",
    )
    _require(
        contract.get("sources") == source_evidence["source_manifests"],
        "contract source manifests drifted",
    )
    semantics = _require_mapping(
        contract.get("semantics"), "contract semantics"
    )
    required_semantics = {
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
        "rollout_fault_tolerance": False,
        "router_health_check_interval_seconds": 1e18,
        "strict_sample_outcome_logs": "redacted",
        "positive_attempt_stream": False,
        "miles_log_passrate": False,
        "default_reward_metrics_before_barrier": False,
        "pre_barrier_reward_aggregates": False,
        "analysis_reward_and_protocol_counted_independently": True,
        "fresh_prompt_sets_disjoint_from_quarantined_grid": True,
        "canary_disjoint_from_grid": True,
    }
    for key, expected in required_semantics.items():
        _require(
            semantics.get(key) == expected,
            f"contract semantic {key} drifted",
        )
    quarantine = _require_mapping(
        contract.get("quarantined_v2r4a"), "contract quarantine"
    )
    _require(
        quarantine.get("report_sha256") == QUARANTINE_REPORT_SHA256
        and quarantine.get("report_file_sha256")
        == QUARANTINE_REPORT_FILE_SHA256
        and quarantine.get("disposition")
        == "entire_grid_quarantined_no_authorization"
        and len(quarantine.get("cells", [])) == 6,
        "contract quarantine binding drifted",
    )
    _require(
        contract.get("quarantine_report")
        == {
            "path": DEFAULT_QUARANTINE_REPORT,
            "report_sha256": QUARANTINE_REPORT_SHA256,
            "file_sha256": QUARANTINE_REPORT_FILE_SHA256,
            "bytes": 17_661,
        },
        "contract quarantine-report file binding drifted",
    )
    return contract, {
        "schema": CONTRACT_SCHEMA,
        "version": CONTRACT_VERSION,
        "contract_sha256": contract_sha,
        "file_sha256": sha256_bytes(raw),
        "bytes": len(raw),
        "plan_sha256": plan["sha256"],
        "analysis_sha256": analysis["sha256"],
        "import_probe": import_probe,
        "endpoint_evaluators": endpoints,
    }


def validate_predecessor_incident(
    raw: bytes, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(
        sha256_bytes(raw) == INCIDENT_REPORT_FILE_SHA256,
        "v2r4c incident report file hash drifted",
    )
    incident = read_json_bytes(raw, "v2r4c preflight incident report")
    report_sha = _require_self_hash(
        incident, "report_sha256", "v2r4c preflight incident report"
    )
    function_call = _require_mapping(
        incident.get("function_call"), "incident function call"
    )
    launcher_ledger = _require_mapping(
        incident.get("launcher_ledger"), "incident launcher ledger"
    )
    boundary = _require_mapping(
        incident.get("gpu_and_outcome_boundary"),
        "incident GPU/outcome boundary",
    )
    disposition = _require_mapping(
        incident.get("disposition"), "incident disposition"
    )
    _require(
        report_sha == INCIDENT_REPORT_SHA256
        and incident.get("schema")
        == "interleaved-v2r4c-preflight-incident-report-v1"
        and incident.get("version") == "v2r4c_production_gate_20260730"
        and function_call.get("function_call_id")
        == PREDECESSOR_INCIDENT_BINDING["preflight_call_id"]
        and function_call.get("input_status") == "TERMINATED"
        and function_call.get("call_graph_children") == []
        and launcher_ledger.get("state") == "preflight_failed"
        and launcher_ledger.get("calls") == []
        and launcher_ledger.get("file_sha256")
        == PREDECESSOR_INCIDENT_BINDING["failed_ledger_file_sha256"]
        and boundary.get("remote_call_children") == 0
        and boundary.get("canary_function_call_spawned") is False
        and boundary.get("grid_function_calls_spawned") == 0
        and boundary.get("canary_root_absent") is True
        and boundary.get("grid_roots_absent") is True
        and boundary.get("outcome_exposure") is False
        and incident.get("verified_absent_roots")
        == list(PREDECESSOR_ROOTS)
        and disposition.get("prompt_artifacts_may_be_reused") is True
        and disposition.get("fresh_version_and_names_required") is True
        and contract.get("predecessor_incident")
        == PREDECESSOR_INCIDENT_BINDING,
        "v2r4c predecessor incident evidence drifted",
    )
    return incident, {
        "schema": incident["schema"],
        "version": incident["version"],
        "report_sha256": report_sha,
        "file_sha256": sha256_bytes(raw),
        "bytes": len(raw),
        "function_call_id": function_call["function_call_id"],
        "call_graph_children": [],
        "launcher_calls": [],
        "verified_absent_roots": list(PREDECESSOR_ROOTS),
        "outcome_exposure": False,
        "prompt_artifact_reuse_authorized": True,
    }


def validate_import_probe_report(
    raw: bytes, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _require_mapping(
        contract.get("import_probe"), "contract import probe"
    )
    _require(
        sha256_bytes(raw) == binding.get("report_file_sha256")
        and len(raw) == binding.get("report_bytes"),
        "v2r4d import-probe report file drifted",
    )
    report = read_json_bytes(raw, "v2r4d import-probe report")
    report_sha = _require_self_hash(
        report, "report_sha256", "v2r4d import-probe report"
    )
    _require(
        report_sha == binding.get("report_sha256")
        and report.get("schema")
        == "interleaved-v2r4d-import-probe-report-v1"
        and report.get("app_id") == binding.get("app_id")
        and report.get("result") == binding.get("result"),
        "v2r4d import-probe report evidence drifted",
    )
    return report, {
        "schema": report["schema"],
        "report_sha256": report_sha,
        "file_sha256": sha256_bytes(raw),
        "bytes": len(raw),
        "app_id": report["app_id"],
        "result": report["result"],
    }


def validate_prompt_manifest(
    raw: bytes, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(
        sha256_bytes(raw) == PROMPT_MANIFEST_FILE_SHA256,
        "fresh prompt manifest file hash drifted",
    )
    manifest = read_json_bytes(raw, "fresh prompt manifest")
    manifest_sha = _require_self_hash(
        manifest, "manifest_sha256", "fresh prompt manifest"
    )
    _require(
        manifest_sha == PROMPT_MANIFEST_SHA256
        and manifest.get("schema") == PROMPT_MANIFEST_SCHEMA
        and manifest.get("version") == PROMPT_SOURCE_VERSION
        and manifest.get("status")
        == "frozen_prompt_data_no_independent_launch_authorization"
        and manifest.get("intersections") == PROMPT_INTERSECTIONS,
        "fresh prompt manifest identity drifted",
    )
    batches = _require_mapping(manifest.get("batches"), "prompt batches")
    _require(
        set(batches) == {"A", "B", "CANARY"},
        "fresh prompt batch set drifted",
    )
    sets: dict[str, set[str]] = {}
    for label in ("A", "B", "CANARY"):
        expected = (
            PROMPT_BATCHES[label] if label in PROMPT_BATCHES else CANARY
        )
        value = _require_mapping(batches[label], f"prompt batch {label}")
        for manifest_key, expected_key in (
            ("logical_path", "path"),
            ("file_sha256", "sha256"),
            ("file_bytes", "bytes"),
            ("rows", "rows"),
            ("rollout_seed", "rollout_seed"),
            ("prompt_set_sha256", "prompt_set_sha256"),
            ("epoch0_prompt_order_sha256", "epoch0_prompt_order_sha256"),
        ):
            _require(
                value.get(manifest_key) == expected[expected_key],
                f"prompt batch {label} {manifest_key} drifted",
            )
        ordered = value.get("ordered_prompt_fingerprints")
        permutation = value.get("epoch0_permutation")
        _require(
            isinstance(ordered, list)
            and len(ordered) == expected["rows"]
            and len(set(ordered)) == expected["rows"]
            and all(
                isinstance(item, str)
                and _SHA256_RE.fullmatch(item) is not None
                for item in ordered
            )
            and isinstance(permutation, list)
            and sorted(permutation) == list(range(expected["rows"])),
            f"prompt batch {label} fingerprint inventory drifted",
        )
        epoch_order = [ordered[index] for index in permutation]
        _require(
            sha256_bytes(canonical_json(sorted(ordered)))
            == expected["prompt_set_sha256"]
            and sha256_bytes(canonical_json(epoch_order))
            == expected["epoch0_prompt_order_sha256"],
            f"prompt batch {label} set/order hash drifted",
        )
        quarters = value.get("rollout_quarters")
        expected_quarters = 1 if label == "CANARY" else 4
        _require(
            isinstance(quarters, list)
            and len(quarters) == expected_quarters,
            f"prompt batch {label} quarter count drifted",
        )
        flattened: list[str] = []
        for rollout_id, quarter_value in enumerate(quarters):
            quarter = _require_mapping(
                quarter_value, f"prompt batch {label} quarter"
            )
            prompts = quarter.get("ordered_prompt_fingerprints")
            _require(
                quarter.get("rollout_id") == rollout_id
                and quarter.get("prompt_count") == 256
                and isinstance(prompts, list)
                and len(prompts) == 256
                and quarter.get("prompt_order_sha256")
                == sha256_bytes(canonical_json(prompts)),
                f"prompt batch {label} quarter {rollout_id} drifted",
            )
            flattened.extend(prompts)
        _require(
            flattened == epoch_order,
            f"prompt batch {label} epoch/quarter order drifted",
        )
        sets[label] = set(ordered)
    _require(
        not sets["A"] & sets["B"]
        and not sets["A"] & sets["CANARY"]
        and not sets["B"] & sets["CANARY"],
        "fresh prompt sets overlap",
    )
    _require(
        contract["prompt_manifest"]["manifest_sha256"] == manifest_sha,
        "contract/prompt manifest binding drifted",
    )
    return manifest, {
        "manifest_sha256": manifest_sha,
        "file_sha256": sha256_bytes(raw),
        "bytes": len(raw),
        "intersections": dict(PROMPT_INTERSECTIONS),
        "prompt_set_sha256": {
            label: (
                PROMPT_BATCHES[label]["prompt_set_sha256"]
                if label in PROMPT_BATCHES
                else CANARY["prompt_set_sha256"]
            )
            for label in ("A", "B", "CANARY")
        },
    }


def authenticate_prompt_files(
    *, read_data_file: Callable[[str], bytes]
) -> list[dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    evidence: list[dict[str, Any]] = []
    for label, expected in (
        ("A", PROMPT_BATCHES["A"]),
        ("B", PROMPT_BATCHES["B"]),
        ("CANARY", CANARY),
    ):
        relative = _data_relative(expected["path"], f"{label} prompt parquet")
        raw = read_data_file(relative)
        _require(
            len(raw) == expected["bytes"]
            and sha256_bytes(raw) == expected["sha256"],
            f"{label} prompt parquet bytes drifted",
        )
        rows = pq.ParquetFile(pa.BufferReader(raw)).metadata.num_rows
        _require(rows == expected["rows"], f"{label} parquet rows drifted")
        evidence.append(
            {
                "label": label,
                "volume_path": relative,
                "rows": rows,
                "bytes": len(raw),
                "sha256": sha256_bytes(raw),
            }
        )
    return evidence


def _validate_preflight(
    value: Mapping[str, Any],
    *,
    phase: str,
    contract: Mapping[str, Any],
    contract_file_sha256: str,
) -> None:
    _require(
        set(value)
        == {
            "schema",
            "version",
            "phase",
            "contract_sha256",
            "contract_file_sha256",
            "grid_roots_absent",
            "canary_state",
            "canary_success_sha256",
            "quarantine",
            "predecessor_incident_sha256",
            "import_probe_report_sha256",
            "predecessor_roots_absent",
            "predecessor_root_count",
            "predecessor_results_absent",
            "predecessor_result_count",
            "prompt_manifest_sha256",
            "fresh_prompt_intersections_authenticated",
            "ray_worker_environment",
            "grid_roots",
        },
        f"{phase} preflight fields drifted",
    )
    expected_roots = [
        f"/rl-checkpoints/{RAW_VOLUME_ROOT}/{cell['run_name']}"
        for cell in _expected_cells()
    ]
    expected_ray = {
        "seed_mode": "sample-index",
        "artifact_root": (
            f"/rl-checkpoints/{RAW_VOLUME_ROOT}/v2r4d-ray-env-preflight"
        ),
        "gpu_allocated": False,
    }
    _require(
        value.get("schema") == PREFLIGHT_SCHEMA
        and value.get("version") == CONTRACT_VERSION
        and value.get("phase") == phase
        and value.get("contract_sha256") == contract["contract_sha256"]
        and value.get("contract_file_sha256") == contract_file_sha256
        and value.get("grid_roots_absent") is True
        and value.get("predecessor_incident_sha256")
        == INCIDENT_REPORT_SHA256
        and value.get("import_probe_report_sha256")
        == contract["import_probe"]["report_sha256"]
        and value.get("predecessor_roots_absent") is True
        and value.get("predecessor_root_count") == len(PREDECESSOR_ROOTS)
        and value.get("predecessor_results_absent") is True
        and value.get("predecessor_result_count") == 0
        and value.get("prompt_manifest_sha256")
        == PROMPT_MANIFEST_SHA256
        and value.get("fresh_prompt_intersections_authenticated") is True
        and value.get("ray_worker_environment") == expected_ray
        and value.get("grid_roots") == expected_roots,
        f"{phase} preflight identity drifted",
    )
    if phase == "canary":
        _require(
            value.get("canary_state") == "absent"
            and value.get("canary_success_sha256") is None,
            "canary preflight state drifted",
        )
    else:
        _require(
            value.get("canary_state") == "success_authenticated"
            and _SHA256_RE.fullmatch(
                str(value.get("canary_success_sha256", ""))
            )
            is not None,
            "grid preflight canary state drifted",
        )
    quarantine = _require_mapping(
        value.get("quarantine"), f"{phase} preflight quarantine"
    )
    bound = _require_mapping(
        contract.get("quarantined_v2r4a"), "contract quarantine"
    )
    expected_grid = [
        {
            **dict(cell),
            "success_marker_present": (
                cell["terminal_state"]
                == "success_quarantined_blinding_violation"
            ),
            "root_identity_authenticated": True,
        }
        for cell in bound["cells"]
    ]
    _require(
        quarantine
        == {
            "grid": expected_grid,
            "report_sha256": QUARANTINE_REPORT_SHA256,
            "report_file_sha256": QUARANTINE_REPORT_FILE_SHA256,
            "entire_grid_quarantined": True,
            "authorization_absent": True,
        },
        f"{phase} preflight quarantine proof drifted",
    )


def validate_canary_ledger(
    raw: bytes,
    *,
    contract: Mapping[str, Any],
    contract_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = read_json_bytes(raw, "v2r4d canary ledger")
    ledger_sha = _require_self_hash(
        ledger, "ledger_sha256", "v2r4d canary ledger"
    )
    _require(
        set(ledger)
        == {
            "schema",
            "version",
            "state",
            "contract_sha256",
            "calls",
            "preflight_call_id",
            "preflight",
            "success_sha256",
            "ledger_sha256",
        },
        "canary ledger fields drifted",
    )
    calls = ledger.get("calls")
    preflight = _require_mapping(
        ledger.get("preflight"), "canary ledger preflight"
    )
    _validate_preflight(
        preflight,
        phase="canary",
        contract=contract,
        contract_file_sha256=contract_file_sha256,
    )
    _require(
        ledger.get("schema") == CANARY_LEDGER_SCHEMA
        and ledger.get("version") == CONTRACT_VERSION
        and ledger.get("state") == "canary_succeeded"
        and ledger.get("contract_sha256") == contract["contract_sha256"]
        and isinstance(calls, list)
        and len(calls) == 1,
        "canary ledger envelope drifted",
    )
    record = _require_mapping(calls[0], "canary ledger call")
    _require(
        set(record) == {"run_name", "function_call_id"}
        and record.get("run_name") == CANARY["run_name"],
        "canary ledger call drifted",
    )
    preflight_id = _require_call_id(
        ledger.get("preflight_call_id"), "canary preflight call"
    )
    call_id = _require_call_id(
        record.get("function_call_id"), "canary call"
    )
    _require(preflight_id != call_id, "canary call IDs are not unique")
    success_sha = _require_sha256(
        ledger.get("success_sha256"), "canary success hash"
    )
    return ledger, {
        "ledger_sha256": ledger_sha,
        "file_sha256": sha256_bytes(raw),
        "bytes": len(raw),
        "preflight_call_id": preflight_id,
        "function_call_id": call_id,
        "success_sha256": success_sha,
    }


def validate_grid_ledger(
    raw: bytes,
    *,
    contract: Mapping[str, Any],
    contract_file_sha256: str,
    canary_ledger_file_sha256: str,
    canary_success_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = read_json_bytes(raw, "v2r4d grid ledger")
    ledger_sha = _require_self_hash(
        ledger, "ledger_sha256", "v2r4d grid ledger"
    )
    _require(
        set(ledger)
        == {
            "schema",
            "version",
            "state",
            "contract_sha256",
            "expected_call_count",
            "canary_ledger_file_sha256",
            "calls",
            "preflight_call_id",
            "preflight",
            "ledger_sha256",
        },
        "grid ledger fields drifted",
    )
    preflight = _require_mapping(
        ledger.get("preflight"), "grid ledger preflight"
    )
    _validate_preflight(
        preflight,
        phase="grid",
        contract=contract,
        contract_file_sha256=contract_file_sha256,
    )
    calls = ledger.get("calls")
    _require(
        ledger.get("schema") == GRID_LEDGER_SCHEMA
        and ledger.get("version") == CONTRACT_VERSION
        and ledger.get("state") == "launched_all"
        and ledger.get("expected_call_count") == 6
        and ledger.get("contract_sha256") == contract["contract_sha256"]
        and ledger.get("canary_ledger_file_sha256")
        == canary_ledger_file_sha256
        and preflight.get("canary_success_sha256")
        == canary_success_sha256
        and isinstance(calls, list)
        and len(calls) == 6,
        "grid ledger envelope drifted",
    )
    ids: list[str] = []
    for observed, expected in zip(calls, _expected_cells(), strict=True):
        record = _require_mapping(observed, "grid ledger call")
        _require(
            set(record)
            == {
                "candidate_step",
                "batch_label",
                "run_name",
                "function_call_id",
            }
            and all(record.get(key) == value for key, value in expected.items()),
            "grid ledger cell identity drifted",
        )
        ids.append(_require_call_id(record.get("function_call_id"), "grid call"))
    preflight_id = _require_call_id(
        ledger.get("preflight_call_id"), "grid preflight call"
    )
    _require(
        len(set(ids)) == 6 and preflight_id not in ids,
        "grid FunctionCall IDs are not unique",
    )
    return ledger, {
        "ledger_sha256": ledger_sha,
        "file_sha256": sha256_bytes(raw),
        "bytes": len(raw),
        "preflight_call_id": preflight_id,
        "call_ids": ids,
    }


def _list_exact_files(
    *,
    root: str,
    expected_files: set[str],
    list_entries: Callable[[str], Sequence[Mapping[str, Any]]],
) -> dict[str, int]:
    observed: dict[str, int] = {}
    prefix = root.rstrip("/") + "/"
    for entry_value in list_entries(root):
        entry = _require_mapping(entry_value, f"volume entry {root}")
        path = _normalized_volume_path(entry.get("path"))
        _require(path.startswith(prefix), "recursive listing escaped run root")
        relative = path.removeprefix(prefix)
        _require(bool(relative), "recursive listing contains empty path")
        entry_type = entry.get("type")
        _require(
            entry_type in {"file", "directory"},
            "recursive listing contains a non-regular entry",
        )
        if entry_type == "file":
            _require(relative not in observed, "duplicate recursive file")
            size = entry.get("bytes")
            _require(
                isinstance(size, int) and not isinstance(size, bool) and size >= 0,
                "recursive file size is invalid",
            )
            observed[relative] = size
    _require(
        set(observed) == expected_files,
        f"recursive regular-file allowlist drifted for {root}: "
        f"{sorted(observed)} != {sorted(expected_files)}",
    )
    return observed


def _parse_jsonl(raw: bytes, label: str) -> list[dict[str, Any]]:
    _require(raw.endswith(b"\n"), f"{label} lacks final newline")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        _require(bool(line), f"{label}:{line_number} is blank")
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label}:{line_number} is invalid JSON") from exc
        rows.append(_require_mapping(value, f"{label}:{line_number}"))
    return rows


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


def _validate_provenance(
    *,
    marker: Mapping[str, Any],
    run_name: str,
    intent_filename: str,
    expected_identity_kind: str,
    contract: Mapping[str, Any],
    contract_file_sha256: str,
    read_checkpoint_file: Callable[[str], bytes],
) -> dict[str, Any]:
    provenance = _require_mapping(marker.get("provenance"), "run provenance")
    _require(
        set(provenance)
        == {
            "root_manifest",
            "launch_manifest",
            "identity_sha256",
            "command_sha256",
        },
        "run provenance fields drifted",
    )
    root_path = (
        f"/rl-checkpoints/{RAW_VOLUME_ROOT}/{run_name}/run_provenance.json"
    )
    _require(
        provenance.get("root_manifest") == root_path,
        "root provenance path drifted",
    )
    launch_path = str(provenance.get("launch_manifest", ""))
    _require(
        re.fullmatch(
            rf"/rl-checkpoints/{re.escape(str(RAW_VOLUME_ROOT))}/"
            rf"{re.escape(run_name)}/provenance/launch_[0-9a-f]{{16}}\.json",
            launch_path,
        )
        is not None,
        "launch provenance path drifted",
    )
    root_raw = read_checkpoint_file(_checkpoint_relative(root_path, "root provenance"))
    launch_raw = read_checkpoint_file(
        _checkpoint_relative(launch_path, "launch provenance")
    )
    intent_path = f"{RAW_VOLUME_ROOT}/{run_name}/{intent_filename}"
    intent_raw = read_checkpoint_file(intent_path)
    root_doc = read_json_bytes(root_raw, "root provenance")
    launch_doc = read_json_bytes(launch_raw, "launch provenance")
    intent = read_json_bytes(intent_raw, "run intent")
    _require(
        set(root_doc)
        == {
            "schema_version",
            "created_at",
            "identity_sha256",
            "identity",
            "initial_command_sha256",
            "initial_command",
        }
        and set(launch_doc)
        == {
            "schema_version",
            "created_at",
            "identity_sha256",
            "command_sha256",
            "command",
        },
        "run provenance document fields drifted",
    )
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
    _require(
        isinstance(command, list)
        and all(isinstance(item, str) for item in command),
        "recorded rollout command drifted",
    )
    command_sha = sha256_bytes(
        json.dumps(command, separators=(",", ":")).encode()
    )
    _require(
        root_doc.get("schema_version") == 1
        and launch_doc.get("schema_version") == 1
        and root_doc.get("identity_sha256") == identity_sha
        and root_doc.get("initial_command_sha256") == command_sha
        and launch_doc.get("identity_sha256") == identity_sha
        and launch_doc.get("command_sha256") == command_sha
        and launch_doc.get("command") == command
        and provenance.get("identity_sha256") == identity_sha
        and provenance.get("command_sha256") == command_sha
        and intent.get("command_sha256") == command_sha,
        "run provenance hash binding drifted",
    )
    _require(
        identity.get("kind") == expected_identity_kind
        and identity.get("version") == CONTRACT_VERSION
        and identity.get("contract_sha256") == contract["contract_sha256"]
        and identity.get("contract_file_sha256")
        == contract_file_sha256
        and identity.get("semantics") == contract["semantics"]
        and identity.get("sources") == contract["sources"],
        "run identity contract binding drifted",
    )
    runtime = _require_mapping(identity.get("runtime"), "runtime identity")
    installed = _require_mapping(
        runtime.get("installed_packages"), "installed package identity"
    )
    _require(
        runtime.get("image") == contract["runtime"]["miles_image"]
        and runtime.get("installed_packages_sha256")
        == sha256_bytes(
            json.dumps(
                installed,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        ),
        "runtime identity drifted",
    )
    _require(
        "--debug-rollout-only" in command
        and "--no-log-passrate" in command
        and "--no-use-fault-tolerance" in command
        and "--dynamic-filter" not in command
        and "--log-passrate" not in command,
        "blinded rollout command flags drifted",
    )
    return {
        "launch_relative": _checkpoint_relative(
            launch_path, "launch provenance"
        ).removeprefix(f"{RAW_VOLUME_ROOT}/{run_name}/"),
        "intent": intent,
        "identity": identity,
        "identity_sha256": identity_sha,
        "command_sha256": command_sha,
        "root_manifest_file_sha256": sha256_bytes(root_raw),
        "launch_manifest_file_sha256": sha256_bytes(launch_raw),
        "intent_file_sha256": sha256_bytes(intent_raw),
    }


def validate_canary_result(
    *,
    function_result: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_file_sha256: str,
    prompt_manifest: Mapping[str, Any],
    read_checkpoint_file: Callable[[str], bytes],
    list_checkpoint_entries: Callable[[str], Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    run_name = CANARY["run_name"]
    marker_path = (
        f"{RAW_VOLUME_ROOT}/{run_name}/_V2R4D_CANARY_SUCCESS.json"
    )
    marker_raw = read_checkpoint_file(marker_path)
    marker = read_json_bytes(marker_raw, "canary success marker")
    _require(
        marker == dict(function_result),
        "canary marker differs from FunctionCall return",
    )
    _require(
        set(marker)
        == {
            "schema",
            "version",
            "contract_sha256",
            "contract_file_sha256",
            "run_name",
            "provenance",
            "artifact",
            "outcome_logging_blinded",
            "fault_tolerance_disabled",
            "router_health_checks_suppressed",
            "positive_attempt_artifacts_absent",
            "reward_metrics_inspected",
            "success_sha256",
        },
        "canary success marker fields drifted",
    )
    success_sha = _require_self_hash(
        marker, "success_sha256", "canary success marker"
    )
    _require(
        marker.get("schema") == CANARY_SUCCESS_SCHEMA
        and marker.get("version") == CONTRACT_VERSION
        and marker.get("contract_sha256") == contract["contract_sha256"]
        and marker.get("contract_file_sha256") == contract_file_sha256
        and marker.get("run_name") == run_name
        and marker.get("outcome_logging_blinded") is True
        and marker.get("fault_tolerance_disabled") is True
        and marker.get("router_health_checks_suppressed") is True
        and marker.get("positive_attempt_artifacts_absent") is True
        and marker.get("reward_metrics_inspected") is False,
        "canary success envelope drifted",
    )
    provenance = _validate_provenance(
        marker=marker,
        run_name=run_name,
        intent_filename="_V2R4D_CANARY_INTENT.json",
        expected_identity_kind="chess_rl_miles_v2r4d_runtime_canary",
        contract=contract,
        contract_file_sha256=contract_file_sha256,
        read_checkpoint_file=read_checkpoint_file,
    )
    identity = provenance["identity"]
    intent = provenance["intent"]
    _require(
        set(intent)
        == {
            "schema",
            "version",
            "contract_sha256",
            "run_name",
            "command_sha256",
        }
        and set(identity)
        == {
            "kind",
            "version",
            "contract_sha256",
            "contract_file_sha256",
            "candidate",
            "canary",
            "semantics",
            "sources",
            "runtime",
        }
        and intent.get("schema") == "interleaved-v2r4d-canary-intent-v1"
        and intent.get("version") == CONTRACT_VERSION
        and intent.get("contract_sha256") == contract["contract_sha256"]
        and intent.get("run_name") == run_name
        and identity.get("canary") == contract["canary"],
        "canary provenance data binding drifted",
    )
    candidate = _require_mapping(
        identity.get("candidate"), "canary candidate"
    )
    expected_candidate = contract["candidates"]["6000"]
    _require(
        candidate.get("step") == 6_000
        and all(
            candidate.get(key) == value
            for key, value in expected_candidate.items()
        )
        and _require_mapping(
            candidate.get("directory_identity"),
            "canary checkpoint directory",
        ).get("manifest_sha256")
        == expected_candidate["hf_directory_manifest_sha256"],
        "canary checkpoint provenance drifted",
    )
    expected_files = {
        "_V2R4D_CANARY_INTENT.json",
        "_V2R4D_CANARY_SUCCESS.json",
        "run_provenance.json",
        provenance["launch_relative"],
        "rollouts/training/rollout_0.jsonl",
    }
    sizes = _list_exact_files(
        root=f"{RAW_VOLUME_ROOT}/{run_name}",
        expected_files=expected_files,
        list_entries=list_checkpoint_entries,
    )
    raw_path = f"{RAW_VOLUME_ROOT}/{run_name}/rollouts/training/rollout_0.jsonl"
    raw = read_checkpoint_file(raw_path)
    _require(
        sizes["rollouts/training/rollout_0.jsonl"] == len(raw),
        "canary listing/file size drifted",
    )
    rows = _parse_jsonl(raw, "canary rollout")
    _require(len(rows) == 2_048, "canary rollout row count drifted")
    fingerprints: list[str] = []
    seed = int(CANARY["rollout_seed"])
    for group_index in range(256):
        siblings = rows[group_index * 8 : (group_index + 1) * 8]
        group_fingerprints = {_prompt_fingerprint(row) for row in siblings}
        _require(
            len(group_fingerprints) == 1,
            "canary sibling prompt identity drifted",
        )
        fingerprints.append(next(iter(group_fingerprints)))
        for sibling_index, row in enumerate(siblings):
            sample_index = group_index * 8 + sibling_index
            metadata = _require_mapping(
                row.get("metadata"), "canary row metadata"
            )
            _require(
                row.get("rollout_id") == 0
                and row.get("group_index") == group_index
                and row.get("sample_index") == sample_index
                and row.get("sampling_seed_sibling_index") == sibling_index
                and row.get("sampling_seed") == seed + sample_index
                and row.get("status") in {"completed", "truncated"}
                and metadata.get("sampling_seed_sibling_index")
                == sibling_index
                and metadata.get("sampling_seed") == seed + sample_index
                and metadata.get("sampling_seed_mode") == "sample-index",
                "canary exact shape/seed drifted",
            )
    expected_ordered = prompt_manifest["batches"]["CANARY"][
        "ordered_prompt_fingerprints"
    ]
    permutation = prompt_manifest["batches"]["CANARY"][
        "epoch0_permutation"
    ]
    expected_epoch = [expected_ordered[index] for index in permutation]
    artifact = _require_mapping(marker.get("artifact"), "canary artifact")
    _require(
        fingerprints == expected_epoch
        and artifact
        == {
            "path": (
                f"/rl-checkpoints/{RAW_VOLUME_ROOT}/{run_name}/"
                "rollouts/training/rollout_0.jsonl"
            ),
            "rows": 2_048,
            "sha256": sha256_bytes(raw),
            "prompt_set_sha256": CANARY["prompt_set_sha256"],
            "epoch0_prompt_order_sha256": (
                CANARY["epoch0_prompt_order_sha256"]
            ),
            "shape_authenticated": True,
            "reward_metrics_inspected": False,
        },
        "canary raw artifact binding drifted",
    )
    return {
        "function_result_authenticated": True,
        "success_sha256": success_sha,
        "success_marker": {
            "volume_path": marker_path,
            "sha256": sha256_bytes(marker_raw),
            "bytes": len(marker_raw),
        },
        "artifact": {
            "volume_path": raw_path,
            "sha256": sha256_bytes(raw),
            "bytes": len(raw),
            "rows": len(rows),
            "shape_authenticated_without_reward_or_output_access": True,
        },
        "provenance": {
            key: value
            for key, value in provenance.items()
            if key not in {"identity", "intent", "launch_relative"}
        },
        "recursive_regular_file_allowlist_authenticated": True,
    }


def validate_grid_cell(
    *,
    record: Mapping[str, Any],
    function_result: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_file_sha256: str,
    prompt_manifest: Mapping[str, Any],
    read_checkpoint_file: Callable[[str], bytes],
    list_checkpoint_entries: Callable[[str], Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    step = int(record["candidate_step"])
    batch = str(record["batch_label"])
    run_name = str(record["run_name"])
    marker_path = (
        f"{RAW_VOLUME_ROOT}/{run_name}/_V2R4D_GATE_SUCCESS.json"
    )
    marker_raw = read_checkpoint_file(marker_path)
    marker = read_json_bytes(marker_raw, f"grid marker {run_name}")
    _require(
        marker == dict(function_result),
        "grid marker differs from FunctionCall return",
    )
    _require(
        set(marker)
        == {
            "schema",
            "version",
            "contract_sha256",
            "contract_file_sha256",
            "run_name",
            "candidate_step",
            "batch_label",
            "provenance",
            "prompt_batch_sha256",
            "prompt_set_sha256",
            "rollout_seed",
            "artifact_records",
            "shape_authenticated",
            "outcome_logging_blinded",
            "positive_attempt_artifacts_absent",
            "reward_metrics_inspected",
            "success_sha256",
        },
        "grid success marker fields drifted",
    )
    success_sha = _require_self_hash(
        marker, "success_sha256", f"grid marker {run_name}"
    )
    expected_batch = contract["prompt_batches"][batch]
    _require(
        marker.get("schema") == GRID_SUCCESS_SCHEMA
        and marker.get("version") == CONTRACT_VERSION
        and marker.get("contract_sha256") == contract["contract_sha256"]
        and marker.get("contract_file_sha256") == contract_file_sha256
        and marker.get("run_name") == run_name
        and marker.get("candidate_step") == step
        and marker.get("batch_label") == batch
        and marker.get("prompt_batch_sha256") == expected_batch["sha256"]
        and marker.get("prompt_set_sha256")
        == expected_batch["prompt_set_sha256"]
        and marker.get("rollout_seed") == expected_batch["rollout_seed"]
        and marker.get("shape_authenticated") is True
        and marker.get("outcome_logging_blinded") is True
        and marker.get("positive_attempt_artifacts_absent") is True
        and marker.get("reward_metrics_inspected") is False,
        "grid success envelope drifted",
    )
    provenance = _validate_provenance(
        marker=marker,
        run_name=run_name,
        intent_filename="_V2R4D_GATE_INTENT.json",
        expected_identity_kind=(
            "chess_rl_miles_v2r4d_production_gate_rollout"
        ),
        contract=contract,
        contract_file_sha256=contract_file_sha256,
        read_checkpoint_file=read_checkpoint_file,
    )
    identity = provenance["identity"]
    intent = provenance["intent"]
    _require(
        set(intent)
        == {
            "schema",
            "version",
            "contract_sha256",
            "contract_file_sha256",
            "candidate_step",
            "batch_label",
            "run_name",
            "command_sha256",
        }
        and set(identity)
        == {
            "kind",
            "version",
            "contract_sha256",
            "contract_file_sha256",
            "contract_path",
            "authorized_cell",
            "candidate",
            "prompt_batch",
            "semantics",
            "sources",
            "runtime",
        }
        and intent.get("schema")
        == "interleaved-v2r4d-gate-cell-intent-v1"
        and intent.get("version") == CONTRACT_VERSION
        and intent.get("contract_sha256") == contract["contract_sha256"]
        and intent.get("contract_file_sha256") == contract_file_sha256
        and intent.get("candidate_step") == step
        and intent.get("batch_label") == batch
        and intent.get("run_name") == run_name
        and identity.get("authorized_cell")
        == {
            "candidate_step": step,
            "batch_label": batch,
            "run_name": run_name,
        },
        "grid authorized-cell provenance drifted",
    )
    candidate = _require_mapping(identity.get("candidate"), "grid candidate")
    expected_candidate = contract["candidates"][str(step)]
    _require(
        candidate.get("step") == step
        and all(
            candidate.get(key) == value
            for key, value in expected_candidate.items()
        )
        and _require_mapping(
            candidate.get("directory_identity"),
            "grid checkpoint directory",
        ).get("manifest_sha256")
        == expected_candidate["hf_directory_manifest_sha256"],
        "grid checkpoint provenance drifted",
    )
    prompt = _require_mapping(
        identity.get("prompt_batch"), "grid prompt provenance"
    )
    _require(
        prompt.get("label") == batch
        and all(
            prompt.get(key) == value for key, value in expected_batch.items()
        )
        and prompt.get("manifest_sha256") == PROMPT_MANIFEST_SHA256
        and prompt.get("manifest_file_sha256")
        == PROMPT_MANIFEST_FILE_SHA256,
        "grid prompt provenance drifted",
    )
    expected_files = {
        "_V2R4D_GATE_INTENT.json",
        "_V2R4D_GATE_SUCCESS.json",
        "run_provenance.json",
        provenance["launch_relative"],
        *{
            f"rollouts/training/rollout_{rollout_id}.jsonl"
            for rollout_id in range(4)
        },
    }
    sizes = _list_exact_files(
        root=f"{RAW_VOLUME_ROOT}/{run_name}",
        expected_files=expected_files,
        list_entries=list_checkpoint_entries,
    )
    artifact_values = marker.get("artifact_records")
    _require(
        isinstance(artifact_values, list) and len(artifact_values) == 4,
        "grid artifact record count drifted",
    )
    quarters = prompt_manifest["batches"][batch]["rollout_quarters"]
    all_rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for rollout_id, (artifact_value, quarter) in enumerate(
        zip(artifact_values, quarters, strict=True)
    ):
        artifact = _require_mapping(
            artifact_value, f"grid artifact {run_name}/{rollout_id}"
        )
        _require(
            set(artifact)
            == {
                "rollout_id",
                "path",
                "rows",
                "sha256",
                "prompt_order_sha256",
            },
            "pre-barrier grid artifact record fields drifted",
        )
        relative = (
            f"{RAW_VOLUME_ROOT}/{run_name}/rollouts/training/"
            f"rollout_{rollout_id}.jsonl"
        )
        absolute = f"/rl-checkpoints/{relative}"
        raw = read_checkpoint_file(relative)
        _require(
            artifact.get("rollout_id") == rollout_id
            and artifact.get("path") == absolute
            and artifact.get("rows") == 2_048
            and artifact.get("sha256") == sha256_bytes(raw)
            and sizes[f"rollouts/training/rollout_{rollout_id}.jsonl"]
            == len(raw),
            f"grid artifact {run_name}/{rollout_id} byte identity drifted",
        )
        rows = _parse_jsonl(raw, f"{run_name}/rollout_{rollout_id}")
        _require(len(rows) == 2_048, "grid rollout row count drifted")
        prompts: list[str] = []
        for local_group in range(256):
            group_index = rollout_id * 256 + local_group
            siblings = rows[local_group * 8 : (local_group + 1) * 8]
            fingerprints = {_prompt_fingerprint(row) for row in siblings}
            _require(
                len(fingerprints) == 1,
                "grid sibling prompt identity drifted",
            )
            prompts.append(next(iter(fingerprints)))
            for sibling_index, row in enumerate(siblings):
                sample_index = group_index * 8 + sibling_index
                _require(
                    row.get("rollout_id") == rollout_id
                    and row.get("group_index") == group_index
                    and row.get("sample_index") == sample_index,
                    "grid row rollout/group/sample identity drifted",
                )
        _require(
            prompts == quarter["ordered_prompt_fingerprints"]
            and artifact.get("prompt_order_sha256")
            == quarter["prompt_order_sha256"]
            == sha256_bytes(canonical_json(prompts)),
            "grid prompt order drifted",
        )
        all_rows.extend(rows)
        artifacts.append(
            {
                "rollout_id": rollout_id,
                "volume_path": relative,
                "rows": len(rows),
                "bytes": len(raw),
                "sha256": sha256_bytes(raw),
                "prompt_order_sha256": artifact["prompt_order_sha256"],
            }
        )
    cell = audit_cell(
        all_rows,
        candidate_step=step,
        batch_label=batch,
        rollout_seed=int(expected_batch["rollout_seed"]),
    )
    _require(
        cell["prompt_set_sha256"] == expected_batch["prompt_set_sha256"]
        and cell["prompt_order_sha256"]
        == expected_batch["epoch0_prompt_order_sha256"],
        "analyzed grid prompt identity drifted",
    )
    return cell, {
        "candidate_step": step,
        "batch_label": batch,
        "run_name": run_name,
        "function_call_id": record["function_call_id"],
        "success_sha256": success_sha,
        "success_marker": {
            "volume_path": marker_path,
            "sha256": sha256_bytes(marker_raw),
            "bytes": len(marker_raw),
        },
        "provenance": {
            key: value
            for key, value in provenance.items()
            if key not in {"identity", "intent", "launch_relative"}
        },
        "artifacts": artifacts,
        "recursive_regular_file_allowlist_authenticated": True,
    }


def validate_quarantine_report(
    raw: bytes, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(
        sha256_bytes(raw) == QUARANTINE_REPORT_FILE_SHA256,
        "v2r4a quarantine report file hash drifted",
    )
    report = read_json_bytes(raw, "v2r4a quarantine report")
    report_sha = _require_self_hash(
        report, "report_sha256", "v2r4a quarantine report"
    )
    _require(
        report_sha == QUARANTINE_REPORT_SHA256
        and report.get("schema")
        == "interleaved-v2r4a-terminal-quarantine-report-v1"
        and report.get("status") == "terminal_quarantined_failed_closed"
        and report["terminal_barrier"]["all_six_terminal"] is True
        and report["terminal_barrier"]["ordered_status_vector"]
        == "S,F,F,S,F,S"
        and report["authorization"]["v2r4a_production_gate_pass"] is False,
        "v2r4a quarantine report identity drifted",
    )
    binding = contract["quarantined_v2r4a"]
    _require(
        binding["report_sha256"] == report_sha
        and binding["report_file_sha256"] == sha256_bytes(raw)
        and binding["ledger_sha256"]
        == report["frozen_bindings"]["launch_ledger"]["ledger_sha256"]
        and binding["ledger_file_sha256"]
        == report["frozen_bindings"]["launch_ledger"]["file_sha256"]
        and binding["contract_sha256"]
        == report["frozen_bindings"]["runtime_contract"]["contract_sha256"],
        "contract/quarantine report binding drifted",
    )
    return report, {
        "report_sha256": report_sha,
        "file_sha256": sha256_bytes(raw),
        "bytes": len(raw),
        "terminal_vector": "S,F,F,S,F,S",
    }


def authenticate_quarantine_roots(
    *,
    report: Mapping[str, Any],
    contract: Mapping[str, Any],
    list_checkpoint_entries: Callable[[str], Sequence[Mapping[str, Any]]],
    read_checkpoint_file: Callable[[str], bytes],
) -> list[dict[str, Any]]:
    report_cells = report["terminal_volume_audit"]["cells"]
    contract_cells = contract["quarantined_v2r4a"]["cells"]
    evidence: list[dict[str, Any]] = []
    for report_cell, contract_cell in zip(
        report_cells, contract_cells, strict=True
    ):
        run_name = contract_cell["run_name"]
        root = f"{RAW_VOLUME_ROOT}/{run_name}"
        entries = list_checkpoint_entries(root)
        files: dict[str, int] = {}
        prefix = root + "/"
        for entry_value in entries:
            entry = _require_mapping(entry_value, "quarantine root entry")
            path = _normalized_volume_path(entry.get("path"))
            _require(
                path.startswith(prefix),
                "quarantine recursive listing escaped root",
            )
            if entry.get("type") == "file":
                relative = path.removeprefix(prefix)
                _require(
                    relative not in files,
                    "duplicate quarantine root file",
                )
                files[relative] = int(entry.get("bytes"))
            else:
                _require(
                    entry.get("type") == "directory",
                    "quarantine root contains non-regular entry",
                )
        records: list[dict[str, Any]] = []
        for relative in sorted(files):
            volume_path = f"{root}/{relative}"
            raw = read_checkpoint_file(volume_path)
            _require(
                len(raw) == files[relative],
                "quarantine listing/file size drifted",
            )
            records.append(
                {
                    "path": f"/{volume_path}",
                    "bytes": len(raw),
                    "sha256": sha256_bytes(raw),
                }
            )
        identity = {
            "root_file_count": len(records),
            "root_total_bytes": sum(item["bytes"] for item in records),
            "root_inventory_sha256": sha256_bytes(canonical_json(records)),
        }
        expected = {
            key: contract_cell[key]
            for key in (
                "root_file_count",
                "root_total_bytes",
                "root_inventory_sha256",
            )
        }
        _require(
            identity == expected
            and all(report_cell[key] == value for key, value in expected.items())
            and report_cell["root"] == f"/{root}",
            f"quarantine root identity drifted: {run_name}",
        )
        marker_name = "_V2R4_GATE_SUCCESS.json"
        marker_record = next(
            (item for item in records if item["path"].endswith("/" + marker_name)),
            None,
        )
        expected_marker = contract_cell["success_marker_sha256"]
        if expected_marker is None:
            _require(
                marker_record is None
                and report_cell["success_marker"] is None,
                "failed quarantine root contains a success marker",
            )
        else:
            _require(
                marker_record is not None
                and marker_record["sha256"] == expected_marker
                and report_cell["success_marker"]["sha256"] == expected_marker,
                "successful quarantine marker byte hash drifted",
            )
            marker_raw = read_checkpoint_file(f"{root}/{marker_name}")
            marker = read_json_bytes(marker_raw, "quarantined success marker")
            _require_self_hash(
                marker, "success_sha256", "quarantined success marker"
            )
        evidence.append(
            {
                **dict(contract_cell),
                **identity,
                "root_identity_authenticated": True,
                "success_marker_authenticated": expected_marker is not None,
            }
        )
    return evidence


def build_final_report(
    *,
    source_evidence: Mapping[str, Any],
    finalizer_source: Mapping[str, Any],
    contract_evidence: Mapping[str, Any],
    import_probe_evidence: Mapping[str, Any],
    predecessor_incident_evidence: Mapping[str, Any],
    prompt_evidence: Mapping[str, Any],
    prompt_files: Sequence[Mapping[str, Any]],
    canary_ledger_evidence: Mapping[str, Any],
    canary_evidence: Mapping[str, Any],
    grid_ledger_evidence: Mapping[str, Any],
    grid_cell_evidence: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    endpoint_ledger_evidence: Mapping[str, Any],
    endpoint_evidence: Sequence[Mapping[str, Any]],
    p2_ledger_evidence: Mapping[str, Any],
    p2_evidence: Sequence[Mapping[str, Any]],
    quarantine_evidence: Mapping[str, Any],
    quarantine_roots: Sequence[Mapping[str, Any]],
    endpoints: Mapping[int, Mapping[str, Mapping[str, Any]]],
    barrier_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    analysis = analyze_grid(cells, endpoints)
    _require(
        analysis.get("schema") == ANALYZER_REPORT_SCHEMA
        and analysis.get("contract_version") == PROMPT_SOURCE_VERSION,
        "frozen v2r4c analyzer emitted the wrong report identity",
    )
    core: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "schema_version": 1,
        "contract_version": CONTRACT_VERSION,
        "status": "complete",
        "inputs_authenticated": True,
        "outcome_barrier": dict(barrier_evidence),
        "sources": {
            **dict(source_evidence),
            "finalizer": dict(finalizer_source),
        },
        "runtime_contract": dict(contract_evidence),
        "remote_import_probe": dict(import_probe_evidence),
        "superseded_v2r4c_preflight_incident": dict(
            predecessor_incident_evidence
        ),
        "fresh_prompt_data": {
            "manifest": dict(prompt_evidence),
            "parquets": [dict(value) for value in prompt_files],
        },
        "runtime_canary": {
            "launch_ledger": dict(canary_ledger_evidence),
            "result": dict(canary_evidence),
        },
        "rollout_gate": {
            "launch_ledger": dict(grid_ledger_evidence),
            "cells": [dict(value) for value in grid_cell_evidence],
        },
        "endpoint_evaluations": {
            "launch_ledger": dict(endpoint_ledger_evidence),
            "results": [dict(value) for value in endpoint_evidence],
        },
        "p2_sft_evaluations": {
            "launch_ledger": dict(p2_ledger_evidence),
            "results": [dict(value) for value in p2_evidence],
        },
        "superseded_v2r4a_quarantine": {
            "report": dict(quarantine_evidence),
            "roots": [dict(value) for value in quarantine_roots],
        },
        "analysis": analysis,
        "analysis_report_sha256": analysis["report_sha256"],
    }
    return {**core, "report_sha256": content_hash(core, "report_sha256")}


def collect_and_finalize(
    *,
    workspace: Path,
    canary_ledger_path: Path,
    grid_ledger_path: Path,
    endpoint_ledger_path: Path,
    p2_ledger_path: Path,
    quarantine_report_path: Path,
    backend: ModalReadOnlyBackend,
    data_volume: str = DATA_VOLUME,
    checkpoint_volume: str = CHECKPOINT_VOLUME,
    results_volume: str = RESULTS_VOLUME,
    contract_path: str = DEFAULT_CONTRACT_PATH,
    incident_path: str = DEFAULT_INCIDENT_PATH,
    import_probe_path: str = DEFAULT_IMPORT_PROBE_PATH,
) -> dict[str, Any]:
    source_evidence = validate_local_sources(workspace)
    finalizer_path = Path(__file__).resolve()
    finalizer_source = {
        "path": str(finalizer_path),
        "sha256": sha256_file(finalizer_path),
        "bytes": finalizer_path.stat().st_size,
    }
    contract_raw = backend.read_volume_file(data_volume, contract_path)
    contract, contract_evidence = validate_runtime_contract(
        contract_raw, source_evidence
    )
    incident_raw = backend.read_volume_file(data_volume, incident_path)
    _, predecessor_incident_evidence = validate_predecessor_incident(
        incident_raw, contract
    )
    import_probe_raw = backend.read_volume_file(
        data_volume, import_probe_path
    )
    _, import_probe_evidence = validate_import_probe_report(
        import_probe_raw, contract
    )
    prompt_raw = backend.read_volume_file(
        data_volume, _data_relative(PROMPT_MANIFEST_PATH, "prompt manifest")
    )
    prompt_manifest, prompt_evidence = validate_prompt_manifest(
        prompt_raw, contract
    )
    prompt_files = authenticate_prompt_files(
        read_data_file=lambda path: backend.read_volume_file(data_volume, path)
    )
    canary_raw, _ = read_json_file(
        canary_ledger_path, "v2r4d canary ledger"
    )
    canary_ledger, canary_ledger_evidence = validate_canary_ledger(
        canary_raw,
        contract=contract,
        contract_file_sha256=contract_evidence["file_sha256"],
    )
    grid_raw, _ = read_json_file(grid_ledger_path, "v2r4d grid ledger")
    grid_ledger, grid_ledger_evidence = validate_grid_ledger(
        grid_raw,
        contract=contract,
        contract_file_sha256=contract_evidence["file_sha256"],
        canary_ledger_file_sha256=sha256_bytes(canary_raw),
        canary_success_sha256=canary_ledger_evidence["success_sha256"],
    )
    endpoint_raw, _ = read_json_file(
        endpoint_ledger_path, "endpoint ledger"
    )
    endpoint_ledger, endpoint_ledger_evidence = validate_endpoint_ledger(
        endpoint_raw, contract
    )
    p2_raw, _ = read_json_file(p2_ledger_path, "P2 ledger")
    p2_ledger, p2_ledger_evidence = validate_p2_ledger(p2_raw, contract)
    quarantine_raw, _ = read_json_file(
        quarantine_report_path, "v2r4a quarantine report"
    )
    quarantine_report, quarantine_evidence = validate_quarantine_report(
        quarantine_raw, contract
    )

    # Shape-only preflight returns and canary result are authenticated first.
    canary_preflight_result = backend.function_result(
        canary_ledger_evidence["preflight_call_id"]
    )
    _require(
        canary_preflight_result == canary_ledger["preflight"],
        "canary preflight FunctionCall differs from ledger",
    )
    canary_result = backend.function_result(
        canary_ledger_evidence["function_call_id"]
    )
    grid_preflight_result = backend.function_result(
        grid_ledger_evidence["preflight_call_id"]
    )
    _require(
        grid_preflight_result == grid_ledger["preflight"],
        "grid preflight FunctionCall differs from ledger",
    )

    # Outcome barrier: do not read any grid JSONL above this line.  A failed,
    # pending, missing, or non-mapping return raises here, before raw outcomes.
    grid_results: list[dict[str, Any]] = []
    for call_id in grid_ledger_evidence["call_ids"]:
        grid_results.append(backend.function_result(call_id))
    all_call_ids = [
        canary_ledger_evidence["preflight_call_id"],
        canary_ledger_evidence["function_call_id"],
        grid_ledger_evidence["preflight_call_id"],
        *grid_ledger_evidence["call_ids"],
    ]
    _require(
        len(set(all_call_ids)) == len(all_call_ids),
        "v2r4d preflight/canary/grid FunctionCall IDs overlap",
    )
    barrier_evidence = {
        "state": "all_six_grid_function_calls_successful_terminal",
        "all_six_terminal_before_first_grid_jsonl_read": True,
        "successful_terminal_call_count": 6,
        "ordered_grid_function_call_ids": list(
            grid_ledger_evidence["call_ids"]
        ),
        "partial_estimator_allowed": False,
    }

    checkpoint_reader = lambda path: backend.read_volume_file(
        checkpoint_volume, path
    )
    checkpoint_lister = lambda root: backend.list_volume_entries(
        checkpoint_volume, root
    )
    results_reader = lambda path: backend.read_volume_file(results_volume, path)
    canary_evidence = validate_canary_result(
        function_result=canary_result,
        contract=contract,
        contract_file_sha256=contract_evidence["file_sha256"],
        prompt_manifest=prompt_manifest,
        read_checkpoint_file=checkpoint_reader,
        list_checkpoint_entries=checkpoint_lister,
    )
    _require(
        canary_evidence["success_sha256"]
        == canary_ledger_evidence["success_sha256"],
        "canary ledger/marker success hash drifted",
    )
    quarantine_roots = authenticate_quarantine_roots(
        report=quarantine_report,
        contract=contract,
        list_checkpoint_entries=checkpoint_lister,
        read_checkpoint_file=checkpoint_reader,
    )
    cells: list[dict[str, Any]] = []
    grid_cell_evidence: list[dict[str, Any]] = []
    for record, function_result in zip(
        grid_ledger["calls"], grid_results, strict=True
    ):
        cell, evidence = validate_grid_cell(
            record=record,
            function_result=function_result,
            contract=contract,
            contract_file_sha256=contract_evidence["file_sha256"],
            prompt_manifest=prompt_manifest,
            read_checkpoint_file=checkpoint_reader,
            list_checkpoint_entries=checkpoint_lister,
        )
        cells.append(cell)
        grid_cell_evidence.append(evidence)

    endpoints: dict[int, dict[str, dict[str, Any]]] = {
        step: {} for step in CANDIDATE_STEPS
    }
    endpoint_evidence: list[dict[str, Any]] = []
    for record in endpoint_ledger["calls"]:
        function_result = backend.function_result(
            str(record["function_call_id"])
        )
        component, normalized, evidence = validate_endpoint_result(
            record, function_result, contract
        )
        evidence["persisted_artifacts"] = authenticate_endpoint_artifacts(
            function_result, read_results_file=results_reader
        )
        endpoints[int(record["step"])][component] = normalized
        endpoint_evidence.append(evidence)
    p2_evidence: list[dict[str, Any]] = []
    for record in p2_ledger["calls"]:
        function_result = backend.function_result(
            str(record["function_call_id"])
        )
        normalized, evidence = validate_p2_result(
            record, function_result, contract
        )
        evidence["persisted_success_file"] = authenticate_p2_success_file(
            record, function_result, read_results_file=results_reader
        )
        step = int(record["kwargs"]["candidate_step"])
        endpoints[step]["p2_sft"] = normalized
        p2_evidence.append(evidence)
    return build_final_report(
        source_evidence=source_evidence,
        finalizer_source=finalizer_source,
        contract_evidence=contract_evidence,
        import_probe_evidence=import_probe_evidence,
        predecessor_incident_evidence=predecessor_incident_evidence,
        prompt_evidence=prompt_evidence,
        prompt_files=prompt_files,
        canary_ledger_evidence=canary_ledger_evidence,
        canary_evidence=canary_evidence,
        grid_ledger_evidence=grid_ledger_evidence,
        grid_cell_evidence=grid_cell_evidence,
        cells=cells,
        endpoint_ledger_evidence=endpoint_ledger_evidence,
        endpoint_evidence=endpoint_evidence,
        p2_ledger_evidence=p2_ledger_evidence,
        p2_evidence=p2_evidence,
        quarantine_evidence=quarantine_evidence,
        quarantine_roots=quarantine_roots,
        endpoints=endpoints,
        barrier_evidence=barrier_evidence,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", type=Path, default=Path(__file__).parents[1]
    )
    parser.add_argument(
        "--canary-ledger", type=Path, default=Path(DEFAULT_CANARY_LEDGER)
    )
    parser.add_argument(
        "--grid-ledger", type=Path, default=Path(DEFAULT_GRID_LEDGER)
    )
    parser.add_argument(
        "--endpoint-ledger", type=Path, default=Path(DEFAULT_ENDPOINT_LEDGER)
    )
    parser.add_argument(
        "--p2-ledger", type=Path, default=Path(DEFAULT_P2_LEDGER)
    )
    parser.add_argument(
        "--quarantine-report",
        type=Path,
        default=Path(DEFAULT_QUARANTINE_REPORT),
    )
    parser.add_argument("--contract-path", default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--incident-path", default=DEFAULT_INCIDENT_PATH)
    parser.add_argument(
        "--import-probe-path", default=DEFAULT_IMPORT_PROBE_PATH
    )
    parser.add_argument("--data-volume", default=DATA_VOLUME)
    parser.add_argument("--checkpoint-volume", default=CHECKPOINT_VOLUME)
    parser.add_argument("--results-volume", default=RESULTS_VOLUME)
    parser.add_argument("--modal-environment", default=None)
    parser.add_argument("--call-timeout", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace = args.workspace.expanduser().resolve()

    def local(value: Path) -> Path:
        return value if value.is_absolute() else workspace / value

    backend = ModalReadOnlyBackend(
        environment_name=args.modal_environment,
        call_timeout=args.call_timeout,
    )
    report = collect_and_finalize(
        workspace=workspace,
        canary_ledger_path=local(args.canary_ledger),
        grid_ledger_path=local(args.grid_ledger),
        endpoint_ledger_path=local(args.endpoint_ledger),
        p2_ledger_path=local(args.p2_ledger),
        quarantine_report_path=local(args.quarantine_report),
        backend=backend,
        data_volume=args.data_volume,
        checkpoint_volume=args.checkpoint_volume,
        results_volume=args.results_volume,
        contract_path=args.contract_path,
        incident_path=args.incident_path,
        import_probe_path=args.import_probe_path,
    )
    summary: dict[str, Any] = {
        "state": "authenticated",
        "report_sha256": report["report_sha256"],
        "authorization": report["analysis"]["authorization"],
        "behavioral_selection": report["analysis"]["behavioral_selection"],
    }
    if args.check_only:
        summary["output"] = None
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    output = local(args.output)
    write_immutable_report(output, report)
    summary["output"] = str(output.resolve())
    summary["output_file_sha256"] = sha256_file(output.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
