"""Import-safe, fresh-prompt v2r4d rollout gate and exact-once launcher."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import modal

# ``modal run path/to/file.py`` mounts this launcher directly under ``/root``
# while the package tree lives at ``/root/chess-rl-miles``. Bootstrap that
# exact package root before importing any ``chess_rl_miles`` module.
_REMOTE_PROJECT_DIRS = (
    Path(__file__).resolve().parent / "chess-rl-miles",
    Path("/root/chess-rl-miles"),
)
for _remote_project_dir in _REMOTE_PROJECT_DIRS:
    if (
        _remote_project_dir.is_dir()
        and str(_remote_project_dir) not in sys.path
    ):
        sys.path.insert(0, str(_remote_project_dir))
        break

from chess_rl_miles.provenance import (
    directory_identity,
    runtime_identity,
    write_run_provenance,
)
from chess_rl_miles.scripts import modal_interleave as shared
from chess_rl_miles.v2r4d_contract_binding import (
    EXPECTED_CONTRACT_FILE_SHA256,
    EXPECTED_CONTRACT_SHA256,
)


base = shared.base
app = modal.App(
    "chess-interleave-v2r4d-gate",
    image=base.image,
    secrets=base.runtime_secrets,
)

VERSION = "v2r4d_production_gate_20260730"
CONTRACT_SCHEMA = (
    "interleaved-v2r4d-production-gate-runtime-contract-v1"
)
BINDING_RELATIVE_PATH = (
    "chess_rl_miles/v2r4d_contract_binding.py"
)
CONTRACT_PATH = (
    "/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/"
    "v2r4d_production_gate_20260730/runtime_contract.json"
)
INCIDENT_PATH = (
    "/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/"
    "v2r4d_production_gate_20260730/"
    "v2r4c_preflight_incident_report.json"
)
IMPORT_PROBE_PATH = (
    "/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/"
    "v2r4d_production_gate_20260730/import_probe_report.json"
)
INCIDENT_SHA256 = (
    "2a2eaeb867d73366b69377e6f965a925eff0c3db8bd8bae881ec4d035e3ec5f0"
)
INCIDENT_FILE_SHA256 = (
    "8c12073b996486318c7df5b107669c4cb3d55952e32b2cb22a3ca68a855a3168"
)
CANARY_PATH = (
    "/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/"
    "v2r4c_production_gate_20260730/canary.parquet"
)
CANARY_SHA256 = (
    "8c714172b9f9b90348673b92705f3c4ddbded404a8ced5c22e38be031e14accb"
)
CANARY_PROMPT_SET_SHA256 = (
    "bac5f41fbed4e41ea511203078a9195c613d0eb04a682a7ea076db56158a8663"
)
CANARY_EPOCH0_ORDER_SHA256 = (
    "1ae1bdece961311cd3fe57c07d4f4a4eb05826d6a7e193c8d37585347f17b73b"
)
CANARY_ROWS = 256
CANARY_SEED = 13_477_620
CANARY_RUN_NAME = "v2r4d-runtime-canary-s6000-seed13477620"
CELL_TIMEOUT_SECONDS = 2 * 60 * 60
ROUTER_HEALTH_INTERVAL_SECONDS = 1e18
CANARY_LEDGER_DEFAULT = (
    "INTERLEAVED_V2R4D_CANARY_LAUNCH_LEDGER.json"
)
GRID_LEDGER_DEFAULT = "INTERLEAVED_V2R4D_GATE_LAUNCH_LEDGER.json"
FINALIZER_PATH = "Eval/finalize_v2r4d_gate.py"
REUSED_FINALIZER_PATH = "Eval/finalize_v2r4_gate.py"
FINALIZER_TEST_PATH = "Eval/tests/test_finalize_v2r4d_gate.py"
ANALYZER_PATH = "Eval/v2r4c_gate_analysis.py"
CORRECTED_ANALYZER_PATH = "Eval/v2r4b_gate_analysis.py"
BASE_ANALYZER_PATH = "Eval/v2r4_gate_analysis.py"

CANDIDATES = {
    step: dict(value)
    for step, value in shared.V2R4_GATE_CANDIDATES.items()
}
PROMPT_MANIFEST_PATH = (
    "/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/"
    "v2r4c_production_gate_20260730/prompt_batches_manifest.json"
)
PROMPT_ARTIFACT_SCHEMA = (
    "interleaved-v2r4c-fresh-prompt-batches-v1"
)
PROMPT_ARTIFACT_VERSION = "v2r4c_production_gate_20260730"
PROMPT_MANIFEST_SHA256 = (
    "83f4718b829b955cb000908c2ecbb9052883d14404114cbca4ecd42988659056"
)
PROMPT_MANIFEST_FILE_SHA256 = (
    "261a313c687bb328d3306301c5477705f6bd8f1b5334d8bccd657102ecfdce60"
)
PROMPT_BATCHES = {
    "A": {
        "path": (
            "/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/"
            "v2r4c_production_gate_20260730/batch_a.parquet"
        ),
        "sha256": (
            "c5f6f208f348b079ee476ecf99c4fad14bba4210ac31c850ed0ce9d801caab61"
        ),
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

PREDECESSOR_INCIDENT = {
    "path": "INTERLEAVED_V2R4C_PREFLIGHT_INCIDENT_REPORT.json",
    "report_sha256": (
        "2a2eaeb867d73366b69377e6f965a925eff0c3db8bd8bae881ec4d035e3ec5f0"
    ),
    "file_sha256": (
        "8c12073b996486318c7df5b107669c4cb3d55952e32b2cb22a3ca68a855a3168"
    ),
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
PREDECESSOR_RUN_NAMES = (
    "v2r4c-runtime-canary-s6000-seed13477620",
    "v2r4c-gate-w190-s6000-batch-a",
    "v2r4c-gate-w190-s6000-batch-b",
    "v2r4c-gate-w190-s8000-batch-a",
    "v2r4c-gate-w190-s8000-batch-b",
    "v2r4c-gate-w190-s9920-batch-a",
    "v2r4c-gate-w190-s9920-batch-b",
)

results_vol = modal.Volume.from_name(
    "chess-rl-eval-results-r6", create_if_missing=False
)


def _run_name(candidate_step: int, batch_label: str) -> str:
    return (
        f"v2r4d-gate-w190-s{candidate_step}-"
        f"batch-{batch_label.lower()}"
    )


CELLS = tuple(
    {
        "candidate_step": step,
        "batch_label": batch,
        "run_name": _run_name(step, batch),
    }
    for step in sorted(CANDIDATES)
    for batch in sorted(PROMPT_BATCHES)
)

QUARANTINED_V2R4A = {
    "report_sha256": (
        "576bf2fb346666f8b9da2d3df5563d9e29e65b60ce1a66c07b74bf6318428947"
    ),
    "report_file_sha256": (
        "b800d3f8289cf1e4d2ef1320efb4185ab62761c576f3dd05b863d11c8d694970"
    ),
    "ledger_sha256": (
        "a09a1a10a651417b3b91294d757a03ce1236e9ede170ad32ff3f2bb193f2f648"
    ),
    "ledger_file_sha256": (
        "065163efa7fa70249ebe0b99889e326ed0f68dad591dec383341027169c720cf"
    ),
    "contract_sha256": (
        "3e201cfd9094815cf72a63058d4225b334e3e5cd77fc0d79fbe6379d48778c9d"
    ),
    "cells": (
        {
            "candidate_step": 6_000,
            "batch_label": "A",
            "run_name": "v2r4a-gate-w190-s6000-batch-a",
            "function_call_id": "fc-01KYT22WT1N54KQFW2HZCR569J",
            "terminal_state": "success_quarantined_blinding_violation",
            "root_file_count": 16,
            "root_total_bytes": 50_471_241,
            "root_inventory_sha256": (
                "af1fd5e5907cf341d6668dbf99336a7475e4d5c7db3867b436633f796ce0d09e"
            ),
            "success_marker_sha256": (
                "d2a23a3dea80a02f4156ddf10ae18b8c6bf0965552fb86aad5e8a52618f2763f"
            ),
        },
        {
            "candidate_step": 6_000,
            "batch_label": "B",
            "run_name": "v2r4a-gate-w190-s6000-batch-b",
            "function_call_id": "fc-01KYT22WZ5FMA8PXX82207G4H0",
            "terminal_state": "watchdog_failure",
            "root_file_count": 3,
            "root_total_bytes": 24_265,
            "root_inventory_sha256": (
                "5df25ddfa827f18b80f928d40792cf13dd4cf3797ee630e99d8196bcb6c0745d"
            ),
            "success_marker_sha256": None,
        },
        {
            "candidate_step": 8_000,
            "batch_label": "A",
            "run_name": "v2r4a-gate-w190-s8000-batch-a",
            "function_call_id": "fc-01KYT22X3A7EG7R6Y83VZ9CBXD",
            "terminal_state": "watchdog_failure",
            "root_file_count": 3,
            "root_total_bytes": 24_268,
            "root_inventory_sha256": (
                "c9ad5a8708c7c939ee1769dabfcc622875847db325dfdeff0517a846dcaf62b5"
            ),
            "success_marker_sha256": None,
        },
        {
            "candidate_step": 8_000,
            "batch_label": "B",
            "run_name": "v2r4a-gate-w190-s8000-batch-b",
            "function_call_id": "fc-01KYT22X68F8KZXV3ENBP8HY17",
            "terminal_state": "success_quarantined_blinding_violation",
            "root_file_count": 16,
            "root_total_bytes": 49_033_603,
            "root_inventory_sha256": (
                "53fafd561cbcfa4766e6b50a4b3758f2a0b0e99a49f85d22fbb3c63e3a7f926d"
            ),
            "success_marker_sha256": (
                "60cd0896b98f4c34a1d09964b3a52fb1dbb067326498a31c8ef74ee0173cb223"
            ),
        },
        {
            "candidate_step": 9_920,
            "batch_label": "A",
            "run_name": "v2r4a-gate-w190-s9920-batch-a",
            "function_call_id": "fc-01KYT22X9KH07VFZGMTR2PMCK7",
            "terminal_state": "watchdog_failure",
            "root_file_count": 3,
            "root_total_bytes": 24_267,
            "root_inventory_sha256": (
                "c30916fe235af0178a0b93af08d593a620bda9da728c600a4e2b7f7c88c1d4bc"
            ),
            "success_marker_sha256": None,
        },
        {
            "candidate_step": 9_920,
            "batch_label": "B",
            "run_name": "v2r4a-gate-w190-s9920-batch-b",
            "function_call_id": "fc-01KYT22XCSR7MEPWA5HERQNKPV",
            "terminal_state": "success_quarantined_blinding_violation",
            "root_file_count": 16,
            "root_total_bytes": 48_703_637,
            "root_inventory_sha256": (
                "b2c76bf69fb471cdf559257751574f1c56ae88c511d20d0ca221738752a064be"
            ),
            "success_marker_sha256": (
                "4ab96a40647a80319aa0369dede1c70ce2cdcf212ce4fbf2ca99ab3ccee7a60f"
            ),
        },
    ),
    "disposition": "entire_grid_quarantined_no_authorization",
    "prebarrier_disclosure": {
        "aggregate_or_prompt_level_outcomes_inspected": False,
        "rollout_jsonl_contents_manually_inspected": False,
        "individual_reward_log_exposures": (
            {
                "candidate_step": 6_000,
                "batch_label": "B",
                "reward": 0,
            },
            {
                "candidate_step": 8_000,
                "batch_label": "A",
                "reward": 0,
            },
        ),
    },
    "postbarrier_disclosure": {
        "complete_six_cell_aggregate_computed": False,
        "frozen_check_only_materialized_outcome_rows": True,
        "diagnostic_row": {
            "candidate_step": 6_000,
            "batch_label": "A",
            "rollout_id": 1,
            "line_number": 388,
            "sample_index": 2_435,
            "puzzle_id": "4DXgm",
            "score": 1,
            "output": "Qh5f7# <call_env>",
            "joint_protocol_valid": False,
        },
        "effect_on_repair": (
            "directly_triggered_direction_neutral_independent_"
            "reward_protocol_counting_only"
        ),
    },
}

SEMANTICS = {
    **shared.V2R4_GATE_SEMANTICS,
    "rollout_fault_tolerance": False,
    "router_health_check_interval_seconds": (
        ROUTER_HEALTH_INTERVAL_SECONDS
    ),
    "strict_sample_outcome_logs": "redacted",
    "positive_attempt_stream": False,
    "miles_log_passrate": False,
    "default_reward_metrics_before_barrier": False,
    "pre_barrier_reward_aggregates": False,
    "strict_success_tail": (
        "require_zero_pending_and_skip_abort_rpc"
    ),
    "hard_cell_timeout_seconds": CELL_TIMEOUT_SECONDS,
    "analysis_reward_and_protocol_counted_independently": True,
    "fresh_prompt_sets_disjoint_from_quarantined_grid": True,
    "canary_disjoint_from_grid": True,
}


@app.function(cpu=1.0, memory=1024, timeout=5 * 60)
def v2r4d_import_probe() -> dict[str, object]:
    """Prove direct-file packaging/imports without remote storage or a GPU."""

    package_path = Path(shared.__file__).resolve()
    project_path = Path(base.PROJECT_DIR).resolve()
    launcher_path = Path(__file__).resolve()
    if not project_path.is_dir():
        raise FileNotFoundError(project_path)
    if project_path not in package_path.parents:
        raise RuntimeError(
            "chess_rl_miles was not imported from the mounted project tree"
        )
    if str(project_path) not in sys.path:
        raise RuntimeError("mounted project tree is absent from sys.path")
    return {
        "schema": "interleaved-v2r4d-import-probe-v1",
        "version": VERSION,
        "launcher_path": str(launcher_path),
        "launcher_sha256": shared._sha256(launcher_path),
        "project_path": str(project_path),
        "package_path": str(package_path),
        "project_on_sys_path": True,
        "volumes_mounted": False,
        "gpu_requested": False,
    }


@app.local_entrypoint()
def v2r4d_import_probe_main() -> None:
    result = v2r4d_import_probe.remote()
    if (
        result.get("schema") != "interleaved-v2r4d-import-probe-v1"
        or result.get("version") != VERSION
        or result.get("project_on_sys_path") is not True
        or result.get("volumes_mounted") is not False
        or result.get("gpu_requested") is not False
    ):
        raise RuntimeError("v2r4d import probe returned an invalid result")
    print(json.dumps(result, sort_keys=True), flush=True)


def _contract_static() -> dict[str, object]:
    return {
        "schema": CONTRACT_SCHEMA,
        "version": VERSION,
        "model_id": shared.MODEL_ID,
        "cells": [dict(cell) for cell in CELLS],
        "candidates": {
            str(step): dict(value)
            for step, value in sorted(CANDIDATES.items())
        },
        "prompt_batches": {
            label: dict(value)
            for label, value in sorted(PROMPT_BATCHES.items())
        },
        "prompt_manifest": {
            "path": PROMPT_MANIFEST_PATH,
            "source_schema": PROMPT_ARTIFACT_SCHEMA,
            "source_version": PROMPT_ARTIFACT_VERSION,
            "manifest_sha256": PROMPT_MANIFEST_SHA256,
            "file_sha256": PROMPT_MANIFEST_FILE_SHA256,
            "reused_after_zero_outcome_preflight_failure": True,
            "intersections": {
                "A_B": 0,
                "A_CANARY": 0,
                "B_CANARY": 0,
                "A_EXCLUDED_PRIOR": 0,
                "B_EXCLUDED_PRIOR": 0,
                "CANARY_EXCLUDED_PRIOR": 0,
            },
        },
        "canary": {
            "run_name": CANARY_RUN_NAME,
            "candidate_step": 6_000,
            "path": CANARY_PATH,
            "sha256": CANARY_SHA256,
            "rows": CANARY_ROWS,
            "prompt_set_sha256": CANARY_PROMPT_SET_SHA256,
            "epoch0_prompt_order_sha256": (
                CANARY_EPOCH0_ORDER_SHA256
            ),
            "rollout_seed": CANARY_SEED,
            "num_rollout": 1,
        },
        "semantics": dict(SEMANTICS),
        "predecessor_incident": dict(PREDECESSOR_INCIDENT),
        "runtime": {
            "miles_image": base.MILES_IMAGE,
            "gpu_type": base.GPU_TYPE,
            "gpus_per_node": base.GPUS_PER_NODE,
            "host_memory_gb": shared.SMALL_MODEL_HOST_MEMORY_GB,
            "max_tokens_per_gpu": (
                shared.SMALL_MODEL_MAX_TOKENS_DEFAULT
            ),
            "sglang_server_concurrency": (
                shared.SGLANG_SERVER_CONCURRENCY_DEFAULT
            ),
        },
        "quarantined_v2r4a": {
            **QUARANTINED_V2R4A,
            "cells": [
                dict(cell) for cell in QUARANTINED_V2R4A["cells"]
            ],
            "prebarrier_disclosure": {
                **QUARANTINED_V2R4A["prebarrier_disclosure"],
                "individual_reward_log_exposures": [
                    dict(value)
                    for value in QUARANTINED_V2R4A[
                        "prebarrier_disclosure"
                    ]["individual_reward_log_exposures"]
                ],
            },
        },
    }


def _require_contract_digest(requested_sha256: str) -> None:
    if (
        EXPECTED_CONTRACT_SHA256 == "0" * 64
        or EXPECTED_CONTRACT_FILE_SHA256 == "0" * 64
        or requested_sha256 != EXPECTED_CONTRACT_SHA256
        or not re.fullmatch(r"[0-9a-f]{64}", requested_sha256)
    ):
        raise ValueError(
            "v2r4d launch requires a non-placeholder frozen contract SHA256"
        )


def _validate_finalizer_contract(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "sha256",
        "reused_dependency",
        "test_source",
    }:
        raise ValueError("v2r4d contract lacks the exact finalizer binding")
    if (
        value.get("path") != FINALIZER_PATH
        or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", "")))
    ):
        raise ValueError("v2r4d finalizer source binding drifted")
    for field, expected_path in (
        ("reused_dependency", REUSED_FINALIZER_PATH),
        ("test_source", FINALIZER_TEST_PATH),
    ):
        nested = value.get(field)
        if (
            not isinstance(nested, dict)
            or set(nested) != {"path", "sha256"}
            or nested.get("path") != expected_path
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(nested.get("sha256", ""))
            )
        ):
            raise ValueError(f"v2r4d finalizer {field} binding drifted")


def _validate_analysis_contract(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "sha256",
        "corrected_dependency",
        "base_dependency",
    }:
        raise ValueError("v2r4d contract lacks the exact analyzer binding")
    if (
        value.get("path") != ANALYZER_PATH
        or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", "")))
    ):
        raise ValueError("v2r4d analyzer source binding drifted")
    for field, expected_path in (
        ("corrected_dependency", CORRECTED_ANALYZER_PATH),
        ("base_dependency", BASE_ANALYZER_PATH),
    ):
        nested = value.get(field)
        if (
            not isinstance(nested, dict)
            or set(nested) != {"path", "sha256"}
            or nested.get("path") != expected_path
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(nested.get("sha256", ""))
            )
        ):
            raise ValueError(f"v2r4d analyzer {field} binding drifted")


def _validate_import_probe_contract(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "report_path",
        "report_sha256",
        "report_file_sha256",
        "report_bytes",
        "app_id",
        "result",
    }:
        raise ValueError("v2r4d contract lacks exact import-probe evidence")
    result = value.get("result")
    if (
        value.get("report_path") != IMPORT_PROBE_PATH
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(value.get("report_sha256", ""))
        )
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(value.get("report_file_sha256", "")),
        )
        or not isinstance(value.get("report_bytes"), int)
        or int(value["report_bytes"]) <= 0
        or not re.fullmatch(
            r"ap-[A-Za-z0-9_-]+", str(value.get("app_id", ""))
        )
        or not isinstance(result, dict)
        or result.get("schema")
        != "interleaved-v2r4d-import-probe-v1"
        or result.get("version") != VERSION
        or result.get("launcher_path") != "/root/modal_v2r4d_gate.py"
        or result.get("launcher_sha256")
        != shared._sha256(Path(__file__).resolve())
        or result.get("project_path") != "/root/chess-rl-miles"
        or result.get("package_path")
        != (
            "/root/chess-rl-miles/chess_rl_miles/"
            "scripts/modal_interleave.py"
        )
        or result.get("project_on_sys_path") is not True
        or result.get("volumes_mounted") is not False
        or result.get("gpu_requested") is not False
    ):
        raise ValueError("v2r4d import-probe contract evidence drifted")


def _load_contract(requested_sha256: str) -> dict[str, object]:
    _require_contract_digest(requested_sha256)
    if not re.fullmatch(
        r"[0-9a-f]{64}", EXPECTED_CONTRACT_FILE_SHA256
    ):
        raise RuntimeError("v2r4d contract file binding is not frozen")
    path = Path(CONTRACT_PATH)
    if (
        not path.is_file()
        or shared._sha256(path) != EXPECTED_CONTRACT_FILE_SHA256
    ):
        raise ValueError("v2r4d runtime-contract file drifted")
    contract = json.loads(path.read_text())
    if not isinstance(contract, dict):
        raise ValueError("v2r4d runtime contract is not an object")
    embedded = contract.pop("contract_sha256", None)
    if (
        embedded != EXPECTED_CONTRACT_SHA256
        or shared._canonical_json_sha256(contract)
        != EXPECTED_CONTRACT_SHA256
    ):
        raise ValueError("v2r4d runtime-contract self-hash drifted")
    for key, expected in _contract_static().items():
        if contract.get(key) != expected:
            raise ValueError(f"v2r4d contract field drifted: {key}")

    sources = contract.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("v2r4d contract lacks source identities")
    actual_project = shared._normalized_source_identity(
        Path(base.PROJECT_DIR),
        excluded_relatives=(BINDING_RELATIVE_PATH,),
    )
    actual_miles = shared._normalized_source_identity(
        Path(base.MILES_DIR)
    )
    if sources.get("chess_rl_miles") != actual_project:
        raise ValueError("v2r4d project source identity drifted")
    if sources.get("miles") != actual_miles:
        raise ValueError("v2r4d Miles source identity drifted")
    plan = contract.get("plan")
    analysis = contract.get("analysis")
    finalizer = contract.get("finalizer")
    import_probe = contract.get("import_probe")
    endpoints = contract.get("endpoint_evaluators")
    if (
        not isinstance(plan, dict)
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(plan.get("sha256", ""))
        )
        or not isinstance(endpoints, dict)
        or set(endpoints) != {"pt_b1_b5", "p2_sft_at_p1"}
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(value))
            for value in endpoints.values()
        )
    ):
        raise ValueError(
            "v2r4d contract lacks frozen plan/analysis/evaluator identities"
        )
    _validate_analysis_contract(analysis)
    _validate_finalizer_contract(finalizer)
    _validate_import_probe_contract(import_probe)
    return {
        **contract,
        "contract_sha256": embedded,
        "contract_file_sha256": EXPECTED_CONTRACT_FILE_SHA256,
        "contract_path": str(path),
    }


def _load_import_probe_report(
    contract: dict[str, object],
) -> dict[str, object]:
    evidence = contract.get("import_probe")
    _validate_import_probe_contract(evidence)
    assert isinstance(evidence, dict)
    path = Path(IMPORT_PROBE_PATH)
    if (
        not path.is_file()
        or shared._sha256(path) != evidence["report_file_sha256"]
        or path.stat().st_size != evidence["report_bytes"]
    ):
        raise ValueError("v2r4d import-probe report file drifted")
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise ValueError("v2r4d import-probe report is not an object")
    embedded = report.pop("report_sha256", None)
    if (
        embedded != evidence["report_sha256"]
        or shared._canonical_json_sha256(report) != embedded
        or report.get("schema")
        != "interleaved-v2r4d-import-probe-report-v1"
        or report.get("app_id") != evidence["app_id"]
        or report.get("result") != evidence["result"]
    ):
        raise ValueError("v2r4d import-probe report evidence drifted")
    return {**report, "report_sha256": embedded}


def _load_predecessor_incident() -> dict[str, object]:
    path = Path(INCIDENT_PATH)
    if (
        not path.is_file()
        or shared._sha256(path) != INCIDENT_FILE_SHA256
    ):
        raise ValueError("v2r4c predecessor incident file drifted")
    incident = json.loads(path.read_text())
    if not isinstance(incident, dict):
        raise ValueError("v2r4c predecessor incident is not an object")
    embedded = incident.pop("report_sha256", None)
    function_call = incident.get("function_call")
    launcher_ledger = incident.get("launcher_ledger")
    boundary = incident.get("gpu_and_outcome_boundary")
    disposition = incident.get("disposition")
    if (
        embedded != INCIDENT_SHA256
        or shared._canonical_json_sha256(incident) != INCIDENT_SHA256
        or incident.get("schema")
        != "interleaved-v2r4c-preflight-incident-report-v1"
        or incident.get("version")
        != PREDECESSOR_INCIDENT["failed_version"]
        or not isinstance(function_call, dict)
        or function_call.get("function_call_id")
        != PREDECESSOR_INCIDENT["preflight_call_id"]
        or function_call.get("call_graph_children") != []
        or function_call.get("input_status") != "TERMINATED"
        or not isinstance(launcher_ledger, dict)
        or launcher_ledger.get("state") != "preflight_failed"
        or launcher_ledger.get("calls") != []
        or launcher_ledger.get("file_sha256")
        != PREDECESSOR_INCIDENT["failed_ledger_file_sha256"]
        or not isinstance(boundary, dict)
        or boundary.get("remote_call_children") != 0
        or boundary.get("canary_function_call_spawned") is not False
        or boundary.get("grid_function_calls_spawned") != 0
        or boundary.get("outcome_exposure") is not False
        or not isinstance(disposition, dict)
        or disposition.get("prompt_artifacts_may_be_reused")
        is not True
        or incident.get("verified_absent_roots")
        != [
            f"chess-rl-miles-interleave/{name}"
            for name in PREDECESSOR_RUN_NAMES
        ]
    ):
        raise ValueError("v2r4c predecessor incident evidence drifted")
    return {**incident, "report_sha256": embedded}


def _validate_predecessor_absence(
    *,
    checkpoint_mount: Path,
    results_mount: Path,
) -> dict[str, object]:
    predecessor_roots = [
        checkpoint_mount
        / "chess-rl-miles-interleave"
        / run_name
        for run_name in PREDECESSOR_RUN_NAMES
    ]
    existing_roots = [
        str(path) for path in predecessor_roots if path.exists()
    ]
    if existing_roots:
        raise FileExistsError(
            "v2r4c predecessor roots appeared after incident freeze: "
            + ", ".join(existing_roots)
        )
    predecessor_results = sorted(
        path.relative_to(results_mount).as_posix()
        for path in results_mount.rglob("*")
        if "v2r4c" in path.relative_to(results_mount).as_posix().lower()
    )
    if predecessor_results:
        raise FileExistsError(
            "v2r4c predecessor result artifacts appeared after incident "
            "freeze: " + ", ".join(predecessor_results)
        )
    return {
        "predecessor_roots_absent": True,
        "predecessor_root_count": len(predecessor_roots),
        "predecessor_results_absent": True,
        "predecessor_result_count": 0,
    }


def _load_prompt_manifest() -> dict[str, object]:
    path = Path(PROMPT_MANIFEST_PATH)
    if (
        not path.is_file()
        or shared._sha256(path) != PROMPT_MANIFEST_FILE_SHA256
    ):
        raise ValueError("v2r4d fresh prompt manifest file drifted")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("v2r4d fresh prompt manifest is not an object")
    embedded = payload.pop("manifest_sha256", None)
    if (
        embedded != PROMPT_MANIFEST_SHA256
        or shared._canonical_json_sha256(payload)
        != PROMPT_MANIFEST_SHA256
        or payload.get("schema") != PROMPT_ARTIFACT_SCHEMA
        or payload.get("version") != PROMPT_ARTIFACT_VERSION
        or payload.get("status")
        != "frozen_prompt_data_no_independent_launch_authorization"
    ):
        raise ValueError("v2r4d fresh prompt manifest self-hash drifted")
    batches = payload.get("batches")
    intersections = payload.get("intersections")
    if (
        not isinstance(batches, dict)
        or set(batches) != {"A", "B", "CANARY"}
        or not isinstance(intersections, dict)
        or set(intersections.values()) != {0}
    ):
        raise ValueError("v2r4d prompt-set disjointness proof drifted")
    expected_sets = {
        "A": {
            "path": PROMPT_BATCHES["A"]["path"],
            "sha256": PROMPT_BATCHES["A"]["sha256"],
            "rows": PROMPT_BATCHES["A"]["rows"],
            "rollout_seed": PROMPT_BATCHES["A"]["rollout_seed"],
            "prompt_set_sha256": (
                PROMPT_BATCHES["A"]["prompt_set_sha256"]
            ),
            "epoch0_prompt_order_sha256": (
                PROMPT_BATCHES["A"]["epoch0_prompt_order_sha256"]
            ),
        },
        "B": {
            "path": PROMPT_BATCHES["B"]["path"],
            "sha256": PROMPT_BATCHES["B"]["sha256"],
            "rows": PROMPT_BATCHES["B"]["rows"],
            "rollout_seed": PROMPT_BATCHES["B"]["rollout_seed"],
            "prompt_set_sha256": (
                PROMPT_BATCHES["B"]["prompt_set_sha256"]
            ),
            "epoch0_prompt_order_sha256": (
                PROMPT_BATCHES["B"]["epoch0_prompt_order_sha256"]
            ),
        },
        "CANARY": {
            "path": CANARY_PATH,
            "sha256": CANARY_SHA256,
            "rows": CANARY_ROWS,
            "rollout_seed": CANARY_SEED,
            "prompt_set_sha256": CANARY_PROMPT_SET_SHA256,
            "epoch0_prompt_order_sha256": (
                CANARY_EPOCH0_ORDER_SHA256
            ),
        },
    }
    for label, expected in expected_sets.items():
        observed = batches.get(label)
        if not isinstance(observed, dict):
            raise ValueError(f"v2r4d prompt set is missing: {label}")
        normalized = {
            "path": observed.get("logical_path"),
            "sha256": observed.get("file_sha256"),
            "rows": observed.get("rows"),
            "rollout_seed": observed.get("rollout_seed"),
            "prompt_set_sha256": observed.get("prompt_set_sha256"),
            "epoch0_prompt_order_sha256": observed.get(
                "epoch0_prompt_order_sha256"
            ),
        }
        if normalized != expected:
            raise ValueError(f"v2r4d prompt set drifted: {label}")
    prompt_sets = {
        label: set(batches[label]["ordered_prompt_fingerprints"])
        for label in ("A", "B", "CANARY")
    }
    if (
        len(prompt_sets["A"]) != 1_024
        or len(prompt_sets["B"]) != 1_024
        or len(prompt_sets["CANARY"]) != 256
        or prompt_sets["A"] & prompt_sets["B"]
        or prompt_sets["A"] & prompt_sets["CANARY"]
        or prompt_sets["B"] & prompt_sets["CANARY"]
    ):
        raise ValueError("v2r4d direct prompt intersection check failed")
    return {**payload, "manifest_sha256": embedded}


def _validate_prompt_data_files() -> None:
    import pyarrow.parquet as pq

    expected = (
        (Path(CANARY_PATH), CANARY_SHA256, CANARY_ROWS),
        *(
            (
                Path(str(PROMPT_BATCHES[label]["path"])),
                str(PROMPT_BATCHES[label]["sha256"]),
                int(PROMPT_BATCHES[label]["rows"]),
            )
            for label in ("A", "B")
        ),
    )
    for path, digest, rows in expected:
        if not path.is_file() or shared._sha256(path) != digest:
            raise ValueError(f"v2r4d prompt parquet drifted: {path}")
        if pq.ParquetFile(path).metadata.num_rows != rows:
            raise ValueError(f"v2r4d prompt row count drifted: {path}")


def _strict_inventory(
    run_root: Path,
    *,
    expected_rollouts: int,
    intent_filename: str,
    launch_manifest: str,
    success_marker_filename: str | None = None,
) -> None:
    launch_path = Path(launch_manifest)
    if (
        launch_path.parent.name != "provenance"
        or launch_path.parent.parent != run_root
        or not re.fullmatch(r"launch_[0-9a-f]{16}\.json", launch_path.name)
    ):
        raise ValueError("strict rollout launch manifest path drifted")
    expected = {
        intent_filename,
        "run_provenance.json",
        f"provenance/{launch_path.name}",
        *{
            f"rollouts/training/rollout_{rollout_id}.jsonl"
            for rollout_id in range(expected_rollouts)
        },
    }
    if success_marker_filename is not None:
        expected.add(success_marker_filename)
    symlinks = sorted(
        path.relative_to(run_root).as_posix()
        for path in run_root.rglob("*")
        if path.is_symlink()
    )
    if symlinks:
        raise RuntimeError(
            "strict rollout emitted prohibited symlinks: "
            + ", ".join(symlinks)
        )
    observed = {
        path.relative_to(run_root).as_posix()
        for path in run_root.rglob("*")
        if path.is_file()
    }
    if observed != expected:
        raise RuntimeError(
            "strict rollout artifact inventory drifted: "
            f"expected={sorted(expected)} observed={sorted(observed)}"
        )


def _validate_canary(
    run_root: Path,
    *,
    launch_manifest: str,
    success_marker_present: bool = False,
) -> dict[str, object]:
    _strict_inventory(
        run_root,
        expected_rollouts=1,
        intent_filename="_V2R4D_CANARY_INTENT.json",
        launch_manifest=launch_manifest,
        success_marker_filename=(
            "_V2R4D_CANARY_SUCCESS.json"
            if success_marker_present
            else None
        ),
    )
    path = run_root / "rollouts" / "training" / "rollout_0.jsonl"
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"canary row {line_number} is not an object"
                )
            rows.append(value)
    if len(rows) != 2_048:
        raise ValueError("canary must contain exactly 2,048 rows")
    fingerprints: list[str] = []
    for group_index in range(256):
        siblings = rows[group_index * 8 : (group_index + 1) * 8]
        group_fingerprints = {
            shared._v2r4_prompt_fingerprint(row)
            for row in siblings
        }
        if len(group_fingerprints) != 1:
            raise ValueError("canary sibling prompt identity drifted")
        fingerprints.append(next(iter(group_fingerprints)))
        for sibling_index, row in enumerate(siblings):
            sample_index = group_index * 8 + sibling_index
            if (
                row.get("rollout_id") != 0
                or row.get("group_index") != group_index
                or row.get("sample_index") != sample_index
                or row.get("sampling_seed_sibling_index")
                != sibling_index
                or row.get("sampling_seed")
                != CANARY_SEED + sample_index
                or row.get("status") not in {"completed", "truncated"}
            ):
                raise ValueError("canary exact shape/seed drifted")
            metadata = row.get("metadata")
            if (
                not isinstance(metadata, dict)
                or metadata.get("sampling_seed")
                != CANARY_SEED + sample_index
                or metadata.get("sampling_seed_sibling_index")
                != sibling_index
                or metadata.get("sampling_seed_mode") != "sample-index"
            ):
                raise ValueError("canary metadata seed drifted")
    prompt_set_sha256 = hashlib.sha256(
        json.dumps(
            sorted(fingerprints),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if (
        len(set(fingerprints)) != 256
        or prompt_set_sha256 != CANARY_PROMPT_SET_SHA256
        or shared._canonical_json_sha256(fingerprints)
        != CANARY_EPOCH0_ORDER_SHA256
    ):
        raise ValueError("canary prompt set drifted")
    return {
        "path": str(path),
        "rows": len(rows),
        "sha256": shared._sha256(path),
        "prompt_set_sha256": prompt_set_sha256,
        "epoch0_prompt_order_sha256": (
            CANARY_EPOCH0_ORDER_SHA256
        ),
        "shape_authenticated": True,
        "reward_metrics_inspected": False,
    }


def _quarantine_root_identity(
    run_root: Path,
    *,
    mount_root: Path = Path("/rl-checkpoints"),
) -> dict[str, object]:
    try:
        logical_root = Path("/") / run_root.relative_to(mount_root)
    except ValueError as exc:
        raise ValueError("quarantine root is outside its volume mount") from exc
    records: list[dict[str, object]] = []
    for path in sorted(run_root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(
                f"quarantine root contains a symlink: {path}"
            )
        if not path.is_file():
            continue
        logical_path = logical_root / path.relative_to(run_root)
        records.append(
            {
                "path": logical_path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": shared._sha256(path),
            }
        )
    return {
        "root_file_count": len(records),
        "root_total_bytes": sum(
            int(record["bytes"]) for record in records
        ),
        "root_inventory_sha256": (
            shared._canonical_json_sha256(records)
        ),
    }


def _validate_quarantine() -> dict[str, object]:
    records: list[dict[str, object]] = []
    for cell in QUARANTINED_V2R4A["cells"]:
        run_root = Path(shared.RAW_RL_ROOT) / str(cell["run_name"])
        if not run_root.is_dir():
            raise FileNotFoundError(run_root)
        marker = run_root / "_V2R4_GATE_SUCCESS.json"
        identity = _quarantine_root_identity(run_root)
        expected_identity = {
            key: cell[key]
            for key in (
                "root_file_count",
                "root_total_bytes",
                "root_inventory_sha256",
            )
        }
        if identity != expected_identity:
            raise RuntimeError(
                "v2r4a quarantine root inventory drifted: "
                f"{cell['run_name']}"
            )
        state = str(cell["terminal_state"])
        expected_marker = cell["success_marker_sha256"]
        if state == "success_quarantined_blinding_violation":
            if (
                not marker.is_file()
                or shared._sha256(marker) != expected_marker
            ):
                raise RuntimeError(
                    "v2r4a success quarantine proof drifted"
                )
            marker_value = json.loads(marker.read_text())
            marker_core = {
                key: value
                for key, value in marker_value.items()
                if key != "success_sha256"
            }
            if marker_value.get(
                "success_sha256"
            ) != shared._canonical_json_sha256(marker_core):
                raise RuntimeError(
                    "v2r4a success marker self-hash drifted"
                )
        elif state == "watchdog_failure":
            if marker.exists() or expected_marker is not None:
                raise RuntimeError(
                    "v2r4a failure quarantine proof drifted"
                )
        else:
            raise ValueError("unknown v2r4a quarantine state")
        records.append(
            {
                **dict(cell),
                "success_marker_present": marker.is_file(),
                "root_identity_authenticated": True,
            }
        )
    return {
        "grid": records,
        "report_sha256": QUARANTINED_V2R4A["report_sha256"],
        "report_file_sha256": (
            QUARANTINED_V2R4A["report_file_sha256"]
        ),
        "entire_grid_quarantined": True,
        "authorization_absent": True,
    }


@app.function(
    cpu=2.0,
    memory=8 * 1024,
    timeout=10 * 60,
    volumes={
        "/data": base.data_vol,
        "/rl-checkpoints": base.ckpt_vol,
        "/results": results_vol,
    },
)
def v2r4d_preflight(
    contract_sha256: str,
    phase: str,
) -> dict[str, object]:
    if phase not in {"canary", "grid"}:
        raise ValueError("phase must be canary or grid")
    base.data_vol.reload()
    base.ckpt_vol.reload()
    results_vol.reload()
    contract = _load_contract(contract_sha256)
    import_probe = _load_import_probe_report(contract)
    predecessor_incident = _load_predecessor_incident()
    predecessor_absence = _validate_predecessor_absence(
        checkpoint_mount=Path("/rl-checkpoints"),
        results_mount=Path("/results"),
    )
    prompt_manifest = _load_prompt_manifest()
    _validate_prompt_data_files()
    quarantine = _validate_quarantine()
    grid_roots = [
        str(Path(shared.RAW_RL_ROOT) / str(cell["run_name"]))
        for cell in contract["cells"]
    ]
    existing_grid = [path for path in grid_roots if Path(path).exists()]
    if existing_grid:
        raise FileExistsError(
            "v2r4d canonical grid roots already exist: "
            + ", ".join(existing_grid)
        )
    canary_root = Path(shared.RAW_RL_ROOT) / CANARY_RUN_NAME
    if phase == "canary" and canary_root.exists():
        raise FileExistsError(canary_root)
    if phase == "grid":
        marker = canary_root / "_V2R4D_CANARY_SUCCESS.json"
        if not marker.is_file():
            raise FileNotFoundError(marker)
        canary_success = json.loads(marker.read_text())
        core = {
            key: value
            for key, value in canary_success.items()
            if key != "success_sha256"
        }
        if (
            canary_success.get("version") != VERSION
            or canary_success.get("contract_sha256")
            != contract_sha256
            or canary_success.get("success_sha256")
            != shared._canonical_json_sha256(core)
        ):
            raise ValueError("v2r4d canary success marker drifted")
        provenance = canary_success.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("v2r4d canary provenance is missing")
        authenticated_artifact = _validate_canary(
            canary_root,
            launch_manifest=str(provenance.get("launch_manifest")),
            success_marker_present=True,
        )
        if canary_success.get("artifact") != authenticated_artifact:
            raise ValueError("v2r4d canary artifact binding drifted")
    ray_worker_environment = shared._verify_ray_worker_gate_environment(
        "v2r4d-ray-env-preflight"
    )
    return {
        "schema": "interleaved-v2r4d-gate-preflight-v1",
        "version": VERSION,
        "phase": phase,
        "contract_sha256": contract_sha256,
        "contract_file_sha256": contract["contract_file_sha256"],
        "grid_roots_absent": True,
        "canary_state": (
            "absent" if phase == "canary" else "success_authenticated"
        ),
        "canary_success_sha256": (
            None
            if phase == "canary"
            else canary_success["success_sha256"]
        ),
        "quarantine": quarantine,
        "predecessor_incident_sha256": predecessor_incident[
            "report_sha256"
        ],
        "import_probe_report_sha256": import_probe["report_sha256"],
        **predecessor_absence,
        "prompt_manifest_sha256": (
            prompt_manifest["manifest_sha256"]
        ),
        "fresh_prompt_intersections_authenticated": True,
        "ray_worker_environment": ray_worker_environment,
        "grid_roots": grid_roots,
    }


def _run_rollout_command(
    *,
    run_name: str,
    command: list[str],
) -> None:
    env = shared._runtime_env(
        run_name=run_name,
        deterministic_seed_mode="sample-index",
    )
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    base._cleanup_runtime()
    base._start_ray_head(env, cpu_threads=int(base.CPU_COUNT))
    env["RAY_ADDRESS"] = base.RAY_ADDRESS
    try:
        result = subprocess.run(
            command,
            env=env,
            cwd=base.PROJECT_DIR,
            timeout=CELL_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        base.ckpt_vol.commit()
        raise RuntimeError(
            f"v2r4d rollout timed out for {run_name}"
        ) from exc
    if result.returncode:
        base.ckpt_vol.commit()
        raise RuntimeError(
            f"v2r4d rollout failed for {run_name}: "
            f"exit {result.returncode}"
        )


@app.function(
    gpu=f"{base.GPU_TYPE}:{base.GPUS_PER_NODE}",
    cpu=base.CPU_COUNT,
    memory=shared.SMALL_MODEL_HOST_MEMORY_MB,
    timeout=3 * 60 * 60,
    volumes={
        "/data": base.data_vol,
        "/rl-checkpoints": base.ckpt_vol,
        shared.PRETRAIN_CKPT_ROOT: shared.pretrain_ckpt_vol,
        base.HF_CACHE_DIR: base.hf_cache,
    },
)
def v2r4d_canary(contract_sha256: str) -> dict[str, object]:
    base.data_vol.reload()
    base.ckpt_vol.reload()
    shared.pretrain_ckpt_vol.reload()
    contract = _load_contract(contract_sha256)
    _load_prompt_manifest()
    source = Path(CANARY_PATH)
    if not source.is_file() or shared._sha256(source) != CANARY_SHA256:
        raise ValueError("v2r4d canary data drifted")
    import pyarrow.parquet as pq

    if pq.ParquetFile(source).metadata.num_rows != CANARY_ROWS:
        raise ValueError("v2r4d canary row count drifted")
    checkpoint = shared._validate_hf_checkpoint(
        str(CANDIDATES[6_000]["hf_path"])
    )
    checkpoint_identity = directory_identity(
        checkpoint, logical_path=str(checkpoint)
    )
    if (
        checkpoint_identity["manifest_sha256"]
        != CANDIDATES[6_000]["hf_directory_manifest_sha256"]
    ):
        raise ValueError("v2r4d canary checkpoint drifted")
    run_root = Path(shared.RAW_RL_ROOT) / CANARY_RUN_NAME
    if run_root.exists():
        raise FileExistsError(run_root)
    command = shared.build_train_command(
        hf_checkpoint=str(checkpoint),
        run_name=CANARY_RUN_NAME,
        model_id=shared.MODEL_ID,
        num_rollout=1,
        dynamic_filter=False,
        rollout_seed=CANARY_SEED,
        save_interval=0,
        eval_interval=0,
        wandb_project="chess_interleave_50m",
        wandb_group="v2r4d_runtime_canary",
        deterministic_inference=True,
        rollout_only=True,
        train_file=CANARY_PATH,
        train_file_sha256=CANARY_SHA256,
        data_source_path=shared.STRICT_GATE_DATA_SOURCE_PATH,
        deterministic_seed_by_sample_index=True,
        fault_tolerance=False,
        rollout_health_check_interval=(
            ROUTER_HEALTH_INTERVAL_SECONDS
        ),
        log_passrate=False,
    )
    run_root.mkdir(parents=True, exist_ok=False)
    shared._atomic_json(
        run_root / "_V2R4D_CANARY_INTENT.json",
        {
            "schema": "interleaved-v2r4d-canary-intent-v1",
            "version": VERSION,
            "contract_sha256": contract_sha256,
            "run_name": CANARY_RUN_NAME,
            "command_sha256": hashlib.sha256(
                json.dumps(command, separators=(",", ":")).encode()
            ).hexdigest(),
        },
    )
    provenance = write_run_provenance(
        run_root=run_root,
        identity={
            "kind": "chess_rl_miles_v2r4d_runtime_canary",
            "version": VERSION,
            "contract_sha256": contract_sha256,
            "contract_file_sha256": contract["contract_file_sha256"],
            "candidate": {
                "step": 6_000,
                **CANDIDATES[6_000],
                "directory_identity": checkpoint_identity,
            },
            "canary": dict(contract["canary"]),
            "semantics": dict(SEMANTICS),
            "sources": dict(contract["sources"]),
            "runtime": runtime_identity(image=base.MILES_IMAGE),
        },
        command=command,
    )
    base.ckpt_vol.commit()
    print("[v2r4d-canary] " + " ".join(command), flush=True)
    _run_rollout_command(run_name=CANARY_RUN_NAME, command=command)
    artifact = _validate_canary(
        run_root,
        launch_manifest=str(provenance["launch_manifest"]),
    )
    core = {
        "schema": "interleaved-v2r4d-canary-success-v1",
        "version": VERSION,
        "contract_sha256": contract_sha256,
        "contract_file_sha256": contract["contract_file_sha256"],
        "run_name": CANARY_RUN_NAME,
        "provenance": provenance,
        "artifact": artifact,
        "outcome_logging_blinded": True,
        "fault_tolerance_disabled": True,
        "router_health_checks_suppressed": True,
        "positive_attempt_artifacts_absent": True,
        "reward_metrics_inspected": False,
    }
    success = {
        **core,
        "success_sha256": shared._canonical_json_sha256(core),
    }
    shared._atomic_json(
        run_root / "_V2R4D_CANARY_SUCCESS.json", success
    )
    base.ckpt_vol.commit()
    return success


@app.function(
    gpu=f"{base.GPU_TYPE}:{base.GPUS_PER_NODE}",
    cpu=base.CPU_COUNT,
    memory=shared.SMALL_MODEL_HOST_MEMORY_MB,
    timeout=3 * 60 * 60,
    volumes={
        "/data": base.data_vol,
        "/rl-checkpoints": base.ckpt_vol,
        shared.PRETRAIN_CKPT_ROOT: shared.pretrain_ckpt_vol,
        base.HF_CACHE_DIR: base.hf_cache,
    },
)
def v2r4d_gate_rollout(
    *,
    candidate_step: int,
    batch_label: str,
    contract_sha256: str,
) -> dict[str, object]:
    normalized_batch = str(batch_label).strip().upper()
    if (
        candidate_step not in CANDIDATES
        or normalized_batch not in PROMPT_BATCHES
    ):
        raise ValueError("v2r4d cell is outside the frozen grid")
    base.data_vol.reload()
    base.ckpt_vol.reload()
    shared.pretrain_ckpt_vol.reload()
    contract = _load_contract(contract_sha256)
    prompt_manifest = _load_prompt_manifest()
    canary_marker = (
        Path(shared.RAW_RL_ROOT)
        / CANARY_RUN_NAME
        / "_V2R4D_CANARY_SUCCESS.json"
    )
    if not canary_marker.is_file():
        raise FileNotFoundError(canary_marker)
    canary_success = json.loads(canary_marker.read_text())
    canary_core = {
        key: value
        for key, value in canary_success.items()
        if key != "success_sha256"
    }
    if (
        canary_success.get("version") != VERSION
        or canary_success.get("contract_sha256") != contract_sha256
        or canary_success.get("success_sha256")
        != shared._canonical_json_sha256(canary_core)
    ):
        raise ValueError("v2r4d gate lacks an authenticated canary")
    candidate = dict(CANDIDATES[candidate_step])
    batch = dict(PROMPT_BATCHES[normalized_batch])
    run_name = _run_name(candidate_step, normalized_batch)
    authorized = [
        cell
        for cell in contract["cells"]
        if cell
        == {
            "candidate_step": candidate_step,
            "batch_label": normalized_batch,
            "run_name": run_name,
        }
    ]
    if len(authorized) != 1:
        raise ValueError("v2r4d contract does not authorize this cell")
    checkpoint = shared._validate_hf_checkpoint(
        str(candidate["hf_path"])
    )
    checkpoint_identity = directory_identity(
        checkpoint, logical_path=str(checkpoint)
    )
    if (
        checkpoint_identity["manifest_sha256"]
        != candidate["hf_directory_manifest_sha256"]
    ):
        raise ValueError("v2r4d candidate checkpoint drifted")
    prompt_path = Path(str(batch["path"]))
    if (
        not prompt_path.is_file()
        or shared._sha256(prompt_path) != batch["sha256"]
    ):
        raise ValueError("v2r4d prompt parquet drifted")
    import pyarrow.parquet as pq

    if pq.ParquetFile(prompt_path).metadata.num_rows != batch["rows"]:
        raise ValueError("v2r4d prompt row count drifted")
    batch_manifest = prompt_manifest.get("batches", {}).get(
        normalized_batch
    )
    if not isinstance(batch_manifest, dict):
        raise ValueError("v2r4d prompt batch manifest missing")
    run_root = Path(shared.RAW_RL_ROOT) / run_name
    if run_root.exists():
        raise FileExistsError(run_root)
    command = shared.build_train_command(
        hf_checkpoint=str(checkpoint),
        run_name=run_name,
        model_id=shared.MODEL_ID,
        num_rollout=4,
        dynamic_filter=False,
        rollout_seed=int(batch["rollout_seed"]),
        save_interval=0,
        eval_interval=0,
        wandb_project="chess_interleave_50m",
        wandb_group="v2r4d_production_gate",
        deterministic_inference=True,
        rollout_only=True,
        train_file=str(prompt_path),
        train_file_sha256=str(batch["sha256"]),
        data_source_path=shared.STRICT_GATE_DATA_SOURCE_PATH,
        deterministic_seed_by_sample_index=True,
        fault_tolerance=False,
        rollout_health_check_interval=(
            ROUTER_HEALTH_INTERVAL_SECONDS
        ),
        log_passrate=False,
    )
    run_root.mkdir(parents=True, exist_ok=False)
    shared._atomic_json(
        run_root / "_V2R4D_GATE_INTENT.json",
        {
            "schema": "interleaved-v2r4d-gate-cell-intent-v1",
            "version": VERSION,
            "contract_sha256": contract_sha256,
            "contract_file_sha256": contract["contract_file_sha256"],
            "candidate_step": candidate_step,
            "batch_label": normalized_batch,
            "run_name": run_name,
            "command_sha256": hashlib.sha256(
                json.dumps(command, separators=(",", ":")).encode()
            ).hexdigest(),
        },
    )
    provenance = write_run_provenance(
        run_root=run_root,
        identity={
            "kind": "chess_rl_miles_v2r4d_production_gate_rollout",
            "version": VERSION,
            "contract_sha256": contract_sha256,
            "contract_file_sha256": contract["contract_file_sha256"],
            "contract_path": contract["contract_path"],
            "authorized_cell": authorized[0],
            "candidate": {
                "step": candidate_step,
                **candidate,
                "directory_identity": checkpoint_identity,
            },
            "prompt_batch": {
                "label": normalized_batch,
                **batch,
                "manifest_sha256": (
                    PROMPT_MANIFEST_SHA256
                ),
                "manifest_file_sha256": (
                    PROMPT_MANIFEST_FILE_SHA256
                ),
            },
            "semantics": dict(SEMANTICS),
            "sources": dict(contract["sources"]),
            "runtime": runtime_identity(image=base.MILES_IMAGE),
        },
        command=command,
    )
    base.ckpt_vol.commit()
    print("[v2r4d-gate] " + " ".join(command), flush=True)
    _run_rollout_command(run_name=run_name, command=command)
    validated_artifact_records = shared._validate_v2r4_gate_artifacts(
        run_root=run_root,
        batch_manifest=batch_manifest,
        rollout_seed=int(batch["rollout_seed"]),
    )
    _strict_inventory(
        run_root,
        expected_rollouts=4,
        intent_filename="_V2R4D_GATE_INTENT.json",
        launch_manifest=str(provenance["launch_manifest"]),
    )
    artifact_records = [
        {
            key: value
            for key, value in record.items()
            if key != "bytes"
        }
        for record in validated_artifact_records
    ]
    core = {
        "schema": "interleaved-v2r4d-gate-cell-success-v1",
        "version": VERSION,
        "contract_sha256": contract_sha256,
        "contract_file_sha256": contract["contract_file_sha256"],
        "run_name": run_name,
        "candidate_step": candidate_step,
        "batch_label": normalized_batch,
        "provenance": provenance,
        "prompt_batch_sha256": batch["sha256"],
        "prompt_set_sha256": batch["prompt_set_sha256"],
        "rollout_seed": batch["rollout_seed"],
        "artifact_records": artifact_records,
        "shape_authenticated": True,
        "outcome_logging_blinded": True,
        "positive_attempt_artifacts_absent": True,
        "reward_metrics_inspected": False,
    }
    success = {
        **core,
        "success_sha256": shared._canonical_json_sha256(core),
    }
    shared._atomic_json(
        run_root / "_V2R4D_GATE_SUCCESS.json", success
    )
    base.ckpt_vol.commit()
    return success


def _persist_ledger(
    path: Path,
    ledger: dict[str, object],
) -> None:
    core = {
        key: value for key, value in ledger.items()
        if key != "ledger_sha256"
    }
    ledger["ledger_sha256"] = shared._canonical_json_sha256(core)
    shared._atomic_json(path, ledger)


def _load_authenticated_canary_ledger(
    path: Path,
    *,
    contract_sha256: str,
) -> dict[str, object]:
    ledger = json.loads(path.read_text())
    if not isinstance(ledger, dict):
        raise ValueError("v2r4d canary ledger is not an object")
    core = {
        key: value
        for key, value in ledger.items()
        if key != "ledger_sha256"
    }
    calls = ledger.get("calls")
    preflight = ledger.get("preflight")
    if (
        ledger.get("ledger_sha256")
        != shared._canonical_json_sha256(core)
        or ledger.get("schema")
        != "interleaved-v2r4d-canary-launch-ledger-v1"
        or ledger.get("version") != VERSION
        or ledger.get("state") != "canary_succeeded"
        or ledger.get("contract_sha256") != contract_sha256
        or not isinstance(calls, list)
        or len(calls) != 1
        or not isinstance(calls[0], dict)
        or calls[0].get("run_name") != CANARY_RUN_NAME
        or not isinstance(preflight, dict)
        or preflight.get("schema")
        != "interleaved-v2r4d-gate-preflight-v1"
        or preflight.get("phase") != "canary"
        or preflight.get("contract_sha256") != contract_sha256
        or preflight.get("grid_roots_absent") is not True
        or preflight.get("canary_state") != "absent"
        or preflight.get("prompt_manifest_sha256")
        != PROMPT_MANIFEST_SHA256
        or not re.fullmatch(
            r"fc-[A-Za-z0-9_-]+",
            str(ledger.get("preflight_call_id", "")),
        )
        or not re.fullmatch(
            r"fc-[A-Za-z0-9_-]+",
            str(calls[0].get("function_call_id", "")),
        )
        or ledger.get("preflight_call_id")
        == calls[0].get("function_call_id")
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(ledger.get("success_sha256", "")),
        )
    ):
        raise ValueError("v2r4d canary launch ledger drifted")
    return ledger


@app.local_entrypoint()
def v2r4d_main(
    action: str,
    contract_sha256: str,
    canary_ledger_path: str = CANARY_LEDGER_DEFAULT,
    grid_ledger_path: str = GRID_LEDGER_DEFAULT,
) -> None:
    _require_contract_digest(contract_sha256)
    if action == "launch-canary":
        path = Path(canary_ledger_path).expanduser().resolve()
        if path.exists():
            raise FileExistsError(path)
        ledger: dict[str, object] = {
            "schema": "interleaved-v2r4d-canary-launch-ledger-v1",
            "version": VERSION,
            "state": "preflight_launching",
            "contract_sha256": contract_sha256,
            "calls": [],
        }
        initial = dict(ledger)
        ledger["ledger_sha256"] = shared._canonical_json_sha256(
            initial
        )
        shared._exclusive_json(path, ledger)
        try:
            preflight = v2r4d_preflight.spawn(
                contract_sha256, "canary"
            )
            ledger["preflight_call_id"] = preflight.object_id
            _persist_ledger(path, ledger)
            result = preflight.get()
        except Exception as exc:
            ledger["state"] = "preflight_failed"
            ledger["error"] = f"{type(exc).__name__}: {exc}"
            _persist_ledger(path, ledger)
            raise
        if (
            result.get("grid_roots_absent") is not True
            or result.get("canary_state") != "absent"
        ):
            ledger["state"] = "preflight_failed"
            _persist_ledger(path, ledger)
            raise RuntimeError("v2r4d canary preflight failed")
        ledger["preflight"] = result
        ledger["state"] = "canary_launching"
        _persist_ledger(path, ledger)
        try:
            call = v2r4d_canary.spawn(contract_sha256)
        except Exception as exc:
            ledger["state"] = "canary_launch_failed"
            ledger["error"] = f"{type(exc).__name__}: {exc}"
            _persist_ledger(path, ledger)
            raise
        ledger["calls"] = [
            {
                "run_name": CANARY_RUN_NAME,
                "function_call_id": call.object_id,
            }
        ]
        ledger["state"] = "canary_running"
        _persist_ledger(path, ledger)
        print(f"SPAWNED v2r4d canary: {call.object_id}", flush=True)
        try:
            success = call.get()
        except Exception as exc:
            ledger["state"] = "canary_failed"
            ledger["error"] = f"{type(exc).__name__}: {exc}"
            _persist_ledger(path, ledger)
            raise
        ledger["state"] = "canary_succeeded"
        ledger["success_sha256"] = success["success_sha256"]
        _persist_ledger(path, ledger)
        print(json.dumps(success, sort_keys=True), flush=True)
        return

    if action == "launch-grid":
        canary_path = Path(
            canary_ledger_path
        ).expanduser().resolve()
        if not canary_path.is_file():
            raise FileNotFoundError(canary_path)
        canary_ledger = _load_authenticated_canary_ledger(
            canary_path,
            contract_sha256=contract_sha256,
        )
        path = Path(grid_ledger_path).expanduser().resolve()
        if path.exists():
            raise FileExistsError(path)
        ledger = {
            "schema": "interleaved-v2r4d-gate-launch-ledger-v1",
            "version": VERSION,
            "state": "preflight_launching",
            "contract_sha256": contract_sha256,
            "expected_call_count": 6,
            "canary_ledger_file_sha256": shared._sha256(canary_path),
            "calls": [],
        }
        initial = dict(ledger)
        ledger["ledger_sha256"] = shared._canonical_json_sha256(
            initial
        )
        shared._exclusive_json(path, ledger)
        try:
            preflight = v2r4d_preflight.spawn(
                contract_sha256, "grid"
            )
            ledger["preflight_call_id"] = preflight.object_id
            _persist_ledger(path, ledger)
            result = preflight.get()
        except Exception as exc:
            ledger["state"] = "preflight_failed"
            ledger["error"] = f"{type(exc).__name__}: {exc}"
            _persist_ledger(path, ledger)
            raise
        if (
            result.get("grid_roots_absent") is not True
            or result.get("canary_state") != "success_authenticated"
            or result.get("canary_success_sha256")
            != canary_ledger["success_sha256"]
        ):
            ledger["state"] = "preflight_failed"
            _persist_ledger(path, ledger)
            raise RuntimeError("v2r4d grid preflight failed")
        ledger["preflight"] = result
        ledger["state"] = "launching"
        _persist_ledger(path, ledger)
        calls = ledger["calls"]
        assert isinstance(calls, list)
        try:
            for cell in CELLS:
                call = v2r4d_gate_rollout.spawn(
                    candidate_step=int(cell["candidate_step"]),
                    batch_label=str(cell["batch_label"]),
                    contract_sha256=contract_sha256,
                )
                calls.append(
                    {
                        **dict(cell),
                        "function_call_id": call.object_id,
                    }
                )
                _persist_ledger(path, ledger)
                print(
                    "SPAWNED v2r4d "
                    f"{cell['candidate_step']}/{cell['batch_label']}: "
                    f"{call.object_id}",
                    flush=True,
                )
        except Exception as exc:
            ledger["state"] = "launch_failed"
            ledger["launched_call_count"] = len(calls)
            ledger["error"] = f"{type(exc).__name__}: {exc}"
            _persist_ledger(path, ledger)
            raise
        ledger["state"] = "launched_all"
        _persist_ledger(path, ledger)
        print(json.dumps(ledger, sort_keys=True), flush=True)
        return

    raise ValueError("action must be launch-canary or launch-grid")
