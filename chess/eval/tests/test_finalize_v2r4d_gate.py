from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from Eval import finalize_v2r4d_gate as finalizer
from chess_rl_miles.scripts import modal_v2r4d_gate as launcher


WORKSPACE = Path(__file__).resolve().parents[2]


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _self_hash(value: dict, field: str) -> dict:
    result = dict(value)
    result[field] = finalizer.content_hash(result, field)
    return result


def _source_evidence() -> dict:
    return {
        "plan": {
            "path": str(WORKSPACE / "INTERLEAVED_V2R4D_GATE_AMENDMENT.md"),
            "sha256": finalizer.sha256_file(
                WORKSPACE / "INTERLEAVED_V2R4D_GATE_AMENDMENT.md"
            ),
        },
        "finalizer": {
            "path": str(WORKSPACE / "Eval/finalize_v2r4d_gate.py"),
            "sha256": finalizer.sha256_file(
                WORKSPACE / "Eval/finalize_v2r4d_gate.py"
            ),
        },
        "finalizer_test": {
            "path": str(
                WORKSPACE / "Eval/tests/test_finalize_v2r4d_gate.py"
            ),
            "sha256": finalizer.sha256_file(
                WORKSPACE / "Eval/tests/test_finalize_v2r4d_gate.py"
            ),
        },
        "launcher": {
            "path": str(
                WORKSPACE
                / (
                    "chess-rl-miles/chess_rl_miles/scripts/"
                    "modal_v2r4d_gate.py"
                )
            ),
            "sha256": finalizer.sha256_file(
                WORKSPACE
                / (
                    "chess-rl-miles/chess_rl_miles/scripts/"
                    "modal_v2r4d_gate.py"
                )
            ),
        },
        "source_manifests": {
            "chess_rl_miles": {"manifest_sha256": "a" * 64},
            "miles": {"manifest_sha256": "b" * 64},
        },
    }


def _contract() -> dict:
    source = _source_evidence()
    core = launcher._contract_static()
    core.update(
        {
            "import_probe": {
                "report_path": launcher.IMPORT_PROBE_PATH,
                "report_sha256": "1" * 64,
                "report_file_sha256": "2" * 64,
                "report_bytes": 1,
                "app_id": "ap-PROBE",
                "result": {
                    "schema": "interleaved-v2r4d-import-probe-v1",
                    "version": launcher.VERSION,
                    "launcher_path": "/root/modal_v2r4d_gate.py",
                    "launcher_sha256": source["launcher"]["sha256"],
                    "project_path": "/root/chess-rl-miles",
                    "package_path": (
                        "/root/chess-rl-miles/chess_rl_miles/"
                        "scripts/modal_interleave.py"
                    ),
                    "project_on_sys_path": True,
                    "volumes_mounted": False,
                    "gpu_requested": False,
                },
            },
            "plan": {
                "path": "INTERLEAVED_V2R4D_GATE_AMENDMENT.md",
                "sha256": source["plan"]["sha256"],
            },
            "analysis": {
                "path": "Eval/v2r4c_gate_analysis.py",
                "sha256": finalizer.ANALYZER_SHA256,
                "corrected_dependency": {
                    "path": "Eval/v2r4b_gate_analysis.py",
                    "sha256": finalizer.CORRECTED_ANALYZER_SHA256,
                },
                "base_dependency": {
                    "path": "Eval/v2r4_gate_analysis.py",
                    "sha256": finalizer.BASE_ANALYZER_SHA256,
                },
            },
            "finalizer": {
                "path": "Eval/finalize_v2r4d_gate.py",
                "sha256": source["finalizer"]["sha256"],
                "reused_dependency": {
                    "path": "Eval/finalize_v2r4_gate.py",
                    "sha256": finalizer.FROZEN_EVIDENCE_SOURCE_SHA256,
                },
                "test_source": {
                    "path": "Eval/tests/test_finalize_v2r4d_gate.py",
                    "sha256": source["finalizer_test"]["sha256"],
                },
            },
            "endpoint_evaluators": {
                "pt_b1_b5": finalizer.ENDPOINT_EVALUATOR_SHA256,
                "p2_sft_at_p1": finalizer.P2_EVALUATOR_SHA256,
            },
            "quarantine_report": {
                "path": finalizer.DEFAULT_QUARANTINE_REPORT,
                "report_sha256": finalizer.QUARANTINE_REPORT_SHA256,
                "file_sha256": finalizer.QUARANTINE_REPORT_FILE_SHA256,
                "bytes": 17_661,
            },
            "sources": source["source_manifests"],
            "construction": {
                "builder_path": "build_v2r4d_runtime_contract.py",
                "builder_sha256": "c" * 64,
                "validated_test_count": 1,
                "known_local_test_exclusion": "test",
            },
        }
    )
    return {
        **core,
        "contract_sha256": finalizer.content_hash(
            core, "contract_sha256"
        ),
    }


def _validate_synthetic_contract(
    monkeypatch: pytest.MonkeyPatch, contract: dict
):
    raw = _json_bytes(contract)
    monkeypatch.setattr(
        finalizer, "CONTRACT_SHA256", contract["contract_sha256"]
    )
    monkeypatch.setattr(
        finalizer,
        "CONTRACT_FILE_SHA256",
        hashlib.sha256(raw).hexdigest(),
    )
    return finalizer.validate_runtime_contract(raw, _source_evidence())


def test_contract_exactly_binds_finalizer_dependency_and_test(
    monkeypatch: pytest.MonkeyPatch,
):
    contract = _contract()
    observed, evidence = _validate_synthetic_contract(
        monkeypatch, contract
    )
    assert observed["finalizer"] == contract["finalizer"]
    assert evidence["analysis_sha256"] == finalizer.ANALYZER_SHA256

    for path, replacement in (
        (("path",), "Eval/not-the-finalizer.py"),
        (("sha256",), "d" * 64),
        (("reused_dependency", "sha256"), "e" * 64),
        (("test_source", "sha256"), "f" * 64),
    ):
        tampered = copy.deepcopy(contract)
        target = tampered["finalizer"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        core = {
            key: value
            for key, value in tampered.items()
            if key != "contract_sha256"
        }
        tampered["contract_sha256"] = finalizer.content_hash(
            core, "contract_sha256"
        )
        raw = _json_bytes(tampered)
        monkeypatch.setattr(
            finalizer, "CONTRACT_SHA256", tampered["contract_sha256"]
        )
        monkeypatch.setattr(
            finalizer,
            "CONTRACT_FILE_SHA256",
            hashlib.sha256(raw).hexdigest(),
        )
        with pytest.raises(ValueError, match="finalizer identity"):
            finalizer.validate_runtime_contract(raw, _source_evidence())


def test_contract_exactly_binds_full_analysis_dependency_chain(
    monkeypatch: pytest.MonkeyPatch,
):
    contract = _contract()
    observed, evidence = _validate_synthetic_contract(
        monkeypatch, contract
    )
    assert observed["analysis"] == contract["analysis"]
    assert evidence["analysis_sha256"] == finalizer.ANALYZER_SHA256

    for path, replacement in (
        (("corrected_dependency", "path"), "Eval/alternate_corrected.py"),
        (("corrected_dependency", "sha256"), "d" * 64),
        (("base_dependency", "path"), "Eval/alternate_base.py"),
        (("base_dependency", "sha256"), "e" * 64),
    ):
        tampered = copy.deepcopy(contract)
        target = tampered["analysis"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        core = {
            key: value
            for key, value in tampered.items()
            if key != "contract_sha256"
        }
        tampered["contract_sha256"] = finalizer.content_hash(
            core, "contract_sha256"
        )
        raw = _json_bytes(tampered)
        monkeypatch.setattr(
            finalizer, "CONTRACT_SHA256", tampered["contract_sha256"]
        )
        monkeypatch.setattr(
            finalizer,
            "CONTRACT_FILE_SHA256",
            hashlib.sha256(raw).hexdigest(),
        )
        with pytest.raises(ValueError, match="analyzer identity"):
            finalizer.validate_runtime_contract(raw, _source_evidence())


def test_local_source_validation_rejects_tampered_base_analyzer(
    tmp_path: Path,
):
    eval_dir = tmp_path / "Eval"
    eval_dir.mkdir()
    for name in ("v2r4c_gate_analysis.py", "v2r4b_gate_analysis.py"):
        (eval_dir / name).write_bytes((WORKSPACE / "Eval" / name).read_bytes())
    base = WORKSPACE / "Eval/v2r4_gate_analysis.py"
    (eval_dir / base.name).write_bytes(base.read_bytes() + b"\n# tampered\n")

    with pytest.raises(ValueError, match="base_analyzer source drifted"):
        finalizer.validate_local_sources(tmp_path)


def test_fully_self_hashed_alternate_prompt_contract_cannot_replace_binding(
    monkeypatch: pytest.MonkeyPatch,
):
    bound = _contract()
    bound_raw = _json_bytes(bound)
    monkeypatch.setattr(
        finalizer, "CONTRACT_SHA256", bound["contract_sha256"]
    )
    monkeypatch.setattr(
        finalizer,
        "CONTRACT_FILE_SHA256",
        hashlib.sha256(bound_raw).hexdigest(),
    )
    alternate = copy.deepcopy(bound)
    alternate["prompt_batches"]["A"]["path"] = "/data/attacker.parquet"
    alternate["prompt_batches"]["A"]["sha256"] = "e" * 64
    core = {
        key: value
        for key, value in alternate.items()
        if key != "contract_sha256"
    }
    alternate["contract_sha256"] = finalizer.content_hash(
        core, "contract_sha256"
    )
    with pytest.raises(ValueError, match="differs from its binding"):
        finalizer.validate_runtime_contract(
            _json_bytes(alternate), _source_evidence()
        )


def test_predecessor_incident_is_semantically_bound(
    monkeypatch: pytest.MonkeyPatch,
):
    raw = (
        WORKSPACE / "INTERLEAVED_V2R4C_PREFLIGHT_INCIDENT_REPORT.json"
    ).read_bytes()
    incident, evidence = finalizer.validate_predecessor_incident(
        raw, _contract()
    )
    assert incident["function_call"]["call_graph_children"] == []
    assert incident["launcher_ledger"]["calls"] == []
    assert evidence["report_sha256"] == finalizer.INCIDENT_REPORT_SHA256
    assert evidence["outcome_exposure"] is False

    tampered = copy.deepcopy(incident)
    tampered["function_call"]["call_graph_children"] = ["fc-CHILD"]
    tampered = _self_hash(tampered, "report_sha256")
    tampered_raw = _json_bytes(tampered)
    monkeypatch.setattr(
        finalizer,
        "INCIDENT_REPORT_SHA256",
        tampered["report_sha256"],
    )
    monkeypatch.setattr(
        finalizer,
        "INCIDENT_REPORT_FILE_SHA256",
        hashlib.sha256(tampered_raw).hexdigest(),
    )
    with pytest.raises(ValueError, match="incident evidence drifted"):
        finalizer.validate_predecessor_incident(
            tampered_raw, _contract()
        )


def test_import_probe_report_is_bound_to_contract_and_launcher():
    contract = _contract()
    core = {
        "schema": "interleaved-v2r4d-import-probe-report-v1",
        "app_id": contract["import_probe"]["app_id"],
        "app_url": "https://modal.com/apps/example",
        "observed_at": "2026-07-30T16:00:00-04:00",
        "result": contract["import_probe"]["result"],
    }
    report = {
        **core,
        "report_sha256": finalizer.content_hash(
            core, "report_sha256"
        ),
    }
    raw = _json_bytes(report)
    contract["import_probe"]["report_sha256"] = report["report_sha256"]
    contract["import_probe"]["report_file_sha256"] = hashlib.sha256(
        raw
    ).hexdigest()
    contract["import_probe"]["report_bytes"] = len(raw)
    observed, evidence = finalizer.validate_import_probe_report(
        raw, contract
    )
    assert observed["result"]["gpu_requested"] is False
    assert evidence["app_id"] == "ap-PROBE"

    tampered = copy.deepcopy(report)
    tampered["result"]["project_on_sys_path"] = False
    tampered = _self_hash(tampered, "report_sha256")
    with pytest.raises(ValueError, match="report file drifted"):
        finalizer.validate_import_probe_report(
            _json_bytes(tampered), contract
        )


def _preflight(contract: dict, phase: str, success_sha: str | None) -> dict:
    bound = contract["quarantined_v2r4a"]
    return {
        "schema": finalizer.PREFLIGHT_SCHEMA,
        "version": finalizer.CONTRACT_VERSION,
        "phase": phase,
        "contract_sha256": contract["contract_sha256"],
        "contract_file_sha256": "d" * 64,
        "grid_roots_absent": True,
        "canary_state": (
            "absent" if phase == "canary" else "success_authenticated"
        ),
        "canary_success_sha256": success_sha,
        "predecessor_incident_sha256": (
            finalizer.INCIDENT_REPORT_SHA256
        ),
        "import_probe_report_sha256": contract["import_probe"][
            "report_sha256"
        ],
        "predecessor_roots_absent": True,
        "predecessor_root_count": len(finalizer.PREDECESSOR_ROOTS),
        "predecessor_results_absent": True,
        "predecessor_result_count": 0,
        "quarantine": {
            "grid": [
                {
                    **dict(cell),
                    "success_marker_present": (
                        cell["terminal_state"]
                        == "success_quarantined_blinding_violation"
                    ),
                    "root_identity_authenticated": True,
                }
                for cell in bound["cells"]
            ],
            "report_sha256": finalizer.QUARANTINE_REPORT_SHA256,
            "report_file_sha256": (
                finalizer.QUARANTINE_REPORT_FILE_SHA256
            ),
            "entire_grid_quarantined": True,
            "authorization_absent": True,
        },
        "prompt_manifest_sha256": finalizer.PROMPT_MANIFEST_SHA256,
        "fresh_prompt_intersections_authenticated": True,
        "ray_worker_environment": {
            "seed_mode": "sample-index",
            "artifact_root": (
                "/rl-checkpoints/chess-rl-miles-interleave/"
                "v2r4d-ray-env-preflight"
            ),
            "gpu_allocated": False,
        },
        "grid_roots": [
            "/rl-checkpoints/chess-rl-miles-interleave/"
            + cell["run_name"]
            for cell in contract["cells"]
        ],
    }


def test_canary_and_grid_ledgers_are_self_hashed_and_exact():
    contract = _contract()
    success_sha = "9" * 64
    canary = _self_hash(
        {
            "schema": finalizer.CANARY_LEDGER_SCHEMA,
            "version": finalizer.CONTRACT_VERSION,
            "state": "canary_succeeded",
            "contract_sha256": contract["contract_sha256"],
            "calls": [
                {
                    "run_name": finalizer.CANARY["run_name"],
                    "function_call_id": "fc-CANARY",
                }
            ],
            "preflight_call_id": "fc-CANARYPREFLIGHT",
            "preflight": _preflight(contract, "canary", None),
            "success_sha256": success_sha,
        },
        "ledger_sha256",
    )
    canary_raw = _json_bytes(canary)
    _, canary_evidence = finalizer.validate_canary_ledger(
        canary_raw,
        contract=contract,
        contract_file_sha256="d" * 64,
    )
    grid = _self_hash(
        {
            "schema": finalizer.GRID_LEDGER_SCHEMA,
            "version": finalizer.CONTRACT_VERSION,
            "state": "launched_all",
            "contract_sha256": contract["contract_sha256"],
            "expected_call_count": 6,
            "canary_ledger_file_sha256": hashlib.sha256(
                canary_raw
            ).hexdigest(),
            "preflight_call_id": "fc-GRIDPREFLIGHT",
            "preflight": _preflight(contract, "grid", success_sha),
            "calls": [
                {**cell, "function_call_id": f"fc-GRID{index}"}
                for index, cell in enumerate(contract["cells"])
            ],
        },
        "ledger_sha256",
    )
    _, grid_evidence = finalizer.validate_grid_ledger(
        _json_bytes(grid),
        contract=contract,
        contract_file_sha256="d" * 64,
        canary_ledger_file_sha256=hashlib.sha256(
            canary_raw
        ).hexdigest(),
        canary_success_sha256=canary_evidence["success_sha256"],
    )
    assert len(grid_evidence["call_ids"]) == 6

    grid["calls"][1]["function_call_id"] = grid["calls"][0][
        "function_call_id"
    ]
    grid = _self_hash(grid, "ledger_sha256")
    with pytest.raises(ValueError, match="not unique"):
        finalizer.validate_grid_ledger(
            _json_bytes(grid),
            contract=contract,
            contract_file_sha256="d" * 64,
            canary_ledger_file_sha256=hashlib.sha256(
                canary_raw
            ).hexdigest(),
            canary_success_sha256=success_sha,
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("predecessor_incident_sha256", "0" * 64),
        ("import_probe_report_sha256", "0" * 64),
        ("predecessor_roots_absent", False),
        ("predecessor_root_count", 6),
        ("predecessor_results_absent", False),
        ("predecessor_result_count", 1),
    ],
)
def test_real_launcher_preflight_proofs_are_exact_and_fail_closed(
    field: str,
    bad_value: object,
):
    contract = _contract()
    preflight = _preflight(contract, "canary", None)
    finalizer._validate_preflight(
        preflight,
        phase="canary",
        contract=contract,
        contract_file_sha256="d" * 64,
    )
    preflight[field] = bad_value
    with pytest.raises(ValueError, match="preflight identity drifted"):
        finalizer._validate_preflight(
            preflight,
            phase="canary",
            contract=contract,
            contract_file_sha256="d" * 64,
        )


def test_recursive_regular_file_allowlist_rejects_any_extra_file():
    root = "chess-rl-miles-interleave/run"
    expected = {"intent.json", "rollouts/training/rollout_0.jsonl"}
    entries = [
        {"path": f"/{root}/{path}", "type": "file", "bytes": 1}
        for path in expected
    ]
    observed = finalizer._list_exact_files(
        root=root,
        expected_files=expected,
        list_entries=lambda _: entries,
    )
    assert set(observed) == expected
    entries.append(
        {
            "path": f"/{root}/logs/reward-summary.json",
            "type": "file",
            "bytes": 1,
        }
    )
    with pytest.raises(ValueError, match="allowlist drifted"):
        finalizer._list_exact_files(
            root=root,
            expected_files=expected,
            list_entries=lambda _: entries,
        )


def _row(
    *,
    prompt: str,
    rollout_id: int,
    group_index: int,
    sibling_index: int,
    seed: int,
) -> dict:
    sample_index = group_index * 8 + sibling_index
    positive = group_index < 20 and sibling_index == 0
    return {
        "input": prompt,
        "FEN": f"fen-{prompt}",
        "PuzzleId": f"id-{prompt}",
        "ground_truth": "e2e4",
        "rollout_id": rollout_id,
        "group_index": group_index,
        "sample_index": sample_index,
        "sampling_seed_sibling_index": sibling_index,
        "sampling_seed": seed + sample_index,
        "status": "completed",
        "score": float(positive),
        "output": "reason </T> text <call_env> e2e4",
        "extracted_moves": "e2e4",
        "metadata": {
            "sampling_seed_sibling_index": sibling_index,
            "sampling_seed": seed + sample_index,
            "sampling_seed_mode": "sample-index",
        },
    }


def test_grid_cell_binds_marker_provenance_raw_shape_and_no_bytes_field():
    contract = _contract()
    step, batch = 6_000, "A"
    run_name = f"v2r4d-gate-w190-s{step}-batch-a"
    seed = 123_000
    files: dict[str, bytes] = {}
    records: list[dict] = []
    quarters: list[dict] = []
    all_fingerprints: list[str] = []
    for rollout_id in range(4):
        rows: list[dict] = []
        prompts: list[str] = []
        for local_group in range(256):
            group = rollout_id * 256 + local_group
            prompt = f"fresh-{group}"
            group_rows = [
                _row(
                    prompt=prompt,
                    rollout_id=rollout_id,
                    group_index=group,
                    sibling_index=sibling,
                    seed=seed,
                )
                for sibling in range(8)
            ]
            rows.extend(group_rows)
            prompts.append(finalizer._prompt_fingerprint(group_rows[0]))
        raw = b"".join(
            json.dumps(row, separators=(",", ":")).encode() + b"\n"
            for row in rows
        )
        relative = (
            f"{finalizer.RAW_VOLUME_ROOT}/{run_name}/rollouts/training/"
            f"rollout_{rollout_id}.jsonl"
        )
        files[relative] = raw
        prompt_hash = hashlib.sha256(
            finalizer.canonical_json(prompts)
        ).hexdigest()
        records.append(
            {
                "rollout_id": rollout_id,
                "path": f"/rl-checkpoints/{relative}",
                "rows": 2_048,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "prompt_order_sha256": prompt_hash,
            }
        )
        quarters.append(
            {
                "rollout_id": rollout_id,
                "prompt_count": 256,
                "ordered_prompt_fingerprints": prompts,
                "prompt_order_sha256": prompt_hash,
            }
        )
        all_fingerprints.extend(prompts)
    contract["prompt_batches"][batch] = {
        **contract["prompt_batches"][batch],
        "rollout_seed": seed,
        "prompt_set_sha256": hashlib.sha256(
            finalizer.canonical_json(sorted(all_fingerprints))
        ).hexdigest(),
        "epoch0_prompt_order_sha256": hashlib.sha256(
            finalizer.canonical_json(all_fingerprints)
        ).hexdigest(),
    }
    prompt_manifest = {
        "batches": {batch: {"rollout_quarters": quarters}}
    }
    command = [
        "python",
        "run.py",
        "--debug-rollout-only",
        "--no-log-passrate",
        "--no-use-fault-tolerance",
        "--no-dynamic-filter",
    ]
    installed = {"torch": "test"}
    identity = {
        "kind": "chess_rl_miles_v2r4d_production_gate_rollout",
        "version": finalizer.CONTRACT_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "contract_file_sha256": "d" * 64,
        "contract_path": "/data/runtime_contract.json",
        "authorized_cell": {
            "candidate_step": step,
            "batch_label": batch,
            "run_name": run_name,
        },
        "candidate": {
            "step": step,
            **contract["candidates"][str(step)],
            "directory_identity": {
                "manifest_sha256": contract["candidates"][str(step)][
                    "hf_directory_manifest_sha256"
                ]
            },
        },
        "prompt_batch": {
            "label": batch,
            **contract["prompt_batches"][batch],
            "manifest_sha256": finalizer.PROMPT_MANIFEST_SHA256,
            "manifest_file_sha256": (
                finalizer.PROMPT_MANIFEST_FILE_SHA256
            ),
        },
        "semantics": contract["semantics"],
        "sources": contract["sources"],
        "runtime": {
            "image": contract["runtime"]["miles_image"],
            "installed_packages": installed,
            "installed_packages_sha256": hashlib.sha256(
                json.dumps(
                    installed,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode()
            ).hexdigest(),
        },
    }
    identity_sha = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    command_sha = hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode()
    ).hexdigest()
    root = f"{finalizer.RAW_VOLUME_ROOT}/{run_name}"
    launch_relative = f"provenance/launch_{command_sha[:16]}.json"
    files[f"{root}/_V2R4D_GATE_INTENT.json"] = _json_bytes(
        {
            "schema": "interleaved-v2r4d-gate-cell-intent-v1",
            "version": finalizer.CONTRACT_VERSION,
            "contract_sha256": contract["contract_sha256"],
            "contract_file_sha256": "d" * 64,
            "candidate_step": step,
            "batch_label": batch,
            "run_name": run_name,
            "command_sha256": command_sha,
        }
    )
    files[f"{root}/run_provenance.json"] = _json_bytes(
        {
            "schema_version": 1,
            "created_at": "test",
            "identity": identity,
            "identity_sha256": identity_sha,
            "initial_command": command,
            "initial_command_sha256": command_sha,
        }
    )
    files[f"{root}/{launch_relative}"] = _json_bytes(
        {
            "schema_version": 1,
            "created_at": "test",
            "identity_sha256": identity_sha,
            "command": command,
            "command_sha256": command_sha,
        }
    )
    marker = _self_hash(
        {
            "schema": finalizer.GRID_SUCCESS_SCHEMA,
            "version": finalizer.CONTRACT_VERSION,
            "contract_sha256": contract["contract_sha256"],
            "contract_file_sha256": "d" * 64,
            "run_name": run_name,
            "candidate_step": step,
            "batch_label": batch,
            "provenance": {
                "root_manifest": (
                    f"/rl-checkpoints/{root}/run_provenance.json"
                ),
                "launch_manifest": (
                    f"/rl-checkpoints/{root}/{launch_relative}"
                ),
                "identity_sha256": identity_sha,
                "command_sha256": command_sha,
            },
            "prompt_batch_sha256": contract["prompt_batches"][batch][
                "sha256"
            ],
            "prompt_set_sha256": contract["prompt_batches"][batch][
                "prompt_set_sha256"
            ],
            "rollout_seed": seed,
            "artifact_records": records,
            "shape_authenticated": True,
            "outcome_logging_blinded": True,
            "positive_attempt_artifacts_absent": True,
            "reward_metrics_inspected": False,
        },
        "success_sha256",
    )
    files[f"{root}/_V2R4D_GATE_SUCCESS.json"] = _json_bytes(marker)
    entries = [
        {"path": "/" + path, "type": "file", "bytes": len(raw)}
        for path, raw in files.items()
        if path.startswith(root + "/")
    ]
    cell, evidence = finalizer.validate_grid_cell(
        record={
            "candidate_step": step,
            "batch_label": batch,
            "run_name": run_name,
            "function_call_id": "fc-GRIDCELL",
        },
        function_result=marker,
        contract=contract,
        contract_file_sha256="d" * 64,
        prompt_manifest=prompt_manifest,
        read_checkpoint_file=files.__getitem__,
        list_checkpoint_entries=lambda _: entries,
    )
    assert cell["rows"] == 8_192
    assert cell["solve_at_8_groups"] == 20
    assert evidence["recursive_regular_file_allowlist_authenticated"] is True
    assert all("bytes" not in record for record in marker["artifact_records"])

    tampered = copy.deepcopy(marker)
    tampered["artifact_records"][0]["bytes"] = len(
        files[records[0]["path"].removeprefix("/rl-checkpoints/")]
    )
    tampered = _self_hash(tampered, "success_sha256")
    files[f"{root}/_V2R4D_GATE_SUCCESS.json"] = _json_bytes(tampered)
    with pytest.raises(ValueError, match="artifact record fields"):
        finalizer.validate_grid_cell(
            record={
                "candidate_step": step,
                "batch_label": batch,
                "run_name": run_name,
                "function_call_id": "fc-GRIDCELL",
            },
            function_result=tampered,
            contract=contract,
            contract_file_sha256="d" * 64,
            prompt_manifest=prompt_manifest,
            read_checkpoint_file=files.__getitem__,
            list_checkpoint_entries=lambda _: entries,
        )

    # Change one complete sibling group's prompt identity, then recompute the
    # raw artifact hash and every enclosing marker hash.  Byte-level
    # authentication alone is insufficient: the frozen prompt order must
    # still reject this internally consistent alternate cell.
    alternate_rows = [
        json.loads(line)
        for line in files[
            f"{root}/rollouts/training/rollout_0.jsonl"
        ].splitlines()
    ]
    for row in alternate_rows[:8]:
        row["input"] = "arbitrary-unbound-prompt"
        row["FEN"] = "arbitrary-unbound-fen"
        row["PuzzleId"] = "arbitrary-unbound-id"
    alternate_raw = b"".join(
        json.dumps(row, separators=(",", ":")).encode() + b"\n"
        for row in alternate_rows
    )
    alternate_marker = copy.deepcopy(marker)
    alternate_marker["artifact_records"][0]["sha256"] = hashlib.sha256(
        alternate_raw
    ).hexdigest()
    alternate_marker = _self_hash(
        alternate_marker, "success_sha256"
    )
    files[f"{root}/rollouts/training/rollout_0.jsonl"] = alternate_raw
    files[f"{root}/_V2R4D_GATE_SUCCESS.json"] = _json_bytes(
        alternate_marker
    )
    alternate_entries = [
        {"path": "/" + path, "type": "file", "bytes": len(raw)}
        for path, raw in files.items()
        if path.startswith(root + "/")
    ]
    with pytest.raises(ValueError, match="prompt order drifted"):
        finalizer.validate_grid_cell(
            record={
                "candidate_step": step,
                "batch_label": batch,
                "run_name": run_name,
                "function_call_id": "fc-GRIDCELL",
            },
            function_result=alternate_marker,
            contract=contract,
            contract_file_sha256="d" * 64,
            prompt_manifest=prompt_manifest,
            read_checkpoint_file=files.__getitem__,
            list_checkpoint_entries=lambda _: alternate_entries,
        )


def test_exact_quarantine_report_is_authenticated():
    raw = (WORKSPACE / finalizer.DEFAULT_QUARANTINE_REPORT).read_bytes()
    report, evidence = finalizer.validate_quarantine_report(
        raw, _contract()
    )
    assert report["terminal_barrier"]["ordered_status_vector"] == "S,F,F,S,F,S"
    assert evidence["report_sha256"] == finalizer.QUARANTINE_REPORT_SHA256
    with pytest.raises(ValueError, match="file hash"):
        finalizer.validate_quarantine_report(raw + b"\n", _contract())


def test_collect_has_hard_terminal_barrier_before_grid_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    contract = _contract()
    grid_ids = [f"fc-GRID{index}" for index in range(6)]
    calls_seen: list[str] = []
    jsonl_reads: list[str] = []

    class Backend:
        def read_volume_file(self, volume: str, path: str) -> bytes:
            if "rollout_" in path and path.endswith(".jsonl"):
                assert calls_seen[-6:] == grid_ids
                jsonl_reads.append(path)
            return b"{}"

        def list_volume_entries(self, volume: str, root: str):
            return []

        def function_result(self, call_id: str) -> dict:
            calls_seen.append(call_id)
            if call_id == "fc-CANARYPREFLIGHT":
                return {"kind": "canary"}
            if call_id == "fc-GRIDPREFLIGHT":
                return {"kind": "grid"}
            return {"call_id": call_id}

    for name in (
        "canary.json",
        "grid.json",
        "endpoint.json",
        "p2.json",
        "quarantine.json",
    ):
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(
        finalizer, "validate_local_sources", lambda workspace: {"ok": True}
    )
    monkeypatch.setattr(
        finalizer,
        "validate_runtime_contract",
        lambda raw, source: (
            contract,
            {"file_sha256": "d" * 64},
        ),
    )
    monkeypatch.setattr(
        finalizer,
        "validate_predecessor_incident",
        lambda raw, value: ({}, {"ok": True}),
    )
    monkeypatch.setattr(
        finalizer,
        "validate_import_probe_report",
        lambda raw, value: ({}, {"ok": True}),
    )
    monkeypatch.setattr(
        finalizer,
        "validate_prompt_manifest",
        lambda raw, value: ({"batches": {}}, {"ok": True}),
    )
    monkeypatch.setattr(
        finalizer, "authenticate_prompt_files", lambda **kwargs: []
    )
    canary_ledger = {
        "preflight": {"kind": "canary"},
        "calls": [{"function_call_id": "fc-CANARY"}],
    }
    monkeypatch.setattr(
        finalizer,
        "validate_canary_ledger",
        lambda *args, **kwargs: (
            canary_ledger,
            {
                "preflight_call_id": "fc-CANARYPREFLIGHT",
                "function_call_id": "fc-CANARY",
                "success_sha256": "a" * 64,
                "file_sha256": "1" * 64,
            },
        ),
    )
    grid_ledger = {
        "preflight": {"kind": "grid"},
        "calls": [
            {**cell, "function_call_id": grid_ids[index]}
            for index, cell in enumerate(contract["cells"])
        ],
    }
    monkeypatch.setattr(
        finalizer,
        "validate_grid_ledger",
        lambda *args, **kwargs: (
            grid_ledger,
            {
                "preflight_call_id": "fc-GRIDPREFLIGHT",
                "call_ids": grid_ids,
            },
        ),
    )
    monkeypatch.setattr(
        finalizer,
        "validate_endpoint_ledger",
        lambda raw, value: ({"calls": []}, {"ok": True}),
    )
    monkeypatch.setattr(
        finalizer,
        "validate_p2_ledger",
        lambda raw, value: ({"calls": []}, {"ok": True}),
    )
    monkeypatch.setattr(
        finalizer,
        "validate_quarantine_report",
        lambda raw, value: ({}, {"ok": True}),
    )

    def canary_result(**kwargs):
        kwargs["read_checkpoint_file"](
            "chess-rl-miles-interleave/canary/rollouts/training/"
            "rollout_0.jsonl"
        )
        return {"success_sha256": "a" * 64}

    monkeypatch.setattr(finalizer, "validate_canary_result", canary_result)
    monkeypatch.setattr(
        finalizer, "authenticate_quarantine_roots", lambda **kwargs: []
    )

    def grid_cell(**kwargs):
        kwargs["read_checkpoint_file"](
            "chess-rl-miles-interleave/grid/rollouts/training/"
            "rollout_0.jsonl"
        )
        return {"cell": True}, {"evidence": True}

    monkeypatch.setattr(finalizer, "validate_grid_cell", grid_cell)
    monkeypatch.setattr(
        finalizer,
        "build_final_report",
        lambda **kwargs: {
            "report_sha256": "f" * 64,
            "barrier": kwargs["barrier_evidence"],
        },
    )
    report = finalizer.collect_and_finalize(
        workspace=WORKSPACE,
        canary_ledger_path=tmp_path / "canary.json",
        grid_ledger_path=tmp_path / "grid.json",
        endpoint_ledger_path=tmp_path / "endpoint.json",
        p2_ledger_path=tmp_path / "p2.json",
        quarantine_report_path=tmp_path / "quarantine.json",
        backend=Backend(),
    )
    assert calls_seen[:3] == [
        "fc-CANARYPREFLIGHT",
        "fc-CANARY",
        "fc-GRIDPREFLIGHT",
    ]
    assert calls_seen[3:9] == grid_ids
    assert jsonl_reads
    assert report["barrier"][
        "all_six_terminal_before_first_grid_jsonl_read"
    ] is True


def test_failed_sixth_call_reads_zero_grid_jsonls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Reuse the barrier test setup, but prove a terminal failure exits before
    # any validator gets a chance to read raw outcomes.
    contract = _contract()
    grid_ids = [f"fc-GRID{index}" for index in range(6)]
    raw_reads: list[str] = []

    class Backend:
        def read_volume_file(self, volume: str, path: str) -> bytes:
            if path.endswith(".jsonl"):
                raw_reads.append(path)
            return b"{}"

        def list_volume_entries(self, volume: str, root: str):
            return []

        def function_result(self, call_id: str) -> dict:
            if call_id == grid_ids[-1]:
                raise RuntimeError("sixth cell failed")
            if call_id in {"fc-CP", "fc-GP"}:
                return {}
            return {"call_id": call_id}

    for name in ("canary", "grid", "endpoint", "p2", "quarantine"):
        (tmp_path / f"{name}.json").write_text("{}")
    monkeypatch.setattr(
        finalizer, "validate_local_sources", lambda workspace: {}
    )
    monkeypatch.setattr(
        finalizer,
        "validate_runtime_contract",
        lambda *args: (contract, {"file_sha256": "d" * 64}),
    )
    monkeypatch.setattr(
        finalizer,
        "validate_predecessor_incident",
        lambda *args: ({}, {}),
    )
    monkeypatch.setattr(
        finalizer,
        "validate_import_probe_report",
        lambda *args: ({}, {}),
    )
    monkeypatch.setattr(
        finalizer,
        "validate_prompt_manifest",
        lambda *args: ({}, {}),
    )
    monkeypatch.setattr(
        finalizer, "authenticate_prompt_files", lambda **kwargs: []
    )
    monkeypatch.setattr(
        finalizer,
        "validate_canary_ledger",
        lambda *args, **kwargs: (
            {"preflight": {}, "calls": [{}]},
            {
                "preflight_call_id": "fc-CP",
                "function_call_id": "fc-C",
                "success_sha256": "a" * 64,
            },
        ),
    )
    monkeypatch.setattr(
        finalizer,
        "validate_grid_ledger",
        lambda *args, **kwargs: (
            {
                "preflight": {},
                "calls": [
                    {**cell, "function_call_id": grid_ids[index]}
                    for index, cell in enumerate(contract["cells"])
                ],
            },
            {
                "preflight_call_id": "fc-GP",
                "call_ids": grid_ids,
            },
        ),
    )
    monkeypatch.setattr(
        finalizer,
        "validate_endpoint_ledger",
        lambda *args: ({"calls": []}, {}),
    )
    monkeypatch.setattr(
        finalizer,
        "validate_p2_ledger",
        lambda *args: ({"calls": []}, {}),
    )
    monkeypatch.setattr(
        finalizer,
        "validate_quarantine_report",
        lambda *args: ({}, {}),
    )
    with pytest.raises(RuntimeError, match="sixth cell failed"):
        finalizer.collect_and_finalize(
            workspace=WORKSPACE,
            canary_ledger_path=tmp_path / "canary.json",
            grid_ledger_path=tmp_path / "grid.json",
            endpoint_ledger_path=tmp_path / "endpoint.json",
            p2_ledger_path=tmp_path / "p2.json",
            quarantine_report_path=tmp_path / "quarantine.json",
            backend=Backend(),
        )
    assert raw_reads == []
