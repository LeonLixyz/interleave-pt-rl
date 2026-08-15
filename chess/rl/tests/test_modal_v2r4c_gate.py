from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from chess_rl_miles.scripts import modal_v2r4c_gate as gate


def _function_source(name: str) -> str:
    source = Path(gate.__file__).read_text()
    tree = ast.parse(source)
    node = next(
        value
        for value in tree.body
        if isinstance(value, ast.FunctionDef) and value.name == name
    )
    rendered = ast.get_source_segment(source, node)
    assert rendered is not None
    return rendered


def test_contract_is_fresh_uniform_six_cell_grid():
    contract = gate._contract_static()

    assert contract["version"] == "v2r4c_production_gate_20260730"
    assert contract["cells"] == [
        {
            "candidate_step": step,
            "batch_label": batch,
            "run_name": (
                f"v2r4c-gate-w190-s{step}-batch-{batch.lower()}"
            ),
        }
        for step in (6_000, 8_000, 9_920)
        for batch in ("A", "B")
    ]
    prompt_manifest = contract["prompt_manifest"]
    assert prompt_manifest["manifest_sha256"] == (
        "83f4718b829b955cb000908c2ecbb9052883d14404114cbca4ecd42988659056"
    )
    assert set(prompt_manifest["intersections"].values()) == {0}
    assert contract["prompt_batches"]["A"]["prompt_set_sha256"] != (
        contract["prompt_batches"]["B"]["prompt_set_sha256"]
    )
    assert contract["canary"]["prompt_set_sha256"] not in {
        value["prompt_set_sha256"]
        for value in contract["prompt_batches"].values()
    }


def test_contract_static_is_exactly_json_round_trip_stable():
    contract = gate._contract_static()

    assert json.loads(json.dumps(contract)) == contract


def test_quarantine_disclosure_is_exact_and_time_separated():
    quarantine = gate._contract_static()["quarantined_v2r4a"]

    assert quarantine["disposition"] == (
        "entire_grid_quarantined_no_authorization"
    )
    assert quarantine["report_sha256"] == (
        "576bf2fb346666f8b9da2d3df5563d9e29e65b60ce1a66c07b74bf6318428947"
    )
    assert quarantine["report_file_sha256"] == (
        "b800d3f8289cf1e4d2ef1320efb4185ab62761c576f3dd05b863d11c8d694970"
    )
    assert [
        value["terminal_state"] for value in quarantine["cells"]
    ] == [
        "success_quarantined_blinding_violation",
        "watchdog_failure",
        "watchdog_failure",
        "success_quarantined_blinding_violation",
        "watchdog_failure",
        "success_quarantined_blinding_violation",
    ]
    prebarrier = quarantine["prebarrier_disclosure"]
    assert prebarrier["aggregate_or_prompt_level_outcomes_inspected"] is False
    assert prebarrier["rollout_jsonl_contents_manually_inspected"] is False
    assert list(prebarrier["individual_reward_log_exposures"]) == [
        {"candidate_step": 6_000, "batch_label": "B", "reward": 0},
        {"candidate_step": 8_000, "batch_label": "A", "reward": 0},
    ]
    postbarrier = quarantine["postbarrier_disclosure"]
    assert postbarrier["complete_six_cell_aggregate_computed"] is False
    assert postbarrier["diagnostic_row"] == {
        "candidate_step": 6_000,
        "batch_label": "A",
        "rollout_id": 1,
        "line_number": 388,
        "sample_index": 2_435,
        "puzzle_id": "4DXgm",
        "score": 1,
        "output": "Qh5f7# <call_env>",
        "joint_protocol_valid": False,
    }
    assert "direction_neutral" in postbarrier["effect_on_repair"]


def test_gate_runtime_is_blinded_and_watchdogs_are_suppressed():
    semantics = gate._contract_static()["semantics"]

    assert semantics["rollout_fault_tolerance"] is False
    assert semantics["router_health_check_interval_seconds"] == 1e18
    assert semantics["strict_sample_outcome_logs"] == "redacted"
    assert semantics["positive_attempt_stream"] is False
    assert semantics["pre_barrier_reward_aggregates"] is False
    assert semantics["miles_log_passrate"] is False
    assert semantics["canary_disjoint_from_grid"] is True
    source = _function_source("v2r4c_gate_rollout")
    assert "fault_tolerance=False" in source
    assert (
        "rollout_health_check_interval=(\n"
        "            ROUTER_HEALTH_INTERVAL_SECONDS\n"
        "        )"
    ) in source
    assert "log_passrate=False" in source
    assert "dynamic_filter=False" in source
    assert "deterministic_seed_by_sample_index=True" in source


def test_gpu_functions_have_hard_timeout_and_no_automatic_retries():
    source = Path(gate.__file__).read_text()
    tree = ast.parse(source)
    for name in ("v2r4c_canary", "v2r4c_gate_rollout"):
        node = next(
            value
            for value in tree.body
            if isinstance(value, ast.FunctionDef)
            and value.name == name
        )
        decorator = next(
            value
            for value in node.decorator_list
            if isinstance(value, ast.Call)
            and ast.unparse(value.func) == "app.function"
        )
        keywords = {
            value.arg: value.value for value in decorator.keywords
        }
        assert "retries" not in keywords
        assert ast.unparse(keywords["timeout"]) == "3 * 60 * 60"
    command_source = _function_source("_run_rollout_command")
    assert "timeout=CELL_TIMEOUT_SECONDS" in command_source


def _make_strict_inventory(tmp_path: Path, rollouts: int) -> tuple[Path, str]:
    run_root = tmp_path / "run"
    training = run_root / "rollouts" / "training"
    provenance = run_root / "provenance"
    training.mkdir(parents=True)
    provenance.mkdir()
    (run_root / "_V2R4C_GATE_INTENT.json").write_text("{}")
    (run_root / "run_provenance.json").write_text("{}")
    launch = provenance / "launch_0123456789abcdef.json"
    launch.write_text("{}")
    for rollout_id in range(rollouts):
        (training / f"rollout_{rollout_id}.jsonl").write_text("")
    return run_root, str(launch)


def test_strict_inventory_is_a_recursive_allowlist(tmp_path):
    run_root, launch = _make_strict_inventory(tmp_path, 4)
    gate._strict_inventory(
        run_root,
        expected_rollouts=4,
        intent_filename="_V2R4C_GATE_INTENT.json",
        launch_manifest=launch,
    )

    logs = run_root / "logs"
    logs.mkdir()
    (logs / "innocuous-looking.json").write_text("{}")
    with pytest.raises(RuntimeError, match="inventory drifted"):
        gate._strict_inventory(
            run_root,
            expected_rollouts=4,
            intent_filename="_V2R4C_GATE_INTENT.json",
            launch_manifest=launch,
        )


def test_quarantine_inventory_uses_volume_relative_absolute_paths(
    tmp_path,
):
    mount = tmp_path / "rl-checkpoints"
    root = mount / "chess-rl-miles-interleave" / "old-run"
    root.mkdir(parents=True)
    (root / "proof.json").write_text("{}")

    identity = gate._quarantine_root_identity(
        root, mount_root=mount
    )
    expected_records = [
        {
            "path": (
                "/chess-rl-miles-interleave/old-run/proof.json"
            ),
            "bytes": 2,
            "sha256": (
                "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
            ),
        }
    ]
    assert identity == {
        "root_file_count": 1,
        "root_total_bytes": 2,
        "root_inventory_sha256": (
            gate.shared._canonical_json_sha256(expected_records)
        ),
    }


def test_all_zero_contract_binding_is_rejected_before_ledger_code(
    monkeypatch,
):
    monkeypatch.setattr(gate, "EXPECTED_CONTRACT_SHA256", "0" * 64)
    monkeypatch.setattr(
        gate, "EXPECTED_CONTRACT_FILE_SHA256", "0" * 64
    )

    with pytest.raises(ValueError, match="non-placeholder"):
        gate._require_contract_digest("0" * 64)
    main_source = _function_source("v2r4c_main")
    assert main_source.index("_require_contract_digest") < (
        main_source.index('if action == "launch-canary"')
    )


def test_contract_requires_exact_finalizer_source_dependency_and_test():
    valid = {
        "path": gate.FINALIZER_PATH,
        "sha256": "1" * 64,
        "reused_dependency": {
            "path": gate.REUSED_FINALIZER_PATH,
            "sha256": "2" * 64,
        },
        "test_source": {
            "path": gate.FINALIZER_TEST_PATH,
            "sha256": "3" * 64,
        },
    }
    gate._validate_finalizer_contract(valid)

    for field in ("sha256", "reused_dependency", "test_source"):
        malformed = json.loads(json.dumps(valid))
        if field == "sha256":
            malformed[field] = "0" * 63
        else:
            malformed[field]["path"] = "unbound.py"
        with pytest.raises(ValueError, match="finalizer"):
            gate._validate_finalizer_contract(malformed)

    missing = dict(valid)
    missing.pop("test_source")
    with pytest.raises(ValueError, match="exact finalizer"):
        gate._validate_finalizer_contract(missing)


def test_contract_requires_full_transitive_analyzer_binding():
    valid = {
        "path": gate.ANALYZER_PATH,
        "sha256": "1" * 64,
        "corrected_dependency": {
            "path": gate.CORRECTED_ANALYZER_PATH,
            "sha256": "2" * 64,
        },
        "base_dependency": {
            "path": gate.BASE_ANALYZER_PATH,
            "sha256": "3" * 64,
        },
    }
    gate._validate_analysis_contract(valid)

    for field in ("sha256", "corrected_dependency", "base_dependency"):
        malformed = json.loads(json.dumps(valid))
        if field == "sha256":
            malformed[field] = "0" * 63
        else:
            malformed[field]["path"] = "unbound.py"
        with pytest.raises(ValueError, match="analyzer"):
            gate._validate_analysis_contract(malformed)

    missing = dict(valid)
    missing.pop("base_dependency")
    with pytest.raises(ValueError, match="exact analyzer"):
        gate._validate_analysis_contract(missing)


def test_canary_ledger_requires_self_hash_and_exact_one_call(tmp_path):
    path = tmp_path / "canary.json"
    contract_sha = "1" * 64
    ledger = {
        "schema": "interleaved-v2r4c-canary-launch-ledger-v1",
        "version": gate.VERSION,
        "state": "canary_succeeded",
        "contract_sha256": contract_sha,
        "preflight_call_id": "fc-PREFLIGHT",
        "preflight": {
            "schema": "interleaved-v2r4c-gate-preflight-v1",
            "phase": "canary",
            "contract_sha256": contract_sha,
            "grid_roots_absent": True,
            "canary_state": "absent",
            "prompt_manifest_sha256": gate.PROMPT_MANIFEST_SHA256,
        },
        "calls": [
            {
                "run_name": gate.CANARY_RUN_NAME,
                "function_call_id": "fc-CANARY",
            }
        ],
        "success_sha256": "2" * 64,
    }
    gate._persist_ledger(path, ledger)
    assert gate._load_authenticated_canary_ledger(
        path, contract_sha256=contract_sha
    )["success_sha256"] == "2" * 64

    tampered = json.loads(path.read_text())
    tampered["calls"].append(dict(tampered["calls"][0]))
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="ledger drifted"):
        gate._load_authenticated_canary_ledger(
            path, contract_sha256=contract_sha
        )


def test_prebarrier_success_marker_omits_rollout_byte_counts():
    source = _function_source("v2r4c_gate_rollout")

    assert 'if key != "bytes"' in source
    assert '"artifact_records": artifact_records' in source


def test_binding_is_the_only_v2r4c_project_manifest_exclusion():
    identity = gate.shared._normalized_source_identity(
        gate.base.PROJECT_LOCAL,
        excluded_relatives=(gate.BINDING_RELATIVE_PATH,),
    )

    assert identity["excluded_relatives"] == [
        "chess_rl_miles/v2r4c_contract_binding.py"
    ]
