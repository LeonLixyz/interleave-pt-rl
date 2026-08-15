from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from Eval import finalize_v2r4_gate as finalizer
from Eval.v2r4_gate_analysis import content_hash


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _self_hash(value: dict, field: str) -> dict:
    result = dict(value)
    result[field] = content_hash(result, field)
    return result


def _contract() -> dict:
    candidates = {
        str(step): {
            "endpoint_checkpoint_sha256": hashlib.sha256(
                f"endpoint-{step}".encode()
            ).hexdigest(),
            "hf_directory_manifest_sha256": hashlib.sha256(
                f"directory-{step}".encode()
            ).hexdigest(),
            "hf_path": f"/pretrain-checkpoints/step_{step}/hf",
            "original_p1_eligible": step == 9_920,
        }
        for step in finalizer.CANDIDATE_STEPS
    }
    batches = {
        label: {
            "epoch0_prompt_order_sha256": hashlib.sha256(
                f"epoch-{label}".encode()
            ).hexdigest(),
            "path": f"/data/batch_{label.lower()}.parquet",
            "prompt_set_sha256": hashlib.sha256(
                f"set-{label}".encode()
            ).hexdigest(),
            "rollout_seed": 1_000 if label == "A" else 2_000,
            "rows": 1_024,
            "sha256": hashlib.sha256(f"bytes-{label}".encode()).hexdigest(),
        }
        for label in finalizer.BATCH_LABELS
    }
    value = {
        "schema": finalizer.CONTRACT_SCHEMA,
        "version": finalizer.CONTRACT_VERSION,
        "model_id": "interleave_47m_qwen3",
        "cells": [
            {
                "candidate_step": step,
                "batch_label": batch,
                "run_name": (
                    f"v2r4a-gate-w190-s{step}-batch-{batch.lower()}"
                ),
            }
            for step in finalizer.CANDIDATE_STEPS
            for batch in finalizer.BATCH_LABELS
        ],
        "candidates": candidates,
        "prompt_batches": batches,
        "prompt_manifest": {
            "path": "/data/prompt_manifest.json",
            "manifest_sha256": "1" * 64,
            "file_sha256": "2" * 64,
        },
        "semantics": {
            "automatic_retries": 0,
            "checkpoint_saves": False,
            "data_source_path": "strict",
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
            "sampling_seed_rule": "sample-index",
            "task_exception_policy": "fail",
            "total_prompt_groups": 1_024,
            "total_rows": 8_192,
        },
        "runtime": {"miles_image": "image@sha256:" + "a" * 64},
        "sources": {
            "chess_rl_miles": {"manifest_sha256": "b" * 64},
            "miles": {"manifest_sha256": "c" * 64},
        },
        "plan": {
            "path": "INTERLEAVED_V2R4A_GATE_AMENDMENT.md",
            "sha256": finalizer.PLAN_SHA256,
        },
        "endpoint_evaluators": {
            "pt_b1_b5": finalizer.ENDPOINT_EVALUATOR_SHA256,
            "p2_sft_at_p1": finalizer.P2_EVALUATOR_SHA256,
        },
    }
    value["contract_sha256"] = content_hash(value, "contract_sha256")
    return value


def _prompt_manifest(contract: dict) -> dict:
    batches = {}
    for label in finalizer.BATCH_LABELS:
        prompts = [
            hashlib.sha256(f"{label}-{index}".encode()).hexdigest()
            for index in range(1_024)
        ]
        quarters = []
        for rollout_id in range(4):
            quarter = prompts[rollout_id * 256 : (rollout_id + 1) * 256]
            quarters.append(
                {
                    "rollout_id": rollout_id,
                    "prompt_count": 256,
                    "ordered_prompt_fingerprints": quarter,
                    "prompt_order_sha256": hashlib.sha256(
                        finalizer.canonical_json(quarter)
                    ).hexdigest(),
                }
            )
        expected = contract["prompt_batches"][label]
        batches[label] = {
            "file_sha256": expected["sha256"],
            "logical_path": expected["path"],
            "prompt_set_sha256": expected["prompt_set_sha256"],
            "epoch0_prompt_order_sha256": (
                expected["epoch0_prompt_order_sha256"]
            ),
            "rollout_seed": expected["rollout_seed"],
            "rows": 1_024,
            "rollout_quarters": quarters,
        }
    value = {
        "schema": "prompt-manifest-test",
        "batches": batches,
    }
    value["manifest_sha256"] = content_hash(value, "manifest_sha256")
    return value


def test_runtime_contract_and_ledgers_are_self_hashed_and_exact(
    monkeypatch: pytest.MonkeyPatch,
):
    contract = _contract()
    raw = _json_bytes(contract)
    monkeypatch.setattr(
        finalizer, "CONTRACT_SHA256", contract["contract_sha256"]
    )
    monkeypatch.setattr(
        finalizer, "CONTRACT_FILE_SHA256", hashlib.sha256(raw).hexdigest()
    )
    observed, evidence = finalizer.validate_runtime_contract(raw)
    assert observed == contract
    assert evidence["contract_sha256"] == contract["contract_sha256"]
    assert evidence["file_sha256"] == hashlib.sha256(raw).hexdigest()

    gate = {
        "schema": finalizer.GATE_LEDGER_SCHEMA,
        "version": finalizer.CONTRACT_VERSION,
        "state": "launched_all",
        "contract_sha256": contract["contract_sha256"],
        "expected_call_count": 6,
        "preflight_call_id": "fc-PREFLIGHT",
        "preflight": {
            "schema": "interleaved-v2r4-gate-preflight-v1",
            "version": finalizer.CONTRACT_VERSION,
            "contract_sha256": contract["contract_sha256"],
            "contract_file_sha256": evidence["file_sha256"],
            "all_six_canonical_roots_absent": True,
            "ray_worker_environment": {
                "artifact_root": (
                    "/rl-checkpoints/chess-rl-miles-interleave/"
                    "v2r4a-ray-env-preflight"
                ),
                "gpu_allocated": False,
                "seed_mode": "sample-index",
            },
            "run_roots": [
                "/rl-checkpoints/chess-rl-miles-interleave/"
                + cell["run_name"]
                for cell in contract["cells"]
            ],
        },
        "calls": [
            {**cell, "function_call_id": f"fc-CALL{index}"}
            for index, cell in enumerate(contract["cells"])
        ],
    }
    gate = _self_hash(gate, "ledger_sha256")
    gate_raw = _json_bytes(gate)
    monkeypatch.setattr(
        finalizer, "GATE_LEDGER_SHA256", gate["ledger_sha256"]
    )
    monkeypatch.setattr(
        finalizer,
        "GATE_LEDGER_FILE_SHA256",
        hashlib.sha256(gate_raw).hexdigest(),
    )
    _, gate_evidence = finalizer.validate_gate_ledger(
        gate_raw, contract, evidence["file_sha256"]
    )
    assert len(gate_evidence["call_ids"]) == 6

    tampered = copy.deepcopy(gate)
    tampered["calls"][0]["batch_label"] = "B"
    tampered = _self_hash(tampered, "ledger_sha256")
    tampered_raw = _json_bytes(tampered)
    monkeypatch.setattr(
        finalizer, "GATE_LEDGER_SHA256", tampered["ledger_sha256"]
    )
    monkeypatch.setattr(
        finalizer,
        "GATE_LEDGER_FILE_SHA256",
        hashlib.sha256(tampered_raw).hexdigest(),
    )
    with pytest.raises(ValueError, match="batch_label"):
        finalizer.validate_gate_ledger(
            tampered_raw, contract, evidence["file_sha256"]
        )


def _row(
    *,
    prompt: str,
    rollout_id: int,
    group: int,
    sibling: int,
    seed: int,
) -> dict:
    sample = group * 8 + sibling
    protocol = sample < 443
    positive = group < 20 and sibling == 0
    return {
        "input": prompt,
        "FEN": f"fen-{prompt}",
        "PuzzleId": f"id-{prompt}",
        "ground_truth": "e2e4",
        "rollout_id": rollout_id,
        "group_index": group,
        "sample_index": sample,
        "sampling_seed_sibling_index": sibling,
        "sampling_seed": seed + sample,
        "status": "completed",
        "score": float(positive),
        "output": (
            "reason </T> text <call_env> e2e4" if protocol else "reason"
        ),
        "extracted_moves": "e2e4" if protocol else "",
        "metadata": {
            "sampling_seed_sibling_index": sibling,
            "sampling_seed": seed + sample,
            "sampling_seed_mode": "sample-index",
        },
    }


def test_gate_cell_rereads_and_hashes_all_artifacts_before_audit():
    contract = _contract()
    step, batch = 6_000, "A"
    run_name = "v2r4a-gate-w190-s6000-batch-a"
    seed = contract["prompt_batches"][batch]["rollout_seed"]
    files: dict[str, bytes] = {}
    records = []
    quarter_records = []
    for rollout_id in range(4):
        rows = []
        prompts = []
        for local_group in range(256):
            group = rollout_id * 256 + local_group
            prompt = f"A-prompt-{group}"
            rows.extend(
                _row(
                    prompt=prompt,
                    rollout_id=rollout_id,
                    group=group,
                    sibling=sibling,
                    seed=seed,
                )
                for sibling in range(8)
            )
            prompts.append(finalizer._prompt_fingerprint(rows[-1]))
        raw = b"".join(
            json.dumps(row, separators=(",", ":")).encode() + b"\n"
            for row in rows
        )
        relative = (
            f"chess-rl-miles-interleave/{run_name}/rollouts/training/"
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
                "bytes": len(raw),
                "prompt_order_sha256": prompt_hash,
            }
        )
        quarter_records.append(
            {
                "rollout_id": rollout_id,
                "prompt_count": 256,
                "ordered_prompt_fingerprints": prompts,
                "prompt_order_sha256": prompt_hash,
            }
        )
    prompt_manifest = {
        "batches": {"A": {"rollout_quarters": quarter_records}}
    }
    identity = {
        "kind": "chess_rl_miles_v2r4_production_gate_rollout",
        "version": finalizer.CONTRACT_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "contract_file_sha256": "d" * 64,
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
            "manifest_sha256": contract["prompt_manifest"][
                "manifest_sha256"
            ],
            "manifest_file_sha256": contract["prompt_manifest"][
                "file_sha256"
            ],
        },
        "semantics": contract["semantics"],
        "sources": contract["sources"],
        "runtime": {"image": contract["runtime"]["miles_image"]},
    }
    command = ["python", "run.py", "--debug-rollout-only"]
    identity_sha = hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()
    command_sha = hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode()
    ).hexdigest()
    root_relative = (
        f"chess-rl-miles-interleave/{run_name}/run_provenance.json"
    )
    launch_relative = (
        f"chess-rl-miles-interleave/{run_name}/provenance/"
        f"launch_{command_sha[:16]}.json"
    )
    files[root_relative] = _json_bytes(
        {
            "identity": identity,
            "identity_sha256": identity_sha,
            "initial_command": command,
            "initial_command_sha256": command_sha,
        }
    )
    files[launch_relative] = _json_bytes(
        {
            "identity_sha256": identity_sha,
            "command": command,
            "command_sha256": command_sha,
        }
    )
    marker = {
        "schema": finalizer.GATE_SUCCESS_SCHEMA,
        "version": finalizer.CONTRACT_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "contract_file_sha256": "d" * 64,
        "run_name": run_name,
        "candidate_step": step,
        "batch_label": batch,
        "provenance": {
            "root_manifest": f"/rl-checkpoints/{root_relative}",
            "launch_manifest": f"/rl-checkpoints/{launch_relative}",
            "identity_sha256": identity_sha,
            "command_sha256": command_sha,
        },
        "prompt_batch_sha256": contract["prompt_batches"][batch]["sha256"],
        "prompt_set_sha256": contract["prompt_batches"][batch][
            "prompt_set_sha256"
        ],
        "rollout_seed": seed,
        "artifact_records": records,
        "shape_authenticated": True,
        "reward_metrics_inspected": False,
    }
    marker["success_sha256"] = content_hash(marker, "success_sha256")
    marker_relative = (
        f"chess-rl-miles-interleave/{run_name}/_V2R4_GATE_SUCCESS.json"
    )
    files[marker_relative] = _json_bytes(marker)
    result = {
        **marker,
        "success_path": f"/rl-checkpoints/{marker_relative}",
    }
    record = {
        "candidate_step": step,
        "batch_label": batch,
        "run_name": run_name,
        "function_call_id": "fc-GATECELL",
    }
    cell, evidence = finalizer.validate_gate_cell(
        record=record,
        function_result=result,
        contract=contract,
        contract_file_sha256="d" * 64,
        prompt_manifest=prompt_manifest,
        read_checkpoint_file=files.__getitem__,
    )
    assert cell["rows"] == 8_192
    assert cell["solve_at_8_groups"] == 20
    assert cell["absolute_gate_pass"] is True
    assert len(evidence["artifacts"]) == 4

    corrupted = dict(files)
    path = records[0]["path"].removeprefix("/rl-checkpoints/")
    corrupted[path] = corrupted[path] + b"{}\n"
    with pytest.raises(ValueError, match="byte identity"):
        finalizer.validate_gate_cell(
            record=record,
            function_result=result,
            contract=contract,
            contract_file_sha256="d" * 64,
            prompt_manifest=prompt_manifest,
            read_checkpoint_file=corrupted.__getitem__,
        )


def _endpoint_result(contract: dict, step: int, component: str) -> dict:
    checkpoint = contract["candidates"][str(step)][
        "endpoint_checkpoint_sha256"
    ]
    base = {
        "schema": finalizer.ENDPOINT_RESULT_SCHEMA,
        "state": "complete",
        "component": component,
        "namespace": finalizer.ENDPOINT_NAMESPACE,
        "experiment_version": finalizer.ENDPOINT_EXPERIMENT_VERSION,
        "endpoint_id": f"v2r4-s{step}",
        "checkpoint_sha256": checkpoint,
        "endpoint": {
            "declared_checkpoint_sha256": checkpoint,
            "training_state": {
                "snapshot_step": step,
                "p2_consumed": False,
            },
        },
        "eval_fingerprint": finalizer.ENDPOINT_FINGERPRINTS[component],
    }
    if component == "losses":
        base.update(
            {
                "datasets": {
                    "pretraining": {
                        "holdout_hash": finalizer.PT_HOLDOUT_SHA256,
                        "records": finalizer.PT_RECORDS,
                        "target_tokens": finalizer.PT_TARGET_TOKENS,
                    }
                },
                "metrics": {
                    "heldout_pretrain_loss": 0.7,
                    "heldout_pretrain_perplexity": 2.0,
                    "heldout_pretrain_token_accuracy": 0.5,
                    "heldout_pretrain_correct_tokens": (
                        finalizer.PT_TARGET_TOKENS // 2
                    ),
                    "heldout_pretrain_target_tokens": (
                        finalizer.PT_TARGET_TOKENS
                    ),
                },
            }
        )
    else:
        benchmark = {
            key: {"avg_reward": 0.01, "pass_at_1": 0.01}
            for key in ("B1", "B2", "B3", "B4", "B5")
        }
        base.update(
            {
                "expected_rows": finalizer.CHESS_ROWS,
                "actual_rows": finalizer.CHESS_ROWS,
                "metrics": {
                    "avg_reward": 0.01,
                    "pass_at_1": 0.01,
                    "b3_avg": 0.01,
                    "b4_avg": 0.01,
                    "b3_b4_avg": 0.01,
                    "benchmarks": benchmark,
                },
            }
        )
    base["result_hash"] = content_hash(base, "result_hash")
    return base


def test_endpoint_normalization_is_finite_flat_and_tamper_evident():
    contract = _contract()
    loss_record = {
        "step": 6_000,
        "component": "losses",
        "function_call_id": "fc-LOSS",
    }
    component, normalized, evidence = finalizer.validate_endpoint_result(
        loss_record, _endpoint_result(contract, 6_000, "losses"), contract
    )
    assert component == "pt"
    assert normalized["metrics"]["target_tokens"] == finalizer.PT_TARGET_TOKENS
    assert evidence["result_hash"]

    chess_record = {
        "step": 6_000,
        "component": "chess",
        "function_call_id": "fc-CHESS",
    }
    result = _endpoint_result(contract, 6_000, "chess")
    component, normalized, _ = finalizer.validate_endpoint_result(
        chess_record, result, contract
    )
    assert component == "chess"
    assert all(
        isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math_isfinite(item)
        for item in normalized["metrics"].values()
    )

    result["metrics"]["b3_b4_avg"] = 0.02
    result["result_hash"] = content_hash(result, "result_hash")
    with pytest.raises(ValueError, match="inconsistent"):
        finalizer.validate_endpoint_result(chess_record, result, contract)


def test_persisted_endpoint_and_raw_chess_artifacts_are_bound(
    monkeypatch: pytest.MonkeyPatch,
):
    contract = _contract()
    result = _endpoint_result(contract, 6_000, "chess")
    result["eval_fingerprint"] = "e" * 64
    root = (
        "endpoint_v2r1_weighted_clean/v2r4-s6000/"
        f"{result['checkpoint_sha256']}/chess_{'e' * 12}"
    )
    generation_path = f"{root}/output/eval/generations/0.jsonl"
    metrics_path = f"{root}/output/eval/generations/metrics.json"
    result["generations"] = f"/results/{generation_path}"
    result["raw_metrics_path"] = f"/results/{metrics_path}"
    result["result_hash"] = content_hash(result, "result_hash")
    monkeypatch.setattr(finalizer, "CHESS_ROWS", 2)
    files = {
        f"{root}/_SUCCESS.json": _json_bytes(result),
        generation_path: b'{"row":1}\n{"row":2}\n',
        metrics_path: b'{"metric":0.5}\n',
    }
    evidence = finalizer.authenticate_endpoint_artifacts(
        result, read_results_file=files.__getitem__
    )
    assert evidence["success_file"]["sha256"]
    assert evidence["raw_chess_artifacts"]["generations"]["rows"] == 2
    assert evidence["raw_chess_artifacts"]["metrics"]["metric_count"] == 1

    tampered = dict(files)
    tampered[generation_path] += b'{"row":3}\n'
    with pytest.raises(ValueError, match="row count"):
        finalizer.authenticate_endpoint_artifacts(
            result, read_results_file=tampered.__getitem__
        )


def math_isfinite(value: float | int) -> bool:
    return value == value and abs(float(value)) != float("inf")


def test_p2_output_requires_all_nested_hash_bindings(
    monkeypatch: pytest.MonkeyPatch,
):
    contract = _contract()
    step = 6_000
    selection = _self_hash(
        {"schema": "selection", "value": 1}, "selection_hash"
    )
    selection["selection_hash"] = finalizer.P2_SELECTION_SHA256
    cache = _self_hash({"schema": "cache", "value": 2}, "cache_shape_hash")
    cache["cache_shape_hash"] = finalizer.P2_CACHE_SHAPE_SHA256
    # The frozen production hashes cannot be synthesized from a tiny fixture;
    # keep this test focused on the finalizer envelope and binding checks.
    monkeypatch.setattr(finalizer, "_require_self_hash", lambda value, field, label: str(value[field]))
    monkeypatch.setattr(
        finalizer.p2_contract,
        "validate_candidate_sft_result",
        lambda *args, **kwargs: dict(args[0]),
    )
    pure = {
        "candidate": {
            "checkpoint_step": step,
            "candidate_id": finalizer._EXPECTED_P2_CANDIDATE_IDS[step],
            "checkpoint_sha256": finalizer._EXPECTED_RECURSIVE_HF[step],
            "training_leg": "p1",
            "has_consumed_p2": False,
        },
        "aggregate": {
            "negative_log_likelihood_sum": 2_759_776.0,
            "correct_supervised_tokens": 1_379_888,
            "supervised_targets": 2_759_776,
            "rows_evaluated": 4_096,
        },
        "metrics": {
            "masked_sft_unweighted_token_ce": 1.0,
            "masked_sft_perplexity": 2.718281828,
            "masked_sft_token_accuracy": 0.5,
        },
        "result_hash": "9" * 64,
    }
    checkpoint_sha = contract["candidates"][str(step)][
        "endpoint_checkpoint_sha256"
    ]
    result = {
        "schema": finalizer.P2_OUTPUT_SCHEMA,
        "state": "complete",
        "runtime_contract_sha256": finalizer.P2_RUNTIME_CONTRACT_SHA256,
        "evaluator_source": {
            "bundle_sha256": finalizer.P2_EVALUATOR_SHA256
        },
        "selection": selection,
        "cache_shape": cache,
        "p2_sft_result": pure,
        "checkpoint": {
            "recursive_hf_identity": finalizer._EXPECTED_RECURSIVE_HF[step],
            "endpoint_checkpoint_sha256": checkpoint_sha,
            "directory_manifest_sha256": contract["candidates"][str(step)][
                "hf_directory_manifest_sha256"
            ],
        },
        "hash_bindings": {
            "selection_hash": finalizer.P2_SELECTION_SHA256,
            "cache_shape_hash": finalizer.P2_CACHE_SHAPE_SHA256,
            "candidate_result_hash": pure["result_hash"],
            "checkpoint_recursive_hf_identity": (
                finalizer._EXPECTED_RECURSIVE_HF[step]
            ),
            "evaluator_source_sha256": finalizer.P2_EVALUATOR_SHA256,
            "runtime_contract_sha256": (
                finalizer.P2_RUNTIME_CONTRACT_SHA256
            ),
        },
        "output_hash": "8" * 64,
    }
    record = {
        "kwargs": {"candidate_step": step},
        "function_call_id": "fc-P2",
        "expected_result_path": "/results/p2/step6000/_SUCCESS.json",
    }
    normalized, evidence = finalizer.validate_p2_result(
        record, result, contract
    )
    assert normalized["metrics"]["cross_entropy"] == 1.0
    assert evidence["candidate_result_hash"] == "9" * 64
    persisted = finalizer.authenticate_p2_success_file(
        record,
        result,
        read_results_file={
            "p2/step6000/_SUCCESS.json": _json_bytes(result)
        }.__getitem__,
    )
    assert persisted["sha256"] == hashlib.sha256(
        _json_bytes(result)
    ).hexdigest()

    result["hash_bindings"]["candidate_result_hash"] = "7" * 64
    with pytest.raises(ValueError, match="hash bindings"):
        finalizer.validate_p2_result(record, result, contract)


def test_quarantine_proves_terminal_failures_and_zero_outcome_inventory(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []
    entries_by_root: dict[str, list[dict]] = {}
    files: dict[str, bytes] = {}
    failures = {}
    for index, (step, batch) in enumerate(
        (
            (step, batch)
            for step in finalizer.CANDIDATE_STEPS
            for batch in finalizer.BATCH_LABELS
        )
    ):
        run_name = f"v2r4-gate-w190-s{step}-batch-{batch.lower()}"
        call_id = f"fc-QUARANTINE{index}"
        calls.append(
            {
                "candidate_step": step,
                "batch_label": batch,
                "run_name": run_name,
                "function_call_id": call_id,
            }
        )
        failures[call_id] = {
            "type": "RuntimeError",
            "message": (
                f"v2r4 rollout gate failed for {run_name}: exit 1"
            ),
        }
        root = f"chess-rl-miles-interleave/{run_name}"
        command = ["python", "run.py", "--debug-rollout-only"]
        identity = {
            "version": finalizer._QUARANTINE_VERSION,
            "contract_sha256": finalizer._QUARANTINE_CONTRACT_SHA256,
            "contract_file_sha256": (
                finalizer._QUARANTINE_CONTRACT_FILE_SHA256
            ),
            "authorized_cell": {
                "candidate_step": step,
                "batch_label": batch,
                "run_name": run_name,
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
        launch_relative = (
            f"provenance/launch_{command_sha[:16]}.json"
        )
        documents = {
            "_V2R4_GATE_INTENT.json": {
                "schema": "interleaved-v2r4-gate-cell-intent-v1",
                "version": finalizer._QUARANTINE_VERSION,
                "contract_sha256": finalizer._QUARANTINE_CONTRACT_SHA256,
                "contract_file_sha256": (
                    finalizer._QUARANTINE_CONTRACT_FILE_SHA256
                ),
                "candidate_step": step,
                "batch_label": batch,
                "run_name": run_name,
            },
            "run_provenance.json": {
                "identity": identity,
                "identity_sha256": identity_sha,
                "initial_command": command,
                "initial_command_sha256": command_sha,
            },
            launch_relative: {
                "identity_sha256": identity_sha,
                "command": command,
                "command_sha256": command_sha,
            },
        }
        directories = {
            "provenance",
            "rollouts",
            "rollouts/validation",
            "rollouts/training",
            "mlflow",
            "logs",
        }
        entries_by_root[root] = [
            {"path": f"{root}/{name}", "type": "directory", "bytes": 0}
            for name in sorted(directories)
        ] + [
            {
                "path": f"{root}/{name}",
                "type": "file",
                "bytes": len(_json_bytes(document)),
            }
            for name, document in documents.items()
        ]
        for name, document in documents.items():
            files[f"{root}/{name}"] = _json_bytes(document)
    ledger = _self_hash(
        {
            "schema": finalizer.GATE_LEDGER_SCHEMA,
            "version": finalizer._QUARANTINE_VERSION,
            "state": "launched_all",
            "expected_call_count": 6,
            "contract_sha256": finalizer._QUARANTINE_CONTRACT_SHA256,
            "preflight": {
                "contract_sha256": (
                    finalizer._QUARANTINE_CONTRACT_SHA256
                ),
                "contract_file_sha256": (
                    finalizer._QUARANTINE_CONTRACT_FILE_SHA256
                ),
                "all_six_canonical_roots_absent": True,
            },
            "calls": calls,
        },
        "ledger_sha256",
    )
    raw = _json_bytes(ledger)
    monkeypatch.setattr(
        finalizer, "_QUARANTINE_LEDGER_SHA256", ledger["ledger_sha256"]
    )
    monkeypatch.setattr(
        finalizer,
        "_QUARANTINE_LEDGER_FILE_SHA256",
        hashlib.sha256(raw).hexdigest(),
    )
    evidence = finalizer.validate_quarantined_v2r4(
        raw,
        terminal_failure=failures.__getitem__,
        list_checkpoint_entries=entries_by_root.__getitem__,
        read_checkpoint_file=files.__getitem__,
    )
    assert evidence["aggregate"] == {
        "calls": 6,
        "terminal_failures": 6,
        "rollout_jsonl_files": 0,
        "success_markers": 0,
        "sampled_prompts": 0,
        "reward_or_outcome_artifacts": 0,
    }

    tampered_entries = copy.deepcopy(entries_by_root)
    first_root = next(iter(tampered_entries))
    tampered_entries[first_root].append(
        {
            "path": f"{first_root}/rollouts/training/rollout_0.jsonl",
            "type": "file",
            "bytes": 2,
        }
    )
    with pytest.raises(ValueError, match="unexpected data"):
        finalizer.validate_quarantined_v2r4(
            raw,
            terminal_failure=failures.__getitem__,
            list_checkpoint_entries=tampered_entries.__getitem__,
            read_checkpoint_file=files.__getitem__,
        )


def test_immutable_writer_refuses_overwrite_and_report_hash_tamper(
    tmp_path: Path,
):
    report = {
        "schema": finalizer.REPORT_SCHEMA,
        "value": 1,
    }
    report["report_sha256"] = content_hash(report, "report_sha256")
    path = tmp_path / "report.json"
    finalizer.write_immutable_report(path, report)
    assert json.loads(path.read_text()) == report
    assert not os.access(path, os.W_OK)
    with pytest.raises(FileExistsError):
        finalizer.write_immutable_report(path, report)

    bad = dict(report)
    bad["value"] = 2
    with pytest.raises(ValueError, match="invalid self hash"):
        finalizer.write_immutable_report(tmp_path / "bad.json", bad)
