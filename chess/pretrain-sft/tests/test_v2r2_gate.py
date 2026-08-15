from __future__ import annotations

import copy

import pytest

from training.v2r2_gate import (
    AUDIT_GROUPS,
    AUDIT_ROWS,
    CONFIRMATION_FIRST_SEED,
    CONTRACT_PLAN_SHA256,
    CONTRACT_SCHEMA,
    CONTRACT_VERSION,
    audit_rollout_rows,
    choose_smallest_eligible_weight,
    is_joint_valid_protocol_row,
    select_first_disjoint_confirmation,
    select_first_disjoint_prompt_confirmation,
    select_first_disjoint_prompt_set,
    self_hash_marker,
    validate_disjoint_audits,
    validate_monolithic_protocol_gate,
    validate_protocol_audit,
    validate_rl_endpoint_gate,
    validate_confirmation_against_prompt_selection,
    validate_self_hashed_marker,
)


def _rows(
    *,
    prompt_prefix: str,
    valid_rows: int = 32,
    valid_groups: int = 16,
    positive_rows: int = 8,
    variance_groups: int = 8,
):
    rows = []
    valid_per_group = valid_rows // valid_groups if valid_groups else 0
    valid_remainder = valid_rows % valid_groups if valid_groups else 0
    for group in range(AUDIT_GROUPS):
        for sample in range(8):
            index = group * 8 + sample
            valid = (
                group < valid_groups
                and sample
                < valid_per_group + int(group < valid_remainder)
            )
            positive = group < variance_groups and sample == 0
            if index >= positive_rows * 8:
                positive = False
            rows.append(
                {
                    "status": "completed",
                    "group_index": group,
                    "input": f"{prompt_prefix}-{group} <T>",
                    "FEN": f"fen-{prompt_prefix}-{group}",
                    "PuzzleId": f"{prompt_prefix}-{group}",
                    "ground_truth": "e2e4",
                    "output": (
                        "reason </T> <call_env> e2e4"
                        if valid
                        else "ordinary continuation"
                    ),
                    "extracted_moves": "e2e4" if valid else "",
                    "score": 1.0 if positive else 0.0,
                }
            )
    assert len(rows) == AUDIT_ROWS
    return rows


def _passing_metrics(prefix: str, seed: int):
    return audit_rollout_rows(_rows(prompt_prefix=prefix), seed=seed)


def test_joint_protocol_requires_order_and_parser_in_same_row():
    assert is_joint_valid_protocol_row(
        {
            "output": "x </T> y <call_env> z",
            "extracted_moves": "e2e4",
        }
    )
    assert not is_joint_valid_protocol_row(
        {
            "output": "x <call_env> y </T>",
            "extracted_moves": "e2e4",
        }
    )
    assert not is_joint_valid_protocol_row(
        {"output": "x </T> y <call_env>", "extracted_moves": ""}
    )


def test_audit_reports_joint_protocol_and_probability_decomposition():
    metrics = _passing_metrics("primary", 42)
    assert metrics["rollout_rows"] == 2048
    assert metrics["prompt_groups"] == 256
    assert metrics["samples_per_group"] == 8
    assert metrics["joint_valid_protocol_rows"] == 32
    assert metrics["joint_valid_protocol_groups"] == 16
    assert metrics["positive_samples"] == 8
    assert metrics["nonzero_variance_groups"] == 8
    assert metrics["p_protocol"] == 32 / 2048
    assert metrics["p_solve_given_protocol"] == 8 / 32
    assert metrics["p_total"] == 8 / 2048
    assert metrics["variance_rate"] == 8 / 256
    assert len(metrics["prompt_fingerprints"]) == 256


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda rows: rows.pop(), "rollout rows"),
        (
            lambda rows: rows[0].__setitem__("status", "failed"),
            "not completed",
        ),
        (
            lambda rows: rows[1].__setitem__("input", "different"),
            "prompt identity changed",
        ),
        (
            lambda rows: rows[0].__setitem__("score", float("nan")),
            "not finite",
        ),
        (
            lambda rows: rows[0].__setitem__("score", 0.5),
            "not binary",
        ),
    ],
)
def test_rollout_shape_and_integrity_fail_closed(mutation, error):
    rows = _rows(prompt_prefix="bad")
    mutation(rows)
    with pytest.raises(ValueError, match=error):
        audit_rollout_rows(rows, seed=42)


def test_positive_score_requires_joint_valid_protocol_row():
    rows = _rows(prompt_prefix="malformed-positive")
    rows[0]["output"] = "missing the required protocol"
    rows[0]["extracted_moves"] = ""
    with pytest.raises(ValueError, match="positive without a joint-valid"):
        audit_rollout_rows(rows, seed=42)


def test_status_policy_is_strict_by_default_and_diagnostic_when_declared():
    rows = _rows(prompt_prefix="status-policy")
    rows[0]["status"] = "truncated"
    with pytest.raises(ValueError, match="not completed"):
        audit_rollout_rows(rows, seed=42)
    metrics = audit_rollout_rows(
        rows,
        seed=42,
        allowed_statuses=("completed", "truncated"),
    )
    assert metrics["status_counts"] == {
        "completed": 2047,
        "truncated": 1,
    }
    assert metrics["positive_samples_by_status"]["truncated"] == 1


def test_protocol_gate_rejects_independent_or_insufficient_counts():
    rows = _rows(prompt_prefix="weak", valid_rows=31, valid_groups=16)
    metrics = audit_rollout_rows(rows, seed=42)
    with pytest.raises(ValueError, match="joint_valid_protocol_rows"):
        validate_protocol_audit(metrics)

    rows = _rows(prompt_prefix="narrow", valid_rows=32, valid_groups=15)
    metrics = audit_rollout_rows(rows, seed=42)
    with pytest.raises(ValueError, match="joint_valid_protocol_groups"):
        validate_protocol_audit(metrics)


def test_two_protocol_audits_must_be_prompt_disjoint():
    primary = _passing_metrics("primary", 42)
    confirmation = _passing_metrics("confirmation", 43)
    pair = validate_disjoint_audits(primary, confirmation)
    assert pair["prompt_intersection"] == 0

    overlap = _passing_metrics("primary", 43)
    with pytest.raises(ValueError, match="overlaps 256"):
        validate_disjoint_audits(primary, overlap)


def test_confirmation_selection_proves_all_lower_seeds_overlapped():
    primary = _passing_metrics("primary", 42)
    seed43 = copy.deepcopy(primary)
    seed43["seed"] = CONFIRMATION_FIRST_SEED
    seed44 = _passing_metrics("confirmation", 44)
    selected = select_first_disjoint_confirmation(
        primary, [seed43, seed44]
    )
    assert selected["confirmation"]["seed"] == 44
    assert selected["attempted_seeds"] == [
        {"seed": 43, "prompt_overlap": 256},
        {"seed": 44, "prompt_overlap": 0},
    ]
    with pytest.raises(ValueError, match="consecutive"):
        select_first_disjoint_confirmation(primary, [seed44])


def test_prompt_selection_is_preserved_when_confirmation_fails_protocol():
    primary = _passing_metrics("primary", 42)
    weak = audit_rollout_rows(
        _rows(
            prompt_prefix="weak-confirmation",
            valid_rows=31,
            valid_groups=16,
        ),
        seed=43,
    )
    selection = select_first_disjoint_prompt_confirmation(primary, [weak])
    assert selection["confirmation"]["seed"] == 43
    assert selection["attempted_seeds"] == [
        {"seed": 43, "prompt_overlap": 0}
    ]
    with pytest.raises(ValueError, match="joint_valid_protocol_rows"):
        select_first_disjoint_confirmation(primary, [weak])


def test_prompt_only_overlap_proof_binds_the_single_generated_confirmation():
    primary = _passing_metrics("primary", 42)
    overlap = copy.deepcopy(primary)
    overlap["seed"] = 43
    selected = _passing_metrics("selected", 44)
    prompt_candidates = [
        {
            "seed": 43,
            "prompt_fingerprints": overlap["prompt_fingerprints"],
            "prompt_set_sha256": overlap["prompt_set_sha256"],
        },
        {
            "seed": 44,
            "prompt_fingerprints": selected["prompt_fingerprints"],
            "prompt_set_sha256": selected["prompt_set_sha256"],
        },
    ]
    proof = select_first_disjoint_prompt_set(primary, prompt_candidates)
    assert proof["attempted_seeds"] == [
        {
            "seed": 43,
            "prompt_overlap": 256,
            "prompt_set_sha256": overlap["prompt_set_sha256"],
        },
        {
            "seed": 44,
            "prompt_overlap": 0,
            "prompt_set_sha256": selected["prompt_set_sha256"],
        },
    ]
    pair = validate_confirmation_against_prompt_selection(
        primary, prompt_candidates, selected
    )
    assert pair["prompt_intersection"] == 0
    wrong = copy.deepcopy(selected)
    wrong["seed"] = 45
    with pytest.raises(ValueError, match="seed differs"):
        validate_confirmation_against_prompt_selection(
            primary, prompt_candidates, wrong
        )


def test_smallest_eligible_weight_is_selected_not_best_outcome():
    passing = {
        96.0: {
            "primary": _passing_metrics("w96-a", 42),
            "confirmation": _passing_metrics("w96-b", 43),
        },
        32.0: {
            "primary": _passing_metrics("w32-a", 42),
            "confirmation": _passing_metrics("w32-b", 43),
        },
    }
    assert choose_smallest_eligible_weight(passing) == 32.0
    with pytest.raises(ValueError, match="unregistered"):
        choose_smallest_eligible_weight({64.0: passing[32.0]})


def test_monolithic_gate_binds_weight_schedule_step_and_manifest():
    primary = _passing_metrics("mono-a", 42)
    confirmation = _passing_metrics("mono-b", 43)
    result = validate_monolithic_protocol_gate(
        selected_weight=32,
        candidate_weight=32,
        candidate_step=2000,
        schedule_total_steps=19840,
        manifest_leg="p1+p2",
        primary=primary,
        confirmation=confirmation,
    )
    assert result["prompt_intersection"] == 0
    with pytest.raises(ValueError, match="19,840"):
        validate_monolithic_protocol_gate(
            selected_weight=32,
            candidate_weight=32,
            candidate_step=2000,
            schedule_total_steps=9920,
            manifest_leg="p1+p2",
            primary=primary,
            confirmation=confirmation,
        )


def test_rl_gate_is_separate_and_requires_both_batches_reward_variance():
    primary = _passing_metrics("rl-a", 42)
    confirmation = _passing_metrics("rl-b", 43)
    assert validate_rl_endpoint_gate(
        primary,
        confirmation,
        dynamic_filter_attempted_groups=8192,
        dynamic_filter_accepted_groups=256,
    )["dynamic_filter_accepted_groups"] == 256

    weak = copy.deepcopy(confirmation)
    weak["positive_samples"] = 7
    with pytest.raises(ValueError, match="positive_samples"):
        validate_rl_endpoint_gate(primary, weak)
    with pytest.raises(ValueError, match=r"\[1, 8192\]"):
        validate_rl_endpoint_gate(
            primary,
            confirmation,
            dynamic_filter_attempted_groups=8193,
            dynamic_filter_accepted_groups=256,
        )


def test_gate_marker_binds_contract_version_and_plan_hash():
    marker = self_hash_marker(
        {
            "contract_schema": CONTRACT_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "contract_plan_sha256": CONTRACT_PLAN_SHA256,
            "approved": True,
        }
    )
    assert validate_self_hashed_marker(marker) == marker
    tampered = dict(marker)
    tampered["approved"] = False
    with pytest.raises(ValueError, match="self-hash"):
        validate_self_hashed_marker(tampered)
