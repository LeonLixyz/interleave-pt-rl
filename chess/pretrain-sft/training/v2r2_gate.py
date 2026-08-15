"""Pure fail-closed validators for the immutable v2r2 staged gate.

This module deliberately has no Modal or filesystem dependency.  Remote gate
code must authenticate artifacts and successful calls before passing their
decoded rollout rows to these validators.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


CONTRACT_SCHEMA = "interleaved-v2r2-staged-gate-contract-v1"
CONTRACT_VERSION = "mix10b_sft90k_3072_v2r2_staged_gate_20260730"
CONTRACT_FROZEN_AT = "2026-07-30T07:42:58-04:00"
CONTRACT_PLAN_SHA256 = (
    "42023d47ad1a9db5ac4dc753ffad5d228366db995849aef98822299ff78a023e"
)

ELIGIBLE_SFT_LOSS_WEIGHTS = (32.0, 96.0, 190.189290837)
PROTOCOL_CANDIDATE_STEP = 2_000
AUDIT_ROWS = 2_048
AUDIT_GROUPS = 256
SAMPLES_PER_GROUP = 8
PROTOCOL_ROWS_MIN = 32
PROTOCOL_GROUPS_MIN = 16
PRIMARY_SEED = 42
CONFIRMATION_FIRST_SEED = 43
RL_POSITIVES_MIN = 8
RL_VARIANCE_GROUPS_MIN = 8
DYNAMIC_FILTER_ACCEPTED_GROUPS = 256
DYNAMIC_FILTER_ATTEMPTED_GROUP_CAP = 8_192


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _binary_score(row: Mapping[str, Any], row_number: int) -> float:
    value = row.get("score", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"rollout row {row_number} score is not numeric")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"rollout row {row_number} score is not finite")
    # Miles' chess reward is binary.  Rejecting other values keeps the
    # probability decomposition exact instead of silently treating an
    # unfamiliar reward scale as either a loss or a success.
    if score not in (0.0, 1.0):
        raise ValueError(
            f"rollout row {row_number} score is not binary (0 or 1)"
        )
    return score


def _parsed_moves(row: Mapping[str, Any]) -> str:
    value = row.get("extracted_moves")
    if value is None:
        reward = row.get("reward")
        if isinstance(reward, Mapping):
            value = reward.get("extracted_moves")
    return str(value or "").strip()


def is_joint_valid_protocol_row(row: Mapping[str, Any]) -> bool:
    """Require ordered delimiters and a parser result in the same row."""

    output = str(row.get("output") or "")
    end_index = output.find("</T>")
    call_index = output.find("<call_env>", end_index + len("</T>"))
    return end_index >= 0 and call_index > end_index and bool(_parsed_moves(row))


def _prompt_fingerprint(row: Mapping[str, Any], row_number: int) -> str:
    prompt = str(row.get("input") or "")
    if not prompt:
        raise ValueError(f"rollout row {row_number} has an empty input")
    identity = {
        "input": prompt,
        "FEN": str(row.get("FEN") or ""),
        "PuzzleId": str(row.get("PuzzleId") or ""),
        "ground_truth": str(row.get("ground_truth") or ""),
    }
    return canonical_json_sha256(identity)


def audit_rollout_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    allowed_statuses: Sequence[str] = ("completed",),
) -> dict[str, Any]:
    """Audit one exact 256x8 rollout without applying a stage threshold."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("rollout seed must be a non-negative integer")
    if isinstance(allowed_statuses, (str, bytes)):
        raise ValueError("allowed_statuses must be a sequence of statuses")
    allowed_status_tuple = tuple(allowed_statuses)
    if (
        not allowed_status_tuple
        or len(set(allowed_status_tuple)) != len(allowed_status_tuple)
        or any(
            not isinstance(status, str) or not status
            for status in allowed_status_tuple
        )
    ):
        raise ValueError("allowed_statuses must be unique nonempty strings")
    allowed_status_set = set(allowed_status_tuple)
    materialized = list(rows)
    if len(materialized) != AUDIT_ROWS:
        raise ValueError(
            f"rollout rows must equal {AUDIT_ROWS}, got {len(materialized)}"
        )

    group_scores: dict[int, list[float]] = defaultdict(list)
    group_protocol: set[int] = set()
    group_prompt_fingerprints: dict[int, str] = {}
    status_counts = {status: 0 for status in allowed_status_tuple}
    positive_status_counts = {status: 0 for status in allowed_status_tuple}
    joint_rows = 0
    positive_samples = 0
    for row_number, row in enumerate(materialized, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"rollout row {row_number} is not an object")
        status = row.get("status")
        if status not in allowed_status_set:
            if allowed_status_tuple == ("completed",):
                raise ValueError(
                    f"rollout row {row_number} is not completed"
                )
            raise ValueError(
                f"rollout row {row_number} has disallowed status "
                f"{status!r}; expected one of {allowed_status_tuple!r}"
            )
        status_counts[str(status)] += 1
        group_index = row.get("group_index")
        if (
            isinstance(group_index, bool)
            or not isinstance(group_index, int)
            or group_index < 0
        ):
            raise ValueError(
                f"rollout row {row_number} has invalid group_index"
            )
        fingerprint = _prompt_fingerprint(row, row_number)
        previous = group_prompt_fingerprints.setdefault(
            group_index, fingerprint
        )
        if previous != fingerprint:
            raise ValueError(
                f"prompt identity changed inside group {group_index}"
            )
        score = _binary_score(row, row_number)
        group_scores[group_index].append(score)
        joint_valid = is_joint_valid_protocol_row(row)
        if score == 1.0 and not joint_valid:
            raise ValueError(
                f"rollout row {row_number} is positive without a "
                "joint-valid protocol parse"
            )
        positive_samples += int(score == 1.0)
        positive_status_counts[str(status)] += int(score == 1.0)
        if joint_valid:
            joint_rows += 1
            group_protocol.add(group_index)

    if len(group_scores) != AUDIT_GROUPS:
        raise ValueError(
            f"prompt groups must equal {AUDIT_GROUPS}, got {len(group_scores)}"
        )
    sizes = {len(scores) for scores in group_scores.values()}
    if sizes != {SAMPLES_PER_GROUP}:
        raise ValueError(
            "every prompt group must contain exactly "
            f"{SAMPLES_PER_GROUP} rows, got {sorted(sizes)}"
        )
    prompt_fingerprints = sorted(group_prompt_fingerprints.values())
    if len(set(prompt_fingerprints)) != AUDIT_GROUPS:
        raise ValueError("rollout contains duplicate prompt groups")
    variance_groups = sum(
        min(scores) < max(scores) for scores in group_scores.values()
    )
    p_protocol = joint_rows / AUDIT_ROWS
    return {
        "seed": seed,
        "rollout_rows": AUDIT_ROWS,
        "prompt_groups": AUDIT_GROUPS,
        "samples_per_group": SAMPLES_PER_GROUP,
        "allowed_statuses": list(allowed_status_tuple),
        "status_counts": status_counts,
        "status_rates": {
            status: count / AUDIT_ROWS
            for status, count in status_counts.items()
        },
        "positive_samples_by_status": positive_status_counts,
        "joint_valid_protocol_rows": joint_rows,
        "joint_valid_protocol_groups": len(group_protocol),
        "positive_samples": positive_samples,
        "nonzero_variance_groups": variance_groups,
        "p_protocol": p_protocol,
        "p_solve_given_protocol": (
            positive_samples / joint_rows if joint_rows else 0.0
        ),
        "p_total": positive_samples / AUDIT_ROWS,
        "variance_rate": variance_groups / AUDIT_GROUPS,
        "prompt_fingerprints": prompt_fingerprints,
        "prompt_set_sha256": canonical_json_sha256(prompt_fingerprints),
    }


def validate_exact_audit(metrics: Mapping[str, Any]) -> dict[str, Any]:
    exact = {
        "rollout_rows": AUDIT_ROWS,
        "prompt_groups": AUDIT_GROUPS,
        "samples_per_group": SAMPLES_PER_GROUP,
    }
    for key, expected in exact.items():
        if metrics.get(key) != expected:
            raise ValueError(f"{key} must equal {expected}")
    prompts = set(metrics.get("prompt_fingerprints", ()))
    if len(prompts) != AUDIT_GROUPS:
        raise ValueError("audit must expose 256 unique prompt fingerprints")
    return dict(metrics)


def validate_protocol_audit(metrics: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_exact_audit(metrics)
    if int(metrics.get("joint_valid_protocol_rows", -1)) < PROTOCOL_ROWS_MIN:
        raise ValueError(
            "joint_valid_protocol_rows must be at least "
            f"{PROTOCOL_ROWS_MIN}"
        )
    if (
        int(metrics.get("joint_valid_protocol_groups", -1))
        < PROTOCOL_GROUPS_MIN
    ):
        raise ValueError(
            "joint_valid_protocol_groups must be at least "
            f"{PROTOCOL_GROUPS_MIN}"
        )
    return validated


def validate_disjoint_prompt_sets(
    primary: Mapping[str, Any],
    confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    primary_valid = validate_exact_audit(primary)
    confirmation_valid = validate_exact_audit(confirmation)
    if primary_valid.get("seed") != PRIMARY_SEED:
        raise ValueError(f"primary seed must equal {PRIMARY_SEED}")
    confirmation_seed = confirmation_valid.get("seed")
    if (
        isinstance(confirmation_seed, bool)
        or not isinstance(confirmation_seed, int)
        or confirmation_seed < CONFIRMATION_FIRST_SEED
    ):
        raise ValueError(
            f"confirmation seed must be >= {CONFIRMATION_FIRST_SEED}"
        )
    primary_prompts = set(primary_valid.get("prompt_fingerprints", ()))
    confirmation_prompts = set(
        confirmation_valid.get("prompt_fingerprints", ())
    )
    if (
        len(primary_prompts) != AUDIT_GROUPS
        or len(confirmation_prompts) != AUDIT_GROUPS
    ):
        raise ValueError("both audits must expose 256 prompt fingerprints")
    intersection = sorted(primary_prompts & confirmation_prompts)
    if intersection:
        raise ValueError(
            f"confirmation audit overlaps {len(intersection)} primary prompts"
        )
    return {
        "primary": primary_valid,
        "confirmation": confirmation_valid,
        "prompt_intersection": 0,
        "combined_prompt_set_sha256": canonical_json_sha256(
            sorted(primary_prompts | confirmation_prompts)
        ),
    }


def validate_disjoint_audits(
    primary: Mapping[str, Any],
    confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    primary_valid = validate_protocol_audit(primary)
    confirmation_valid = validate_protocol_audit(confirmation)
    return validate_disjoint_prompt_sets(primary_valid, confirmation_valid)


def select_first_disjoint_prompt_confirmation(
    primary: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the first prompt-disjoint seed without outcome cherry-picking."""

    primary_valid = validate_exact_audit(primary)
    expected_seed = CONFIRMATION_FIRST_SEED
    primary_prompts = set(primary_valid.get("prompt_fingerprints", ()))
    if len(primary_prompts) != AUDIT_GROUPS:
        raise ValueError("primary audit must expose 256 prompt fingerprints")
    attempted: list[dict[str, int]] = []
    for candidate in candidates:
        candidate_valid = validate_exact_audit(candidate)
        seed = candidate.get("seed")
        if seed != expected_seed:
            raise ValueError(
                f"confirmation candidates must be consecutive from "
                f"{CONFIRMATION_FIRST_SEED}; expected {expected_seed}, got {seed}"
            )
        prompts = set(candidate_valid.get("prompt_fingerprints", ()))
        if len(prompts) != AUDIT_GROUPS:
            raise ValueError(
                f"confirmation seed {seed} lacks 256 prompt fingerprints"
            )
        overlap = len(primary_prompts & prompts)
        attempted.append({"seed": expected_seed, "prompt_overlap": overlap})
        if overlap == 0:
            validated = validate_disjoint_prompt_sets(
                primary_valid, candidate_valid
            )
            return {
                **validated,
                "selection_rule": (
                    "smallest_integer_at_least_43_with_zero_primary_"
                    "prompt_fingerprint_overlap"
                ),
                "attempted_seeds": attempted,
            }
        expected_seed += 1
    raise ValueError("no disjoint confirmation candidate was supplied")


def select_first_disjoint_prompt_set(
    primary: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select from authenticated prompt-set proofs; no rollout is required."""

    primary_valid = validate_exact_audit(primary)
    primary_prompts = set(primary_valid.get("prompt_fingerprints", ()))
    expected_seed = CONFIRMATION_FIRST_SEED
    attempted: list[dict[str, Any]] = []
    for candidate in candidates:
        seed = candidate.get("seed")
        if seed != expected_seed:
            raise ValueError(
                f"prompt-set candidates must be consecutive from "
                f"{CONFIRMATION_FIRST_SEED}; expected {expected_seed}, "
                f"got {seed}"
            )
        raw_prompts = candidate.get("prompt_fingerprints", ())
        if not isinstance(raw_prompts, Sequence) or isinstance(
            raw_prompts, (str, bytes)
        ):
            raise ValueError(f"prompt-set seed {seed} lacks fingerprints")
        prompts = set(raw_prompts)
        if len(prompts) != AUDIT_GROUPS:
            raise ValueError(
                f"prompt-set seed {seed} must contain 256 unique prompts"
            )
        prompt_set_sha256 = canonical_json_sha256(sorted(prompts))
        recorded_hash = candidate.get("prompt_set_sha256")
        if recorded_hash not in (None, prompt_set_sha256):
            raise ValueError(f"prompt-set seed {seed} hash mismatch")
        overlap = len(primary_prompts & prompts)
        attempted.append(
            {
                "seed": expected_seed,
                "prompt_overlap": overlap,
                "prompt_set_sha256": prompt_set_sha256,
            }
        )
        if overlap == 0:
            return {
                "selection_rule": (
                    "smallest_integer_at_least_43_with_zero_primary_"
                    "prompt_fingerprint_overlap"
                ),
                "primary_seed": PRIMARY_SEED,
                "primary_prompt_set_sha256": primary_valid.get(
                    "prompt_set_sha256"
                ),
                "confirmation_seed": expected_seed,
                "confirmation_prompt_set_sha256": prompt_set_sha256,
                "confirmation_prompt_fingerprints": sorted(prompts),
                "prompt_intersection": 0,
                "attempted_seeds": attempted,
            }
        expected_seed += 1
    raise ValueError("no disjoint prompt-set candidate was supplied")


def validate_confirmation_against_prompt_selection(
    primary: Mapping[str, Any],
    prompt_candidates: Sequence[Mapping[str, Any]],
    confirmation: Mapping[str, Any],
    *,
    require_protocol: bool = True,
) -> dict[str, Any]:
    """Bind the one generated confirmation to the selected prompt proof."""

    selection = select_first_disjoint_prompt_set(primary, prompt_candidates)
    confirmation_valid = validate_exact_audit(confirmation)
    if confirmation_valid.get("seed") != selection["confirmation_seed"]:
        raise ValueError("confirmation seed differs from prompt selection")
    if (
        confirmation_valid.get("prompt_set_sha256")
        != selection["confirmation_prompt_set_sha256"]
        or sorted(confirmation_valid.get("prompt_fingerprints", ()))
        != selection["confirmation_prompt_fingerprints"]
    ):
        raise ValueError("confirmation prompts differ from prompt selection")
    pair = (
        validate_disjoint_audits(primary, confirmation_valid)
        if require_protocol
        else validate_disjoint_prompt_sets(primary, confirmation_valid)
    )
    return {**pair, "prompt_selection": selection}


def select_first_disjoint_confirmation(
    primary: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the first disjoint seed and require both protocol thresholds."""

    selection = select_first_disjoint_prompt_confirmation(primary, candidates)
    validate_protocol_audit(selection["primary"])
    validate_protocol_audit(selection["confirmation"])
    return selection


def choose_smallest_eligible_weight(
    passing_audits: Mapping[float, Mapping[str, Any]],
) -> float:
    eligible = []
    for raw_weight, audit_pair in passing_audits.items():
        weight = float(raw_weight)
        if weight not in ELIGIBLE_SFT_LOSS_WEIGHTS:
            raise ValueError(f"unregistered SFT loss weight {weight}")
        if not isinstance(audit_pair, Mapping):
            raise ValueError(f"weight {weight} audit pair is not an object")
        validate_disjoint_audits(
            audit_pair.get("primary", {}),
            audit_pair.get("confirmation", {}),
        )
        eligible.append(weight)
    if not eligible:
        raise ValueError("no eligible SFT loss weight passed the protocol gate")
    return min(eligible)


def validate_monolithic_protocol_gate(
    *,
    selected_weight: float,
    candidate_weight: float,
    candidate_step: int,
    schedule_total_steps: int,
    manifest_leg: str,
    primary: Mapping[str, Any],
    confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    if float(candidate_weight) != float(selected_weight):
        raise ValueError("monolithic canary weight differs from selected weight")
    if candidate_step != PROTOCOL_CANDIDATE_STEP:
        raise ValueError("monolithic canary must stop at step 2000")
    if schedule_total_steps != 19_840:
        raise ValueError("monolithic canary must use the 19,840-step schedule")
    if manifest_leg != "p1+p2":
        raise ValueError("monolithic canary must use the full P1+P2 manifest")
    return validate_disjoint_audits(primary, confirmation)


def validate_rl_endpoint_gate(
    primary: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    *,
    dynamic_filter_attempted_groups: int | None = None,
    dynamic_filter_accepted_groups: int | None = None,
) -> dict[str, Any]:
    pair = validate_disjoint_prompt_sets(primary, confirmation)
    for label, metrics in (
        ("primary", pair["primary"]),
        ("confirmation", pair["confirmation"]),
    ):
        if int(metrics.get("positive_samples", -1)) < RL_POSITIVES_MIN:
            raise ValueError(
                f"{label} positive_samples must be at least {RL_POSITIVES_MIN}"
            )
        if (
            int(metrics.get("nonzero_variance_groups", -1))
            < RL_VARIANCE_GROUPS_MIN
        ):
            raise ValueError(
                f"{label} nonzero_variance_groups must be at least "
                f"{RL_VARIANCE_GROUPS_MIN}"
            )
    if (
        dynamic_filter_attempted_groups is None
        and dynamic_filter_accepted_groups is None
    ):
        return pair
    if dynamic_filter_accepted_groups != DYNAMIC_FILTER_ACCEPTED_GROUPS:
        raise ValueError(
            "dynamic filter must fill exactly "
            f"{DYNAMIC_FILTER_ACCEPTED_GROUPS} accepted groups"
        )
    if (
        isinstance(dynamic_filter_attempted_groups, bool)
        or not isinstance(dynamic_filter_attempted_groups, int)
        or not 0 < dynamic_filter_attempted_groups <= DYNAMIC_FILTER_ATTEMPTED_GROUP_CAP
    ):
        raise ValueError(
            "dynamic filter attempted groups must be in [1, 8192]"
        )
    return {
        **pair,
        "dynamic_filter_attempted_groups": dynamic_filter_attempted_groups,
        "dynamic_filter_accepted_groups": dynamic_filter_accepted_groups,
    }


def self_hash_marker(core: Mapping[str, Any]) -> dict[str, Any]:
    marker = dict(core)
    if "gate_sha256" in marker:
        raise ValueError("unhashed marker core cannot contain gate_sha256")
    marker["gate_sha256"] = canonical_json_sha256(marker)
    return marker


def validate_self_hashed_marker(marker: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(marker)
    recorded = value.pop("gate_sha256", None)
    if recorded != canonical_json_sha256(value):
        raise ValueError("gate marker self-hash mismatch")
    if value.get("contract_schema") != CONTRACT_SCHEMA:
        raise ValueError("gate marker contract schema mismatch")
    if value.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("gate marker contract version mismatch")
    if value.get("contract_plan_sha256") != CONTRACT_PLAN_SHA256:
        raise ValueError("gate marker plan hash mismatch")
    return dict(marker)
