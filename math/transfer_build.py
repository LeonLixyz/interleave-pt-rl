"""Verify, match, and serialize fixed-policy D0/D1 math generations.

The builder consumes the exact token IDs written by :mod:`transfer_generate`.
It never re-tokenizes decoded output.  Its pure functions are deliberately
usable with an injected verifier so schema, matching, checksums, and masks can
be tested without GPUs or the ``math-verify`` package.

The production matcher aggregates candidates with identical matching
statistics and solves the resulting sparse mixed-integer linear program with a
pinned SciPy/HiGHS runtime.  It maximizes the common assistant-token budget,
matches document counts in every difficulty stratum, enforces the processed
token tolerance, and prefers exact per-stratum assistant-token equality when
that is feasible at the maximum global budget.  Every solver result is checked
again with integer arithmetic before any corpus is written.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from transfer_generate import (
    DEFAULT_EOS_TOKEN_ID,
    GENERATION_SCHEMA_VERSION,
    canonical_json_bytes,
    derive_sample_seed,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    validate_generation_record,
    validate_prompt_record,
    write_jsonl_atomic,
    write_manifest_atomic,
)


VERIFICATION_SCHEMA_VERSION = "math-transfer-verification-v1"
CORPUS_SCHEMA_VERSION = "math-transfer-corpus-v1"
MATCH_SCHEMA_VERSION = "math-transfer-match-v1"
CANONICAL_VERIFIER_ID = "verl-math-verify-compatible-v1"
REQUIRED_MATH_VERIFY_VERSION = "0.5.2"
REQUIRED_SCIPY_VERSION = "1.17.1"
DEFAULT_ASSISTANT_TOKEN_CAP = 200_000_000
DEFAULT_MATCH_SELECTION_SEED = 20_260_710


class MatchingSolverError(RuntimeError):
    """Raised when the production matcher cannot prove an optimal solution."""


def _math_verify_version() -> str | None:
    try:
        return importlib.metadata.version("math-verify")
    except importlib.metadata.PackageNotFoundError:
        return None


def require_math_verify_version(
    expected: str = REQUIRED_MATH_VERIFY_VERSION,
    *,
    version_getter: Callable[[], str | None] | None = None,
) -> str:
    """Fail closed unless the verifier runtime exactly matches the protocol."""

    actual = (version_getter or _math_verify_version)()
    if actual != expected:
        raise RuntimeError(
            f"math-verify version mismatch: required {expected!r}, found {actual!r}"
        )
    return actual


def canonical_math_verify_worker(ground_truth: Any, model_output: str) -> float:
    """The exact parse/verify procedure used by the pinned verl RL scorer.

    This mirrors ``verl.utils.reward_score.math_verify._verify_in_subprocess``:
    gold is wrapped in ``\\boxed{...}``, gold uses LaTeX extraction, and model
    output uses expression followed by LaTeX extraction.
    """

    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

    ground_truth_boxed = "\\boxed{" + str(ground_truth) + "}"
    extracted_gold = parse(ground_truth_boxed, (LatexExtractionConfig(),))
    extracted_pred = parse(
        str(model_output), (ExprExtractionConfig(), LatexExtractionConfig())
    )
    if not extracted_gold or not extracted_pred:
        return 0.0
    return float(
        any(verify(gold, pred) for pred in extracted_pred for gold in extracted_gold)
    )


_VERIFY_POOL: concurrent.futures.ProcessPoolExecutor | None = None


def _default_verify_pool() -> concurrent.futures.ProcessPoolExecutor:
    global _VERIFY_POOL
    if _VERIFY_POOL is None:
        _VERIFY_POOL = concurrent.futures.ProcessPoolExecutor(
            max_workers=4, mp_context=multiprocessing.get_context("spawn")
        )
    return _VERIFY_POOL


def verify_completion_detailed(
    model_output: str,
    ground_truth: Any,
    *,
    timeout_s: float = 30.0,
    runner: Callable[[Any, str], float | bool] | None = None,
    executor: concurrent.futures.Executor | None = None,
) -> dict[str, Any]:
    """Return a non-lossy verification outcome.

    ``correct`` and ``incorrect`` are semantic results.  ``timeout`` and
    ``error`` are infrastructure/parser failures and must never be silently
    counted as incorrect examples during corpus matching.

    Supplying ``runner`` executes it directly and is the local pure-function
    test path.  Production calls use a process pool so the timeout is real.
    """

    verifier_version = _math_verify_version()
    try:
        if runner is not None:
            score = float(runner(ground_truth, model_output))
        else:
            pool = executor or _default_verify_pool()
            future = pool.submit(canonical_math_verify_worker, ground_truth, model_output)
            score = float(future.result(timeout=timeout_s))
        status = "correct" if score > 0.5 else "incorrect"
        return {
            "schema_version": VERIFICATION_SCHEMA_VERSION,
            "verifier_id": CANONICAL_VERIFIER_ID,
            "math_verify_version": verifier_version,
            "status": status,
            "score": score,
            "error_type": None,
            "error_message": None,
        }
    except (TimeoutError, concurrent.futures.TimeoutError) as exc:
        return {
            "schema_version": VERIFICATION_SCHEMA_VERSION,
            "verifier_id": CANONICAL_VERIFIER_ID,
            "math_verify_version": verifier_version,
            "status": "timeout",
            "score": None,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:500] or None,
        }
    except Exception as exc:  # verifier errors are data, not implicit negatives.
        return {
            "schema_version": VERIFICATION_SCHEMA_VERSION,
            "verifier_id": CANONICAL_VERIFIER_ID,
            "math_verify_version": verifier_version,
            "status": "error",
            "score": None,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:500] or None,
        }


def prompt_map(prompts: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for prompt in prompts:
        validate_prompt_record(prompt)
        uid = str(prompt["problem_uid"])
        if uid in out:
            raise ValueError(f"duplicate problem_uid in prompt manifest: {uid}")
        out[uid] = dict(prompt)
    if not out:
        raise ValueError("prompt manifest is empty")
    return out


def verify_generation_records(
    records: Iterable[Mapping[str, Any]],
    prompts_by_uid: Mapping[str, Mapping[str, Any]],
    *,
    timeout_s: float = 30.0,
    runner: Callable[[Any, str], float | bool] | None = None,
    executor: concurrent.futures.Executor | None = None,
) -> list[dict[str, Any]]:
    """Validate provenance and attach detailed verification to every record."""

    verified: list[dict[str, Any]] = []
    seen_request_keys: set[tuple[str, str, int]] = set()
    sorted_records = sorted(
        records,
        key=lambda row: (
            str(row.get("arm", "")),
            str(row.get("problem_uid", "")),
            int(row.get("sample_index", -1)),
        ),
    )
    for raw_record in sorted_records:
        record = dict(raw_record)
        validate_generation_record(record)
        uid = str(record["problem_uid"])
        try:
            prompt = prompts_by_uid[uid]
        except KeyError as exc:
            raise ValueError(f"generation references unknown problem_uid: {uid}") from exc
        if list(record["prompt_token_ids"]) != list(prompt["prompt_token_ids"]):
            raise ValueError(f"prompt token mismatch for {uid}")
        if record["dedup_group"] != prompt["dedup_group"]:
            raise ValueError(f"dedup group mismatch for {uid}")
        expected_seed_key = (
            str(record["arm"]),
            uid,
            int(record["sample_index"]),
        )
        if expected_seed_key in seen_request_keys:
            raise ValueError(f"duplicate generation request: {expected_seed_key}")
        seen_request_keys.add(expected_seed_key)
        record["verification"] = verify_completion_detailed(
            record["response_text"],
            prompt["ground_truth"],
            timeout_s=timeout_s,
            runner=runner,
            executor=executor,
        )
        record["ground_truth_sha256"] = sha256_bytes(
            canonical_json_bytes(prompt["ground_truth"])
        )
        verified.append(record)
    return verified


def validate_paired_generation_records(
    d0_records: Sequence[Mapping[str, Any]],
    d1_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require D0/D1 to differ only in policy/model output provenance.

    Every ``(problem_uid, sample_index)`` request must exist in both arms with
    the same prompt IDs, derived seed, tokenizer, and sampling configuration.
    """

    def index_arm(
        records: Sequence[Mapping[str, Any]], arm: str
    ) -> dict[tuple[str, int], Mapping[str, Any]]:
        indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
        for record in records:
            validate_generation_record(record)
            if record["arm"] != arm:
                raise ValueError(f"expected {arm}, found {record['arm']}")
            key = (str(record["problem_uid"]), int(record["sample_index"]))
            if key in indexed:
                raise ValueError(f"duplicate {arm} request: {key}")
            indexed[key] = record
        return indexed

    indexed = {"D0": index_arm(d0_records, "D0"), "D1": index_arm(d1_records, "D1")}
    keys_d0, keys_d1 = set(indexed["D0"]), set(indexed["D1"])
    if keys_d0 != keys_d1:
        missing_d0 = sorted(keys_d1 - keys_d0)[:10]
        missing_d1 = sorted(keys_d0 - keys_d1)[:10]
        raise ValueError(
            f"unpaired request sets; missing_D0={missing_d0}, missing_D1={missing_d1}"
        )
    tokenizer_hashes: set[str] = set()
    sampling_hashes: set[str] = set()
    for key in sorted(keys_d0):
        left, right = indexed["D0"][key], indexed["D1"][key]
        for field in ("prompt_token_ids", "sample_seed", "sampling_config", "tokenizer_sha256"):
            if left[field] != right[field]:
                raise ValueError(f"paired generation mismatch for {key}: {field}")
        tokenizer_hashes.add(str(left["tokenizer_sha256"]))
        sampling_hashes.add(sha256_bytes(canonical_json_bytes(left["sampling_config"])))
        expected_seed = derive_sample_seed(
            int(left["sampling_config"]["base_seed"]), key[0], key[1]
        )
        if int(left["sample_seed"]) != expected_seed:
            raise ValueError(f"noncanonical derived seed for paired request {key}")
    if len(tokenizer_hashes) != 1:
        raise ValueError(f"paired run used multiple tokenizer hashes: {sorted(tokenizer_hashes)}")
    if len(sampling_hashes) != 1:
        raise ValueError("paired run used multiple sampling configurations")
    model_hashes = {
        arm: sorted({str(record["model_bundle_sha256"]) for record in records})
        for arm, records in (("D0", d0_records), ("D1", d1_records))
    }
    if any(len(values) != 1 for values in model_hashes.values()):
        raise ValueError(f"each arm must use exactly one model bundle: {model_hashes}")
    if model_hashes["D0"] == model_hashes["D1"]:
        raise ValueError("D0 and D1 use the same model bundle")
    return {
        "request_count_per_arm": len(keys_d0),
        "tokenizer_sha256": next(iter(tokenizer_hashes)),
        "sampling_config_sha256": next(iter(sampling_hashes)),
        "model_bundle_sha256": model_hashes,
    }


def _candidate_id(record: Mapping[str, Any]) -> str:
    identity = {
        "arm": record["arm"],
        "problem_uid": record["problem_uid"],
        "response_sha256": record["response_sha256"],
        "sample_index": record["sample_index"],
    }
    return sha256_bytes(canonical_json_bytes(identity))


def cap_and_deduplicate_candidates(
    verified_records: Iterable[Mapping[str, Any]],
    prompts_by_uid: Mapping[str, Mapping[str, Any]],
    *,
    per_prompt_cap: int = 8,
    eos_token_id: int = DEFAULT_EOS_TOKEN_ID,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep verified outputs, remove duplicates, and enforce a semantic-prompt cap."""

    if per_prompt_cap <= 0:
        raise ValueError("per_prompt_cap must be positive")
    accepted = rejected = duplicate = capped = 0
    seen_outputs: set[tuple[str, str, str]] = set()
    by_semantic_prompt: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw_record in sorted(
        verified_records,
        key=lambda row: (
            str(row["arm"]),
            str(row["dedup_group"]),
            int(row["sample_index"]),
            str(row["problem_uid"]),
            str(row["response_sha256"]),
        ),
    ):
        record = dict(raw_record)
        verification = record.get("verification") or {}
        if verification.get("status") != "correct":
            rejected += 1
            continue
        key = (
            str(record["arm"]),
            str(record["dedup_group"]),
            str(record["response_sha256"]),
        )
        if key in seen_outputs:
            duplicate += 1
            continue
        seen_outputs.add(key)
        prompt = prompts_by_uid[str(record["problem_uid"])]
        response_ids = list(record["response_token_ids"])
        eos_appended = not response_ids or response_ids[-1] != eos_token_id
        assistant_token_count = len(response_ids) + int(eos_appended)
        candidate = dict(record)
        candidate.update(
            {
                "candidate_id": _candidate_id(record),
                "difficulty_bin": prompt["difficulty_bin"],
                "assistant_token_count": assistant_token_count,
                "processed_token_count": len(prompt["prompt_token_ids"])
                + assistant_token_count,
                "eos_token_id": int(eos_token_id),
                "eos_appended": eos_appended,
            }
        )
        by_semantic_prompt[(str(record["arm"]), str(record["dedup_group"]))].append(
            candidate
        )

    candidates: list[dict[str, Any]] = []
    for key in sorted(by_semantic_prompt):
        rows = by_semantic_prompt[key]
        rows.sort(
            key=lambda row: (
                int(row["sample_index"]),
                str(row["problem_uid"]),
                str(row["response_sha256"]),
            )
        )
        candidates.extend(rows[:per_prompt_cap])
        capped += max(0, len(rows) - per_prompt_cap)
    candidates.sort(
        key=lambda row: (
            str(row["arm"]),
            str(row["difficulty_bin"]),
            str(row["dedup_group"]),
            str(row["candidate_id"]),
        )
    )
    accepted = len(candidates)
    return candidates, {
        "selected_candidates": accepted,
        "non_correct": rejected,
        "duplicate_outputs": duplicate,
        "over_cap": capped,
    }


def require_scipy_version(
    expected: str = REQUIRED_SCIPY_VERSION,
    *,
    version_getter: Callable[[], str | None] | None = None,
) -> str:
    """Fail closed unless the exact preregistered MILP runtime is installed."""

    if version_getter is None:
        try:
            actual = importlib.metadata.version("scipy")
        except importlib.metadata.PackageNotFoundError:
            actual = None
    else:
        actual = version_getter()
    if actual != expected:
        raise RuntimeError(
            f"scipy version mismatch: required {expected!r}, found {actual!r}"
        )
    return actual


def _stable_selection_key(candidate_id: str, selection_seed: int) -> tuple[str, str]:
    digest = sha256_bytes(
        canonical_json_bytes(
            {"candidate_id": candidate_id, "selection_seed": int(selection_seed)}
        )
    )
    return digest, candidate_id


def _prepare_match_buckets(
    candidates: Iterable[Mapping[str, Any]],
    *,
    arms: tuple[str, str],
    selection_seed: int,
) -> tuple[list[dict[str, Any]], list[str], str, int]:
    """Validate, canonicalize, and aggregate matching-equivalent candidates."""

    allowed_arms = set(arms)
    if len(allowed_arms) != 2:
        raise ValueError(f"arms must contain two distinct labels: {arms!r}")
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    seen_candidate_ids: set[str] = set()
    input_identity: list[dict[str, Any]] = []
    candidate_count = 0
    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        arm = str(candidate.get("arm", ""))
        if arm not in allowed_arms:
            raise ValueError(f"unexpected candidate arm {arm!r}; expected one of {arms!r}")
        candidate_id = str(candidate.get("candidate_id", ""))
        if not candidate_id:
            raise ValueError("matching candidate is missing candidate_id")
        if candidate_id in seen_candidate_ids:
            raise ValueError(f"duplicate matching candidate_id: {candidate_id}")
        seen_candidate_ids.add(candidate_id)
        stratum = str(candidate.get("difficulty_bin", ""))
        if not stratum:
            raise ValueError(f"candidate {candidate_id} is missing difficulty_bin")
        try:
            assistant_tokens = int(candidate["assistant_token_count"])
            processed_tokens = int(candidate["processed_token_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"candidate {candidate_id} has invalid token counts") from exc
        if assistant_tokens <= 0:
            raise ValueError(
                f"candidate {candidate_id} assistant_token_count must be positive"
            )
        if processed_tokens < assistant_tokens:
            raise ValueError(
                f"candidate {candidate_id} processed_token_count is smaller than "
                "assistant_token_count"
            )
        candidate["arm"] = arm
        candidate["candidate_id"] = candidate_id
        candidate["difficulty_bin"] = stratum
        candidate["assistant_token_count"] = assistant_tokens
        candidate["processed_token_count"] = processed_tokens
        grouped[(arm, stratum, assistant_tokens, processed_tokens)].append(candidate)
        input_identity.append(
            {
                "arm": arm,
                "candidate_id": candidate_id,
                "difficulty_bin": stratum,
                "assistant_token_count": assistant_tokens,
                "processed_token_count": processed_tokens,
            }
        )
        candidate_count += 1
    if not candidate_count:
        raise ValueError("matching candidate set is empty")

    arm_order = {arm: index for index, arm in enumerate(arms)}
    buckets: list[dict[str, Any]] = []
    for key in sorted(
        grouped,
        key=lambda value: (arm_order[value[0]], value[1], value[2], value[3]),
    ):
        arm, stratum, assistant_tokens, processed_tokens = key
        rows = sorted(
            grouped[key],
            key=lambda row: _stable_selection_key(
                str(row["candidate_id"]), selection_seed
            ),
        )
        bucket_identity = {
            "arm": arm,
            "difficulty_bin": stratum,
            "assistant_token_count": assistant_tokens,
            "processed_token_count": processed_tokens,
            "selection_seed": int(selection_seed),
        }
        buckets.append(
            {
                **bucket_identity,
                "capacity": len(rows),
                "rows": rows,
                "tie_weight": 1
                + int(sha256_bytes(canonical_json_bytes(bucket_identity))[:12], 16)
                % 1_000_003,
            }
        )
    strata = sorted({str(bucket["difficulty_bin"]) for bucket in buckets})
    input_identity.sort(key=lambda row: (str(row["arm"]), str(row["candidate_id"])))
    input_sha256 = sha256_bytes(canonical_json_bytes(input_identity))
    return buckets, strata, input_sha256, candidate_count


def _matching_metrics(
    buckets: Sequence[Mapping[str, Any]],
    counts: Sequence[int],
    *,
    arms: tuple[str, str],
    strata: Sequence[str],
) -> dict[str, Any]:
    assistant = {arm: 0 for arm in arms}
    processed = {arm: 0 for arm in arms}
    documents = {arm: 0 for arm in arms}
    by_stratum = {
        stratum: {
            "assistant": {arm: 0 for arm in arms},
            "processed": {arm: 0 for arm in arms},
            "documents": {arm: 0 for arm in arms},
        }
        for stratum in strata
    }
    for bucket, raw_count in zip(buckets, counts, strict=True):
        count = int(raw_count)
        arm = str(bucket["arm"])
        stratum = str(bucket["difficulty_bin"])
        assistant_tokens = count * int(bucket["assistant_token_count"])
        processed_tokens = count * int(bucket["processed_token_count"])
        assistant[arm] += assistant_tokens
        processed[arm] += processed_tokens
        documents[arm] += count
        by_stratum[stratum]["assistant"][arm] += assistant_tokens
        by_stratum[stratum]["processed"][arm] += processed_tokens
        by_stratum[stratum]["documents"][arm] += count
    arm_a, arm_b = arms
    return {
        "assistant": assistant,
        "processed": processed,
        "documents": documents,
        "by_stratum": by_stratum,
        "stratum_assistant_abs_delta": sum(
            abs(values["assistant"][arm_a] - values["assistant"][arm_b])
            for values in by_stratum.values()
        ),
        "processed_abs_delta": abs(processed[arm_a] - processed[arm_b]),
    }


def _solver_version_details() -> dict[str, str]:
    import scipy
    from scipy.optimize._highspy import _core as highs_core

    return {
        "scipy": str(scipy.__version__),
        "highs": ".".join(
            str(value)
            for value in (
                highs_core.HIGHS_VERSION_MAJOR,
                highs_core.HIGHS_VERSION_MINOR,
                highs_core.HIGHS_VERSION_PATCH,
            )
        ),
    }


def _solve_matching_milp(
    buckets: Sequence[Mapping[str, Any]],
    strata: Sequence[str],
    *,
    arms: tuple[str, str],
    phase: str,
    objective_kind: str,
    max_processed_token_relative_delta: float,
    assistant_token_cap: int | None,
    fixed_assistant_budget: int | None = None,
    exact_assistant_tokens_per_stratum: bool = False,
    max_stratum_assistant_abs_delta: int | None = None,
    max_processed_abs_delta: int | None = None,
    require_every_shared_stratum: bool = False,
    solver_time_limit_s: float | None = 3_600.0,
    allow_infeasible: bool = False,
) -> tuple[list[int] | None, dict[str, Any]]:
    """Solve one lexicographic phase of the sparse aggregated MILP."""

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_array

    if objective_kind not in {
        "assistant_budget",
        "feasibility",
        "stratum_imbalance",
        "processed_delta",
        "tie_break",
    }:
        raise ValueError(f"unknown MILP objective {objective_kind!r}")
    arm_a, arm_b = arms
    select_variable_count = len(buckets)
    lower_bounds = [0.0] * select_variable_count
    upper_bounds = [float(bucket["capacity"]) for bucket in buckets]
    integrality = [1] * select_variable_count
    objective = [0.0] * select_variable_count
    rows: list[dict[int, float]] = []
    row_lower: list[float] = []
    row_upper: list[float] = []

    def add_constraint(
        coefficients: Mapping[int, float], lower: float, upper: float
    ) -> None:
        rows.append(dict(coefficients))
        row_lower.append(float(lower))
        row_upper.append(float(upper))

    def coefficient_row(
        value_field: str,
        *,
        stratum: str | None = None,
        one_arm: str | None = None,
    ) -> dict[int, float]:
        coefficients: dict[int, float] = {}
        for index, bucket in enumerate(buckets):
            if stratum is not None and bucket["difficulty_bin"] != stratum:
                continue
            arm = str(bucket["arm"])
            if one_arm is not None:
                if arm != one_arm:
                    continue
                sign = 1.0
            else:
                sign = 1.0 if arm == arm_a else -1.0
            value = 1 if value_field == "documents" else int(bucket[value_field])
            coefficients[index] = sign * value
        return coefficients

    shared_strata: set[str] = set()
    for stratum in strata:
        count_difference = coefficient_row("documents", stratum=stratum)
        add_constraint(count_difference, 0.0, 0.0)
        available_arms = {
            str(bucket["arm"])
            for bucket in buckets
            if bucket["difficulty_bin"] == stratum
        }
        if available_arms == set(arms):
            shared_strata.add(stratum)
            if require_every_shared_stratum:
                add_constraint(
                    coefficient_row("documents", stratum=stratum, one_arm=arm_a),
                    1.0,
                    np.inf,
                )

    assistant_difference = coefficient_row("assistant_token_count")
    add_constraint(assistant_difference, 0.0, 0.0)
    assistant_arm_a = coefficient_row("assistant_token_count", one_arm=arm_a)
    if assistant_token_cap is not None:
        add_constraint(assistant_arm_a, -np.inf, float(assistant_token_cap))
    if fixed_assistant_budget is not None:
        add_constraint(
            assistant_arm_a,
            float(fixed_assistant_budget),
            float(fixed_assistant_budget),
        )

    processed_difference = coefficient_row("processed_token_count")
    processed_arm_a = coefficient_row("processed_token_count", one_arm=arm_a)
    processed_arm_b = coefficient_row("processed_token_count", one_arm=arm_b)
    ratio = 1.0 + float(max_processed_token_relative_delta)
    add_constraint(
        {
            **processed_arm_a,
            **{
                index: -ratio * value
                for index, value in processed_arm_b.items()
            },
        },
        -np.inf,
        0.0,
    )
    add_constraint(
        {
            **processed_arm_b,
            **{
                index: -ratio * value
                for index, value in processed_arm_a.items()
            },
        },
        -np.inf,
        0.0,
    )
    if max_processed_abs_delta is not None:
        add_constraint(
            processed_difference,
            -float(max_processed_abs_delta),
            float(max_processed_abs_delta),
        )

    stratum_difference_rows = {
        stratum: coefficient_row("assistant_token_count", stratum=stratum)
        for stratum in strata
    }
    if exact_assistant_tokens_per_stratum:
        for coefficients in stratum_difference_rows.values():
            add_constraint(coefficients, 0.0, 0.0)

    stratum_auxiliary_indices: list[int] = []
    needs_stratum_auxiliaries = (
        objective_kind == "stratum_imbalance"
        or max_stratum_assistant_abs_delta is not None
    )
    if needs_stratum_auxiliaries:
        max_assistant_total = sum(
            int(bucket["capacity"]) * int(bucket["assistant_token_count"])
            for bucket in buckets
        )
        for coefficients in stratum_difference_rows.values():
            auxiliary_index = len(lower_bounds)
            stratum_auxiliary_indices.append(auxiliary_index)
            lower_bounds.append(0.0)
            upper_bounds.append(float(max_assistant_total))
            integrality.append(0)
            objective.append(0.0)
            add_constraint({**coefficients, auxiliary_index: -1.0}, -np.inf, 0.0)
            add_constraint(
                {
                    **{index: -value for index, value in coefficients.items()},
                    auxiliary_index: -1.0,
                },
                -np.inf,
                0.0,
            )
        if max_stratum_assistant_abs_delta is not None:
            add_constraint(
                {index: 1.0 for index in stratum_auxiliary_indices},
                -np.inf,
                float(max_stratum_assistant_abs_delta),
            )

    processed_auxiliary_index: int | None = None
    if objective_kind == "processed_delta":
        processed_auxiliary_index = len(lower_bounds)
        max_processed_total = sum(
            int(bucket["capacity"]) * int(bucket["processed_token_count"])
            for bucket in buckets
        )
        lower_bounds.append(0.0)
        upper_bounds.append(float(max_processed_total))
        integrality.append(0)
        objective.append(0.0)
        add_constraint(
            {**processed_difference, processed_auxiliary_index: -1.0},
            -np.inf,
            0.0,
        )
        add_constraint(
            {
                **{index: -value for index, value in processed_difference.items()},
                processed_auxiliary_index: -1.0,
            },
            -np.inf,
            0.0,
        )

    if objective_kind == "assistant_budget":
        for index, bucket in enumerate(buckets):
            if bucket["arm"] == arm_a:
                objective[index] = -float(bucket["assistant_token_count"])
    elif objective_kind == "stratum_imbalance":
        for index in stratum_auxiliary_indices:
            objective[index] = 1.0
    elif objective_kind == "processed_delta":
        assert processed_auxiliary_index is not None
        objective[processed_auxiliary_index] = 1.0
    elif objective_kind == "tie_break":
        for index, bucket in enumerate(buckets):
            objective[index] = float(bucket["tie_weight"])

    matrix_row: list[int] = []
    matrix_column: list[int] = []
    matrix_data: list[float] = []
    for row_index, coefficients in enumerate(rows):
        for column_index, value in coefficients.items():
            if value:
                matrix_row.append(row_index)
                matrix_column.append(column_index)
                matrix_data.append(float(value))
    matrix = coo_array(
        (matrix_data, (matrix_row, matrix_column)),
        shape=(len(rows), len(lower_bounds)),
    ).tocsc()
    options: dict[str, Any] = {"mip_rel_gap": 0.0, "presolve": True}
    if solver_time_limit_s is not None:
        if solver_time_limit_s <= 0:
            raise ValueError("solver_time_limit_s must be positive or None")
        options["time_limit"] = float(solver_time_limit_s)
    result = milp(
        c=np.asarray(objective, dtype=np.float64),
        integrality=np.asarray(integrality, dtype=np.uint8),
        bounds=Bounds(
            np.asarray(lower_bounds, dtype=np.float64),
            np.asarray(upper_bounds, dtype=np.float64),
        ),
        constraints=LinearConstraint(
            matrix,
            np.asarray(row_lower, dtype=np.float64),
            np.asarray(row_upper, dtype=np.float64),
        ),
        options=options,
    )
    phase_summary = {
        "phase": phase,
        "objective": objective_kind,
        "status": int(result.status),
        "success": bool(result.success),
        "message": str(result.message),
        "variables": len(lower_bounds),
        "integer_selection_variables": select_variable_count,
        "constraints": len(rows),
        "mip_rel_gap_requested": 0.0,
        "presolve": True,
        "time_limit_s": solver_time_limit_s,
        "mip_node_count": (
            None
            if getattr(result, "mip_node_count", None) is None
            else int(result.mip_node_count)
        ),
        "mip_gap": (
            None if getattr(result, "mip_gap", None) is None else float(result.mip_gap)
        ),
    }
    if int(result.status) == 2 and allow_infeasible:
        phase_summary["proven_infeasible"] = True
        return None, phase_summary
    if int(result.status) != 0 or not result.success or result.x is None:
        raise MatchingSolverError(
            f"MILP phase {phase!r} did not prove optimality: "
            f"status={result.status}, message={result.message!r}, "
            f"mip_gap={getattr(result, 'mip_gap', None)!r}"
        )
    raw_selection = np.asarray(result.x[:select_variable_count], dtype=np.float64)
    rounded_selection = np.rint(raw_selection).astype(np.int64)
    if np.max(np.abs(raw_selection - rounded_selection), initial=0.0) > 1e-5:
        raise MatchingSolverError(
            f"MILP phase {phase!r} returned nonintegral selection variables"
        )
    counts = [int(value) for value in rounded_selection]
    for count, bucket in zip(counts, buckets, strict=True):
        if count < 0 or count > int(bucket["capacity"]):
            raise MatchingSolverError(
                f"MILP phase {phase!r} returned a count outside bucket bounds"
            )
    phase_summary["objective_value"] = float(result.fun)
    return counts, phase_summary


def enforce_processed_token_tolerance(
    processed_tokens: Mapping[str, int],
    *,
    arms: tuple[str, str] = ("D0", "D1"),
    max_relative_delta: float = 0.001,
) -> float:
    """Fail closed when paired processed-token exposure exceeds tolerance."""

    if not 0 <= max_relative_delta <= 1:
        raise ValueError("max_relative_delta must be in [0, 1]")
    left, right = (int(processed_tokens[arm]) for arm in arms)
    if left <= 0 or right <= 0:
        raise ValueError(f"processed-token totals must be positive: {dict(processed_tokens)}")
    relative_delta = abs(left - right) / min(left, right)
    if relative_delta > max_relative_delta:
        raise ValueError(
            "processed-token mismatch exceeds tolerance: "
            f"{arms[0]}={left:,}, {arms[1]}={right:,}, "
            f"relative_delta={relative_delta:.6f} > {max_relative_delta:.6f}"
        )
    return relative_delta


def match_equal_assistant_tokens(
    candidates: Iterable[Mapping[str, Any]],
    *,
    arms: tuple[str, str] = ("D0", "D1"),
    max_states_per_arm_and_stratum: int = 250_000,
    max_processed_token_relative_delta: float = 0.001,
    require_every_shared_stratum: bool = False,
    assistant_token_cap: int | None = DEFAULT_ASSISTANT_TOKEN_CAP,
    selection_seed: int = DEFAULT_MATCH_SELECTION_SEED,
    solver_time_limit_s: float | None = 3_600.0,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Select an optimal, protocol-valid D0/D1 pair with sparse MILP.

    Matching is lexicographic and fail-closed:

    1. maximize the common assistant-token budget (up to the protocol cap),
    2. use exact assistant-token equality in every stratum if feasible at that
       maximum budget; otherwise minimize total per-stratum imbalance,
    3. minimize the absolute processed-token difference, and
    4. apply a seeded deterministic tie-break without changing 1--3.

    ``max_states_per_arm_and_stratum`` is retained only for CLI compatibility
    with pilot commands from the retired subset-DP matcher and has no effect.
    """

    del max_states_per_arm_and_stratum
    require_scipy_version()
    if not 0 <= max_processed_token_relative_delta <= 1:
        raise ValueError("max_processed_token_relative_delta must be in [0, 1]")
    if assistant_token_cap is not None and assistant_token_cap <= 0:
        raise ValueError("assistant_token_cap must be positive or None")
    buckets, strata, input_sha256, candidate_count = _prepare_match_buckets(
        candidates, arms=arms, selection_seed=selection_seed
    )
    arm_a, arm_b = arms
    phases: list[dict[str, Any]] = []

    primary_counts, phase = _solve_matching_milp(
        buckets,
        strata,
        arms=arms,
        phase="maximize_common_assistant_budget",
        objective_kind="assistant_budget",
        max_processed_token_relative_delta=max_processed_token_relative_delta,
        assistant_token_cap=assistant_token_cap,
        require_every_shared_stratum=require_every_shared_stratum,
        solver_time_limit_s=solver_time_limit_s,
    )
    phases.append(phase)
    assert primary_counts is not None
    primary_metrics = _matching_metrics(
        buckets, primary_counts, arms=arms, strata=strata
    )
    assistant_budget = int(primary_metrics["assistant"][arm_a])
    if assistant_budget <= 0:
        raise ValueError("matcher selected no positive D0/D1 corpus")
    if assistant_budget != int(primary_metrics["assistant"][arm_b]):
        raise MatchingSolverError("primary MILP violated global assistant equality")

    strict_counts, phase = _solve_matching_milp(
        buckets,
        strata,
        arms=arms,
        phase="exact_per_stratum_minimize_processed_delta",
        objective_kind="processed_delta",
        max_processed_token_relative_delta=max_processed_token_relative_delta,
        assistant_token_cap=assistant_token_cap,
        fixed_assistant_budget=assistant_budget,
        exact_assistant_tokens_per_stratum=True,
        require_every_shared_stratum=require_every_shared_stratum,
        solver_time_limit_s=solver_time_limit_s,
        allow_infeasible=True,
    )
    phases.append(phase)
    exact_per_stratum = strict_counts is not None
    if exact_per_stratum:
        assert strict_counts is not None
        balanced_counts = strict_counts
        balanced_metrics = _matching_metrics(
            buckets, balanced_counts, arms=arms, strata=strata
        )
        stratum_abs_delta = 0
    else:
        imbalance_counts, phase = _solve_matching_milp(
            buckets,
            strata,
            arms=arms,
            phase="minimize_per_stratum_assistant_imbalance",
            objective_kind="stratum_imbalance",
            max_processed_token_relative_delta=max_processed_token_relative_delta,
            assistant_token_cap=assistant_token_cap,
            fixed_assistant_budget=assistant_budget,
            require_every_shared_stratum=require_every_shared_stratum,
            solver_time_limit_s=solver_time_limit_s,
        )
        phases.append(phase)
        assert imbalance_counts is not None
        imbalance_metrics = _matching_metrics(
            buckets, imbalance_counts, arms=arms, strata=strata
        )
        stratum_abs_delta = int(imbalance_metrics["stratum_assistant_abs_delta"])
        balanced_counts, phase = _solve_matching_milp(
            buckets,
            strata,
            arms=arms,
            phase="global_match_minimize_processed_delta",
            objective_kind="processed_delta",
            max_processed_token_relative_delta=max_processed_token_relative_delta,
            assistant_token_cap=assistant_token_cap,
            fixed_assistant_budget=assistant_budget,
            max_stratum_assistant_abs_delta=stratum_abs_delta,
            require_every_shared_stratum=require_every_shared_stratum,
            solver_time_limit_s=solver_time_limit_s,
        )
        phases.append(phase)
        assert balanced_counts is not None
        balanced_metrics = _matching_metrics(
            buckets, balanced_counts, arms=arms, strata=strata
        )
    processed_abs_delta = int(balanced_metrics["processed_abs_delta"])

    final_counts, phase = _solve_matching_milp(
        buckets,
        strata,
        arms=arms,
        phase="deterministic_tie_break",
        objective_kind="tie_break",
        max_processed_token_relative_delta=max_processed_token_relative_delta,
        assistant_token_cap=assistant_token_cap,
        fixed_assistant_budget=assistant_budget,
        exact_assistant_tokens_per_stratum=exact_per_stratum,
        max_stratum_assistant_abs_delta=(
            None if exact_per_stratum else stratum_abs_delta
        ),
        max_processed_abs_delta=processed_abs_delta,
        require_every_shared_stratum=require_every_shared_stratum,
        solver_time_limit_s=solver_time_limit_s,
    )
    phases.append(phase)
    assert final_counts is not None
    final_metrics = _matching_metrics(buckets, final_counts, arms=arms, strata=strata)

    selected: dict[str, list[dict[str, Any]]] = {arm_a: [], arm_b: []}
    for bucket, count in zip(buckets, final_counts, strict=True):
        selected[str(bucket["arm"])].extend(bucket["rows"][:count])
    for arm in selected:
        selected[arm].sort(
            key=lambda row: (
                str(row["difficulty_bin"]),
                str(row.get("dedup_group", "")),
                str(row["candidate_id"]),
            )
        )
    if not selected[arm_a] or not selected[arm_b]:
        raise ValueError("matcher selected no positive D0/D1 corpus")

    assistant_totals = {
        arm: sum(int(row["assistant_token_count"]) for row in rows)
        for arm, rows in selected.items()
    }
    processed_totals = {
        arm: sum(int(row["processed_token_count"]) for row in rows)
        for arm, rows in selected.items()
    }
    if assistant_totals != final_metrics["assistant"]:
        raise MatchingSolverError("selected rows disagree with aggregate assistant totals")
    if processed_totals != final_metrics["processed"]:
        raise MatchingSolverError("selected rows disagree with aggregate processed totals")
    if assistant_totals[arm_a] != assistant_totals[arm_b]:
        raise MatchingSolverError("final selection violates global assistant equality")
    if assistant_totals[arm_a] != assistant_budget:
        raise MatchingSolverError("deterministic tie-break changed the optimal budget")
    if assistant_token_cap is not None and assistant_budget > assistant_token_cap:
        raise MatchingSolverError("final selection exceeds assistant-token cap")
    processed_relative_delta = enforce_processed_token_tolerance(
        processed_totals,
        arms=arms,
        max_relative_delta=max_processed_token_relative_delta,
    )

    available_by_stratum = {
        stratum: {
            arm: sum(
                int(bucket["capacity"])
                for bucket in buckets
                if bucket["difficulty_bin"] == stratum and bucket["arm"] == arm
            )
            for arm in arms
        }
        for stratum in strata
    }
    stratum_summaries: dict[str, Any] = {}
    for stratum in strata:
        values = final_metrics["by_stratum"][stratum]
        documents = values["documents"]
        if documents[arm_a] != documents[arm_b]:
            raise MatchingSolverError(
                f"final selection violates document equality in {stratum!r}"
            )
        if (
            require_every_shared_stratum
            and all(available_by_stratum[stratum][arm] > 0 for arm in arms)
            and documents[arm_a] <= 0
        ):
            raise MatchingSolverError(
                f"final selection omitted required shared stratum {stratum!r}"
            )
        assistant_values = values["assistant"]
        processed_values = values["processed"]
        stratum_summaries[stratum] = {
            "status": "matched" if documents[arm_a] else "matched_zero_documents",
            "document_count_per_arm": documents[arm_a],
            "assistant_tokens": assistant_values,
            "assistant_tokens_per_arm": (
                assistant_values[arm_a]
                if assistant_values[arm_a] == assistant_values[arm_b]
                else None
            ),
            "assistant_token_abs_delta": abs(
                assistant_values[arm_a] - assistant_values[arm_b]
            ),
            "processed_tokens": processed_values,
            "processed_tokens_per_arm": (
                processed_values[arm_a]
                if processed_values[arm_a] == processed_values[arm_b]
                else None
            ),
            "available": available_by_stratum[stratum],
        }
    if exact_per_stratum and any(
        summary["assistant_token_abs_delta"] for summary in stratum_summaries.values()
    ):
        raise MatchingSolverError("strict per-stratum solution failed exact validation")
    measured_stratum_abs_delta = sum(
        summary["assistant_token_abs_delta"] for summary in stratum_summaries.values()
    )
    if measured_stratum_abs_delta != int(final_metrics["stratum_assistant_abs_delta"]):
        raise MatchingSolverError("stratum imbalance summary is internally inconsistent")
    if measured_stratum_abs_delta > stratum_abs_delta:
        raise MatchingSolverError("tie-break worsened the optimal stratum imbalance")
    if int(final_metrics["processed_abs_delta"]) > processed_abs_delta:
        raise MatchingSolverError("tie-break worsened the optimal processed-token balance")

    selected_identity = {
        arm: [str(row["candidate_id"]) for row in selected[arm]] for arm in arms
    }
    summary = {
        "schema_version": MATCH_SCHEMA_VERSION,
        "arms": [arm_a, arm_b],
        "assistant_tokens_per_arm": assistant_budget,
        "assistant_token_cap": assistant_token_cap,
        "selected_documents": {arm: len(rows) for arm, rows in selected.items()},
        "processed_tokens": processed_totals,
        "processed_token_relative_delta": processed_relative_delta,
        "max_processed_token_relative_delta": max_processed_token_relative_delta,
        "exact_assistant_tokens_per_stratum": exact_per_stratum,
        "stratum_assistant_abs_delta": measured_stratum_abs_delta,
        "strata": stratum_summaries,
        "solver": "scipy_milp_highs_sparse_aggregated_v1",
        "solver_versions": _solver_version_details(),
        "solver_phases": phases,
        "candidate_count": candidate_count,
        "aggregated_bucket_count": len(buckets),
        "aggregation_ratio": len(buckets) / candidate_count,
        "selection_seed": int(selection_seed),
        "matching_input_sha256": input_sha256,
        "selected_candidate_ids_sha256": sha256_bytes(
            canonical_json_bytes(selected_identity)
        ),
    }
    return selected, summary


def _atomic_raw_array(path: Path, chunks: Iterable[np.ndarray], dtype: np.dtype[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        for chunk in chunks:
            np.asarray(chunk, dtype=dtype).tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_training_corpus(
    selected_records: Iterable[Mapping[str, Any]],
    prompts_by_uid: Mapping[str, Mapping[str, Any]],
    *,
    arm: str,
    output_dir: str | os.PathLike[str],
    eos_token_id: int = DEFAULT_EOS_TOKEN_ID,
    vocab_size: int = 100_278,
    prompt_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Write raw uint32 tokens and raw bool assistant/EOS loss masks."""

    rows = [dict(row) for row in selected_records]
    if not rows:
        raise ValueError("cannot write an empty transfer corpus")
    if any(row["arm"] != arm for row in rows):
        raise ValueError(f"selected corpus contains an arm other than {arm}")
    rows.sort(
        key=lambda row: (
            str(row["difficulty_bin"]),
            str(row["dedup_group"]),
            str(row["candidate_id"]),
        )
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    token_path = destination / "token_ids_00000.npy"
    mask_path = destination / "labels_mask_00000.npy"

    token_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []
    selected_metadata: list[dict[str, Any]] = []
    offset = 0
    loss_bearing_tokens = 0
    for row in rows:
        validate_generation_record(row)
        prompt = prompts_by_uid[str(row["problem_uid"])]
        prompt_ids = [int(value) for value in prompt["prompt_token_ids"]]
        if prompt_ids != list(row["prompt_token_ids"]):
            raise ValueError(f"prompt token drift for {row['problem_uid']}")
        response_ids = [int(value) for value in row["response_token_ids"]]
        eos_appended = not response_ids or response_ids[-1] != eos_token_id
        assistant_ids = response_ids + ([int(eos_token_id)] if eos_appended else [])
        document_ids = prompt_ids + assistant_ids
        if not document_ids or min(document_ids) < 0 or max(document_ids) >= vocab_size:
            raise ValueError(
                f"token ID outside [0, {vocab_size}) for candidate {row['candidate_id']}"
            )
        mask = np.concatenate(
            (
                np.zeros(len(prompt_ids), dtype=np.bool_),
                np.ones(len(assistant_ids), dtype=np.bool_),
            )
        )
        token_chunks.append(np.asarray(document_ids, dtype=np.uint32))
        mask_chunks.append(mask)
        selected_metadata.append(
            {
                "schema_version": CORPUS_SCHEMA_VERSION,
                "arm": arm,
                "candidate_id": row["candidate_id"],
                "problem_uid": row["problem_uid"],
                "dedup_group": row["dedup_group"],
                "difficulty_bin": row["difficulty_bin"],
                "sample_index": row["sample_index"],
                "sample_seed": row["sample_seed"],
                "source_response_sha256": row["response_sha256"],
                "response_text": row["response_text"],
                "verification": row["verification"],
                "token_offset_start": offset,
                "token_offset_end": offset + len(document_ids),
                "prompt_token_count": len(prompt_ids),
                "assistant_token_count": len(assistant_ids),
                "processed_token_count": len(document_ids),
                "eos_appended": eos_appended,
                "model_bundle_sha256": row["model_bundle_sha256"],
                "tokenizer_sha256": row["tokenizer_sha256"],
            }
        )
        offset += len(document_ids)
        loss_bearing_tokens += len(assistant_ids)

    _atomic_raw_array(token_path, token_chunks, np.dtype(np.uint32))
    _atomic_raw_array(mask_path, mask_chunks, np.dtype(np.bool_))
    if token_path.stat().st_size // np.dtype(np.uint32).itemsize != offset:
        raise AssertionError("raw token file size does not match token count")
    if mask_path.stat().st_size // np.dtype(np.bool_).itemsize != offset:
        raise AssertionError("raw mask file size does not match token count")
    selected_artifact = write_jsonl_atomic(destination / "selected.jsonl", selected_metadata)
    file_artifacts = [
        {
            "path": token_path.name,
            "format": "raw_headerless_uint32",
            "bytes": token_path.stat().st_size,
            "sha256": sha256_file(token_path),
        },
        {
            "path": mask_path.name,
            "format": "raw_headerless_bool",
            "bytes": mask_path.stat().st_size,
            "sha256": sha256_file(mask_path),
        },
        selected_artifact,
    ]
    manifest = write_manifest_atomic(
        destination / "manifest.json",
        {
            "artifact_kind": "assistant_masked_transfer_corpus",
            "corpus_schema_version": CORPUS_SCHEMA_VERSION,
            "arm": arm,
            "document_count": len(rows),
            "processed_tokens": offset,
            "loss_bearing_assistant_tokens": loss_bearing_tokens,
            "masked_prompt_tokens": offset - loss_bearing_tokens,
            "eos_token_id": int(eos_token_id),
            "vocab_size": int(vocab_size),
            "prompt_manifest_sha256": prompt_manifest_sha256,
            "model_bundle_sha256": sorted(
                {str(row["model_bundle_sha256"]) for row in rows}
            ),
            "tokenizer_sha256": sorted({str(row["tokenizer_sha256"]) for row in rows}),
            "files": file_artifacts,
        },
    )
    return manifest


def _expand_generation_paths(value: str) -> list[str]:
    path = Path(value)
    if path.is_dir():
        files = sorted(str(candidate) for candidate in path.glob("part-*.jsonl"))
    else:
        files = sorted(glob.glob(value))
    if not files:
        raise FileNotFoundError(f"no generation JSONL files matched {value!r}")
    return files


def validate_generation_artifact_directory(
    directory: str | os.PathLike[str],
    *,
    expected_arm: str,
    expected_prompt_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify the generator manifest and every declared JSONL part checksum."""

    root = Path(directory)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing generation manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content_id = manifest.pop("content_id", None)
    expected_content_id = sha256_bytes(canonical_json_bytes(manifest))
    manifest["content_id"] = content_id
    if content_id != expected_content_id:
        raise ValueError(f"generation manifest content_id mismatch: {manifest_path}")
    if manifest.get("artifact_kind") != "fixed_policy_generations":
        raise ValueError(f"not a generation artifact manifest: {manifest_path}")
    if manifest.get("arm") != expected_arm:
        raise ValueError(
            f"generation manifest arm mismatch: expected {expected_arm}, got {manifest.get('arm')}"
        )
    if manifest.get("prompt_manifest_sha256") != expected_prompt_manifest_sha256:
        raise ValueError(f"generation used a different frozen prompt manifest: {manifest_path}")
    declared_parts = manifest.get("parts") or []
    if not declared_parts:
        raise ValueError(f"generation manifest declares no parts: {manifest_path}")
    verified_rows = 0
    verified_paths: list[str] = []
    for part in declared_parts:
        path = root / str(part["path"])
        if not path.is_file():
            raise FileNotFoundError(f"missing declared generation part: {path}")
        if path.stat().st_size != int(part["bytes"]):
            raise ValueError(f"generation part byte-count mismatch: {path}")
        if sha256_file(path) != part["sha256"]:
            raise ValueError(f"generation part checksum mismatch: {path}")
        row_count = sum(1 for _ in read_jsonl(path))
        if row_count != int(part["rows"]):
            raise ValueError(f"generation part row-count mismatch: {path}")
        verified_rows += row_count
        verified_paths.append(str(path))
    if verified_rows != int(manifest.get("generation_count", -1)):
        raise ValueError(f"generation_count mismatch: {manifest_path}")
    return {
        "manifest_verified": True,
        "manifest_path": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
        "content_id": content_id,
        "paths": verified_paths,
        "rows": verified_rows,
    }


def _load_generation_paths(paths: Sequence[str]) -> list[dict[str, Any]]:
    return [record for path in paths for record in read_jsonl(path)]


def build_pair(
    *,
    prompt_manifest_path: str,
    d0_generations: str,
    d1_generations: str,
    output_root: str,
    per_prompt_cap: int = 8,
    eos_token_id: int = DEFAULT_EOS_TOKEN_ID,
    vocab_size: int = 100_278,
    max_states: int = 250_000,
    max_processed_token_relative_delta: float = 0.001,
    assistant_token_cap: int | None = DEFAULT_ASSISTANT_TOKEN_CAP,
    match_selection_seed: int = DEFAULT_MATCH_SELECTION_SEED,
    solver_time_limit_s: float | None = 3_600.0,
    timeout_s: float = 30.0,
    runner: Callable[[Any, str], float | bool] | None = None,
) -> dict[str, Any]:
    """End-to-end local build path; ``runner`` enables dependency-free tests."""

    prompts = prompt_map(read_jsonl(prompt_manifest_path))
    prompt_manifest_sha256 = sha256_file(prompt_manifest_path)
    generation_paths = {
        "D0": _expand_generation_paths(d0_generations),
        "D1": _expand_generation_paths(d1_generations),
    }
    generation_descriptors: dict[str, dict[str, Any]] = {}
    for arm, value in (("D0", d0_generations), ("D1", d1_generations)):
        if Path(value).is_dir():
            generation_descriptors[arm] = validate_generation_artifact_directory(
                value,
                expected_arm=arm,
                expected_prompt_manifest_sha256=prompt_manifest_sha256,
            )
            generation_paths[arm] = generation_descriptors[arm]["paths"]
        else:
            generation_descriptors[arm] = {
                "manifest_verified": False,
                "paths": generation_paths[arm],
            }
    raw_by_arm = {
        arm: _load_generation_paths(generation_paths[arm]) for arm in ("D0", "D1")
    }
    pairing_provenance = validate_paired_generation_records(
        raw_by_arm["D0"], raw_by_arm["D1"]
    )
    verified_by_arm: dict[str, list[dict[str, Any]]] = {}
    candidates: list[dict[str, Any]] = []
    candidate_stats: dict[str, Any] = {}
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    for arm in ("D0", "D1"):
        records = raw_by_arm[arm]
        if any(record.get("arm") != arm for record in records):
            raise ValueError(f"{arm} input contains another arm")
        verified = verify_generation_records(
            records, prompts, timeout_s=timeout_s, runner=runner
        )
        verified_by_arm[arm] = verified
        write_jsonl_atomic(root / f"verified_{arm}.jsonl", verified)
        arm_candidates, stats = cap_and_deduplicate_candidates(
            verified,
            prompts,
            per_prompt_cap=per_prompt_cap,
            eos_token_id=eos_token_id,
        )
        candidates.extend(arm_candidates)
        candidate_stats[arm] = stats

    selected, match_summary = match_equal_assistant_tokens(
        candidates,
        max_states_per_arm_and_stratum=max_states,
        max_processed_token_relative_delta=max_processed_token_relative_delta,
        assistant_token_cap=assistant_token_cap,
        selection_seed=match_selection_seed,
        solver_time_limit_s=solver_time_limit_s,
    )
    manifests = {
        arm: write_training_corpus(
            selected[arm],
            prompts,
            arm=arm,
            output_dir=root / arm,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            prompt_manifest_sha256=prompt_manifest_sha256,
        )
        for arm in ("D0", "D1")
    }
    pair_manifest = write_manifest_atomic(
        root / "pair_manifest.json",
        {
            "artifact_kind": "matched_D0_D1_pair",
            "paired_generation_provenance": pairing_provenance,
            "match": match_summary,
            "candidate_stats": candidate_stats,
            "prompt_manifest_path": Path(prompt_manifest_path).name,
            "prompt_manifest_sha256": prompt_manifest_sha256,
            "generation_inputs": {
                arm: {
                    "descriptor": {
                        key: value
                        for key, value in generation_descriptors[arm].items()
                        if key != "paths"
                    },
                    "parts": [
                        {"path": Path(path).name, "sha256": sha256_file(path)}
                        for path in generation_paths[arm]
                    ],
                }
                for arm in ("D0", "D1")
            },
            "corpora": {arm: manifest["content_id"] for arm, manifest in manifests.items()},
        },
    )
    return pair_manifest


# ---------------------------------------------------------------------------
# Optional Modal wrapper. Pure imports and tests do not require Modal.
# ---------------------------------------------------------------------------

try:
    import modal
except ImportError:  # pragma: no cover - local pure-function path remains usable.
    modal = None


if modal is not None:
    _build_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "numpy>=1.26",
            f"scipy=={REQUIRED_SCIPY_VERSION}",
            f"math-verify=={REQUIRED_MATH_VERIFY_VERSION}",
        )
        .add_local_python_source("transfer_generate")
    )
    app = modal.App("math-transfer-build", image=_build_image)
    checkpoint_volume = modal.Volume.from_name(
        "olmo-core-checkpoints-v2", create_if_missing=True, version=2
    )

    @app.function(
        cpu=8,
        memory=32 * 1024,
        timeout=60 * 60 * 6,
        volumes={"/checkpoints": checkpoint_volume},
    )
    def build_pair_remote(
        prompt_manifest_path: str,
        d0_generations: str,
        d1_generations: str,
        output_root: str,
        per_prompt_cap: int = 8,
        eos_token_id: int = DEFAULT_EOS_TOKEN_ID,
        vocab_size: int = 100_278,
        max_states: int = 250_000,
        max_processed_token_relative_delta: float = 0.001,
        assistant_token_cap: int | None = DEFAULT_ASSISTANT_TOKEN_CAP,
        match_selection_seed: int = DEFAULT_MATCH_SELECTION_SEED,
        solver_time_limit_s: float | None = 3_600.0,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        """CPU artifact smoke/build with the preregistered verifier version."""

        require_math_verify_version()
        require_scipy_version()
        checkpoint_volume.reload()
        pair_manifest = build_pair(
            prompt_manifest_path=prompt_manifest_path,
            d0_generations=d0_generations,
            d1_generations=d1_generations,
            output_root=output_root,
            per_prompt_cap=per_prompt_cap,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            max_states=max_states,
            max_processed_token_relative_delta=max_processed_token_relative_delta,
            assistant_token_cap=assistant_token_cap,
            match_selection_seed=match_selection_seed,
            solver_time_limit_s=solver_time_limit_s,
            timeout_s=timeout_s,
        )
        checkpoint_volume.commit()
        return pair_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--d0-generations", required=True)
    parser.add_argument("--d1-generations", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--per-prompt-cap", type=int, default=8)
    parser.add_argument("--eos-token-id", type=int, default=DEFAULT_EOS_TOKEN_ID)
    parser.add_argument("--vocab-size", type=int, default=100_278)
    parser.add_argument("--max-states", type=int, default=250_000)
    parser.add_argument("--max-processed-token-relative-delta", type=float, default=0.001)
    parser.add_argument(
        "--assistant-token-cap", type=int, default=DEFAULT_ASSISTANT_TOKEN_CAP
    )
    parser.add_argument(
        "--match-selection-seed", type=int, default=DEFAULT_MATCH_SELECTION_SEED
    )
    parser.add_argument("--solver-time-limit-s", type=float, default=3_600.0)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    options = parser.parse_args()
    result = build_pair(
        prompt_manifest_path=options.prompt_manifest,
        d0_generations=options.d0_generations,
        d1_generations=options.d1_generations,
        output_root=options.output_root,
        per_prompt_cap=options.per_prompt_cap,
        eos_token_id=options.eos_token_id,
        vocab_size=options.vocab_size,
        max_states=options.max_states,
        max_processed_token_relative_delta=options.max_processed_token_relative_delta,
        assistant_token_cap=options.assistant_token_cap,
        match_selection_seed=options.match_selection_seed,
        solver_time_limit_s=options.solver_time_limit_s,
        timeout_s=options.timeout_s,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
