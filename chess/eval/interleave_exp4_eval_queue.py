"""Pure live-registry parsing for Experiment 4 RL evaluation stages.

The existing eight-stage core queue remains byte-for-byte pinned. Experiment 4
is mutable by design, so this helper only registers a final RL stage after its
controller has published a stable run ID and pretraining endpoint.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}")
_EXP4_MATRIX = {
    (filter_code, method)
    for filter_code in ("U", "D")
    for method in ("sft", "distill", "scratch")
}


def _conversion_origin(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if value.startswith("/pretrain-checkpoints/"):
        return value
    if value.startswith("/checkpoints/"):
        return "/pretrain-checkpoints/" + value.removeprefix("/checkpoints/")
    return None


def _planned_final_rl_stages(
    exp4: Mapping[str, Any],
) -> dict[str, tuple[str, str, str]]:
    """Return the immutable six-arm matrix keyed by planned RL stage ID."""

    arms = exp4.get("arms")
    if not isinstance(arms, list) or len(arms) != len(_EXP4_MATRIX):
        raise ValueError("registry.exp4 must contain exactly six planned arms")

    planned: dict[str, tuple[str, str, str]] = {}
    seen_arm_ids: set[str] = set()
    observed_matrix: set[tuple[str, str]] = set()
    for arm in arms:
        if not isinstance(arm, Mapping):
            raise ValueError("registry.exp4 arm must be an object")
        arm_id = str(arm.get("arm_id") or "")
        filter_code = str(arm.get("filter") or "").upper()
        method = str(arm.get("method") or "").lower()
        stages = arm.get("stages")
        rl = stages.get("rl") if isinstance(stages, Mapping) else None
        stage_id = str(rl.get("stage_id") or "") if isinstance(rl, Mapping) else ""
        if not arm_id or arm_id in seen_arm_ids:
            raise ValueError(f"duplicate or empty Exp4 arm_id {arm_id!r}")
        if not stage_id or stage_id in planned:
            raise ValueError(f"duplicate or empty Exp4 RL stage_id {stage_id!r}")
        seen_arm_ids.add(arm_id)
        observed_matrix.add((filter_code, method))
        planned[stage_id] = (arm_id, filter_code, method)

    if observed_matrix != _EXP4_MATRIX:
        raise ValueError(
            f"registry.exp4 arm matrix mismatch: "
            f"{observed_matrix} != {_EXP4_MATRIX}"
        )
    return planned


def flatten_exp4_rl_registry(
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return every fully registered Exp4 final-RL stage.

    Arms with a null ``run_id`` are intentionally omitted: they are planned,
    not discoverable training runs. A non-null run ID fails closed unless the
    controller also provides a usable clean-HF pretraining endpoint.
    """

    exp4 = registry.get("exp4")
    if exp4 is None:
        return []
    if not isinstance(exp4, Mapping):
        raise ValueError("registry.exp4 must be an object")
    planned = _planned_final_rl_stages(exp4)
    registrations = exp4.get("final_rl_runs", [])
    if not isinstance(registrations, list):
        raise ValueError("registry.exp4.final_rl_runs must be a list")
    if len(registrations) > len(planned):
        raise ValueError("registry.exp4 has more final RL runs than planned arms")
    fixed = exp4.get("fixed", {})
    if not isinstance(fixed, Mapping):
        raise ValueError("registry.exp4.fixed must be an object")
    fixed_rl = registry.get("fixed_rl")
    rl_app = exp4.get("rl_app")
    if not isinstance(fixed_rl, Mapping) or not isinstance(rl_app, Mapping):
        raise ValueError("registry exact Exp4 RL provenance contract is missing")
    rl_function_ids = rl_app.get("function_ids")
    if not isinstance(rl_function_ids, Mapping):
        raise ValueError("registry Exp4 RL function revisions are missing")
    default_target = int(fixed.get("final_rl_steps", 0))
    offset = int(fixed.get("rl1_steps", 0))
    seed = int(fixed.get("final_rl_seed", -1))
    if default_target <= 0 or offset < 0 or seed < 0:
        raise ValueError("registry.exp4 fixed RL settings are invalid")

    raw_root = str(registry["rl_raw_root"]).rstrip("/")
    if not raw_root.startswith("/rl-checkpoints/"):
        raise ValueError("registry RL root is outside the evaluator mount")
    model = str(registry["model_id"])
    experiment_version = str(registry["experiment_version"])
    exp4_version = str(exp4.get("version") or "")
    if not exp4_version:
        raise ValueError("registry.exp4.version must be non-empty")

    stages: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    seen_stage_ids: set[str] = set()
    for registration in registrations:
        if not isinstance(registration, Mapping):
            raise ValueError("Exp4 final RL registration must be an object")
        stage_id = str(registration.get("stage_id") or "")
        if stage_id not in planned:
            raise ValueError(f"unplanned Exp4 RL stage_id {stage_id!r}")
        run_name = registration.get("run_id")
        if registration.get("run_name") != run_name:
            raise ValueError(f"{stage_id} run_id/run_name mismatch")
        if not isinstance(run_name, str) or not _RUN_NAME_RE.fullmatch(run_name):
            raise ValueError(f"{stage_id} has invalid RL run_id")
        arm_id = str(registration["arm"])
        filter_code = str(registration["filter"]).upper()
        method = str(registration["method"]).lower()
        expected_arm, expected_filter, expected_method = planned[stage_id]
        if (arm_id, filter_code, method) != (
            expected_arm,
            expected_filter,
            expected_method,
        ):
            raise ValueError(
                f"{stage_id} registration does not match planned arm contract"
            )
        if filter_code not in {"U", "D"}:
            raise ValueError(f"unsupported Exp4 filter {filter_code}")
        if method not in {"sft", "distill", "scratch"}:
            raise ValueError(f"unsupported Exp4 method {method}")
        if stage_id in seen_stage_ids:
            raise ValueError(f"duplicate Exp4 RL stage_id {stage_id}")
        if run_name in seen_runs:
            raise ValueError(f"duplicate Exp4 RL run_id {run_name}")
        seen_stage_ids.add(stage_id)
        seen_runs.add(run_name)

        target = int(registration.get("target_steps", default_target))
        if target != default_target:
            raise ValueError(
                f"{stage_id} target {target} differs from fixed {default_target}"
            )
        effective_offset = int(registration.get("effective_step_offset", -1))
        rollout_seed = int(registration.get("rollout_seed", -1))
        save_interval = int(registration.get("save_interval", -1))
        eval_interval = int(registration.get("eval_interval", -1))
        if (
            effective_offset != offset
            or rollout_seed != seed
            or save_interval != 40
            or eval_interval != 40
        ):
            raise ValueError(f"{stage_id} final RL contract differs from fixed v1")
        endpoint = _conversion_origin(registration.get("origin_hf"))
        if endpoint is None:
            raise ValueError(
                f"{stage_id} is registered without a usable pretrain endpoint"
            )
        identity_hashes: dict[str, str] = {}
        for hash_field in (
            "origin_hf_sha256",
            "method_plan_sha256",
            "call_contract_sha256",
        ):
            digest = registration.get(hash_field)
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ValueError(f"{stage_id} has invalid {hash_field}")
            identity_hashes[hash_field] = digest
        expected_provenance = {
            "app_name": str(rl_app["name"]),
            "app_id": str(rl_app["app_id"]),
            "function_id": str(rl_function_ids["train_hf"]),
            "source_tree_sha256": fixed_rl["source_tree_sha256"],
            "miles_source_tree_sha256": fixed_rl["miles_source_tree_sha256"],
            "checkpoint_publication": fixed_rl["checkpoint_publication"],
            "runtime_image": fixed_rl["runtime_image"],
            "runtime_packages": fixed_rl["runtime_packages"],
            "installed_packages_sha256": fixed_rl[
                "installed_packages_sha256"
            ],
            "num_rollout": target,
            "rollout_seed": rollout_seed,
            "dynamic_filter": filter_code == "D",
            "save_interval": save_interval,
            "max_tokens_per_gpu": fixed_rl["max_tokens_per_gpu"],
            "gradient_checkpointing": fixed_rl["gradient_checkpointing"],
            "host_memory_gb": fixed_rl["host_memory_gb"],
            "sglang_server_concurrency": fixed_rl[
                "sglang_server_concurrency"
            ],
            "gate_calls": fixed_rl["gate_calls"],
        }
        if registration.get("expected_provenance") != expected_provenance:
            raise ValueError(
                f"{stage_id} expected RL provenance differs from frozen registry"
            )
        stages.append(
            {
                "experiment_version": experiment_version,
                "exp4_version": exp4_version,
                "model": model,
                "experiment": "E4",
                "arm": arm_id,
                "filter": filter_code,
                "filter_mode": (
                    "dynamic" if filter_code == "D" else "unfiltered"
                ),
                "method": method,
                "phase": str(registration.get("phase") or "RL"),
                "stage_id": stage_id,
                "run_name": run_name,
                "target_step": target,
                "effective_step_offset": effective_offset,
                "rollout_seed": rollout_seed,
                "dynamic_filter": filter_code == "D",
                "registry_status": str(
                    registration.get("status", "registered")
                ),
                "raw_checkpoint_root": f"{raw_root}/{run_name}",
                "conversion_origin_hf": endpoint,
                "conversion_origin_fallback": False,
                **identity_hashes,
            }
        )
    return stages
