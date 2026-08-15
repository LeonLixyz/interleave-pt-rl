from __future__ import annotations

import copy
import hashlib
import json

import pytest

from Eval import modal_eval_all_rl_ckpts as evaluator
from Eval.interleave_live_schema import build_live_feed


def test_historical_r6_fingerprint_is_unchanged() -> None:
    assert evaluator.PRODUCTION_FINGERPRINT == (
        "467f569b87ae80ba83d6dabd6e499293b374d0160a3e6d3ea577fd858ba618c1"
    )


def test_core_interleave_fingerprint_is_unchanged() -> None:
    assert evaluator.INTERLEAVE_PRODUCTION_FINGERPRINT == (
        "aba2e260f3a82a24e9ad37b21a562e2259dd943c8aa9cc8c3a9d7607bd921f6d"
    )


def test_interleave_packaged_helper_hashes_are_pinned() -> None:
    assert (
        hashlib.sha256(evaluator.INTERLEAVE_QUEUE_LOCAL.read_bytes()).hexdigest()
        == evaluator.INTERLEAVE_QUEUE_SHA256
    )
    assert (
        hashlib.sha256(
            evaluator.INTERLEAVE_DASHBOARD_SCHEMA_LOCAL.read_bytes()
        ).hexdigest()
        == evaluator.INTERLEAVE_DASHBOARD_SCHEMA_SHA256
    )
    assert (
        hashlib.sha256(evaluator.MILES_CONVERTER_LOCAL.read_bytes()).hexdigest()
        == evaluator.MILES_CONVERTER_SHA256
    )
    assert (
        hashlib.sha256(
            evaluator.INTERLEAVE_EXP4_QUEUE_LOCAL.read_bytes()
        ).hexdigest()
        == evaluator.INTERLEAVE_EXP4_QUEUE_SHA256
    )
    assert (
        hashlib.sha256(
            evaluator.INTERLEAVE_LIVE_SCHEMA_LOCAL.read_bytes()
        ).hexdigest()
        == evaluator.INTERLEAVE_LIVE_SCHEMA_SHA256
    )


def test_interleave_success_requires_identity_rows_and_final_table(
    tmp_path,
) -> None:
    marker = tmp_path / "_SUCCESS.json"
    payload = {
        "namespace": "interleave_v1",
        "fingerprint": "eval-fingerprint",
        "checkpoint_identity": "checkpoint-identity",
        "expected_rows": 23_680,
        "actual_rows": 23_680,
        "table_metrics": {
            "pass_at_1": 0.2,
            "avg_reward": 0.2,
            "b3_avg": 0.2,
            "b4_avg": 0.3,
            "b3_b4_avg": 0.25,
        },
    }
    marker.write_text(json.dumps(payload))

    assert evaluator._valid_interleave_success(
        marker,
        fingerprint="eval-fingerprint",
        checkpoint_identity="checkpoint-identity",
    )
    payload["actual_rows"] = 1
    marker.write_text(json.dumps(payload))
    assert not evaluator._valid_interleave_success(
        marker,
        fingerprint="eval-fingerprint",
        checkpoint_identity="checkpoint-identity",
    )


def test_live_registry_pointer_and_content_address_are_both_verified(
    tmp_path,
) -> None:
    registry = json.loads(evaluator.INTERLEAVE_REGISTRY_LOCAL.read_text())
    feed = build_live_feed(
        registry,
        {"stages": {}},
        generated_at="2026-07-30T08:00:00+00:00",
    )
    root = tmp_path / "interleave"
    pointer = root / "latest.json"
    version = root / "feeds" / f"{feed['payload_sha256']}.json"
    version.parent.mkdir(parents=True)
    pointer.write_text(json.dumps(feed))
    version.write_text(json.dumps(feed))

    assert evaluator._load_live_interleave_registry(pointer) == registry
    version.unlink()
    try:
        evaluator._load_live_interleave_registry(pointer)
    except ValueError as exc:
        assert "content-addressed" in str(exc)
    else:
        raise AssertionError("missing content-addressed feed was accepted")


def test_live_exp4_registration_expands_448_to_670_without_core_rehash(
    tmp_path,
) -> None:
    registry = json.loads(evaluator.INTERLEAVE_REGISTRY_LOCAL.read_text())
    registrations = []
    for index, arm in enumerate(registry["exp4"]["arms"]):
        registrations.append(
            {
                "stage_id": arm["stages"]["rl"]["stage_id"],
                "run_id": f"exp4-final-{index}",
                "run_name": f"exp4-final-{index}",
                "model": "interleave_47m_qwen3",
                "experiment": "EXP4",
                "arm": arm["arm_id"],
                "filter": arm["filter"],
                "filter_mode": (
                    "dynamic" if arm["filter"] == "D" else "unfiltered"
                ),
                "method": arm["method"],
                "phase": "RL",
                "target_steps": 1500,
                "effective_step_offset": 1500,
                "rollout_seed": 43,
                "dynamic_filter": arm["filter"] == "D",
                "save_interval": 40,
                "eval_interval": 40,
                "origin_hf": f"/checkpoints/exp4/{index}/final",
                "origin_hf_sha256": "a" * 64,
                "method_plan_sha256": "b" * 64,
                "call_contract_sha256": "c" * 64,
                "expected_provenance": {
                    "app_name": registry["exp4"]["rl_app"]["name"],
                    "app_id": registry["exp4"]["rl_app"]["app_id"],
                    "function_id": registry["exp4"]["rl_app"]["function_ids"][
                        "train_hf"
                    ],
                    "source_tree_sha256": registry["fixed_rl"][
                        "source_tree_sha256"
                    ],
                    "miles_source_tree_sha256": registry["fixed_rl"][
                        "miles_source_tree_sha256"
                    ],
                    "checkpoint_publication": registry["fixed_rl"][
                        "checkpoint_publication"
                    ],
                    "runtime_image": registry["fixed_rl"]["runtime_image"],
                    "runtime_packages": registry["fixed_rl"][
                        "runtime_packages"
                    ],
                    "installed_packages_sha256": registry["fixed_rl"][
                        "installed_packages_sha256"
                    ],
                    "num_rollout": 1500,
                    "rollout_seed": 43,
                    "dynamic_filter": arm["filter"] == "D",
                    "save_interval": 40,
                    "max_tokens_per_gpu": registry["fixed_rl"][
                        "max_tokens_per_gpu"
                    ],
                    "gradient_checkpointing": registry["fixed_rl"][
                        "gradient_checkpointing"
                    ],
                    "host_memory_gb": registry["fixed_rl"][
                        "host_memory_gb"
                    ],
                    "sglang_server_concurrency": registry["fixed_rl"][
                        "sglang_server_concurrency"
                    ],
                    "gate_calls": registry["fixed_rl"]["gate_calls"],
                },
                "status": "submitted",
            }
        )
    registry["exp4"]["final_rl_runs"] = registrations
    feed = build_live_feed(
        registry,
        {"stages": {}},
        generated_at="2026-07-30T08:00:00+00:00",
    )
    root = tmp_path / "interleave"
    pointer = root / "latest.json"
    version = root / "feeds" / f"{feed['payload_sha256']}.json"
    version.parent.mkdir(parents=True)
    pointer.write_text(json.dumps(feed))
    version.write_text(json.dumps(feed))

    stages, expected_exp4 = evaluator._current_interleave_stages(pointer)
    assert expected_exp4 == 6
    assert len(stages) == 14
    assert sum(
        len(evaluator.cadence_steps(stage["target_step"]))
        for stage in stages
    ) == 670
    _, core_fingerprint = evaluator._interleave_stage_profile_settings(
        "production", stages[0]
    )
    _, exp4_fingerprint = evaluator._interleave_stage_profile_settings(
        "production", stages[-1]
    )
    assert core_fingerprint == evaluator.INTERLEAVE_PRODUCTION_FINGERPRINT
    assert exp4_fingerprint != core_fingerprint
    for field, replacement in (
        ("origin_hf_sha256", "d" * 64),
        ("method_plan_sha256", "e" * 64),
        ("call_contract_sha256", "f" * 64),
    ):
        mutated = dict(stages[-1])
        mutated[field] = replacement
        _, mutated_fingerprint = evaluator._interleave_stage_profile_settings(
            "production", mutated
        )
        assert mutated_fingerprint != exp4_fingerprint

    collision_registry = copy.deepcopy(registry)
    core_run_name = evaluator.INTERLEAVE_STAGES[0]["run_name"]
    collision_registry["exp4"]["final_rl_runs"][0]["run_id"] = core_run_name
    collision_registry["exp4"]["final_rl_runs"][0]["run_name"] = core_run_name
    collision_feed = build_live_feed(
        collision_registry,
        {"stages": {}},
        generated_at="2026-07-30T08:01:00+00:00",
    )
    collision_root = tmp_path / "collision" / "interleave"
    collision_pointer = collision_root / "latest.json"
    collision_version = (
        collision_root
        / "feeds"
        / f"{collision_feed['payload_sha256']}.json"
    )
    collision_version.parent.mkdir(parents=True)
    collision_pointer.write_text(json.dumps(collision_feed))
    collision_version.write_text(json.dumps(collision_feed))
    with pytest.raises(ValueError, match="collide"):
        evaluator._current_interleave_stages(collision_pointer)
