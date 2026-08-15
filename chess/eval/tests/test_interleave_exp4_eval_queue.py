from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from Eval.interleave_eval_queue import cadence_steps
from Eval.interleave_exp4_eval_queue import flatten_exp4_rl_registry


ROOT = Path(__file__).resolve().parents[2]


def _registry() -> dict:
    return json.loads((ROOT / "INTERLEAVED_CORE_REGISTRY.json").read_text())


def _register_exp4_runs(registry: dict) -> None:
    registrations = []
    for index, arm in enumerate(registry["exp4"]["arms"]):
        pretrain = arm["stages"]["pretrain"]
        rl = arm["stages"]["rl"]
        endpoint = f"/checkpoints/interleave_50m/exp4/test/{index}/p2/final"
        run_id = f"exp4-final-{index}"
        pretrain["endpoint"] = endpoint
        rl["run_id"] = run_id
        rl["status"] = "submitted"
        registrations.append(
            {
                "stage_id": rl["stage_id"],
                "run_id": run_id,
                "run_name": run_id,
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
                "origin_hf": endpoint,
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


def test_planned_exp4_arms_are_not_discoverable_runs() -> None:
    assert flatten_exp4_rl_registry(_registry()) == []


def test_six_registered_exp4_arms_add_222_cadence_points() -> None:
    registry = _registry()
    _register_exp4_runs(registry)
    stages = flatten_exp4_rl_registry(registry)

    assert len(stages) == 6
    assert sum(len(cadence_steps(stage["target_step"])) for stage in stages) == 222
    assert {stage["filter"] for stage in stages} == {"U", "D"}
    assert {stage["method"] for stage in stages} == {
        "sft",
        "distill",
        "scratch",
    }
    assert all(stage["effective_step_offset"] == 1500 for stage in stages)
    assert all(
        stage["conversion_origin_hf"].startswith("/pretrain-checkpoints/")
        for stage in stages
    )
    assert all(
        all(
            field in stage
            for field in (
                "origin_hf_sha256",
                "method_plan_sha256",
                "call_contract_sha256",
            )
        )
        for stage in stages
    )


def test_registered_run_fails_closed_without_pretrain_endpoint() -> None:
    registry = _registry()
    _register_exp4_runs(registry)
    broken = copy.deepcopy(registry)
    broken["exp4"]["final_rl_runs"][0]["origin_hf"] = None
    with pytest.raises(ValueError, match="pretrain endpoint"):
        flatten_exp4_rl_registry(broken)


def test_registered_run_fails_closed_on_expected_provenance_drift() -> None:
    registry = _registry()
    _register_exp4_runs(registry)
    broken = copy.deepcopy(registry)
    broken["exp4"]["final_rl_runs"][0]["expected_provenance"][
        "source_tree_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="expected RL provenance"):
        flatten_exp4_rl_registry(broken)


def test_registration_must_match_its_planned_arm_contract() -> None:
    registry = _registry()
    _register_exp4_runs(registry)
    broken = copy.deepcopy(registry)
    broken["exp4"]["final_rl_runs"][1]["arm"] = "EXP4-U-SFT"
    broken["exp4"]["final_rl_runs"][1]["filter"] = "U"
    broken["exp4"]["final_rl_runs"][1]["method"] = "sft"
    with pytest.raises(ValueError, match="planned arm contract"):
        flatten_exp4_rl_registry(broken)


def test_unplanned_seventh_registration_is_rejected() -> None:
    registry = _registry()
    _register_exp4_runs(registry)
    extra = copy.deepcopy(registry["exp4"]["final_rl_runs"][-1])
    extra.update(
        {
            "stage_id": "exp4-unplanned-rl1500",
            "run_id": "exp4-unplanned-final",
            "run_name": "exp4-unplanned-final",
        }
    )
    registry["exp4"]["final_rl_runs"].append(extra)
    with pytest.raises(ValueError, match="more final RL runs"):
        flatten_exp4_rl_registry(registry)
