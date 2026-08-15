from __future__ import annotations

import json
import ast
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from Eval import interleave_endpoint_eval as schema
from Eval import modal_eval_all_rl_ckpts as base_evaluator
from Eval import modal_eval_interleaved_endpoints as evaluator


def _write_endpoint(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(dict(schema.EXPECTED_MODEL_CONFIG))
    )
    (path / "generation_config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "interleaved_training_state.json").write_text(
        json.dumps({"global_step": 1})
    )
    (path / "tokenizer_config.json").write_text("{}")
    (path / "tokenizer.py").write_text("# tokenizer v1\n")
    (path / "vocab.json").write_text("{}")


def test_self_contained_chess_contract_matches_existing_evaluator():
    assert evaluator.UPSTREAM_GIT_SHA == base_evaluator.UPSTREAM_GIT_SHA
    assert evaluator.EVAL_CODE_SHA256 == base_evaluator.EVAL_CODE_SHA256
    assert evaluator.PRODUCTION_SETTINGS == base_evaluator.PRODUCTION_SETTINGS
    assert evaluator.REMOTE_VERL_ROOT == base_evaluator.REMOTE_VERL_ROOT
    assert evaluator.REMOTE_EVAL_DATA == base_evaluator.REMOTE_EVAL_DATA


def test_v1_profile_remains_the_default():
    assert evaluator.ENDPOINT_EVAL_PROFILE == "v1"
    assert evaluator.APP_NAME == "chess-interleave-endpoint-eval"
    assert evaluator.ENDPOINT_NAMESPACE == "endpoint_v1"
    assert evaluator.EXPERIMENT_VERSION == schema.EXPERIMENT_VERSION
    assert evaluator.INCLUDE_EXP4 is True
    assert evaluator.REUSE_AUTHENTICATED_PT_HOLDOUT is False
    assert str(evaluator.EVAL_ARTIFACT_ROOT).endswith(
        "50m_interleaved_mix10b_sft90k_v1/endpoint_eval_v1"
    )
    assert evaluator.REMOTE_PROFILE_CONTRACT["profile"] == "v1"
    assert (
        evaluator.REMOTE_PROFILE_ENV[
            "CHESS_INTERLEAVE_ENDPOINT_EVAL_PROFILE"
        ]
        == "v1"
    )


def test_v2_wrapper_has_separate_app_data_and_result_roots():
    code = """
import json
from Eval import modal_eval_interleaved_endpoints_v2 as evaluator
print(json.dumps({
    "profile": evaluator.ENDPOINT_EVAL_PROFILE,
    "app": evaluator.APP_NAME,
    "namespace": evaluator.ENDPOINT_NAMESPACE,
    "experiment": evaluator.EXPERIMENT_VERSION,
    "data_root": str(evaluator.DATA_ARTIFACT_ROOT),
    "eval_root": str(evaluator.EVAL_ARTIFACT_ROOT),
    "pt_holdout": str(evaluator.PT_HOLDOUT_PATH),
    "sft_audit": str(evaluator.SFT_AUDIT_PATH),
    "result_root": str(evaluator.configured_endpoint_root("p1", "a" * 64)),
    "include_exp4": evaluator.INCLUDE_EXP4,
    "reuse_pt": evaluator.REUSE_AUTHENTICATED_PT_HOLDOUT,
    "expected_exp4": sorted(evaluator.EXPECTED_EXP4_CELLS),
    "remote_contract": evaluator.REMOTE_PROFILE_CONTRACT,
    "remote_env": evaluator.REMOTE_PROFILE_ENV,
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment.pop("CHESS_INTERLEAVE_ENDPOINT_EVAL_PROFILE", None)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(
        [line for line in completed.stdout.splitlines() if line.strip()][-1]
    )

    assert value["profile"] == "v2"
    assert value["app"] == "chess-interleave-endpoint-eval-v2r1"
    assert value["app"] != evaluator.APP_NAME
    assert value["namespace"] == "endpoint_v2r1_weighted_clean"
    assert value["namespace"] != evaluator.ENDPOINT_NAMESPACE
    assert value["experiment"] == schema.V2_EXPERIMENT_VERSION
    assert value["data_root"].endswith(
        "50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate"
    )
    assert value["eval_root"].endswith(
        "50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/"
        "endpoint_eval_v2r1"
    )
    assert value["sft_audit"].startswith(value["eval_root"] + "/")
    assert value["pt_holdout"] == str(evaluator.PT_HOLDOUT_PATH)
    assert value["result_root"].startswith(
        "/results/endpoint_v2r1_weighted_clean/"
    )
    assert value["result_root"] != str(
        evaluator._endpoint_root("p1", "a" * 64)
    )
    assert value["include_exp4"] is False
    assert value["reuse_pt"] is True
    assert value["expected_exp4"] == []
    assert value["remote_contract"]["profile"] == "v2"
    assert value["remote_contract"]["namespace"] == value["namespace"]
    assert value["remote_contract"]["experiment_version"] == value["experiment"]
    assert (
        value["remote_env"]["CHESS_INTERLEAVE_ENDPOINT_EVAL_PROFILE"]
        == "v2"
    )
    assert (
        value["remote_env"]["CHESS_INTERLEAVE_ENDPOINT_EVAL_NAMESPACE"]
        == value["namespace"]
    )


def test_remote_profile_contract_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    for environment_key in evaluator.REMOTE_PROFILE_ENV:
        monkeypatch.delenv(environment_key, raising=False)
    with pytest.raises(RuntimeError, match="refusing all artifact access"):
        evaluator._assert_remote_profile_contract()

    for environment_key, value in evaluator.REMOTE_PROFILE_ENV.items():
        monkeypatch.setenv(environment_key, value)
    assert (
        evaluator._assert_remote_profile_contract()
        == evaluator.REMOTE_PROFILE_CONTRACT
    )

    monkeypatch.setenv(
        "CHESS_INTERLEAVE_ENDPOINT_EVAL_NAMESPACE",
        "endpoint_v2r1_weighted_clean",
    )
    with pytest.raises(RuntimeError, match="namespace"):
        evaluator._assert_remote_profile_contract()


def test_all_remote_entrypoints_are_profile_guarded_and_all_images_bind_env():
    source_path = Path(evaluator.__file__).resolve()
    source = source_path.read_text()
    tree = ast.parse(source)
    remote_entrypoints = {
        "prepare_holdouts",
        "eval_losses_one",
        "eval_chess_one",
        "watch_and_enqueue",
        "status_endpoints",
    }
    observed: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in remote_entrypoints:
                observed[node.name] = {
                    ast.unparse(decorator) for decorator in node.decorator_list
                }
    assert set(observed) == remote_entrypoints
    assert all(
        "_remote_profile_guard" in decorators
        for decorators in observed.values()
    )
    # loss, chess, and control images each bake the complete contract into
    # their runtime environment before any remote module import.
    assert source.count("**REMOTE_PROFILE_ENV") == 3


def test_checkpoint_identity_hashes_custom_tokenizer_and_vocab(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_endpoint(checkpoint)
    original = schema.checkpoint_fingerprint(checkpoint)
    (checkpoint / "tokenizer.py").write_text("# tokenizer v2\n")
    tokenizer_changed = schema.checkpoint_fingerprint(checkpoint)
    assert tokenizer_changed != original
    (checkpoint / "vocab.json").write_text('{"K": 1}')
    assert schema.checkpoint_fingerprint(checkpoint) != tokenizer_changed


def test_declared_exp4_digest_must_equal_computed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint_mount = tmp_path / "checkpoints"
    results_mount = tmp_path / "results"
    checkpoint = checkpoint_mount / "endpoint"
    _write_endpoint(checkpoint)
    monkeypatch.setattr(evaluator, "CHECKPOINT_MOUNT", checkpoint_mount)
    monkeypatch.setattr(evaluator, "RESULTS_MOUNT", results_mount)
    endpoint = {
        "endpoint_id": "exp4-u-hard-sft-abcdef123456",
        "checkpoint_path": str(checkpoint),
        "declared_checkpoint_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="declared checkpoint digest mismatch"):
        evaluator._cached_checkpoint_fingerprint(endpoint)

    endpoint["declared_checkpoint_sha256"] = schema.checkpoint_fingerprint(
        checkpoint
    )
    observed, _ = evaluator._cached_checkpoint_fingerprint(endpoint)
    assert observed == endpoint["declared_checkpoint_sha256"]


def test_component_state_has_finite_retry_and_terminal_attempts(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(evaluator, "RESULTS_MOUNT", tmp_path)
    endpoint_id = "p1"
    checkpoint_sha = "1" * 64
    fingerprint = "2" * 64
    root = evaluator._component_dir(
        endpoint_id, checkpoint_sha, "losses", fingerprint
    )
    root.mkdir(parents=True)

    evaluator._atomic_json(
        root / "_RUNNING.json",
        {"unix_time": time.time()},
    )
    assert (
        evaluator._component_state(
            endpoint_id, checkpoint_sha, "losses", fingerprint
        )
        == "running"
    )
    evaluator._atomic_json(
        root / "_RUNNING.json",
        {"unix_time": time.time() - evaluator.RUNNING_LEASE_SECONDS - 1},
    )
    evaluator._atomic_json(
        root / "_FAILED.json",
        {
            "attempt": 1,
            "retry_after_unix": time.time() + 60,
        },
    )
    assert (
        evaluator._component_state(
            endpoint_id, checkpoint_sha, "losses", fingerprint
        )
        == "retry_wait"
    )
    evaluator._atomic_json(
        root / "_FAILED.json",
        {
            "attempt": 1,
            "retry_after_unix": time.time() - 1,
        },
    )
    assert (
        evaluator._component_state(
            endpoint_id, checkpoint_sha, "losses", fingerprint
        )
        == "failed_retryable"
    )
    evaluator._atomic_json(
        root / "_FAILED.json",
        {
            "attempt": evaluator.MAX_COMPONENT_ATTEMPTS,
            "retry_after_unix": time.time() - 1,
        },
    )
    assert (
        evaluator._component_state(
            endpoint_id, checkpoint_sha, "losses", fingerprint
        )
        == "failed_terminal"
    )
