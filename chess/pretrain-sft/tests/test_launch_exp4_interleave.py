from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from modal_scripts import launch_exp4_interleave as launcher

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


def _load_endpoint_eval_module():
    path = Path(__file__).resolve().parents[2] / "Eval/interleave_endpoint_eval.py"
    spec = importlib.util.spec_from_file_location("interleave_endpoint_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_rollout_pair(root: Path, rollout_id: int, payload: str) -> None:
    (root / f"rollout_{rollout_id}.jsonl").write_text(payload + "\n", encoding="utf-8")
    (root / f"rollout_{rollout_id}.summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sampling_scope": ("all_completed_attempts_before_dynamic_filter"),
                "rollout_id": rollout_id,
                "step": rollout_id,
                "attempted_groups": 256,
                "positive_completed_samples": 1,
            }
        ),
        encoding="utf-8",
    )


def test_rollout_inventory_hashes_every_jsonl_and_summary(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    source = raw_root / "run/rollouts/all_attempts_positive"
    source.mkdir(parents=True)
    _write_rollout_pair(source, 0, '{"row":0}')
    _write_rollout_pair(source, 1, '{"row":1}')
    monkeypatch.setattr(launcher, "RAW_RL_ROOT", raw_root)

    first = launcher._rollout_inventory(source, expected_files=2)
    assert len(first["files"]) == 4
    assert all(launcher._is_sha256(row["sha256"]) for row in first["files"])

    (source / "rollout_1.jsonl").write_text('{"row":"mutated"}\n', encoding="utf-8")
    second = launcher._rollout_inventory(source, expected_files=2)
    assert first["inventory_sha256"] != second["inventory_sha256"]


def test_rollout_inventory_rejects_an_unpaired_or_missing_stage(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    source = raw_root / "run/rollouts/all_attempts_positive"
    source.mkdir(parents=True)
    _write_rollout_pair(source, 0, '{"row":0}')
    (source / "rollout_0.summary.json").unlink()
    monkeypatch.setattr(launcher, "RAW_RL_ROOT", raw_root)

    with pytest.raises(RuntimeError, match="incomplete"):
        launcher._rollout_inventory(source, expected_files=1)


def test_checkpoint_identity_matches_endpoint_evaluator_and_requires_custom_assets(
    tmp_path, monkeypatch
):
    checkpoint_mount = tmp_path / "checkpoints"
    checkpoint = checkpoint_mount / "endpoint"
    checkpoint.mkdir(parents=True)
    config = {
        "model_type": "qwen3",
        "vocab_size": 85,
        "max_position_embeddings": 3072,
        "hidden_size": 512,
        "head_dim": 128,
        "num_hidden_layers": 12,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "intermediate_size": 1536,
        "tie_word_embeddings": True,
    }
    assets = {
        "config.json": json.dumps(config).encode(),
        "generation_config.json": b"{}",
        "model.safetensors": b"weights",
        "model.safetensors.index.json": b"{}",
        "interleaved_training_state.json": b'{"global_step":9920}',
        "tokenizer.py": b"# custom tokenizer\n",
        "tokenizer_config.json": b"{}",
        "vocab.json": b"{}",
        "merges.txt": b"",
        "special_tokens_map.json": b"{}",
        "added_tokens.json": b"{}",
        "training.log": b"must not enter endpoint identity",
        "complete.json": b"must not enter endpoint identity",
    }
    for name, payload in assets.items():
        (checkpoint / name).write_bytes(payload)
    monkeypatch.setattr(launcher, "CHECKPOINT_MOUNT", checkpoint_mount)
    endpoint_eval = _load_endpoint_eval_module()

    launcher_files = [
        path.relative_to(checkpoint).as_posix()
        for path in launcher._checkpoint_files(checkpoint)
    ]
    evaluator_files = [
        path.relative_to(checkpoint).as_posix()
        for path in endpoint_eval.checkpoint_files(checkpoint)
    ]
    assert launcher_files == evaluator_files
    assert launcher._checkpoint_fingerprint(
        checkpoint
    ) == endpoint_eval.checkpoint_fingerprint(checkpoint)
    assert "training.log" not in launcher_files
    assert "complete.json" not in launcher_files

    original = launcher._checkpoint_fingerprint(checkpoint)
    (checkpoint / "training.log").write_bytes(b"unrelated log mutation")
    assert launcher._checkpoint_fingerprint(checkpoint) == original
    (checkpoint / "interleaved_training_state.json").write_bytes(
        b'{"global_step":9921}'
    )
    assert launcher._checkpoint_fingerprint(checkpoint) != original
    (checkpoint / "tokenizer.py").unlink()
    with pytest.raises(FileNotFoundError, match="tokenizer.py"):
        launcher._checkpoint_files(checkpoint)


def test_rl_run_provenance_is_required_and_bound_to_exact_u_identity(
    tmp_path, monkeypatch
):
    checkpoint_mount = tmp_path / "checkpoints"
    p1 = checkpoint_mount / "p1"
    p1.mkdir(parents=True)
    (p1 / "config.json").write_text('{"model":"p1"}\n', encoding="utf-8")
    (p1.parent / "config.yaml").write_text(
        json.dumps(
            {
                "model": {
                    "attn_implementation": "sdpa",
                    "flash_attention_version": "2.8.3",
                },
                "training": {
                    "torch_compile": "none",
                    "total_steps": 9920,
                    "local_batch_size": 21,
                    "gradient_accumulation_steps": 1,
                },
                "provenance": {
                    "experiment_version": launcher.EXPERIMENT_VERSION,
                    "attention_backend": "sdpa",
                    "flash_attention_version": "2.8.3",
                    "torch_compile_mode": "none",
                    "source_tree_sha256": (
                        launcher.UPSTREAM_PRETRAIN_SOURCE_TREE_SHA256
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    (p1 / "interleaved_training_state.json").write_text(
        json.dumps(
            {
                "global_step": 9920,
                "local_batch_size": 21,
                "world_size": 8,
                "gradient_accumulation_steps": 1,
                "attention_backend": "sdpa",
                "torch_compile_mode": "none",
                "runtime_provenance": {
                    "attention_backend": "sdpa",
                    "torch_compile_mode": "none",
                    "flash_attention_version": None,
                },
            }
        ),
        encoding="utf-8",
    )
    p1_files = sorted(path for path in p1.iterdir() if path.is_file())
    origin_files = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": launcher._sha256_file(path),
        }
        for path in p1_files
    ]
    p1_rows = [
        f"{item['path']}\t{item['bytes']}\t{item['sha256']}\n" for item in origin_files
    ]
    origin = {
        "logical_path": "/pretrain-checkpoints/p1",
        "file_count": len(origin_files),
        "total_bytes": sum(item["bytes"] for item in origin_files),
        "manifest_sha256": launcher._sha256_bytes("".join(p1_rows).encode()),
        "files": origin_files,
    }
    raw_root = tmp_path / "raw"
    run_root = raw_root / "core-e1-u-rl1-seed42"
    (run_root / "provenance").mkdir(parents=True)
    identity = {
        "kind": "chess_rl_miles_interleave_run",
        "run": {
            "app_name": "chess-interleave-rl",
            "run_name": "core-e1-u-rl1-seed42",
            "model_id": "interleave_47m_qwen3",
            "num_rollout": 1500,
            "dynamic_filter": False,
            "rollout_seed": 42,
            "save_interval": 40,
            "eval_interval": 0,
            "canary": False,
        },
        "policy_update_profile": {
            "name": "small-model-h200",
            "max_tokens_per_gpu": 65536,
            "gradient_checkpointing": False,
            "train_backend": "fsdp",
            "actor_num_nodes": 1,
            "actor_num_gpus_per_node": 8,
            "gpu_type": "H200",
        },
        "fixed_rl_semantics": {
            "rollout_batch_size": 256,
            "samples_per_prompt": 8,
            "global_batch_size": 2048,
            "policy_loss_agg_mode": "token-mean",
            "advantage_estimator": "grpo",
            "cispo": False,
            "optimizer": "adamw",
            "lr": 1e-5,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_eps": 1e-8,
            "weight_decay": 0.01,
            "kl_loss_coef": 0.001,
            "rollout_max_prompt_len": 512,
            "rollout_max_response_len": 2560,
            "rollout_max_context_len": 3072,
        },
        "balanced_data": {
            "logical_path": "/data/balanced.parquet",
            "sha256": launcher.BALANCED_DATA_SHA256,
        },
        "origin_hf": origin,
        "sources": {
            "chess_rl_miles": {"manifest_sha256": HASH_A},
            "miles": {"manifest_sha256": HASH_B},
        },
        "runtime": {
            "image": "pinned-image",
            "packages": {"torch": "2.9.0"},
        },
    }
    identity_sha = launcher._provenance_identity_sha256(identity)
    command = ["python", "-m", "train"]
    command_sha = launcher._command_sha256(command)
    (run_root / "run_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity_sha256": identity_sha,
                "identity": identity,
                "initial_command_sha256": command_sha,
                "initial_command": command,
            }
        ),
        encoding="utf-8",
    )
    launch = run_root / f"provenance/launch_{command_sha[:16]}.json"
    launch_payload = {
        "schema_version": 1,
        "identity_sha256": identity_sha,
        "command_sha256": command_sha,
        "command": command,
    }
    launch.write_text(json.dumps(launch_payload), encoding="utf-8")
    monkeypatch.setattr(launcher, "CHECKPOINT_MOUNT", checkpoint_mount)
    monkeypatch.setattr(launcher, "P1_CHECKPOINT", p1)
    monkeypatch.setattr(launcher, "RAW_RL_ROOT", raw_root)

    provenance = launcher._validate_rl_run_provenance("U")
    assert provenance.identity_sha256 == identity_sha
    assert len(provenance.launch_manifest_paths) == 1
    assert launcher._is_sha256(provenance.bundle_sha256)

    launch.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid RL launch provenance"):
        launcher._validate_rl_run_provenance("U")

    launch.write_text(json.dumps(launch_payload), encoding="utf-8")
    config_path = p1.parent / "config.yaml"
    config = json.loads(config_path.read_text())
    config["provenance"]["source_tree_sha256"] = HASH_C
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="P1 frozen production config mismatch"):
        launcher._validate_rl_run_provenance("U")


def test_replay_artifact_validation_binds_output_manifest_and_filter(
    tmp_path, monkeypatch
):
    exp4_root = tmp_path / "exp4"
    root = exp4_root / "u/positive-replay/fingerprint"
    root.mkdir(parents=True)
    replay = root / "positive_replay.jsonl"
    replay.write_text('{"selected":true}\n', encoding="utf-8")
    replay_sha = launcher._sha256_file(replay)
    manifest = root / "positive_replay.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "exp4_positive_rollout_replay",
                "config": {"filter_setting": "U"},
                "output": {
                    "path": str(replay.resolve()),
                    "sha256": replay_sha,
                    "rows": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    copied_root = root / "rl_run_provenance/run_provenance.json"
    copied_root.parent.mkdir(parents=True)
    copied_root.write_text('{"root":true}\n', encoding="utf-8")
    copied_launch = (
        root / "rl_run_provenance/provenance" / "launch_1111111111111111.json"
    )
    copied_launch.parent.mkdir(parents=True)
    copied_launch.write_text('{"launch":true}\n', encoding="utf-8")
    identity_sha = "3" * 64
    bundle = {
        "run_name": "core-e1-u-rl1-seed42",
        "identity_sha256": identity_sha,
        "root_manifest_sha256": launcher._sha256_file(copied_root),
        "launch_manifests": [
            {
                "path": copied_launch.name,
                "sha256": launcher._sha256_file(copied_launch),
                "command_sha256": "1" * 64,
            }
        ],
    }
    bundle_sha = launcher._content_fingerprint("exp4-rl-run-provenance-bundle", bundle)
    contract = launcher._replay_contract(
        filter_setting="U",
        run_name="core-e1-u-rl1-seed42",
        policy_checkpoint="/checkpoints/policy",
        policy_checkpoint_sha256=HASH_A,
        rollout_inventory_sha256=HASH_B,
        rl_run_provenance_identity_sha256=identity_sha,
        rl_run_provenance_bundle_sha256=bundle_sha,
        source_tree_sha256=HASH_C,
    )
    artifact = root / "artifact_manifest.json"
    artifact.write_text(
        json.dumps(
            {
                "kind": "exp4_positive_replay_artifact",
                "state": "complete",
                "fingerprint": launcher._content_fingerprint(
                    "exp4-positive-replay", contract
                ),
                "contract": contract,
                "replay_sha256": replay_sha,
                "replay_manifest_sha256": launcher._sha256_file(manifest),
                "rl_run_provenance": {
                    "identity_sha256": identity_sha,
                    "bundle_sha256": bundle_sha,
                    "bundle_contract": bundle,
                    "copied_files": [
                        {
                            "path": str(copied_root.relative_to(root)),
                            "sha256": launcher._sha256_file(copied_root),
                        },
                        {
                            "path": str(copied_launch.relative_to(root)),
                            "sha256": launcher._sha256_file(copied_launch),
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "EXP4_ROOT", exp4_root)

    validated = launcher._validate_replay_artifact(replay, manifest, filter_setting="U")
    assert validated.rows == 1
    assert validated.artifact_sha256 == launcher._sha256_file(artifact)

    replay.write_text('{"selected":"mutated"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="manifest contract mismatch"):
        launcher._validate_replay_artifact(replay, manifest, filter_setting="U")


def _method_contract(filter_setting: str, method: str = "hard-sft"):
    return launcher._method_contract(
        method=method,
        filter_setting=filter_setting,
        replay_sha256=HASH_A,
        replay_manifest_sha256=HASH_B,
        replay_artifact_sha256=HASH_C,
        p1_checkpoint_sha256=(None if method == "scratch-replay" else HASH_D),
        teacher_checkpoint_sha256=HASH_E if method == "soft-kl" else None,
        p2_manifest_sha256="1" * 64,
        source_tree_sha256="2" * 64,
    )


def test_u_and_d_arms_have_distinct_immutable_method_identities():
    unfiltered = _method_contract("U")
    dynamic = _method_contract("D")
    assert unfiltered["filter_setting"] == "U"
    assert dynamic["filter_setting"] == "D"
    assert launcher._content_fingerprint(
        "exp4-method-plan", unfiltered
    ) != launcher._content_fingerprint("exp4-method-plan", dynamic)
    assert launcher._rollout_source("U") != launcher._rollout_source("D")
    assert launcher._teacher_checkpoint("U") != launcher._teacher_checkpoint("D")


def test_methods_bind_exact_topology_and_forbid_wrong_weight_sources():
    hard = _method_contract("U", "hard-sft")
    soft = _method_contract("U", "soft-kl")
    scratch = _method_contract("U", "scratch-replay")
    assert hard["topology"] == {
        "gpu_type": "H200",
        "world_size": 8,
        "local_batch_size": 21,
        "global_batch_size": 168,
        "gradient_accumulation_steps": 1,
    }
    assert hard["teacher_checkpoint_sha256"] is None
    assert soft["teacher_checkpoint_sha256"] == HASH_E
    assert scratch["p1_checkpoint_sha256"] is None
    assert scratch["teacher_checkpoint_sha256"] is None
    assert hard["runtime_backend"] == {
        "attention": "sdpa",
        "flash_attention_config_version": "2.8.3",
        "flash_attention_runtime_version": None,
        "torch_compile": "none",
    }
    assert hard["upstream_pretrain_contract"]["source_tree_sha256"] == (
        launcher.UPSTREAM_PRETRAIN_SOURCE_TREE_SHA256
    )
    with pytest.raises(ValueError, match="must not bind/load P1"):
        launcher._method_contract(
            method="scratch-replay",
            filter_setting="U",
            replay_sha256=HASH_A,
            replay_manifest_sha256=HASH_B,
            replay_artifact_sha256=HASH_C,
            teacher_checkpoint_sha256=None,
            p2_manifest_sha256="1" * 64,
            source_tree_sha256="2" * 64,
            p1_checkpoint_sha256=HASH_D,
        )


def test_dry_run_is_read_only_and_exposes_content_addressed_prefixes():
    extract = launcher._dry_run(
        action="extract",
        method="",
        filter_setting="U",
        replay_path="",
        replay_manifest_path="",
    )
    assert extract["required_rollout_jsonl_summary_pairs"] == 1500
    assert extract["rl_run_name"] == "core-e1-u-rl1-seed42"
    train = launcher._dry_run(
        action="train",
        method="soft-kl",
        filter_setting="D",
        replay_path="/checkpoints/replay.jsonl",
        replay_manifest_path="/checkpoints/replay.manifest.json",
    )
    assert train["gpus"] == "H200:8"
    assert train["gradient_accumulation_steps"] == 1
    assert "/d/soft-kl" in train["output_prefix"]
    assert train["runtime_backend"]["attention"] == (
        launcher.PRODUCTION_ATTENTION_BACKEND
    )
    assert train["runtime_backend"]["torch_compile"] == "none"
    assert train["upstream_pretrain_source_tree_sha256"] == (
        launcher.UPSTREAM_PRETRAIN_SOURCE_TREE_SHA256
    )


def test_remote_attention_contract_is_embedded_and_fail_closed(tmp_path, monkeypatch):
    missing = tmp_path / "missing.py"
    monkeypatch.delenv("CHESS_EXP4_ATTENTION_BACKEND", raising=False)
    monkeypatch.delenv("CHESS_EXP4_FLASH_ATTENTION_VERSION", raising=False)
    monkeypatch.delenv("CHESS_EXP4_TORCH_COMPILE_MODE", raising=False)
    with pytest.raises(RuntimeError, match="embedded attention contract"):
        launcher._read_main_launcher_backend(missing)

    monkeypatch.setenv("CHESS_EXP4_ATTENTION_BACKEND", "flash_attention_2")
    monkeypatch.setenv("CHESS_EXP4_FLASH_ATTENTION_VERSION", "2.8.3")
    monkeypatch.setenv("CHESS_EXP4_TORCH_COMPILE_MODE", "none")
    with pytest.raises(RuntimeError, match="drifted"):
        launcher._read_main_launcher_backend(missing)

    monkeypatch.setenv("CHESS_EXP4_ATTENTION_BACKEND", "sdpa")
    assert launcher._read_main_launcher_backend(missing) == ("sdpa", "2.8.3")


def test_transfer_command_uses_identical_batching_and_teacher_only_for_kl(tmp_path):
    replay = launcher.ReplayArtifact(
        replay_path=tmp_path / "positive.jsonl",
        manifest_path=tmp_path / "manifest.json",
        artifact_path=tmp_path / "artifact.json",
        replay_sha256=HASH_A,
        manifest_sha256=HASH_B,
        artifact_sha256=HASH_C,
        rows=100,
        filter_setting="U",
    )

    def plan(method: str):
        return launcher.MethodPlan(
            method=method,
            filter_setting="U",
            fingerprint="f" * 64,
            root=tmp_path / method,
            contract={},
            replay=replay,
            p1_checkpoint=Path("/checkpoints/p1"),
            teacher_checkpoint=(
                Path("/checkpoints/teacher") if method == "soft-kl" else None
            ),
            p2_manifest_sha256="1" * 64,
        )

    hard = launcher._transfer_command(plan("hard-sft"))
    soft = launcher._transfer_command(plan("soft-kl"))
    assert hard[hard.index("--num_processes") + 1] == "8"
    assert hard[hard.index("--local-batch-size") + 1] == "21"
    assert "--teacher-checkpoint" not in hard
    assert soft[soft.index("--teacher-checkpoint") + 1] == "/checkpoints/teacher"
    assert hard[hard.index("--save-interval") + 1] == "200"
    assert (
        hard[hard.index("--attn-implementation") + 1]
        == launcher.PRODUCTION_ATTENTION_BACKEND
    )
    assert (
        hard[hard.index("--flash-attention-version") + 1]
        == launcher.PINNED_FLASH_ATTENTION_VERSION
    )
