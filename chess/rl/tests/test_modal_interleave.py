from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from chess_rl_miles.scripts import modal_interleave
from chess_rl_miles.scripts.modal_interleave import (
    BALANCED_TRAIN_FILE,
    BALANCED_TRAIN_SHA256,
    POLICY_UPDATE_PROFILE,
    RAW_RL_ROOT,
    SGLANG_SERVER_CONCURRENCY_DEFAULT,
    SMALL_MODEL_HOST_MEMORY_GB,
    SMALL_MODEL_MAX_TOKENS_DEFAULT,
    _commit_new_checkpoint_if_ready,
    _latest_complete_checkpoint_step,
    _run_training_with_checkpoint_commits,
    _safe_component,
    _validate_checkpoint_context,
    _validate_hf_checkpoint,
    build_train_command,
)


def test_modal_app_secrets_are_not_conditioned_on_local_env_file():
    source = Path(modal_interleave.__file__).read_text()
    app_block = source[source.index("app = modal.App(") : source.index("\n\n\ndef _safe_component")]
    assert "*base.runtime_secrets" not in app_block
    assert 'modal.Secret.from_name("huggingface-secret")' in app_block
    assert 'modal.Secret.from_name("wandb-interleave-pt-rl")' in app_block


def _values(command: list[str], option: str) -> list[str]:
    return [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == option
    ]


def _command(**overrides) -> list[str]:
    kwargs = {
        "hf_checkpoint": "/pretrain-checkpoints/interleave_50m/p1/final",
        "run_name": "e1_u_rl1",
        "num_rollout": 1500,
        "dynamic_filter": False,
        "rollout_seed": 42,
    }
    kwargs.update(overrides)
    return build_train_command(**kwargs)


def _write_complete_checkpoint(root: Path, step: int) -> None:
    checkpoint = root / f"iter_{step:07d}"
    for name in ("model", "optimizer", "lr_scheduler"):
        payload_root = checkpoint / name
        payload_root.mkdir(parents=True)
        (payload_root / ".metadata").write_bytes(
            f"{name} distributed checkpoint metadata".encode()
        )
        (payload_root / "__0_0.distcp").write_bytes(
            f"{name} distributed checkpoint payload".encode()
    )
    (checkpoint / "rng_rank_00000.pt").write_bytes(b"rng-rank-0")
    (checkpoint / "rollout_state.pt").write_bytes(b"rollout-state")
    (checkpoint / "meta.json").write_text(
        json.dumps(
            {
                "iteration": step,
                "rollout_id": step - 1,
                "next_rollout_id": step,
            }
        )
    )
    payload = []
    files = []
    for name in ("model", "optimizer", "lr_scheduler"):
        files.extend(
            path for path in (checkpoint / name).rglob("*") if path.is_file()
        )
    files.extend(
        [
            checkpoint / "meta.json",
            checkpoint / "rng_rank_00000.pt",
            checkpoint / "rollout_state.pt",
        ]
    )
    for path in sorted(files):
        payload.append(
            {
                "path": path.relative_to(checkpoint).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    marker_core = {
        "schema": "miles-fsdp-checkpoint-commit-v1",
        "iteration": step,
        "optimizer_included": True,
        "rng_included": True,
        "rollout_state_included": True,
        "world_size": 1,
        "payload": payload,
    }
    (checkpoint / modal_interleave.CHECKPOINT_COMMIT_MARKER).write_text(
        json.dumps(
            {
                **marker_core,
                "commit_sha256": modal_interleave._canonical_json_sha256(
                    marker_core
                ),
            }
        )
    )
    # Miles writes this last, after all checkpoint payloads.
    (root / "latest_checkpointed_iteration.txt").write_text(str(step))


class _FakeVolume:
    def __init__(self, *, failures: int = 0):
        self.attempts = 0
        self.commits = 0
        self.failures = failures

    def commit(self) -> None:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise RuntimeError("transient commit failure")
        self.commits += 1


def test_command_pins_verified_optimized_miles_settings():
    command = _command()
    assert _values(command, "--train-file") == [BALANCED_TRAIN_FILE]
    assert _values(command, "--train-file-sha256") == [BALANCED_TRAIN_SHA256]
    assert _values(command, "--save-dir") == [RAW_RL_ROOT]
    assert _values(command, "--rollout-batch-size") == ["256"]
    assert _values(command, "--n-samples-per-prompt") == ["8"]
    assert _values(command, "--global-batch-size") == ["2048"]
    assert _values(command, "--policy-loss-agg-mode") == ["token-mean"]
    assert _values(command, "--lr") == ["1e-5"]
    assert _values(command, "--adam-beta1") == ["0.9"]
    assert _values(command, "--adam-beta2") == ["0.999"]
    assert _values(command, "--adam-eps") == ["1e-8"]
    assert _values(command, "--weight-decay") == ["0.01"]
    assert _values(command, "--kl-loss-coef") == ["0.001"]
    # low_var_kl (= k3) matches the original verl chess RL setup; Miles'
    # own default is the signed, higher-variance k1 estimator.
    assert _values(command, "--kl-loss-type") == ["low_var_kl"]
    assert _values(command, "--save-interval") == ["40"]
    assert _values(command, "--small-model-profile") == [
        POLICY_UPDATE_PROFILE
    ]
    assert _values(command, "--max-tokens-per-gpu") == [
        str(SMALL_MODEL_MAX_TOKENS_DEFAULT)
    ]
    assert _values(command, "--sglang-server-concurrency") == [
        str(SGLANG_SERVER_CONCURRENCY_DEFAULT)
    ]
    assert _values(command, "--sglang-dtype") == ["bfloat16"]
    assert _values(command, "--rollout-max-prompt-len") == ["512"]
    assert _values(command, "--rollout-max-response-len") == ["2560"]
    assert _values(command, "--rollout-max-context-len") == ["3072"]
    assert "--no-gradient-checkpointing" in command
    assert "--gradient-checkpointing" not in command
    assert "--no-cispo" in command
    assert "--batched-rollout" in command
    assert "--sglang-token-id-only" in command
    assert "--sglang-enable-deterministic-inference" not in command
    assert "--debug-rollout-only" not in command
    assert "--dynamic-filter" not in command
    assert "--prepare-sft" not in command
    assert "--spec" not in command


def test_command_binds_exact_initial_adam_import_without_changing_rl_optimizer():
    command = _command(
        initial_adam_checkpoint="/pretrain-checkpoints/mixed_sft3/resume/step_00036848",
        initial_adam_completion_sha256="a" * 64,
        initial_adam_source_tree_sha256="b" * 64,
        initial_adam_step=36_848,
    )
    assert _values(command, "--initial-adam-checkpoint") == [
        "/pretrain-checkpoints/mixed_sft3/resume/step_00036848"
    ]
    assert _values(command, "--initial-adam-completion-sha256") == ["a" * 64]
    assert _values(command, "--initial-adam-source-tree-sha256") == ["b" * 64]
    assert _values(command, "--initial-adam-step") == ["36848"]
    assert _values(command, "--lr") == ["1e-5"]
    assert _values(command, "--adam-beta2") == ["0.999"]
    assert _values(command, "--weight-decay") == ["0.01"]


def test_command_rejects_partial_initial_adam_import():
    with pytest.raises(ValueError, match="requires checkpoint"):
        _command(
            initial_adam_checkpoint="/pretrain-checkpoints/source",
            initial_adam_step=36_848,
        )


def test_dynamic_stage_uses_explicit_seed_and_filter():
    command = _command(dynamic_filter=True, rollout_seed=43)
    assert "--dynamic-filter" in command
    assert _values(command, "--rollout-seed") == ["43"]


def test_native_2048_context_override_is_explicit():
    command = _command(
        model_id=modal_interleave.CONTEXT2048_MODEL_ID,
        rollout_max_prompt_len=512,
        rollout_max_response_len=1536,
        rollout_max_context_len=2048,
    )
    assert _values(command, "--rollout-max-prompt-len") == ["512"]
    assert _values(command, "--rollout-max-response-len") == ["1536"]
    assert _values(command, "--rollout-max-context-len") == ["2048"]
    assert _values(command, "--sglang-context-length") == ["2048"]


def test_native_2048_profile_rejects_legacy_3072_geometry():
    with pytest.raises(ValueError, match="requires the pinned rollout geometry"):
        _command(model_id=modal_interleave.CONTEXT2048_MODEL_ID)


def test_checkpoint_context_preflight_rejects_positions_beyond_native_limit(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 2048}))
    assert _validate_checkpoint_context(tmp_path, requested_context_len=2048) == 2048
    with pytest.raises(ValueError, match="exceeds checkpoint native context"):
        _validate_checkpoint_context(tmp_path, requested_context_len=3072)


def test_exact_context_profile_rejects_larger_native_context(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"max_position_embeddings": 4096})
    )
    with pytest.raises(ValueError, match="requires checkpoint native context"):
        _validate_checkpoint_context(
            tmp_path,
            requested_context_len=2048,
            require_exact=True,
        )


def test_rollout_context_override_rejects_invalid_geometry():
    with pytest.raises(ValueError, match="smaller than"):
        _command(
            rollout_max_prompt_len=2048,
            rollout_max_response_len=1536,
            rollout_max_context_len=2048,
        )


def test_diagnostic_mode_enables_exact_per_sample_sglang_seeds():
    command = _command(
        deterministic_inference=True,
        rollout_only=True,
    )
    assert "--sglang-enable-deterministic-inference" in command
    assert "--debug-rollout-only" in command


def test_v2r4_gate_command_is_exact_once_rollout_only():
    batch = modal_interleave.V2R4_GATE_BATCHES["A"]
    candidate = modal_interleave.V2R4_GATE_CANDIDATES[6_000]
    command = _command(
        hf_checkpoint=candidate["hf_path"],
        run_name="v2r4a-gate-w190-s6000-batch-a",
        num_rollout=4,
        rollout_seed=batch["rollout_seed"],
        save_interval=0,
        deterministic_inference=True,
        rollout_only=True,
        train_file=batch["path"],
        train_file_sha256=batch["sha256"],
        data_source_path=modal_interleave.STRICT_GATE_DATA_SOURCE_PATH,
        deterministic_seed_by_sample_index=True,
        fault_tolerance=False,
        rollout_health_check_interval=1e18,
    )

    assert _values(command, "--num-rollout") == ["4"]
    assert _values(command, "--train-file") == [batch["path"]]
    assert _values(command, "--train-file-sha256") == [batch["sha256"]]
    assert _values(command, "--data-source-path") == [
        modal_interleave.STRICT_GATE_DATA_SOURCE_PATH
    ]
    assert _values(command, "--save-interval") == ["0"]
    assert "--sglang-enable-deterministic-inference" in command
    assert "--chess-deterministic-seed-by-sample-index" in command
    assert "--debug-rollout-only" in command
    assert "--no-use-fault-tolerance" in command
    assert _values(command, "--rollout-health-check-interval") == [
        "1e+18"
    ]
    assert "--dynamic-filter" not in command
    assert "--" not in command


def test_v2r4_contract_declares_exact_six_cell_grid_and_zero_retries():
    contract = modal_interleave._v2r4_contract_static()

    assert contract["cells"] == [
        {
            "candidate_step": step,
            "batch_label": batch,
            "run_name": (
                f"v2r4a-gate-w190-s{step}-batch-{batch.lower()}"
            ),
        }
        for step in (6_000, 8_000, 9_920)
        for batch in ("A", "B")
    ]
    assert contract["semantics"]["automatic_retries"] == 0

    tree = ast.parse(Path(modal_interleave.__file__).read_text())
    gate = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "v2r4_gate_rollout"
    )
    function_decorator = next(
        decorator
        for decorator in gate.decorator_list
        if isinstance(decorator, ast.Call)
        and ast.unparse(decorator.func) == "app.function"
    )
    assert {
        keyword.arg for keyword in function_decorator.keywords
    }.isdisjoint({"retries"})

    source = ast.get_source_segment(
        Path(modal_interleave.__file__).read_text(),
        gate,
    )
    assert source is not None
    assert "fault_tolerance=False" in source
    assert "rollout_health_check_interval=1e18" in source


def test_v2r4_contract_loader_binds_file_sources_runtime_and_exact_digest(
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "project"
    miles = tmp_path / "miles"
    (project / "chess_rl_miles").mkdir(parents=True)
    miles.mkdir()
    (project / "chess_rl_miles" / "gate.py").write_text("VALUE = 1\n")
    (project / modal_interleave.V2R4_GATE_BINDING_RELATIVE_PATH).write_text(
        'EXPECTED = "outside-tree-contract-binding"\n'
    )
    (miles / "rollout.py").write_text("VALUE = 2\n")
    monkeypatch.setattr(modal_interleave.base, "PROJECT_DIR", str(project))
    monkeypatch.setattr(modal_interleave.base, "MILES_DIR", str(miles))

    core = {
        **modal_interleave._v2r4_contract_static(),
        "plan": {
            "path": "INTERLEAVED_V2R4A_GATE_AMENDMENT.md",
            "sha256": "1" * 64,
        },
        "endpoint_evaluators": {
            "pt_b1_b5": "2" * 64,
            "p2_sft_at_p1": "3" * 64,
        },
        "sources": {
            "chess_rl_miles": modal_interleave._normalized_source_identity(
                project,
                excluded_relatives=(
                    modal_interleave.V2R4_GATE_BINDING_RELATIVE_PATH,
                ),
            ),
            "miles": modal_interleave._normalized_source_identity(miles),
        },
    }
    contract_sha256 = modal_interleave._canonical_json_sha256(core)
    payload = {**core, "contract_sha256": contract_sha256}
    contract_path = tmp_path / "runtime_contract.json"
    contract_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    contract_file_sha256 = hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest()
    monkeypatch.setattr(
        modal_interleave,
        "V2R4_GATE_CONTRACT_MANIFEST",
        str(contract_path),
    )
    monkeypatch.setattr(
        modal_interleave,
        "V2R4_EXPECTED_CONTRACT_SHA256",
        contract_sha256,
    )
    monkeypatch.setattr(
        modal_interleave,
        "V2R4_EXPECTED_CONTRACT_FILE_SHA256",
        contract_file_sha256,
    )

    loaded = modal_interleave._load_v2r4_gate_contract(contract_sha256)
    assert loaded["contract_sha256"] == contract_sha256
    assert loaded["contract_file_sha256"] == contract_file_sha256

    (project / "chess_rl_miles" / "gate.py").write_text("VALUE = 9\n")
    with pytest.raises(ValueError, match="source identity drifted"):
        modal_interleave._load_v2r4_gate_contract(contract_sha256)


def test_v2r4_contract_loader_rejects_any_other_well_formed_digest(
    monkeypatch,
):
    monkeypatch.setattr(
        modal_interleave,
        "V2R4_EXPECTED_CONTRACT_SHA256",
        "a" * 64,
    )
    with pytest.raises(ValueError, match="exact frozen contract"):
        modal_interleave._load_v2r4_gate_contract("b" * 64)


def test_v2r4_exact_once_ledger_intent_uses_exclusive_create(tmp_path):
    ledger = tmp_path / "ledger.json"
    modal_interleave._exclusive_json(ledger, {"state": "launching"})
    assert json.loads(ledger.read_text()) == {"state": "launching"}

    with pytest.raises(FileExistsError):
        modal_interleave._exclusive_json(ledger, {"state": "duplicate"})
    assert json.loads(ledger.read_text()) == {"state": "launching"}


def test_sample_index_seeding_requires_deterministic_inference():
    with pytest.raises(ValueError, match="requires deterministic inference"):
        _command(deterministic_seed_by_sample_index=True)


def test_v2r4a_ray_head_inherits_sample_index_seed_mode(monkeypatch):
    monkeypatch.delenv(
        "CHESS_RL_MILES_DETERMINISTIC_SEED_MODE",
        raising=False,
    )
    plain = modal_interleave._runtime_env(run_name="plain")
    assert "CHESS_RL_MILES_DETERMINISTIC_SEED_MODE" not in plain

    gate = modal_interleave._runtime_env(
        run_name="gate",
        deterministic_seed_mode="sample-index",
    )
    assert gate["CHESS_RL_MILES_DETERMINISTIC_SEED_MODE"] == "sample-index"

    with pytest.raises(ValueError, match="unsupported deterministic seed"):
        modal_interleave._runtime_env(
            run_name="bad",
            deterministic_seed_mode="group-index",
        )

    source = Path(modal_interleave.__file__).read_text()
    gate_env = source.index(
        'deterministic_seed_mode="sample-index"', source.index(
            "def v2r4_gate_rollout"
        )
    )
    ray_start = source.index(
        "base._start_ray_head", source.index("def v2r4_gate_rollout")
    )
    assert gate_env < ray_start


def test_v2r4_cell_artifact_validator_authenticates_all_four_files(
    tmp_path,
):
    run_root = tmp_path / "run"
    training = run_root / "rollouts" / "training"
    training.mkdir(parents=True)
    rollout_seed = 1_567_877_051
    quarters = []
    for rollout_id in range(4):
        expected_prompts = []
        path = training / f"rollout_{rollout_id}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for local_group in range(256):
                group_index = rollout_id * 256 + local_group
                prompt = f"prompt-{group_index}"
                identity_row = {
                    "input": prompt,
                    "FEN": f"fen-{group_index}",
                    "PuzzleId": f"puzzle-{group_index}",
                    "ground_truth": "['e2e4']",
                }
                expected_prompts.append(
                    modal_interleave._v2r4_prompt_fingerprint(identity_row)
                )
                for sibling_index in range(8):
                    sample_index = group_index * 8 + sibling_index
                    row = {
                        **identity_row,
                        "rollout_id": rollout_id,
                        "group_index": group_index,
                        "sample_index": sample_index,
                        "sampling_seed": rollout_seed + sample_index,
                        "sampling_seed_sibling_index": sibling_index,
                        "status": "completed",
                        "metadata": {
                            "sampling_seed": rollout_seed + sample_index,
                            "sampling_seed_sibling_index": sibling_index,
                            "sampling_seed_mode": "sample-index",
                        },
                    }
                    handle.write(json.dumps(row) + "\n")
        quarters.append(
            {
                "rollout_id": rollout_id,
                "ordered_prompt_fingerprints": expected_prompts,
            }
        )

    records = modal_interleave._validate_v2r4_gate_artifacts(
        run_root=run_root,
        batch_manifest={"rollout_quarters": quarters},
        rollout_seed=rollout_seed,
    )

    assert [record["rollout_id"] for record in records] == [0, 1, 2, 3]
    assert all(record["rows"] == 2_048 for record in records)
    assert all(len(record["sha256"]) == 64 for record in records)


def test_resume_restores_optimizer_step_and_rollout_cursor():
    command = _command(
        resume_path=f"{RAW_RL_ROOT}/e1_u_rl1",
        resume_step=840,
    )
    assert _values(command, "--load") == [f"{RAW_RL_ROOT}/e1_u_rl1"]
    separator = command.index("--")
    assert command[separator + 1 :] == [
        "--ckpt-step",
        "840",
        "--start-rollout-id",
        "840",
    ]


def test_canary_can_disable_forced_final_checkpoint_explicitly():
    assert _values(_command(canary=True, save_interval=0), "--save-interval") == ["0"]


def test_validated_logical_file_path_preserves_mount_alias(tmp_path):
    volume = tmp_path / "volume"
    volume.mkdir()
    parquet = volume / "train.parquet"
    parquet.write_bytes(b"immutable")
    alias = tmp_path / "data"
    alias.symlink_to(volume, target_is_directory=True)
    logical = alias / parquet.name

    assert modal_interleave._validated_logical_file_path(
        str(logical),
        name="training parquet",
    ) == str(logical)
    assert logical.resolve(strict=True) == parquet

    with pytest.raises(ValueError, match="must be an absolute path"):
        modal_interleave._validated_logical_file_path(
            "relative.parquet",
            name="training parquet",
        )


def test_precision_resume_canary_can_checkpoint_update_one():
    assert _values(_command(canary=True, save_interval=1), "--save-interval") == ["1"]


def test_production_and_precision_gate_share_data_source_and_failure_mode():
    assert modal_interleave._training_source_contract(
        canary=False,
        rollout_only=False,
    ) == (modal_interleave.PRECISION_RESUME_DATA_SOURCE_PATH, False)
    assert modal_interleave._training_source_contract(
        canary=True,
        rollout_only=False,
    ) == (modal_interleave.DEFAULT_DATA_SOURCE_PATH, True)


def test_small_model_profile_allows_only_staged_token_budgets():
    assert SMALL_MODEL_HOST_MEMORY_GB == 192
    command = _command(max_tokens_per_gpu=131_072)
    assert _values(command, "--max-tokens-per-gpu") == ["131072"]

    with pytest.raises(ValueError, match="65,536, 131,072"):
        _command(max_tokens_per_gpu=32_768)


def test_small_model_profile_fails_closed_on_host_memory_drift(monkeypatch):
    monkeypatch.setattr(modal_interleave, "SMALL_MODEL_HOST_MEMORY_GB", 384)
    with pytest.raises(RuntimeError, match="exactly 192 GB host memory"):
        _command()


def test_sglang_server_concurrency_allows_only_staged_values():
    command = _command(sglang_server_concurrency=256)
    assert _values(command, "--sglang-server-concurrency") == ["256"]

    with pytest.raises(ValueError, match="128, 256"):
        _command(sglang_server_concurrency=512)


@pytest.mark.parametrize("value", ["../bad", "/absolute", "has space", ""])
def test_safe_component_rejects_unsafe_names(value):
    with pytest.raises(ValueError):
        _safe_component(value, name="run_name")


def test_hf_checkpoint_validation(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "tokenizer_config.json").write_text("{}")
    assert _validate_hf_checkpoint(checkpoint) == checkpoint

    (checkpoint / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="model weights"):
        _validate_hf_checkpoint(checkpoint)


def _write_tokenizer_vocab(
    path: Path,
    vocab: dict[str, int],
    *,
    custom_export_layout: bool = True,
) -> None:
    payload = (
        vocab
        if custom_export_layout
        else {"model": {"type": "WordLevel", "vocab": vocab}}
    )
    path.write_text(json.dumps(payload))


def test_rl_tokenizer_contract_accepts_exact_85_mapping(tmp_path):
    vocab = dict(modal_interleave.EXPECTED_RL_VOCAB_85)
    tokenizer = tmp_path / "vocab.json"
    _write_tokenizer_vocab(tokenizer, vocab)
    evidence = modal_interleave._validate_rl_tokenizer_vocab(tokenizer)
    assert evidence["vocab_size"] == 85
    assert evidence["token_ids"] == modal_interleave.EXPECTED_RL_TOKEN_IDS
    assert evidence["vocab_mapping_sha256"] == (
        modal_interleave.EXPECTED_RL_VOCAB_MAPPING_SHA256
    )

    tokenizer_json = tmp_path / "tokenizer.json"
    _write_tokenizer_vocab(
        tokenizer_json,
        vocab,
        custom_export_layout=False,
    )
    assert modal_interleave._validate_rl_tokenizer_vocab(
        tokenizer_json
    )["token_ids"] == modal_interleave.EXPECTED_RL_TOKEN_IDS


def test_rl_tokenizer_contract_rejects_81_tokenizer(tmp_path):
    vocab = {
        token: index
        for index, token in enumerate(
            ["<bos>", "<eos>", "<unk>", *[f"token_{i}" for i in range(3, 81)]]
        )
    }
    tokenizer = tmp_path / "vocab.json"
    _write_tokenizer_vocab(tokenizer, vocab)
    with pytest.raises(RuntimeError, match="exact 85-token"):
        modal_interleave._validate_rl_tokenizer_vocab(tokenizer)


def test_rl_tokenizer_contract_rejects_wrong_special_id(tmp_path):
    vocab = dict(modal_interleave.EXPECTED_RL_VOCAB_85)
    vocab["<T>"], vocab["</T>"] = vocab["</T>"], vocab["<T>"]
    tokenizer = tmp_path / "vocab.json"
    _write_tokenizer_vocab(tokenizer, vocab)
    with pytest.raises(RuntimeError, match="complete exact 85-token"):
        modal_interleave._validate_rl_tokenizer_vocab(tokenizer)


def test_rl_tokenizer_contract_rejects_ordinary_token_swap(tmp_path):
    vocab = dict(modal_interleave.EXPECTED_RL_VOCAB_85)
    vocab["a1"], vocab["b1"] = vocab["b1"], vocab["a1"]
    tokenizer = tmp_path / "vocab.json"
    _write_tokenizer_vocab(tokenizer, vocab)
    with pytest.raises(RuntimeError, match="complete exact 85-token"):
        modal_interleave._validate_rl_tokenizer_vocab(tokenizer)


def test_rl_tokenizer_contract_matches_real_production_lan_tokenizer(tmp_path):
    pretrain_sft_root = Path(__file__).resolve().parents[2] / "pretrain-sft"
    sys.path.insert(0, str(pretrain_sft_root))
    try:
        from llm_tokens.chess.lan_tokenizer_sft import LanTokenizerSFT

        tokenizer = LanTokenizerSFT(
            {
                "include_move_numbers": False,
                "keep_result": False,
                "include_env_tokens": True,
                "include_reward_tokens": False,
            }
        )
        real_vocab = tokenizer.get_vocab()
    finally:
        sys.path.remove(str(pretrain_sft_root))
    assert real_vocab == modal_interleave.EXPECTED_RL_VOCAB_85
    vocab_json = tmp_path / "vocab.json"
    tokenizer_json = tmp_path / "tokenizer.json"
    _write_tokenizer_vocab(vocab_json, real_vocab)
    _write_tokenizer_vocab(
        tokenizer_json,
        real_vocab,
        custom_export_layout=False,
    )
    assert modal_interleave._validate_rl_tokenizer_vocab(
        vocab_json
    )["vocab_mapping_sha256"] == modal_interleave._validate_rl_tokenizer_vocab(
        tokenizer_json
    )["vocab_mapping_sha256"]


def test_rl_tokenizer_config_requires_bos_padding_and_model_pad_id_zero():
    tokenizer_config = {
        "tokenizer_class": "HFTokenizerWrapper",
        "auto_map": {
            "AutoTokenizer": ["tokenizer.HFTokenizerWrapper", None],
        },
        "use_fast": False,
        "lan_tokenizer_class": "LanTokenizerSFT",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<bos>",
        "env_token": "<call_env>",
        "env_id": 84,
        "model_max_length": 2_048,
        "lan_config": {
            "include_move_numbers": False,
            "include_black_tripledots": False,
            "bos": "<bos>",
            "eos": "<eos>",
            "unk": "<unk>",
            "pad": "<bos>",
            "keep_result": False,
            "include_env_tokens": True,
            "include_reward_tokens": False,
        },
    }
    model_config = {
        "vocab_size": 85,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 0,
        "max_position_embeddings": 2_048,
    }
    modal_interleave._validate_rl_tokenizer_and_model_config(
        tokenizer_config,
        model_config,
    )

    wrong_tokenizer = {**tokenizer_config, "pad_token": "<eos>"}
    with pytest.raises(RuntimeError, match="pad_token drifted"):
        modal_interleave._validate_rl_tokenizer_and_model_config(
            wrong_tokenizer,
            model_config,
        )
    wrong_model = {**model_config, "pad_token_id": 1}
    with pytest.raises(RuntimeError, match="pad_token_id drifted"):
        modal_interleave._validate_rl_tokenizer_and_model_config(
            tokenizer_config,
            wrong_model,
        )

    wrong_wrapper = {**tokenizer_config, "use_fast": True}
    with pytest.raises(RuntimeError, match="use_fast drifted"):
        modal_interleave._validate_rl_tokenizer_and_model_config(
            wrong_wrapper,
            model_config,
        )

    wrong_lan = {
        **tokenizer_config,
        "lan_config": {
            **tokenizer_config["lan_config"],
            "include_reward_tokens": True,
        },
    }
    with pytest.raises(RuntimeError, match="LAN tokenizer configuration"):
        modal_interleave._validate_rl_tokenizer_and_model_config(
            wrong_lan,
            model_config,
        )


def test_complete_checkpoint_step_requires_payload_before_tracker(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "latest_checkpointed_iteration.txt").write_text("40")
    assert _latest_complete_checkpoint_step(run_root) is None

    _write_complete_checkpoint(run_root, 40)
    assert _latest_complete_checkpoint_step(run_root) == 40

    metadata_path = run_root / "iter_0000040" / "meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["next_rollout_id"] = 39
    metadata_path.write_text(json.dumps(metadata))
    assert _latest_complete_checkpoint_step(run_root) is None


def test_incremental_commit_retries_and_publishes_each_step_once(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    _write_complete_checkpoint(run_root, 40)
    volume = _FakeVolume(failures=1)

    published = _commit_new_checkpoint_if_ready(
        run_root=run_root,
        volume=volume,
        published_through_step=0,
    )
    assert published == 0
    assert volume.attempts == 1

    published = _commit_new_checkpoint_if_ready(
        run_root=run_root,
        volume=volume,
        published_through_step=published,
    )
    assert published == 40
    assert volume.commits == 1

    assert (
        _commit_new_checkpoint_if_ready(
            run_root=run_root,
            volume=volume,
            published_through_step=published,
        )
        == 40
    )
    assert volume.attempts == 2


def test_training_runner_commits_new_checkpoints_while_process_runs(
    tmp_path, monkeypatch
):
    run_root = tmp_path / "run"
    run_root.mkdir()
    volume = _FakeVolume()

    class FakeProcess:
        def __init__(self):
            self.wait_calls = 0

        def wait(self, timeout):
            assert timeout == 0.01
            self.wait_calls += 1
            if self.wait_calls == 1:
                _write_complete_checkpoint(run_root, 40)
                raise subprocess.TimeoutExpired(["miles"], timeout)
            if self.wait_calls == 2:
                _write_complete_checkpoint(run_root, 80)
                raise subprocess.TimeoutExpired(["miles"], timeout)
            return 0

    process = FakeProcess()

    def fake_popen(command, *, env, cwd, start_new_session):
        assert command == ["miles"]
        assert env == {"RAY_ADDRESS": "127.0.0.1:6379"}
        assert cwd == "/root/project"
        assert start_new_session is True
        return process

    monkeypatch.setattr(modal_interleave.subprocess, "Popen", fake_popen)
    returncode, published = _run_training_with_checkpoint_commits(
        ["miles"],
        env={"RAY_ADDRESS": "127.0.0.1:6379"},
        cwd="/root/project",
        run_root=run_root,
        volume=volume,
        poll_seconds=0.01,
    )

    assert returncode == 0
    assert published == 80
    assert volume.commits == 2


def _gate_runtime_identity() -> dict[str, object]:
    return {
        "image": modal_interleave.base.MILES_IMAGE,
        "python": "test-python",
        "platform": "test-platform",
        "packages": {},
        "installed_packages": {},
        "installed_packages_sha256": "a" * 64,
        "modal_app_name": modal_interleave.APP_NAME,
        "modal_app_id": "ap-test",
        "modal_image_id": "im-test",
        "os_package_count": 1,
        "os_package_inventory_sha256": "b" * 64,
    }


def test_precision_gate_contract_binds_exact_2048_commands(tmp_path, monkeypatch):
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    train_file = tmp_path / "train.parquet"
    train_file.write_bytes(b"immutable-filtered-data")
    source = {
        "sources": {"chess_rl_miles": {}, "miles": {}},
        "source_sha256": "c" * 64,
    }
    runtime = _gate_runtime_identity()
    monkeypatch.setattr(
        modal_interleave,
        "_precision_gate_source_identity",
        lambda: source,
    )
    monkeypatch.setattr(
        modal_interleave,
        "_hydrated_modal_runtime_identity",
        lambda: runtime,
    )

    contract = modal_interleave._precision_gate_contract(
        checkpoint=checkpoint,
        origin_authentication={"schema": "test"},
        model_id=modal_interleave.CONTEXT2048_MODEL_ID,
        train_file=str(train_file),
        train_file_sha256=hashlib.sha256(train_file.read_bytes()).hexdigest(),
        rollout_seed=42,
        wandb_project="test-project",
        max_tokens_per_gpu=131_072,
        sglang_server_concurrency=128,
        lr="1e-5",
        kl_loss_type="low_var_kl",
        rollout_max_prompt_len=512,
        rollout_max_response_len=1_536,
        rollout_max_context_len=2_048,
    )
    assert modal_interleave._validate_precision_gate_contract(contract)
    assert contract["semantics"]["data_source_path"] == (
        modal_interleave.PRECISION_RESUME_DATA_SOURCE_PATH
    )
    assert contract["semantics"]["production_and_gate_share_data_source"] is True
    assert contract["semantics"]["fault_tolerance"] is False
    first, second = modal_interleave._precision_gate_commands(contract)
    for command in (first, second):
        assert _values(command, "--rollout-max-prompt-len") == ["512"]
        assert _values(command, "--rollout-max-response-len") == ["1536"]
        assert _values(command, "--rollout-max-context-len") == ["2048"]
        assert _values(command, "--sglang-dtype") == ["bfloat16"]
        assert _values(command, "--data-source-path") == [
            modal_interleave.PRECISION_RESUME_DATA_SOURCE_PATH
        ]
        assert "--fp16" not in command
    assert _values(first, "--num-rollout") == ["1"]
    assert _values(second, "--num-rollout") == ["2"]
    assert _values(second, "--ckpt-step") == ["1"]
    assert _values(second, "--start-rollout-id") == ["1"]


def test_precision_gate_binds_initial_adam_identity_into_both_processes(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    train_file = tmp_path / "train.parquet"
    train_file.write_bytes(b"immutable-filtered-data")
    monkeypatch.setattr(
        modal_interleave,
        "_precision_gate_source_identity",
        lambda: {
            "sources": {"chess_rl_miles": {}, "miles": {}},
            "source_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        modal_interleave,
        "_hydrated_modal_runtime_identity",
        _gate_runtime_identity,
    )
    contract = modal_interleave._precision_gate_contract(
        checkpoint=checkpoint,
        origin_authentication={"schema": "test"},
        model_id=modal_interleave.CONTEXT2048_MODEL_ID,
        train_file=str(train_file),
        train_file_sha256=hashlib.sha256(train_file.read_bytes()).hexdigest(),
        rollout_seed=42,
        wandb_project="test-project",
        max_tokens_per_gpu=131_072,
        sglang_server_concurrency=128,
        lr="1e-5",
        kl_loss_type="low_var_kl",
        rollout_max_prompt_len=512,
        rollout_max_response_len=1_536,
        rollout_max_context_len=2_048,
        initial_adam_checkpoint="/pretrain-checkpoints/source/step_00036848",
        initial_adam_completion_sha256="d" * 64,
        initial_adam_source_tree_sha256="e" * 64,
        initial_adam_step=36_848,
    )
    assert contract["initial_optimizer_state"] == {
        "mode": "continue_adam_moments_and_parameter_steps",
        "checkpoint": "/pretrain-checkpoints/source/step_00036848",
        "completion_sha256": "d" * 64,
        "source_tree_sha256": "e" * 64,
        "source_step": 36_848,
        "destination_hyperparameters_preserved": True,
    }
    first, second = modal_interleave._precision_gate_commands(contract)
    for command in (first, second):
        assert _values(command, "--initial-adam-step") == ["36848"]
        assert _values(command, "--initial-adam-completion-sha256") == [
            "d" * 64
        ]
    assert modal_interleave._validate_precision_gate_contract(contract)


def test_precision_gate_rejects_non_native_context_before_data_access(tmp_path):
    with pytest.raises(ValueError, match="exact native-2048"):
        modal_interleave._precision_gate_contract(
            checkpoint=tmp_path,
            origin_authentication={},
            model_id=modal_interleave.MODEL_ID,
            train_file=str(tmp_path / "missing"),
            train_file_sha256="0" * 64,
            rollout_seed=42,
            wandb_project="test",
            max_tokens_per_gpu=131_072,
            sglang_server_concurrency=128,
            lr="1e-5",
            kl_loss_type="low_var_kl",
            rollout_max_prompt_len=512,
            rollout_max_response_len=2_560,
            rollout_max_context_len=3_072,
        )


def test_matching_deployment_authenticates_local_source_and_runtime(monkeypatch):
    source = {
        "sources": {"chess_rl_miles": {}, "miles": {}},
        "source_sha256": "d" * 64,
    }
    identity = {
        "schema": "chess-rl-miles-modal-deployment-identity-v1",
        "precision_resume_gate_version": (
            modal_interleave.PRECISION_RESUME_GATE_VERSION
        ),
        **source,
        "runtime": _gate_runtime_identity(),
    }

    class DeployedIdentity:
        def remote(self):
            return identity

    monkeypatch.setattr(
        modal_interleave,
        "_local_precision_gate_source_identity",
        lambda: source,
    )
    monkeypatch.setattr(
        modal_interleave,
        "_deployed_function",
        lambda name: DeployedIdentity(),
    )
    assert modal_interleave._require_matching_deployment() == identity


def test_controller_uses_persistent_deployed_function_handles_only():
    source = Path(modal_interleave.__file__).read_text()
    main_source = source[source.index("def main(") :]
    assert "_require_matching_deployment()" in main_source
    assert '_deployed_function("train_hf").spawn' in main_source
    assert '_deployed_function("precision_resume_gate_leg")' in main_source
    assert '"finalize_precision_resume_gate"\n            ).spawn' in main_source
    for forbidden in (
        "train_hf.spawn(",
        "precision_resume_gate_leg.spawn(",
        "finalize_precision_resume_gate.spawn(",
    ):
        assert forbidden not in main_source


def test_precision_gate_publication_is_atomic_and_idempotent(tmp_path):
    gate_root = tmp_path / "gate"
    contract = {"contract_sha256": "a" * 64}
    success_core = {
        "schema": "chess-rl-miles-precision-resume-gate-success-v1",
        "contract_sha256": "a" * 64,
        "passed": True,
        # JSON object keys are strings on disk. This models the real W&B
        # evidence shape and guards idempotence across serialization.
        "wandb": {
            "train_steps": {0: {"loss": 1.0}, 1: {"loss": 0.5}},
            "rollout_steps": {0: {"pass@1": 0.1}, 1: {"pass@1": 0.2}},
        },
    }
    success = modal_interleave._self_hashed_payload(
        success_core,
        hash_key="success_sha256",
    )
    assert modal_interleave._publish_precision_gate_result(
        gate_root,
        contract=contract,
        success=success,
    ) == "published"
    assert modal_interleave._publish_precision_gate_result(
        gate_root,
        contract=contract,
        success=success,
    ) == "authenticated-existing"
    assert sorted(path.name for path in gate_root.iterdir()) == [
        "CONTRACT.json",
        "PASSED.json",
    ]
    drifted = dict(success)
    drifted["passed"] = False
    with pytest.raises((RuntimeError, ValueError)):
        modal_interleave._publish_precision_gate_result(
            gate_root,
            contract=contract,
            success=drifted,
        )


def test_precision_gate_finalizer_has_bounded_retries_and_authenticated_reuse(
    tmp_path,
    monkeypatch,
):
    assert modal_interleave.PRECISION_FINALIZER_MAX_RETRIES == 2
    assert modal_interleave.PRECISION_FINALIZER_RETRY_DELAY_SECONDS > 0
    source = Path(modal_interleave.__file__).read_text()
    function_index = source.index("def finalize_precision_resume_gate(")
    decorator = source[source.rfind("@app.function(", 0, function_index) : function_index]
    assert "retries=modal.Retries(" in decorator
    assert "max_retries=PRECISION_FINALIZER_MAX_RETRIES" in decorator
    assert "initial_delay=PRECISION_FINALIZER_RETRY_DELAY_SECONDS" in decorator

    gate_root = tmp_path / "gate"
    contract = {"contract_sha256": "f" * 64}
    success = modal_interleave._self_hashed_payload(
        {
            "schema": "chess-rl-miles-precision-resume-gate-success-v1",
            "contract_sha256": "f" * 64,
            "passed": True,
        },
        hash_key="success_sha256",
    )
    gate_root.mkdir()
    modal_interleave._exclusive_json(gate_root / "PASSED.json", success)
    authenticated = []
    monkeypatch.setattr(
        modal_interleave,
        "_require_precision_resume_gate",
        lambda *, contract: authenticated.append(contract),
    )

    reused = modal_interleave._reuse_authenticated_precision_gate_result(
        gate_root=gate_root,
        contract=contract,
    )
    assert authenticated == [contract]
    assert reused == {
        **success,
        "passed_path": str(gate_root / "PASSED.json"),
    }

    (gate_root / "PASSED.json").unlink()
    authenticated.clear()
    assert (
        modal_interleave._reuse_authenticated_precision_gate_result(
            gate_root=gate_root,
            contract=contract,
        )
        is None
    )
    assert authenticated == []


def test_precision_gate_publication_recovers_complete_staging(tmp_path):
    gate_root = tmp_path / "gate"
    contract = {"contract_sha256": "b" * 64}
    success = modal_interleave._self_hashed_payload(
        {"contract_sha256": "b" * 64, "passed": True},
        hash_key="success_sha256",
    )
    staging = tmp_path / f".{gate_root.name}.old.incomplete"
    staging.mkdir()
    modal_interleave._exclusive_json(staging / "CONTRACT.json", contract)
    modal_interleave._exclusive_json(staging / "PASSED.json", success)

    assert modal_interleave._publish_precision_gate_result(
        gate_root,
        contract=contract,
        success=success,
    ) == "recovered-staging"
    assert gate_root.is_dir()
    assert not staging.exists()


def test_precision_gate_publication_quarantines_incomplete_staging(tmp_path):
    gate_root = tmp_path / "gate"
    contract = {"contract_sha256": "c" * 64}
    success = modal_interleave._self_hashed_payload(
        {"contract_sha256": "c" * 64, "passed": True},
        hash_key="success_sha256",
    )
    staging = tmp_path / f".{gate_root.name}.old.incomplete"
    staging.mkdir()
    modal_interleave._exclusive_json(staging / "CONTRACT.json", contract)

    assert modal_interleave._publish_precision_gate_result(
        gate_root,
        contract=contract,
        success=success,
    ) == "published"
    quarantined = list(tmp_path.glob(f"{staging.name}.quarantine.*"))
    assert len([path for path in quarantined if path.is_dir()]) == 1
    assert len(list(tmp_path.glob(f"{staging.name}.quarantine.*.reason.txt"))) == 1


def test_precision_gate_publication_quarantines_markerless_final(tmp_path):
    gate_root = tmp_path / "gate"
    contract = {"contract_sha256": "d" * 64}
    success = modal_interleave._self_hashed_payload(
        {"contract_sha256": "d" * 64, "passed": True},
        hash_key="success_sha256",
    )
    gate_root.mkdir()
    modal_interleave._exclusive_json(
        gate_root / "CONTRACT.json",
        contract,
    )

    assert modal_interleave._publish_precision_gate_result(
        gate_root,
        contract=contract,
        success=success,
    ) == "published"
    modal_interleave._validate_published_precision_gate_files(
        gate_root,
        contract=contract,
        success=success,
    )
    quarantined = list(tmp_path.glob(".gate.quarantine.*"))
    assert len([path for path in quarantined if path.is_dir()]) == 1
    assert len(
        [path for path in quarantined if path.name.endswith(".reason.txt")]
    ) == 1
