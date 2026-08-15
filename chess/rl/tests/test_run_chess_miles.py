import hashlib
import sys

import pytest

from chess_rl_miles.data import DEFAULT_TRAIN_FILE
from chess_rl_miles.scripts import run_chess_miles


def _build_command(
    monkeypatch,
    tmp_path,
    *,
    save_interval: int,
    batched_rollout: bool | None = None,
    token_id_only: bool | None = None,
    extra_cli_args: list[str] | None = None,
):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    default_train_file = data_dir / DEFAULT_TRAIN_FILE
    default_train_file.write_bytes(b"balanced chess RL test data")
    monkeypatch.setattr(
        run_chess_miles,
        "DEFAULT_TRAIN_FILE_SHA256",
        hashlib.sha256(default_train_file.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_chess_miles",
            "--hf-checkpoint",
            str(tmp_path / "model"),
            "--model-id",
            "test-model",
            "--miles-dir",
            str(tmp_path / "miles"),
            "--project-dir",
            str(tmp_path / "project"),
            "--data-dir",
            str(data_dir),
            "--save-dir",
            str(tmp_path / "artifacts"),
            "--save-interval",
            str(save_interval),
        ],
    )
    if batched_rollout is not None:
        sys.argv.append("--batched-rollout" if batched_rollout else "--no-batched-rollout")
    if token_id_only is not None:
        sys.argv.append("--sglang-token-id-only" if token_id_only else "--no-sglang-token-id-only")
    if extra_cli_args:
        sys.argv.extend(extra_cli_args)
    args = run_chess_miles.parse_args()
    return run_chess_miles.build_command(args)


def _option_values(command: list[str], option: str) -> list[str]:
    return [command[index + 1] for index, value in enumerate(command[:-1]) if value == option]


def test_zero_save_interval_omits_periodic_save_option(monkeypatch, tmp_path):
    command, _ = _build_command(monkeypatch, tmp_path, save_interval=0)

    assert "--save-interval" not in command
    save_paths = _option_values(command, "--save")
    assert len(save_paths) == 1
    assert save_paths[0].endswith("/test-model/checkpoints")


def test_positive_save_interval_is_forwarded_once(monkeypatch, tmp_path):
    command, _ = _build_command(monkeypatch, tmp_path, save_interval=20)

    assert _option_values(command, "--save-interval") == ["20"]


def test_fast_rollout_defaults_enable_refactor_batching_and_token_id_server(monkeypatch, tmp_path):
    command, env = _build_command(monkeypatch, tmp_path, save_interval=0)

    assert _option_values(command, "--prompt-data") == [
        str((tmp_path / "data" / "train_v4_dataset_balanced_multi_turn.parquet").resolve())
    ]
    assert _option_values(command, "--data-source-path") == [
        "miles.rollout.data_source.RolloutDataSourceWithBuffer"
    ]
    assert _option_values(command, "--rollout-seed") == ["42"]
    assert _option_values(command, "--rollout-function-path") == [
        "chess_rl_miles.batched_rollout.ChessBatchedRolloutFn"
    ]
    assert "--sglang-skip-tokenizer-init" in command
    assert _option_values(command, "--sglang-server-concurrency") == ["128"]
    assert _option_values(command, "--sglang-dtype") == ["bfloat16"]
    assert "--sglang-context-length" not in command
    assert _option_values(command, "--sglang-cuda-graph-backend-prefill") == ["disabled"]
    assert _option_values(command, "--rollout-health-check-interval") == ["30.0"]
    assert "--sglang-disable-piecewise-cuda-graph" not in command
    assert "--gradient-checkpointing" in command
    assert "--use-fault-tolerance" in command
    assert env["CHESS_RL_MILES_SMALL_MODEL_PROFILE"] == "standard"
    assert env["MILES_EXPERIMENTAL_ROLLOUT_REFACTOR"] == "1"


def test_explicit_sglang_context_length_matches_rollout_context(monkeypatch, tmp_path):
    command, _ = _build_command(
        monkeypatch,
        tmp_path,
        save_interval=0,
        extra_cli_args=[
            "--rollout-max-context-len",
            "2048",
            "--sglang-context-length",
            "2048",
        ],
    )
    assert _option_values(command, "--sglang-context-length") == ["2048"]


def test_sglang_context_length_rejects_mismatch(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="must equal"):
        _build_command(
            monkeypatch,
            tmp_path,
            save_interval=0,
            extra_cli_args=["--sglang-context-length", "2048"],
        )


def test_kl_loss_type_defaults_and_override_are_forwarded(monkeypatch, tmp_path):
    default_command, _ = _build_command(monkeypatch, tmp_path, save_interval=0)
    assert _option_values(default_command, "--kl-loss-type") == ["low_var_kl"]

    override_command, _ = _build_command(
        monkeypatch,
        tmp_path,
        save_interval=0,
        extra_cli_args=["--kl-loss-type", "k1"],
    )
    assert _option_values(override_command, "--kl-loss-type") == ["k1"]


def test_invalid_kl_loss_type_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _build_command(
            monkeypatch,
            tmp_path,
            save_interval=0,
            extra_cli_args=["--kl-loss-type", "not-an-estimator"],
        )


def test_fast_rollout_can_be_disabled(monkeypatch, tmp_path):
    command, env = _build_command(
        monkeypatch,
        tmp_path,
        save_interval=0,
        batched_rollout=False,
        token_id_only=False,
    )

    assert "--rollout-function-path" not in command
    assert "--sglang-skip-tokenizer-init" not in command
    assert "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR" not in env


def test_deterministic_inference_flag_is_forwarded_to_miles(
    monkeypatch, tmp_path
):
    default_command, _ = _build_command(
        monkeypatch, tmp_path, save_interval=0
    )
    assert "--sglang-enable-deterministic-inference" not in default_command
    assert "--debug-rollout-only" not in default_command

    deterministic_command, _ = _build_command(
        monkeypatch,
        tmp_path,
        save_interval=0,
        extra_cli_args=[
            "--sglang-enable-deterministic-inference",
            "--debug-rollout-only",
        ],
    )
    assert (
        deterministic_command.count(
            "--sglang-enable-deterministic-inference"
        )
        == 1
    )
    assert deterministic_command.count("--debug-rollout-only") == 1


def test_exact_once_data_source_is_forwarded(monkeypatch, tmp_path):
    command, env = _build_command(
        monkeypatch,
        tmp_path,
        save_interval=0,
        extra_cli_args=[
            "--data-source-path",
            "chess_rl_miles.gate_data_source.StrictEpochRolloutDataSource",
            "--chess-deterministic-seed-by-sample-index",
        ],
    )
    assert _option_values(command, "--data-source-path") == [
        "chess_rl_miles.gate_data_source.StrictEpochRolloutDataSource"
    ]
    assert (
        env["CHESS_RL_MILES_DETERMINISTIC_SEED_MODE"]
        == "sample-index"
    )


def test_initial_adam_import_is_forwarded_as_one_complete_identity(
    monkeypatch, tmp_path
):
    command, _ = _build_command(
        monkeypatch,
        tmp_path,
        save_interval=0,
        extra_cli_args=[
            "--initial-adam-checkpoint",
            "/pretrain-checkpoints/source/step_00036848",
            "--initial-adam-completion-sha256",
            "a" * 64,
            "--initial-adam-source-tree-sha256",
            "b" * 64,
            "--initial-adam-step",
            "36848",
        ],
    )
    assert _option_values(command, "--initial-adam-checkpoint") == [
        "/pretrain-checkpoints/source/step_00036848"
    ]
    assert _option_values(command, "--initial-adam-completion-sha256") == [
        "a" * 64
    ]
    assert _option_values(command, "--initial-adam-source-tree-sha256") == [
        "b" * 64
    ]
    assert _option_values(command, "--initial-adam-step") == ["36848"]


def test_partial_initial_adam_import_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="requires checkpoint"):
        _build_command(
            monkeypatch,
            tmp_path,
            save_interval=0,
            extra_cli_args=[
                "--initial-adam-checkpoint",
                "/pretrain-checkpoints/source",
            ],
        )


def test_rollout_health_watchdog_can_be_disabled(monkeypatch, tmp_path):
    command, _ = _build_command(
        monkeypatch,
        tmp_path,
        save_interval=0,
        extra_cli_args=[
            "--no-use-fault-tolerance",
            "--rollout-health-check-interval",
            "1e18",
        ],
    )

    assert "--use-fault-tolerance" not in command
    assert _option_values(
        command, "--rollout-health-check-interval"
    ) == ["1e+18"]


def test_explicit_train_file_checksum_and_rollout_seed_are_forwarded(
    monkeypatch,
    tmp_path,
    capsys,
):
    custom_train_file = tmp_path / "filtered.parquet"
    custom_train_file.write_bytes(b"fixed filtered dataset")
    expected_sha256 = hashlib.sha256(custom_train_file.read_bytes()).hexdigest()

    command, _ = _build_command(
        monkeypatch,
        tmp_path,
        save_interval=0,
        extra_cli_args=[
            "--train-file",
            str(custom_train_file),
            "--train-file-sha256",
            expected_sha256,
            "--rollout-seed",
            "43",
        ],
    )

    assert _option_values(command, "--prompt-data") == [str(custom_train_file.resolve())]
    assert _option_values(command, "--rollout-seed") == ["43"]
    output = capsys.readouterr().out
    assert f"sha256={expected_sha256}" in output
    assert "status=verified" in output


def test_train_file_checksum_mismatch_fails_before_launch(monkeypatch, tmp_path):
    custom_train_file = tmp_path / "wrong.parquet"
    custom_train_file.write_bytes(b"unexpected bytes")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _build_command(
            monkeypatch,
            tmp_path,
            save_interval=0,
            extra_cli_args=[
                "--train-file",
                str(custom_train_file),
                "--train-file-sha256",
                "0" * 64,
            ],
        )


def test_small_model_h200_profile_disables_checkpointing_and_raises_token_budget(
    monkeypatch,
    tmp_path,
):
    command, env = _build_command(
        monkeypatch,
        tmp_path,
        save_interval=0,
        extra_cli_args=[
            "--small-model-profile",
            "small-model-h200",
            "--no-gradient-checkpointing",
            "--max-tokens-per-gpu",
            "65536",
        ],
    )

    assert "--gradient-checkpointing" not in command
    assert _option_values(command, "--max-tokens-per-gpu") == ["65536"]
    assert env["CHESS_RL_MILES_SMALL_MODEL_PROFILE"] == "small-model-h200"


@pytest.mark.parametrize(
    "invalid_args, match",
    [
        (["--max-tokens-per-gpu", "32768"], "must be one of"),
        (["--actor-num-gpus-per-node", "4"], "exactly 1 node x 8 GPUs"),
        (["--gradient-checkpointing"], "must be disabled"),
    ],
)
def test_small_model_h200_profile_fails_closed(
    monkeypatch,
    tmp_path,
    invalid_args,
    match,
):
    with pytest.raises(ValueError, match=match):
        _build_command(
            monkeypatch,
            tmp_path,
            save_interval=0,
            extra_cli_args=[
                "--small-model-profile",
                "small-model-h200",
                "--no-gradient-checkpointing",
                "--max-tokens-per-gpu",
                "65536",
                *invalid_args,
            ],
        )
