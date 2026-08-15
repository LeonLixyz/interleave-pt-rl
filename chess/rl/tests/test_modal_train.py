import inspect
from pathlib import Path

import pytest

from chess_rl_miles.data import DEFAULT_TRAIN_FILE, DEFAULT_TRAIN_FILE_SHA256
from chess_rl_miles.scripts import modal_train
from chess_rl_miles.scripts.modal_train import (
    _choose_specs,
    _ignore_modal_source_path,
    _resolve_resume_checkpoint,
    _validate_resume_modes,
)

PRODUCTION_SPECS = [
    "6p5e18|32m|0.200|0.013",
    "6p5e18|32m|0.400|0.013",
    "6p5e18|410m|0.750|0.148",
    "6p5e18|410m|1.000|0.148",
]


def _option_values(command: list[str], option: str) -> list[str]:
    return [command[index + 1] for index, value in enumerate(command[:-1]) if value == option]


@pytest.mark.parametrize("spec", PRODUCTION_SPECS)
def test_production_specs_are_accepted(spec):
    assert _choose_specs(spec) == [spec]


def test_modal_train_defaults_to_balanced_data_and_deterministic_rollout_seed():
    parameters = inspect.signature(modal_train.train_one.get_raw_f()).parameters

    assert parameters["train_file"].default == (
        f"{modal_train.DATA_DIR}/{DEFAULT_TRAIN_FILE}"
    )
    assert parameters["rollout_seed"].default == 42
    assert DEFAULT_TRAIN_FILE_SHA256 == (
        "bcf131d88cd9916f6ec12a12f8e85901a05e0dabc5d5a1fdebd8ea0b0ae86c30"
    )


@pytest.mark.parametrize(
    "relative",
    [
        ".env",
        ".venv/bin/python",
        "pkg/__pycache__/module.cpython-311.pyc",
        "pkg/module.pyc",
        "wandb/run/files/config.yaml",
        "tests/fixture.py",
        "dashboard/node_modules/pkg/index.js",
        "dashboard/dist/index.js",
    ],
)
def test_modal_source_mount_excludes_secrets_caches_and_artifacts(relative):
    assert _ignore_modal_source_path(Path(relative))


@pytest.mark.parametrize(
    "relative",
    [
        "chess_rl_miles/rollout.py",
        "chess_rl_miles/scripts/modal_interleave.py",
        "miles/backends/training_utils/loss.py",
        "tools/convert_fsdp_to_hf.py",
        "pyproject.toml",
    ],
)
def test_modal_source_mount_keeps_runtime_source(relative):
    assert not _ignore_modal_source_path(Path(relative))


def test_modal_image_mounts_use_the_explicit_source_ignore_policy():
    source = Path(modal_train.__file__).read_text()
    assert source.count("ignore=_ignore_modal_source_path") == 2


def _make_checkpoint(save_path: Path, step: int) -> None:
    (save_path / f"iter_{step:07d}" / "model").mkdir(parents=True)
    (save_path / "latest_checkpointed_iteration.txt").write_text(str(step))


def test_strict_resume_rejects_missing_tracker(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint tracker is missing"):
        _resolve_resume_checkpoint(
            tmp_path / "new-run",
            resume_from_save=True,
            resume_if_available=False,
        )


def test_resume_if_available_starts_fresh_without_tracker(tmp_path):
    assert _resolve_resume_checkpoint(
        tmp_path / "new-run",
        resume_from_save=False,
        resume_if_available=True,
    ) == (None, None)


@pytest.mark.parametrize("resume_from_save,resume_if_available", [(True, False), (False, True)])
def test_resume_modes_select_valid_tracked_checkpoint(
    tmp_path,
    resume_from_save,
    resume_if_available,
):
    save_path = tmp_path / "run"
    _make_checkpoint(save_path, 40)

    assert _resolve_resume_checkpoint(
        save_path,
        resume_from_save=resume_from_save,
        resume_if_available=resume_if_available,
    ) == (save_path, 40)


def test_resume_rejects_tracker_with_missing_checkpoint(tmp_path):
    save_path = tmp_path / "run"
    save_path.mkdir()
    (save_path / "latest_checkpointed_iteration.txt").write_text("40")

    with pytest.raises(FileNotFoundError, match="Checkpoint step 40 is incomplete"):
        _resolve_resume_checkpoint(
            save_path,
            resume_from_save=False,
            resume_if_available=True,
        )


def test_resume_rejects_invalid_tracker(tmp_path):
    save_path = tmp_path / "run"
    save_path.mkdir()
    (save_path / "latest_checkpointed_iteration.txt").write_text("not-a-step")

    with pytest.raises(RuntimeError, match="Invalid checkpoint tracker"):
        _resolve_resume_checkpoint(
            save_path,
            resume_from_save=True,
            resume_if_available=False,
        )


def test_explicit_resume_step_must_exist(tmp_path):
    save_path = tmp_path / "run"
    _make_checkpoint(save_path, 40)

    with pytest.raises(FileNotFoundError, match="Checkpoint step 20 is incomplete"):
        _resolve_resume_checkpoint(
            save_path,
            resume_from_save=True,
            resume_if_available=False,
            resume_ckpt_step=20,
        )


def test_resume_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_resume_modes(
            resume_from_save=True,
            resume_if_available=True,
            resume_ckpt_step=0,
        )


def test_explicit_step_requires_a_resume_mode():
    with pytest.raises(ValueError, match="requires --resume-from-save"):
        _validate_resume_modes(
            resume_from_save=False,
            resume_if_available=False,
            resume_ckpt_step=20,
        )


def test_resume_if_available_wires_fresh_then_retry_commands(tmp_path, monkeypatch):
    class FakeVolume:
        def __init__(self):
            self.reload_count = 0
            self.commit_count = 0

        def reload(self):
            self.reload_count += 1

        def commit(self):
            self.commit_count += 1

    checkpoint_volume = FakeVolume()
    sft_volume = FakeVolume()
    data_volume = FakeVolume()
    commands = []

    monkeypatch.setattr(modal_train, "CKPT_DIR", str(tmp_path))
    monkeypatch.setattr(modal_train, "ckpt_vol", checkpoint_volume)
    monkeypatch.setattr(modal_train, "sft_vol", sft_volume)
    monkeypatch.setattr(modal_train, "data_vol", data_volume)
    monkeypatch.setattr(modal_train, "_cleanup_runtime", lambda: None)
    monkeypatch.setattr(modal_train, "_start_ray_head", lambda *args, **kwargs: None)

    def fake_call(command, **kwargs):
        del kwargs
        commands.append(command)
        return 0

    monkeypatch.setattr(modal_train.subprocess, "call", fake_call)
    train_one = modal_train.train_one.get_raw_f()
    kwargs = {
        "run_name_suffix": "fresh-retry-test",
        "io_layout": "flat",
        "resume_if_available": True,
    }

    train_one("6p5e18|680m|1.000|0.296", **kwargs)
    assert "--load" not in commands[-1]
    assert _option_values(commands[-1], "--train-file") == [
        "/data/chess-rl-data/train_v4_dataset_balanced_multi_turn.parquet"
    ]
    assert _option_values(commands[-1], "--rollout-seed") == ["42"]

    model_id = "C6p5e18_680m_alpha1.000_beta0.296"
    save_path = tmp_path / "chess-rl-miles" / f"{model_id}_fresh-retry-test"
    _make_checkpoint(save_path, 20)

    train_one("6p5e18|680m|1.000|0.296", **kwargs)
    load_index = commands[-1].index("--load")
    assert commands[-1][load_index + 1] == str(save_path)
    assert checkpoint_volume.reload_count == 2

    train_one(
        "6p5e18|680m|1.000|0.296",
        **kwargs,
        train_file="/data/chess-rl-data/filtered.parquet",
        train_file_sha256="a" * 64,
        rollout_seed=43,
    )
    assert _option_values(commands[-1], "--train-file") == [
        "/data/chess-rl-data/filtered.parquet"
    ]
    assert _option_values(commands[-1], "--train-file-sha256") == ["a" * 64]
    assert _option_values(commands[-1], "--rollout-seed") == ["43"]
