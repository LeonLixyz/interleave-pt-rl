from types import SimpleNamespace
import json

import pytest
from miles.utils.types import Sample

from chess_rl_miles import io


def _args():
    return SimpleNamespace(reward_key="score")


def _positive_sample() -> Sample:
    return Sample(
        group_index=17,
        index=139,
        prompt="p <T>",
        tokens=[1, 2, 3, 4, 5, 6],
        response="r </T> M <call_env>",
        response_length=4,
        label="['a1a2']",
        reward={"score": 1.0, "first_move_score": 1.0},
        loss_mask=[1, 1, 1, 0],
        status=Sample.Status.COMPLETED,
        metadata={
            "PuzzleId": "abc",
            "Rating": 1234,
            "nested": {"preserved": True},
        },
        train_metadata={"advantage": 0.25},
        session_id="chess-group-17",
        weight_versions=["1499"],
    )


def test_positive_row_preserves_identity_metadata_and_exact_tokens(monkeypatch):
    monkeypatch.setattr(io, "compute_rollout_step", lambda args, rollout_id: 1500)
    sample = _positive_sample()

    row = io._flat_rollout_row(
        _args(),
        sample,
        rollout_id=1499,
        split="training",
        index=3,
    )

    assert row["index"] == row["artifact_index"] == 3
    assert row["group_index"] == 17
    assert row["sample_index"] == 139
    assert row["step"] == 1500
    assert row["metadata"] == sample.metadata
    assert row["train_metadata"] == {"advantage": 0.25}
    assert row["reward"] == sample.reward
    assert row["prompt_token_ids"] == [1, 2]
    assert row["response_token_ids"] == [3, 4, 5, 6]
    assert row["response_loss_mask"] == [1, 1, 1, 0]
    assert len(row["token_ids_sha256"]) == 64


def test_nonpositive_row_omits_large_token_artifact(monkeypatch):
    monkeypatch.setattr(io, "compute_rollout_step", lambda args, rollout_id: 1)
    sample = _positive_sample()
    sample.reward = {"score": 0.0}

    row = io._flat_rollout_row(
        _args(),
        sample,
        rollout_id=0,
        split="training",
        index=0,
    )

    assert "prompt_token_ids" not in row
    assert "response_token_ids" not in row
    assert "response_loss_mask" not in row


def test_positive_row_fails_closed_on_loss_mask_drift(monkeypatch):
    monkeypatch.setattr(io, "compute_rollout_step", lambda args, rollout_id: 1)
    sample = _positive_sample()
    sample.loss_mask = [1, 1]

    with pytest.raises(ValueError, match="loss-mask length"):
        io._flat_rollout_row(
            _args(),
            sample,
            rollout_id=0,
            split="training",
            index=0,
        )


def test_jsonl_write_is_atomic(tmp_path):
    path = tmp_path / "rollout.jsonl"
    io._write_jsonl(path, [{"index": 1}, {"index": 2}])

    assert [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ] == [{"index": 1}, {"index": 2}]
    assert not path.with_suffix(".jsonl.tmp").exists()


@pytest.mark.parametrize(
    "data_source_path, expected_handled",
    [
        (io.STRICT_EXACT_ONCE_DATA_SOURCE, True),
        (io.STRICT_EXHAUSTIVE_DATA_SOURCE, True),
        (
            "miles.rollout.data_source.RolloutDataSourceWithBuffer",
            False,
        ),
    ],
)
def test_rollout_logger_suppresses_default_metrics_only_for_strict_gate(
    monkeypatch,
    tmp_path,
    data_source_path,
    expected_handled,
):
    monkeypatch.setenv("CHESS_RL_MILES_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setattr(io, "compute_rollout_step", lambda args, rollout_id: 1)
    args = SimpleNamespace(
        reward_key="score",
        data_source_path=data_source_path,
    )

    handled = io.log_rollout_data(
        0,
        args,
        [_positive_sample()],
        {},
        1.0,
    )

    assert handled is expected_handled
    assert (
        tmp_path / "rollouts" / "training" / "rollout_0.jsonl"
    ).is_file()


def test_all_attempt_stream_keeps_all_one_group_dropped_by_dynamic_filter(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("CHESS_RL_MILES_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setattr(io, "compute_rollout_step", lambda args, rollout_id: 1)

    accepted_mixed_group = [_positive_sample(), _positive_sample()]
    accepted_mixed_group[0].group_index = 1
    accepted_mixed_group[1].group_index = 1
    accepted_mixed_group[1].reward = {"score": 0.0}

    # A nonzero-variance dynamic filter drops this all-one group, but the
    # pre-filter attempt logger must retain both successful siblings.
    dropped_all_one_group = [_positive_sample(), _positive_sample()]
    for offset, sample in enumerate(dropped_all_one_group):
        sample.group_index = 2
        sample.index = 200 + offset

    count = io.log_all_attempts_positive(
        0,
        _args(),
        [accepted_mixed_group, dropped_all_one_group],
    )

    path = (
        tmp_path
        / "rollouts"
        / "all_attempts_positive"
        / "rollout_0.jsonl"
    )
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert count == 3
    assert [row["group_index"] for row in rows] == [1, 2, 2]
    assert all(
        row["sampling_scope"]
        == "all_completed_attempts_before_dynamic_filter"
        for row in rows
    )
    assert all("response_token_ids" in row for row in rows)
    summary = json.loads(
        (
            tmp_path
            / "rollouts"
            / "all_attempts_positive"
            / "rollout_0.summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["attempted_groups"] == 2
    assert summary["attempted_samples"] == 4
    assert summary["positive_completed_samples"] == 3
    assert summary["group_success_count_histogram"] == {"1": 1, "2": 1}
