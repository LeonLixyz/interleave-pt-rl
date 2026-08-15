from argparse import Namespace
from pathlib import Path

import pytest

from miles.backends.experimental.fsdp_utils.checkpoint import (
    CHECKPOINT_COMMIT_MARKER,
    _reconcile_checkpoint_tracker,
    _validate_checkpoint_commit_marker,
    _validate_resume_precision_metadata,
    _rename_directory_noreplace,
    _write_checkpoint_commit_marker,
    load_committed_rollout_state,
)
from miles.backends.experimental.fsdp_utils.precision import precision_contract


def _metadata(args):
    return {
        "precision_contract": precision_contract(args),
        "runtime_precision_verified": {
            "fp32_accumulated_reduced_gradients": True,
            "fp32_adam_state": True,
            "low_precision_actor_forward": True,
        },
    }


def test_resume_accepts_complete_fp32_master_runtime_evidence():
    args = Namespace(fp16=False)
    _validate_resume_precision_metadata(_metadata(args), args)


def test_resume_rejects_legacy_checkpoint_without_precision_contract():
    with pytest.raises(RuntimeError, match="precision contract mismatch"):
        _validate_resume_precision_metadata({}, Namespace(fp16=False))


def test_resume_rejects_checkpoint_without_runtime_gradient_evidence():
    args = Namespace(fp16=False)
    metadata = _metadata(args)
    metadata["runtime_precision_verified"]["fp32_accumulated_reduced_gradients"] = False
    with pytest.raises(RuntimeError, match="lacks successful runtime precision verification"):
        _validate_resume_precision_metadata(metadata, args)


def _write_payload(checkpoint: Path) -> None:
    for name in ("model", "optimizer", "lr_scheduler"):
        root = checkpoint / name
        root.mkdir(parents=True)
        (root / ".metadata").write_bytes(f"{name}-metadata".encode())
        (root / "__0_0.distcp").write_bytes(f"{name}-payload".encode())
    (checkpoint / "meta.json").write_text('{"iteration": 1}\n')
    (checkpoint / "rng_rank_00000.pt").write_bytes(b"rng-rank-0")
    (checkpoint / "rollout_state.pt").write_bytes(b"rollout-state")


def test_checkpoint_commit_marker_authenticates_complete_payload(tmp_path):
    checkpoint = tmp_path / "iter_0000001"
    _write_payload(checkpoint)

    marker = _write_checkpoint_commit_marker(
        checkpoint,
        iteration=1,
        include_optimizer=True,
        include_rng=True,
        include_rollout_state=True,
        world_size=1,
    )

    assert (checkpoint / CHECKPOINT_COMMIT_MARKER).is_file()
    assert marker["iteration"] == 1
    assert _validate_checkpoint_commit_marker(
        checkpoint,
        iteration=1,
        require_optimizer=True,
        require_rng=True,
        require_rollout_state=True,
        expected_world_size=1,
    ) == marker


def test_checkpoint_commit_marker_rejects_truncated_payload(tmp_path):
    checkpoint = tmp_path / "iter_0000001"
    _write_payload(checkpoint)
    _write_checkpoint_commit_marker(
        checkpoint,
        iteration=1,
        include_optimizer=True,
        include_rng=True,
        include_rollout_state=True,
        world_size=1,
    )
    # Same-size corruption proves the marker authenticates contents, not only
    # file presence and byte length.
    (checkpoint / "model" / "__0_0.distcp").write_bytes(b"model-payloae")

    with pytest.raises(RuntimeError, match="no longer matches commit marker"):
        _validate_checkpoint_commit_marker(
            checkpoint,
            iteration=1,
            require_optimizer=True,
            require_rng=True,
            require_rollout_state=True,
            expected_world_size=1,
        )


def test_checkpoint_commit_marker_requires_all_rank_rng_files(tmp_path):
    checkpoint = tmp_path / "iter_0000001"
    _write_payload(checkpoint)

    with pytest.raises(FileNotFoundError, match="RNG checkpoint inventory"):
        _write_checkpoint_commit_marker(
            checkpoint,
            iteration=1,
            include_optimizer=True,
            include_rng=True,
            include_rollout_state=True,
            world_size=2,
        )


def test_checkpoint_commit_marker_rejects_undeclared_root_entry(tmp_path):
    checkpoint = tmp_path / "iter_0000001"
    _write_payload(checkpoint)
    _write_checkpoint_commit_marker(
        checkpoint,
        iteration=1,
        include_optimizer=True,
        include_rng=True,
        include_rollout_state=True,
        world_size=1,
    )
    (checkpoint / "undeclared.txt").write_text("not authenticated")
    with pytest.raises(RuntimeError, match="not fully bound"):
        _validate_checkpoint_commit_marker(
            checkpoint,
            iteration=1,
            require_optimizer=True,
            require_rng=True,
            require_rollout_state=True,
            expected_world_size=1,
        )


def test_atomic_directory_publish_never_replaces_existing_destination(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "value").write_text("candidate")
    (destination / "value").write_text("winner")

    with pytest.raises(FileExistsError):
        _rename_directory_noreplace(source, destination)
    assert (destination / "value").read_text() == "winner"
    assert (source / "value").read_text() == "candidate"


def test_reconcile_quarantines_final_directory_without_commit_marker(tmp_path):
    root = tmp_path / "checkpoints"
    checkpoint = root / "iter_0000001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "partial-shard").write_bytes(b"forensic payload")

    assert _reconcile_checkpoint_tracker(
        root,
        expected_world_size=1,
        require_optimizer=True,
        require_rng=True,
        require_rollout_state=True,
    ) is None

    assert not checkpoint.exists()
    quarantined = list((root / "_quarantine").iterdir())
    assert len(quarantined) == 1
    assert quarantined[0].name.startswith("iter_0000001.")
    assert (quarantined[0] / "partial-shard").read_bytes() == b"forensic payload"
