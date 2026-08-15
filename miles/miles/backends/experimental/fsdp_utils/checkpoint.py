from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import random
import shutil
import ctypes
import errno
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import numpy as np
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

from .precision import (
    assert_fp32_training_state,
    precision_contract,
)

logger = logging.getLogger(__name__)
CHECKPOINT_COMMIT_MARKER = "COMMITTED.json"
ROLLOUT_STATE_FILE = "rollout_state.pt"
VOLUME_COMMIT_LOCK_SUFFIX = ".volume-commit.lock"


@contextlib.contextmanager
def _checkpoint_volume_commit_lock(root: Path):
    """Exclude a whole-Volume commit while checkpoint shards are staging."""

    unresolved = root.expanduser().resolve(strict=False)
    unresolved.parent.mkdir(parents=True, exist_ok=True)
    lock_path = unresolved.parent / f".{unresolved.name}{VOLUME_COMMIT_LOCK_SUFFIX}"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


class ModelState(Stateful):
    """Wrapper for model state only."""

    def __init__(self, model):
        self.model = model

    def state_dict(self):
        model_state_dict, _ = get_state_dict(self.model, optimizers=[])
        return {"model": model_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(self.model, optimizers=[], model_state_dict=state_dict["model"], optim_state_dict=None)


class OptimizerState(Stateful):
    """Wrapper for optimizer state only."""

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        _, optimizer_state_dict = get_state_dict(self.model, optimizers=self.optimizer)
        return {"optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model, optimizers=self.optimizer, model_state_dict=None, optim_state_dict=state_dict["optim"]
        )


class LRSchedulerState(Stateful):
    """Wrapper for LR scheduler state only."""

    def __init__(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def state_dict(self):
        return {"lr_scheduler": self.lr_scheduler.state_dict()}

    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])


def _read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse checkpoint metadata at {path}")
        return {}


def _canonical_json_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_checkpoint_metadata(path: Path, metadata: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _fsync_tree(root: Path) -> None:
    """Make every existing payload file and directory durable before commit."""

    if not root.is_dir():
        raise FileNotFoundError(root)
    directories = [root]
    for path in sorted(root.rglob("*")):
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        elif path.is_dir():
            directories.append(path)
    for directory in reversed(directories):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _checkpoint_payload_manifest(
    checkpoint_dir: Path,
    *,
    include_optimizer: bool,
    include_rng: bool,
    include_rollout_state: bool,
    world_size: int,
) -> list[dict[str, Any]]:
    roots = [checkpoint_dir / "model"]
    if include_optimizer:
        roots.extend(
            [
                checkpoint_dir / "optimizer",
                checkpoint_dir / "lr_scheduler",
            ]
        )
    files: list[Path] = []
    for root in roots:
        if not (root / ".metadata").is_file():
            raise FileNotFoundError(
                f"[FSDP] Distributed-checkpoint metadata is missing: {root / '.metadata'}"
            )
        files.extend(path for path in root.rglob("*") if path.is_file())
    metadata_path = checkpoint_dir / "meta.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"[FSDP] Checkpoint metadata is missing: {metadata_path}")
    files.append(metadata_path)
    if include_rng:
        expected_rng_paths = [
            checkpoint_dir / f"rng_rank_{rank:05d}.pt"
            for rank in range(world_size)
        ]
        observed_rng_paths = sorted(checkpoint_dir.glob("rng_rank_*.pt"))
        if observed_rng_paths != expected_rng_paths:
            raise FileNotFoundError(
                "[FSDP] Distributed RNG checkpoint inventory mismatch: "
                f"expected={expected_rng_paths} actual={observed_rng_paths}"
            )
        files.extend(expected_rng_paths)
    if include_rollout_state:
        rollout_state_path = checkpoint_dir / ROLLOUT_STATE_FILE
        if not rollout_state_path.is_file():
            raise FileNotFoundError(
                f"[FSDP] Authenticated rollout cursor is missing: {rollout_state_path}"
            )
        files.append(rollout_state_path)

    # Authenticate the complete checkpoint inventory, not merely the files we
    # know how to load.  Otherwise an undeclared file or directory could be
    # added after COMMITTED.json was written without changing any recorded
    # hash.  Symlinks are forbidden so every authenticated path names bytes
    # physically contained by this checkpoint tree.
    observed_files: set[Path] = set()
    observed_directories: set[Path] = set()
    for path in checkpoint_dir.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(
                f"[FSDP] Symlinks are forbidden in committed checkpoints: {path}"
            )
        if path.is_file():
            observed_files.add(path)
        elif path.is_dir():
            observed_directories.add(path)
        else:
            raise RuntimeError(
                f"[FSDP] Unsupported checkpoint filesystem entry: {path}"
            )

    expected_files = set(files)
    marker_path = checkpoint_dir / CHECKPOINT_COMMIT_MARKER
    if marker_path.is_file():
        expected_files.add(marker_path)
    if observed_files != expected_files:
        unexpected = sorted(
            path.relative_to(checkpoint_dir).as_posix()
            for path in observed_files - expected_files
        )
        missing = sorted(
            path.relative_to(checkpoint_dir).as_posix()
            for path in expected_files - observed_files
        )
        raise RuntimeError(
            "[FSDP] Checkpoint file inventory is not fully bound by the commit "
            f"marker: unexpected={unexpected} missing={missing}"
        )

    expected_directories: set[Path] = set()
    for path in expected_files:
        parent = path.parent
        while parent != checkpoint_dir:
            expected_directories.add(parent)
            parent = parent.parent
    if observed_directories != expected_directories:
        unexpected = sorted(
            path.relative_to(checkpoint_dir).as_posix()
            for path in observed_directories - expected_directories
        )
        missing = sorted(
            path.relative_to(checkpoint_dir).as_posix()
            for path in expected_directories - observed_directories
        )
        raise RuntimeError(
            "[FSDP] Checkpoint directory inventory is not fully bound by the "
            f"commit marker: unexpected={unexpected} missing={missing}"
        )

    rows = []
    for path in sorted(set(files)):
        size = path.stat().st_size
        if size <= 0:
            raise RuntimeError(f"[FSDP] Empty checkpoint payload file: {path}")
        row: dict[str, Any] = {
            "path": path.relative_to(checkpoint_dir).as_posix(),
            "bytes": size,
            "sha256": _sha256_file(path),
        }
        rows.append(row)
    return rows


def _write_checkpoint_commit_marker(
    checkpoint_dir: Path,
    *,
    iteration: int,
    include_optimizer: bool,
    include_rng: bool,
    include_rollout_state: bool,
    world_size: int,
) -> dict[str, Any]:
    core = {
        "schema": "miles-fsdp-checkpoint-commit-v1",
        "iteration": int(iteration),
        "optimizer_included": bool(include_optimizer),
        "rng_included": bool(include_rng),
        "rollout_state_included": bool(include_rollout_state),
        "world_size": int(world_size),
        "payload": _checkpoint_payload_manifest(
            checkpoint_dir,
            include_optimizer=include_optimizer,
            include_rng=include_rng,
            include_rollout_state=include_rollout_state,
            world_size=world_size,
        ),
    }
    marker = {**core, "commit_sha256": _canonical_json_sha256(core)}
    _write_checkpoint_metadata(
        checkpoint_dir / CHECKPOINT_COMMIT_MARKER,
        marker,
    )
    return marker


def _validate_checkpoint_commit_marker(
    checkpoint_dir: Path,
    *,
    iteration: int,
    require_optimizer: bool,
    require_rng: bool,
    require_rollout_state: bool,
    expected_world_size: int,
) -> dict[str, Any]:
    marker_path = checkpoint_dir / CHECKPOINT_COMMIT_MARKER
    marker = _read_checkpoint_metadata(marker_path)
    if not marker:
        raise RuntimeError(
            f"[FSDP] Checkpoint commit marker is missing or invalid: {marker_path}"
        )
    core = {key: value for key, value in marker.items() if key != "commit_sha256"}
    if marker.get("commit_sha256") != _canonical_json_sha256(core):
        raise RuntimeError(f"[FSDP] Checkpoint commit marker hash mismatch: {marker_path}")
    if marker.get("schema") != "miles-fsdp-checkpoint-commit-v1" or marker.get(
        "iteration"
    ) != int(iteration):
        raise RuntimeError(f"[FSDP] Checkpoint commit marker identity mismatch: {marker_path}")
    if require_optimizer and marker.get("optimizer_included") is not True:
        raise RuntimeError(f"[FSDP] Checkpoint commit marker lacks optimizer state: {marker_path}")
    if require_rng and marker.get("rng_included") is not True:
        raise RuntimeError(f"[FSDP] Checkpoint commit marker lacks RNG state: {marker_path}")
    if require_rollout_state and marker.get("rollout_state_included") is not True:
        raise RuntimeError(
            f"[FSDP] Checkpoint commit marker lacks rollout cursor state: {marker_path}"
        )
    if marker.get("world_size") != int(expected_world_size):
        raise RuntimeError(
            f"[FSDP] Checkpoint world size mismatch: expected={expected_world_size} "
            f"actual={marker.get('world_size')}"
        )

    actual_payload = _checkpoint_payload_manifest(
        checkpoint_dir,
        include_optimizer=bool(marker.get("optimizer_included")),
        include_rng=bool(marker.get("rng_included")),
        include_rollout_state=bool(marker.get("rollout_state_included")),
        world_size=expected_world_size,
    )
    recorded_payload = marker.get("payload")
    if recorded_payload != actual_payload:
        raise RuntimeError(
            f"[FSDP] Checkpoint payload no longer matches commit marker: {checkpoint_dir}"
        )
    return marker


def _reconcile_checkpoint_tracker(
    root_path: Path,
    *,
    expected_world_size: int,
    require_optimizer: bool,
    require_rng: bool,
    require_rollout_state: bool,
) -> int | None:
    """Repair a stale pointer only from fully authenticated committed roots."""

    incomplete = sorted(root_path.glob(".iter_*.incomplete"))
    for staging_dir in incomplete:
        try:
            step = int(
                staging_dir.name.removeprefix(".iter_").removesuffix(
                    ".incomplete"
                )
            )
        except ValueError as exc:
            raise RuntimeError(
                f"[FSDP] Invalid checkpoint staging directory name: {staging_dir}"
            ) from exc
        checkpoint_dir = root_path / f"iter_{step:07d}"
        if checkpoint_dir.exists():
            _quarantine_checkpoint_path(
                root_path,
                staging_dir,
                reason="final checkpoint path already exists",
            )
            continue
        try:
            _validate_checkpoint_commit_marker(
                staging_dir,
                iteration=step,
                require_optimizer=require_optimizer,
                require_rng=require_rng,
                require_rollout_state=require_rollout_state,
                expected_world_size=expected_world_size,
            )
            metadata = _read_checkpoint_metadata(staging_dir / "meta.json")
            if (
                metadata.get("iteration") != step
                or metadata.get("next_rollout_id") != step
                or metadata.get("world_size") != expected_world_size
            ):
                raise RuntimeError(
                    f"[FSDP] Staged checkpoint accounting mismatch: {staging_dir}"
                )
        except Exception as exc:
            _quarantine_checkpoint_path(
                root_path,
                staging_dir,
                reason=f"not an authenticated complete checkpoint: {type(exc).__name__}: {exc}",
            )
        else:
            try:
                _rename_directory_noreplace(staging_dir, checkpoint_dir)
            except FileExistsError:
                # A competing publisher won after the earlier existence
                # check. Preserve its immutable final and retain this staging
                # tree as forensic evidence rather than replacing either.
                _quarantine_checkpoint_path(
                    root_path,
                    staging_dir,
                    reason="a concurrent publisher created the immutable final checkpoint",
                )
            else:
                _fsync_directory(root_path)
                logger.warning(
                    "[FSDP] Recovered authenticated staged checkpoint %s",
                    checkpoint_dir,
                )

    committed_steps: list[int] = []
    for checkpoint_dir in sorted(root_path.glob("iter_*")):
        if not checkpoint_dir.is_dir():
            continue
        try:
            step = int(checkpoint_dir.name.removeprefix("iter_"))
        except ValueError as exc:
            raise RuntimeError(
                f"[FSDP] Invalid checkpoint directory name: {checkpoint_dir}"
            ) from exc
        if not (checkpoint_dir / CHECKPOINT_COMMIT_MARKER).is_file():
            # Modal Volumes do not support Linux renameat2(RENAME_NOREPLACE)
            # for directories.  New checkpoints are therefore written to an
            # exclusively-created final step directory and become visible to
            # readers only when COMMITTED.json is written last.  A terminated
            # writer can leave an uncommitted final directory; preserve it for
            # diagnosis and continue only after moving it out of the committed
            # namespace.
            _quarantine_checkpoint_path(
                root_path,
                checkpoint_dir,
                reason="checkpoint writer terminated before commit marker",
            )
            continue
        _validate_checkpoint_commit_marker(
            checkpoint_dir,
            iteration=step,
            require_optimizer=require_optimizer,
            require_rng=require_rng,
            require_rollout_state=require_rollout_state,
            expected_world_size=expected_world_size,
        )
        metadata = _read_checkpoint_metadata(checkpoint_dir / "meta.json")
        if (
            metadata.get("iteration") != step
            or metadata.get("next_rollout_id") != step
            or metadata.get("world_size") != expected_world_size
        ):
            raise RuntimeError(
                f"[FSDP] Committed checkpoint accounting mismatch: {checkpoint_dir}"
            )
        committed_steps.append(step)

    tracker_file = root_path / "latest_checkpointed_iteration.txt"
    if tracker_file.exists():
        try:
            tracker_step = int(tracker_file.read_text().strip())
        except ValueError as exc:
            raise RuntimeError(
                f"[FSDP] Invalid checkpoint tracker: {tracker_file}"
            ) from exc
        if tracker_step not in committed_steps:
            raise RuntimeError(
                "[FSDP] Checkpoint tracker does not select an authenticated committed checkpoint: "
                f"tracker={tracker_step} committed={committed_steps}"
            )
    else:
        tracker_step = None

    if not committed_steps:
        return None
    latest_step = max(committed_steps)
    if tracker_step != latest_step:
        _atomic_text(tracker_file, f"{latest_step}\n")
        logger.warning(
            "[FSDP] Reconciled stale checkpoint tracker from %s to committed step %s",
            tracker_step,
            latest_step,
        )
    return latest_step


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one immutable directory without replacing a peer.

    A preceding ``exists()`` check is not sufficient: another process can
    publish the destination between that check and ``rename``.  Linux Modal
    workers provide ``renameat2(RENAME_NOREPLACE)``; the Darwin branch keeps
    local tests and development equally fail-closed.
    """

    if source.parent != destination.parent:
        raise ValueError(
            "immutable checkpoint publication requires a same-directory rename: "
            f"{source} -> {destination}"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            -100,  # AT_FDCWD
            source_bytes,
            -100,
            destination_bytes,
            1,  # RENAME_NOREPLACE
        )
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(
            source_bytes,
            destination_bytes,
            0x00000004,  # RENAME_EXCL
        )
    else:
        raise RuntimeError(
            "platform lacks atomic no-replace directory rename; refusing "
            "unsafe checkpoint publication"
        )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error,
            f"refusing to replace immutable checkpoint: {destination}",
            str(destination),
        )
    raise OSError(error, os.strerror(error), f"{source} -> {destination}")


def _quarantine_checkpoint_path(root_path: Path, path: Path, *, reason: str) -> Path:
    """Move a non-committable path aside without deleting forensic evidence."""

    quarantine_root = root_path / "_quarantine"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    destination = quarantine_root / f"{path.name}.{time.time_ns()}"
    os.replace(path, destination)
    _fsync_directory(quarantine_root)
    _fsync_directory(root_path)
    logger.warning(
        "[FSDP] Quarantined checkpoint path %s as %s: %s",
        path,
        destination,
        reason,
    )
    return destination


def load_committed_rollout_state(
    root_path: Path,
    *,
    rollout_id: int,
) -> dict[str, Any]:
    """Load a rollout cursor only after authenticating its whole checkpoint.

    The rollout manager is a separate Ray actor and does not join the FSDP
    process group.  It therefore derives the expected world size from the
    self-authenticated commit marker, then validates every committed payload
    hash before allowing pickle deserialization.
    """

    if isinstance(rollout_id, bool) or not isinstance(rollout_id, int) or rollout_id < 0:
        raise ValueError(f"invalid rollout_id for committed cursor: {rollout_id!r}")
    checkpoint_step = rollout_id + 1
    checkpoint_dir = Path(root_path).expanduser() / f"iter_{checkpoint_step:07d}"
    marker = _read_checkpoint_metadata(
        checkpoint_dir / CHECKPOINT_COMMIT_MARKER
    )
    world_size = marker.get("world_size")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise RuntimeError(
            f"[FSDP] Invalid committed world size for rollout cursor: {world_size!r}"
        )
    _validate_checkpoint_commit_marker(
        checkpoint_dir,
        iteration=checkpoint_step,
        require_optimizer=True,
        require_rng=True,
        require_rollout_state=True,
        expected_world_size=world_size,
    )
    state = torch.load(
        checkpoint_dir / ROLLOUT_STATE_FILE,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(state, dict):
        raise RuntimeError(
            f"[FSDP] Authenticated rollout state is not a dictionary: {checkpoint_dir}"
        )
    return state


def _validate_resume_precision_metadata(metadata: dict[str, Any], args: Any) -> None:
    expected_precision = precision_contract(args)
    actual_precision = metadata.get("precision_contract")
    if actual_precision != expected_precision:
        raise RuntimeError(
            "[FSDP] Resume precision contract mismatch: "
            f"expected={expected_precision} actual={actual_precision}. "
            "BF16-master checkpoints cannot resume under the FP32-master contract."
        )
    runtime_verified = metadata.get("runtime_precision_verified")
    expected_runtime = {
        "fp32_accumulated_reduced_gradients": True,
        "fp32_adam_state": True,
        "low_precision_actor_forward": True,
    }
    if runtime_verified != expected_runtime:
        raise RuntimeError(
            "[FSDP] Resume checkpoint lacks successful runtime precision verification: "
            f"expected={expected_runtime} actual={runtime_verified}"
        )


def load(actor: Any) -> dict[str, Any] | None:
    """Load checkpoint from disk.

    Loads model weights and optionally optimizer state from separate directories.
    This allows loading weights without optimizer or deleting optimizer before loading.
    """
    load_root = getattr(actor.args, "load", None)
    if load_root is None:
        return None

    root_path = Path(load_root).expanduser()
    explicit_step = getattr(actor.args, "ckpt_step", None) is not None
    if explicit_step and (
        getattr(actor.args, "no_load_optim", False)
        or getattr(actor.args, "no_load_rng", False)
    ):
        raise RuntimeError(
            "[FSDP] Resumable training requires optimizer, scheduler, and RNG state; "
            "--no-load-optim/--no-load-rng are forbidden for an explicit checkpoint step"
        )
    if not root_path.exists():
        if explicit_step:
            raise FileNotFoundError(f"[FSDP] Explicit resume root does not exist: {root_path}")
        logger.info(f"[FSDP] Checkpoint directory {root_path} not found; using fresh HF initialization.")
        return None

    reconciliation: list[Any] = [None, None]
    if dist.get_rank() == 0:
        try:
            reconciliation[0] = _reconcile_checkpoint_tracker(
                root_path,
                expected_world_size=dist.get_world_size(),
                require_optimizer=not getattr(
                    actor.args,
                    "no_load_optim",
                    False,
                ),
                require_rng=not getattr(actor.args, "no_load_rng", False),
                require_rollout_state=bool(
                    getattr(actor.args, "rollout_global_dataset", False)
                ),
            )
        except Exception as exc:
            reconciliation[1] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(reconciliation, src=0)
    if reconciliation[1] is not None:
        raise RuntimeError(str(reconciliation[1]))

    target_step = getattr(actor.args, "ckpt_step", None)
    if target_step is not None and int(target_step) != reconciliation[0]:
        raise RuntimeError(
            "[FSDP] Explicit resume step is not the latest authenticated committed checkpoint: "
            f"requested={target_step} reconciled={reconciliation[0]}. "
            "Rebuild the launch command from the reconciled pointer."
        )
    if target_step is None:
        target_step = reconciliation[0]
        if target_step is None:
            logger.info(
                f"[FSDP] No committed distributed checkpoint under {root_path}; skipping load."
            )
            return None

    checkpoint_dir = root_path / f"iter_{target_step:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    required_paths = [model_dir, checkpoint_dir / "meta.json"]
    if not getattr(actor.args, "no_load_optim", False):
        required_paths.extend([optimizer_dir, lr_scheduler_dir])
    required_paths.extend(
        [
            model_dir / ".metadata",
            checkpoint_dir / CHECKPOINT_COMMIT_MARKER,
        ]
    )
    if not getattr(actor.args, "no_load_optim", False):
        required_paths.extend(
            [
                optimizer_dir / ".metadata",
                lr_scheduler_dir / ".metadata",
            ]
        )
    if not getattr(actor.args, "no_load_rng", False):
        required_paths.append(
            checkpoint_dir / f"rng_rank_{dist.get_rank():05d}.pt"
        )
    if getattr(actor.args, "rollout_global_dataset", False):
        required_paths.append(checkpoint_dir / ROLLOUT_STATE_FILE)
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "[FSDP] Resume checkpoint is incomplete; missing required paths: "
            + ", ".join(str(path) for path in missing_paths)
        )

    metadata = _read_checkpoint_metadata(checkpoint_dir / "meta.json")
    if not metadata:
        raise RuntimeError(f"[FSDP] Resume checkpoint metadata is missing or invalid: {checkpoint_dir / 'meta.json'}")
    _validate_resume_precision_metadata(metadata, actor.args)
    _validate_checkpoint_commit_marker(
        checkpoint_dir,
        iteration=int(target_step),
        require_optimizer=not getattr(actor.args, "no_load_optim", False),
        require_rng=not getattr(actor.args, "no_load_rng", False),
        require_rollout_state=bool(
            getattr(actor.args, "rollout_global_dataset", False)
        ),
        expected_world_size=dist.get_world_size(),
    )

    # Load model weights (always)
    model_state = ModelState(actor.model)
    state_dict = {"model_state": model_state}

    try:
        dcp.load(state_dict=state_dict, checkpoint_id=str(model_dir))
        logger.info(f"[FSDP] Loaded model from {model_dir}")
    except Exception as e:
        raise RuntimeError(f"[FSDP] Required model load failed from {model_dir}") from e

    # Load optimizer state (optional)
    load_optimizer = not getattr(actor.args, "no_load_optim", False) and hasattr(actor, "optimizer")
    if load_optimizer and optimizer_dir.exists():
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        try:
            dcp.load(state_dict=optim_state_dict, checkpoint_id=str(optimizer_dir))
            logger.info(f"[FSDP] Loaded optimizer from {optimizer_dir}")
        except Exception as e:
            raise RuntimeError(f"[FSDP] Required optimizer load failed from {optimizer_dir}") from e
    elif load_optimizer:
        raise FileNotFoundError(f"[FSDP] Required optimizer checkpoint not found at {optimizer_dir}")

    # Load LR scheduler state (optional)
    load_lr_scheduler = (
        not getattr(actor.args, "no_load_optim", False)
        and hasattr(actor, "lr_scheduler")
        and lr_scheduler_dir.exists()
    )
    if load_lr_scheduler:
        lr_scheduler_state = LRSchedulerState(actor.lr_scheduler)
        lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
        try:
            dcp.load(state_dict=lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))
            logger.info(f"[FSDP] Loaded LR scheduler from {lr_scheduler_dir}")
        except Exception as e:
            raise RuntimeError(f"[FSDP] Required LR scheduler load failed from {lr_scheduler_dir}") from e
    elif not getattr(actor.args, "no_load_optim", False) and hasattr(actor, "lr_scheduler"):
        raise FileNotFoundError(f"[FSDP] Required LR scheduler checkpoint not found at {lr_scheduler_dir}")

    rng_state = None
    rng_path = checkpoint_dir / f"rng_rank_{dist.get_rank():05d}.pt"
    if rng_path.exists():
        # COMMITTED.json authenticated this exact file before deserialization.
        # NumPy RNG state contains NumPy reconstruction objects, so PyTorch's
        # weights-only unpickler cannot decode it.
        rng_state = torch.load(
            rng_path,
            map_location="cpu",
            weights_only=False,
        )
        if (
            rng_state.get("rank") != dist.get_rank()
            or rng_state.get("world_size") != dist.get_world_size()
        ):
            raise RuntimeError(
                "[FSDP] Distributed RNG checkpoint rank/world-size mismatch: "
                f"path={rng_path} rank={rng_state.get('rank')} "
                f"world_size={rng_state.get('world_size')}"
            )

    return {
        "rng": rng_state,
        "metadata": metadata,
        "iteration": target_step,
    }


def finalize_load(actor: Any, checkpoint_payload: dict[str, Any] | None) -> None:
    if checkpoint_payload is None:
        dist.barrier()
        return

    if checkpoint_payload.get("rng") is not None and not getattr(actor.args, "no_load_rng", False):
        rng_state = checkpoint_payload["rng"]
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])

    metadata = checkpoint_payload.get("metadata") or {}
    iteration = checkpoint_payload.get("iteration")
    requested_start_rollout = getattr(actor.args, "start_rollout_id", None)
    if metadata:
        actor.global_step = int(metadata.get("global_step", actor.global_step))
        actor.micro_step = int(metadata.get("micro_step", actor.micro_step))
    elif iteration is not None:
        metadata = {"next_rollout_id": iteration}
    else:
        raise RuntimeError("[FSDP] Resume payload lacks a rollout cursor")

    expected_start_rollout = int(metadata.get("next_rollout_id", iteration))
    if (
        requested_start_rollout is not None
        and int(requested_start_rollout) != expected_start_rollout
    ):
        raise RuntimeError(
            f"[FSDP] Resume rollout cursor mismatch: start_rollout_id={requested_start_rollout} "
            f"checkpoint_next_rollout_id={expected_start_rollout}"
        )
    actor.args.start_rollout_id = expected_start_rollout
    expected_global_step = int(metadata.get("global_step", -1))
    expected_micro_step = int(metadata.get("micro_step", -1))
    if expected_global_step < expected_start_rollout or expected_micro_step < expected_global_step:
        raise RuntimeError(
            f"[FSDP] Resume accounting mismatch: global_step={expected_global_step} "
            f"micro_step={expected_micro_step} next_rollout_id={expected_start_rollout}"
        )
    saved_weight_version = int(metadata.get("weight_version_at_save", -1))
    next_weight_version = int(metadata.get("next_weight_version", -1))
    if (
        saved_weight_version != expected_global_step
        or next_weight_version != expected_global_step + 1
    ):
        raise RuntimeError(
            "[FSDP] Resume rollout-weight accounting mismatch: "
            f"global_step={expected_global_step} "
            f"weight_version_at_save={saved_weight_version} "
            f"next_weight_version={next_weight_version}"
        )
    scheduler_step = int(actor.lr_scheduler.state_dict().get("last_epoch", actor.global_step))
    if not getattr(actor.args, "no_load_optim", False) and scheduler_step != actor.global_step:
        raise RuntimeError(
            f"[FSDP] Resume scheduler/accounting mismatch: scheduler_last_epoch={scheduler_step} "
            f"global_step={actor.global_step}"
        )

    assert_fp32_training_state(
        actor.model,
        actor.optimizer,
        where="after finalized checkpoint resume",
        require_optimizer_state=not getattr(actor.args, "no_load_optim", False),
    )

    torch.cuda.synchronize()
    dist.barrier()


def _save_with_volume_commit_lock_held(
    actor: Any,
    iteration: int,
    *,
    rollout_state: dict[str, Any] | None = None,
) -> None:
    """Save checkpoint to disk.

    Saves model weights and optimizer state to separate directories.
    This allows loading weights without optimizer or deleting optimizer before loading.
    """
    torch.cuda.synchronize()
    if getattr(actor.args, "no_save_optim", False):
        raise RuntimeError(
            "[FSDP] Resumable training requires optimizer and scheduler state; "
            "--no-save-optim is forbidden"
        )
    assert_fp32_training_state(
        actor.model,
        actor.optimizer,
        where=f"checkpoint serialization for iteration {iteration + 1}",
        require_optimizer_state=True,
    )
    if not getattr(actor, "_gradient_precision_verified", False):
        raise RuntimeError(
            "[FSDP] Refusing to save a training checkpoint before FP32 accumulated/reduced gradients were verified"
        )
    if not getattr(actor, "_optimizer_precision_verified", False):
        raise RuntimeError("[FSDP] Refusing to save before FP32 Adam state was verified")
    required_forward_paths = {"actor", "actor_train"}
    if getattr(actor, "ref_model", None) is not None:
        required_forward_paths.add("ref")
    verified_forward_paths = set(
        getattr(actor, "_forward_precision_verified", set())
    )
    missing_forward_paths = sorted(
        required_forward_paths - verified_forward_paths
    )
    if missing_forward_paths:
        raise RuntimeError(
            "[FSDP] Refusing to save before every required BF16 forward was verified: "
            f"missing={missing_forward_paths} verified={sorted(verified_forward_paths)}"
        )

    base_dir = Path(actor.args.save).expanduser()
    step_id = iteration + 1
    checkpoint_dir = base_dir / f"iter_{step_id:07d}"
    legacy_staging_dir = base_dir / f".iter_{step_id:07d}.incomplete"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"
    require_rollout_state = bool(
        getattr(actor.args, "rollout_global_dataset", False)
    )
    if require_rollout_state:
        if not isinstance(rollout_state, dict):
            raise RuntimeError(
                "[FSDP] A global-dataset checkpoint requires rollout_state"
            )
        expected_rollout_identity = {
            "schema": "miles-rollout-data-source-v1",
            "rollout_id": iteration,
            "next_rollout_id": step_id,
        }
        mismatches = {
            key: {"expected": value, "actual": rollout_state.get(key)}
            for key, value in expected_rollout_identity.items()
            if rollout_state.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "[FSDP] Rollout cursor does not match checkpoint identity: "
                f"{mismatches}"
            )
    elif rollout_state is not None:
        raise RuntimeError(
            "[FSDP] rollout_state was supplied without --rollout-global-dataset"
        )

    # Only rank zero owns filesystem namespace changes. Broadcasting the
    # outcome before the barrier ensures no nonzero rank writes a shard unless
    # rank zero acquired the immutable final step directory.
    setup_result: list[str | None] = [None]
    if dist.get_rank() == 0:
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            latest = _reconcile_checkpoint_tracker(
                base_dir,
                expected_world_size=dist.get_world_size(),
                require_optimizer=True,
                require_rng=True,
                require_rollout_state=require_rollout_state,
            )
            if latest is not None and latest >= step_id:
                raise FileExistsError(
                    "[FSDP] Refusing to overwrite or go behind an authenticated checkpoint: "
                    f"requested={step_id} latest={latest}"
                )
            if checkpoint_dir.exists() or legacy_staging_dir.exists():
                raise FileExistsError(
                    "[FSDP] Checkpoint reconciliation left a colliding path: "
                    f"final={checkpoint_dir.exists()} "
                    f"legacy_staging={legacy_staging_dir.exists()}"
                )
            # mkdir(exist_ok=False) is the no-replace namespace claim.  The
            # directory is intentionally uncommitted while DCP shards are
            # written; readers accept it only after COMMITTED.json appears.
            checkpoint_dir.mkdir(parents=True, exist_ok=False)
            model_dir.mkdir(parents=True, exist_ok=True)
            optimizer_dir.mkdir(parents=True, exist_ok=True)
            lr_scheduler_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            setup_result[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(setup_result, src=0)
    if setup_result[0] is not None:
        raise RuntimeError(
            f"[FSDP] Checkpoint staging setup failed: {setup_result[0]}"
        )
    dist.barrier()
    # Save model weights
    model_state = ModelState(actor.model)
    state_dict = {"model_state": model_state}
    dcp.save(state_dict, checkpoint_id=str(model_dir))

    # Save optimizer state (skip if --no-save-optim is set)
    save_optimizer_state = not getattr(actor.args, "no_save_optim", False)
    if save_optimizer_state and hasattr(actor, "optimizer") and actor.optimizer is not None:
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        dcp.save(optim_state_dict, checkpoint_id=str(optimizer_dir))

    # Save LR scheduler state (skip if --no-save-optim is set)
    if save_optimizer_state and hasattr(actor, "lr_scheduler") and actor.lr_scheduler is not None:
        lr_scheduler_state = LRSchedulerState(actor.lr_scheduler)
        lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
        dcp.save(lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))

    rng_state = {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }
    _atomic_torch_save(
        checkpoint_dir / f"rng_rank_{dist.get_rank():05d}.pt",
        rng_state,
    )
    dist.barrier()

    if dist.get_rank() == 0:
        metadata = {
            "iteration": step_id,
            "rollout_id": iteration,
            "next_rollout_id": iteration + 1,
            "global_step": actor.global_step,
            "micro_step": actor.micro_step,
            "weight_version_at_save": actor.weight_updater.weight_version,
            "next_weight_version": actor.global_step + 1,
            "world_size": dist.get_world_size(),
            "timestamp": time.time(),
            "precision_contract": precision_contract(actor.args),
            "runtime_precision_verified": {
                "fp32_accumulated_reduced_gradients": actor._gradient_precision_verified,
                "fp32_adam_state": actor._optimizer_precision_verified,
                "low_precision_actor_forward": required_forward_paths.issubset(
                    actor._forward_precision_verified
                ),
            },
            "initial_adam_import": getattr(
                actor,
                "_initial_adam_import_evidence",
                None,
            ),
            "initial_adam_step_progression": getattr(
                actor,
                "_initial_adam_step_progression",
                None,
            ),
            "forward_output_dtypes": {
                name: actor._forward_precision_dtypes[name]
                for name in sorted(required_forward_paths)
            },
            "required_forward_paths": sorted(required_forward_paths),
            "rollout_state_summary": (
                {
                    key: rollout_state[key]
                    for key in (
                        "schema",
                        "rollout_id",
                        "next_rollout_id",
                        "dataset_length",
                        "sample_offset",
                        "epoch_id",
                        "sample_group_index",
                        "sample_index",
                    )
                }
                if require_rollout_state
                else None
            ),
        }
        _write_checkpoint_metadata(checkpoint_dir / "meta.json", metadata)
        if require_rollout_state:
            _atomic_torch_save(
                checkpoint_dir / ROLLOUT_STATE_FILE,
                rollout_state,
            )

        _fsync_tree(checkpoint_dir)

        _write_checkpoint_commit_marker(
            checkpoint_dir,
            iteration=step_id,
            include_optimizer=save_optimizer_state,
            include_rng=True,
            include_rollout_state=require_rollout_state,
            world_size=dist.get_world_size(),
        )

    # No rank may still be writing a DCP shard when rank zero writes the commit
    # marker.  The final directory can be present before this barrier, but no
    # reader may treat it as resumable without its authenticated marker.
    dist.barrier()
    publication_result: list[str | None] = [None]
    if dist.get_rank() == 0:
        try:
            _fsync_directory(base_dir)

            # Publish the tracker last with an atomic rename. Readers therefore
            # observe either the previous committed checkpoint or this
            # marker-authenticated checkpoint, never an uncommitted directory.
            tracker_file = base_dir / "latest_checkpointed_iteration.txt"
            _atomic_text(tracker_file, f"{step_id}\n")
            logger.info(f"[FSDP] Saved checkpoint to {checkpoint_dir}")
        except Exception as exc:
            publication_result[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(publication_result, src=0)
    if publication_result[0] is not None:
        raise RuntimeError(
            "[FSDP] Immutable checkpoint publication failed: "
            f"{publication_result[0]}"
        )
    dist.barrier()


def save(
    actor: Any,
    iteration: int,
    *,
    rollout_state: dict[str, Any] | None = None,
) -> None:
    """Save without allowing the launcher to accept an uncommitted directory.

    Rank zero owns the advisory lock. Other ranks can enter the implementation
    immediately because its first filesystem namespace operation is a rank-zero
    setup followed by a broadcast; they cannot write shards until rank zero has
    acquired the lock and exclusively created the final step directory. The
    directory is resumable only after COMMITTED.json is written last.
    """

    base_dir = Path(actor.args.save).expanduser()
    lock = (
        _checkpoint_volume_commit_lock(base_dir)
        if dist.get_rank() == 0
        else contextlib.nullcontext()
    )
    with lock:
        _save_with_volume_commit_lock_held(
            actor,
            iteration,
            rollout_state=rollout_state,
        )
