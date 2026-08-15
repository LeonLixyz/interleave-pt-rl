"""Crash-safe publication and validation for Accelerator checkpoints.

Checkpoint contents are written into a temporary directory, authenticated by a
completion marker, and atomically renamed to an immutable step directory.  An
atomic JSON pointer identifies the newest committed step.  Readers never infer
completeness from ``trainer_state.json`` alone.
"""
from __future__ import annotations

import contextlib
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


CHECKPOINTS_DIRECTORY = "resume_checkpoints"
CHECKPOINT_COMPLETE_FILE = ".complete.json"
LATEST_CHECKPOINT_POINTER = "latest_checkpoint.json"
LATEST_CHECKPOINT_SYMLINK = "latest"
VOLUME_COMMIT_LOCK_SUFFIX = ".volume-commit.lock"
CHECKPOINT_SCHEMA = "interleaved-accelerator-checkpoint-v1"
CHECKPOINT_POINTER_SCHEMA = "interleaved-accelerator-latest-pointer-v1"
HF_EXPORT_COMPLETE_FILE = ".complete.json"
HF_EXPORT_SCHEMA = "interleaved-hf-export-v1"
DIAGNOSTIC_SNAPSHOT_SCHEMA = "interleaved-diagnostic-snapshot-v1"


@contextlib.contextmanager
def checkpoint_volume_commit_lock(root: Path):
    """Serialize run-tree mutations with a whole-Volume commit.

    Modal commits snapshot the entire mounted Volume.  The trainer holds this
    per-run advisory lock while it creates checkpoint or export staging trees,
    and the launcher holds the same lock while validating and committing.  A
    committed Volume therefore cannot contain a transient staging directory.
    """

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
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    """Flush one already-written regular file to its backing filesystem."""

    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Flush directory-entry changes on the POSIX hosts used for training."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_tree(
    root: Path,
    *,
    allow_symlinks: bool = False,
) -> None:
    """Flush every file, then every directory from the leaves upward."""

    if not root.is_dir() or root.is_symlink():
        raise NotADirectoryError(root)
    directories: list[Path] = []
    for current, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        directory_names.sort()
        file_names.sort()
        current_path = Path(current)
        directories.append(current_path)
        for directory_name in directory_names:
            child = current_path / directory_name
            if child.is_symlink():
                if allow_symlinks:
                    continue
                raise RuntimeError(
                    f"checkpoint directories may not contain symlinks: {child}"
                )
        for file_name in file_names:
            child = current_path / file_name
            if child.is_symlink():
                if allow_symlinks:
                    continue
                raise RuntimeError(
                    "checkpoint directories may contain only regular files: "
                    f"{child}"
                )
            if not child.is_file():
                raise RuntimeError(
                    "checkpoint directories may contain only regular files: "
                    f"{child}"
                )
            _fsync_file(child)
    for directory in reversed(directories):
        _fsync_directory(directory)


def _write_durable_json(target: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace JSON and durably publish its directory entry."""

    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
    _fsync_file(temporary)
    os.replace(temporary, target)
    _fsync_directory(target.parent)


def _rename_directory_noreplace(
    temporary: Path,
    final: Path,
    *,
    allow_serialized_fallback: bool = False,
) -> None:
    """Atomically publish a directory without replacing an existing target."""

    if temporary.parent != final.parent:
        raise ValueError(
            "immutable publication requires a same-directory rename: "
            f"{temporary} -> {final}"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    temporary_bytes = os.fsencode(temporary)
    final_bytes = os.fsencode(final)
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
            temporary_bytes,
            -100,
            final_bytes,
            1,  # RENAME_NOREPLACE
        )
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(
            temporary_bytes,
            final_bytes,
            0x00000004,  # RENAME_EXCL
        )
    else:
        raise RuntimeError(
            "platform lacks atomic no-replace directory rename; refusing "
            "unsafe immutable publication"
        )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error,
            f"refusing to replace immutable directory: {final}",
            str(final),
        )
    unsupported = {
        errno.EINVAL,
        errno.ENOSYS,
        getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
        errno.EOPNOTSUPP,
    }
    if error in unsupported:
        if not allow_serialized_fallback:
            raise RuntimeError(
                "filesystem rejects atomic no-replace rename and the caller "
                "did not attest that checkpoint_volume_commit_lock is held"
            )
        # Modal Volume supports same-directory rename but currently rejects
        # renameat2(RENAME_NOREPLACE).  The trainer holds its cross-container
        # checkpoint_volume_commit_lock across this check and rename, so every
        # writer using this code observes the target as absent before the
        # ordinary atomic rename.  Never enable this path outside that lock.
        if final.exists() or final.is_symlink():
            raise FileExistsError(
                errno.EEXIST,
                f"refusing to replace immutable directory: {final}",
                str(final),
            )
        os.rename(temporary, final)
        return
    raise OSError(error, os.strerror(error), f"{temporary} -> {final}")


def _safetensors_fp32_identity(root: Path) -> dict[str, Any]:
    """Inspect tensor headers, shard inventory, and the optional HF index."""

    from safetensors import safe_open

    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors weights under {root}")
    tensor_to_shard: dict[str, str] = {}
    dtype_counts: dict[str, int] = {}
    allowed_nonfloating = {
        "BOOL",
        "I8",
        "I16",
        "I32",
        "I64",
        "U8",
        "U16",
        "U32",
        "U64",
    }
    for shard in shards:
        if shard.is_symlink():
            raise RuntimeError(f"safetensors shard may not be a symlink: {shard}")
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in tensor_to_shard:
                    raise RuntimeError(
                        f"safetensors tensor appears in multiple shards: {name}"
                    )
                dtype = str(handle.get_slice(name).get_dtype())
                tensor_to_shard[name] = shard.name
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    bad = {
        dtype: count
        for dtype, count in dtype_counts.items()
        if dtype != "F32" and dtype not in allowed_nonfloating
    }
    if bad:
        raise RuntimeError(
            f"safetensors contain non-FP32 floating tensors under {root}: {bad}"
        )
    if dtype_counts.get("F32", 0) == 0:
        raise RuntimeError(f"safetensors contain no FP32 tensors under {root}")

    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in weight_map.items()
        ):
            raise RuntimeError(f"invalid safetensors index: {index_path}")
        if weight_map != tensor_to_shard:
            raise RuntimeError(
                f"safetensors index does not match shard headers: {index_path}"
            )
        if set(weight_map.values()) != {path.name for path in shards}:
            raise RuntimeError(
                f"safetensors index shard inventory drifted: {index_path}"
            )
    elif len(shards) != 1 or shards[0].name != "model.safetensors":
        raise RuntimeError(
            "sharded HF safetensors require model.safetensors.index.json; "
            f"found {[path.name for path in shards]}"
        )
    return {
        "shards": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in shards
        ],
        "tensor_count": len(tensor_to_shard),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "weight_map_sha256": hashlib.sha256(
            canonical_json(tensor_to_shard)
        ).hexdigest(),
    }


def inspect_accelerator_checkpoint_fp32(checkpoint: Path) -> dict[str, Any]:
    """Inspect persisted Accelerator model and Adam tensors, not config claims."""

    checkpoint = checkpoint.resolve(strict=True)
    model_files = sorted(checkpoint.glob("model*.safetensors"))
    unsupported_model_files = sorted(
        [*checkpoint.glob("pytorch_model*.bin"), *checkpoint.glob("model*.bin")]
    )
    if unsupported_model_files:
        raise RuntimeError(
            "unsupported non-safetensors Accelerator model payloads: "
            f"{[path.name for path in unsupported_model_files]}"
        )
    if not model_files:
        raise FileNotFoundError(
            f"Accelerator checkpoint contains no model safetensors: {checkpoint}"
        )
    model_evidence = _safetensors_fp32_identity(checkpoint)

    optimizer_files = sorted(checkpoint.glob("optimizer*.bin"))
    if not optimizer_files:
        raise FileNotFoundError(
            f"Accelerator checkpoint contains no optimizer payload: {checkpoint}"
        )
    optimizer_tensor_count = 0
    adam_tensor_count = 0
    optimizer_files_evidence: list[dict[str, Any]] = []
    for optimizer_file in optimizer_files:
        try:
            optimizer_state = torch.load(
                optimizer_file,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"unsupported Accelerator optimizer payload: {optimizer_file}"
            ) from exc
        if not isinstance(optimizer_state, Mapping):
            raise RuntimeError(
                f"Accelerator optimizer payload is not a mapping: {optimizer_file}"
            )
        parameter_states = optimizer_state.get("state")
        if not isinstance(parameter_states, Mapping) or not parameter_states:
            raise RuntimeError(
                f"Accelerator optimizer payload has no initialized state: {optimizer_file}"
            )
        parameter_groups = optimizer_state.get("param_groups")
        if not isinstance(parameter_groups, list) or not parameter_groups:
            raise RuntimeError(
                f"Accelerator optimizer payload has no parameter groups: {optimizer_file}"
            )
        parameter_ids: list[Any] = []
        for group_index, group in enumerate(parameter_groups):
            if not isinstance(group, Mapping) or not isinstance(
                group.get("params"), list
            ):
                raise RuntimeError(
                    "Accelerator optimizer parameter group is invalid: "
                    f"{optimizer_file}:{group_index}"
                )
            parameter_ids.extend(group["params"])
        if (
            not parameter_ids
            or any(
                isinstance(parameter_id, bool)
                or not isinstance(parameter_id, int)
                for parameter_id in parameter_ids
            )
            or len(set(parameter_ids)) != len(parameter_ids)
        ):
            raise RuntimeError(
                f"Accelerator optimizer parameter inventory is invalid: {optimizer_file}"
            )
        if set(parameter_states) != set(parameter_ids):
            raise RuntimeError(
                "Accelerator optimizer state does not cover every parameter: "
                f"{optimizer_file}"
            )
        for parameter_id, state in parameter_states.items():
            if not isinstance(state, Mapping):
                raise RuntimeError(
                    f"optimizer state {parameter_id!r} is not a mapping in {optimizer_file}"
                )
            for adam_name in ("exp_avg", "exp_avg_sq"):
                tensor = state.get(adam_name)
                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError(
                        f"optimizer state {parameter_id!r} lacks {adam_name} in {optimizer_file}"
                    )
                if tensor.dtype is not torch.float32:
                    raise RuntimeError(
                        "persisted Adam tensor is not FP32: "
                        f"{optimizer_file}:{parameter_id}:{adam_name}={tensor.dtype}"
                    )
                adam_tensor_count += 1
            for name, value in state.items():
                if not isinstance(value, torch.Tensor):
                    continue
                optimizer_tensor_count += 1
                if value.is_floating_point() and value.dtype is not torch.float32:
                    raise RuntimeError(
                        "persisted floating optimizer tensor is not FP32: "
                        f"{optimizer_file}:{parameter_id}:{name}={value.dtype}"
                    )
        optimizer_files_evidence.append(
            {
                "path": optimizer_file.name,
                "bytes": optimizer_file.stat().st_size,
                "sha256": sha256_file(optimizer_file),
                "parameter_state_count": len(parameter_states),
                "parameter_group_count": len(parameter_groups),
            }
        )
    return {
        "schema": "interleaved-accelerator-persisted-fp32-v1",
        "model": model_evidence,
        "optimizer_files": optimizer_files_evidence,
        "optimizer_tensor_count": optimizer_tensor_count,
        "adam_moment_tensor_count": adam_tensor_count,
        "model_floating_dtype": "float32",
        "adam_moment_dtype": "float32",
    }


def directory_file_identity(
    root: Path,
    *,
    excluded_relative_paths: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative in excluded_relative_paths:
            continue
        if path.is_symlink():
            target = os.readlink(path)
            target_bytes = os.fsencode(target)
            files.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "target": target,
                    "bytes": len(target_bytes),
                    "sha256": hashlib.sha256(target_bytes).hexdigest(),
                }
            )
            continue
        if not path.is_file():
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise RuntimeError(f"checkpoint directory is empty: {root}")
    return {
        "file_count": len(files),
        "total_bytes": sum(int(row["bytes"]) for row in files),
        "files": files,
        "manifest_sha256": hashlib.sha256(canonical_json(files)).hexdigest(),
    }


def checkpoint_directory(root: Path, step: int) -> Path:
    if isinstance(step, bool) or int(step) < 0:
        raise ValueError(f"checkpoint step must be non-negative, got {step!r}")
    return root / CHECKPOINTS_DIRECTORY / f"step_{int(step):08d}"


def temporary_checkpoint_directory(root: Path, step: int) -> Path:
    final = checkpoint_directory(root, step)
    return final.with_name(f".{final.name}.tmp")


def write_completion_marker(checkpoint: Path, *, step: int) -> dict[str, Any]:
    """Durably write the last file inside a complete checkpoint directory.

    Every Accelerator payload file and its directory entry is flushed before
    the completion marker is created.  A present marker therefore never
    authenticates payload that was still only in the process page cache when
    this function returned.
    """

    marker_path = checkpoint / CHECKPOINT_COMPLETE_FILE
    if marker_path.exists():
        raise FileExistsError(f"checkpoint completion marker already exists: {marker_path}")
    marker_temporary = marker_path.with_suffix(marker_path.suffix + ".tmp")
    if marker_temporary.exists():
        raise FileExistsError(
            f"stale temporary checkpoint completion marker: {marker_temporary}"
        )
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(
            f"checkpoint cannot be committed without trainer state: {state_path}"
        )
    _fsync_directory_tree(checkpoint)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, Mapping):
        raise ValueError(f"checkpoint trainer state is not a mapping: {state_path}")
    if int(state.get("global_step", -1)) != int(step):
        raise ValueError(
            "checkpoint step disagrees with trainer state: "
            f"{step} != {state.get('global_step')!r}"
        )
    core: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "global_step": int(step),
        "trainer_state_sha256": sha256_file(state_path),
        "checkpoint_identity": directory_file_identity(checkpoint),
        "persisted_precision": inspect_accelerator_checkpoint_fp32(checkpoint),
    }
    marker = {
        **core,
        "marker_sha256": hashlib.sha256(canonical_json(core)).hexdigest(),
    }
    _write_durable_json(marker_path, marker)
    return marker


def publish_checkpoint_directory(
    temporary: Path,
    final: Path,
    *,
    allow_serialized_fallback: bool = False,
) -> Path:
    """Durably rename one authenticated temporary checkpoint into place."""

    if temporary.parent != final.parent:
        raise ValueError(
            "checkpoint publication requires a same-directory atomic rename: "
            f"{temporary} -> {final}"
        )
    validate_completed_checkpoint(temporary)

    # Repeat the tree flush here so callers cannot publish a marker written by
    # an implementation that omitted the payload durability barrier.  Flush
    # both ancestors before rename in case the checkpoint container directory
    # was newly created in this transaction.
    _fsync_directory_tree(temporary)
    _fsync_directory(temporary.parent)
    _fsync_directory(temporary.parent.parent)
    _rename_directory_noreplace(
        temporary,
        final,
        allow_serialized_fallback=allow_serialized_fallback,
    )
    _fsync_directory(final.parent)
    return final


def validate_completed_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Validate the completion marker and every persisted checkpoint file."""

    checkpoint = checkpoint.resolve(strict=True)
    marker_path = checkpoint / CHECKPOINT_COMPLETE_FILE
    if not marker_path.is_file():
        raise RuntimeError(
            "resume checkpoint is not committed: missing authenticated "
            f"completion marker {marker_path}"
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(marker, dict):
        raise ValueError(f"invalid checkpoint completion marker: {marker_path}")
    recorded = marker.pop("marker_sha256", None)
    expected_hash = hashlib.sha256(canonical_json(marker)).hexdigest()
    if recorded != expected_hash:
        raise RuntimeError(f"checkpoint completion marker hash drifted: {marker_path}")
    if marker.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError(
            f"unsupported checkpoint completion schema: {marker.get('schema')!r}"
        )
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"committed checkpoint lost trainer state: {state_path}")
    if marker.get("trainer_state_sha256") != sha256_file(state_path):
        raise RuntimeError(f"checkpoint trainer state hash drifted: {state_path}")
    observed_identity = directory_file_identity(
        checkpoint,
        excluded_relative_paths=frozenset({CHECKPOINT_COMPLETE_FILE}),
    )
    if marker.get("checkpoint_identity") != observed_identity:
        raise RuntimeError(f"checkpoint file identity drifted: {checkpoint}")
    observed_precision = inspect_accelerator_checkpoint_fp32(checkpoint)
    if marker.get("persisted_precision") != observed_precision:
        raise RuntimeError(
            f"checkpoint persisted precision evidence drifted: {checkpoint}"
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError(f"invalid checkpoint trainer state: {state_path}")
    if int(state.get("global_step", -1)) != int(marker.get("global_step", -2)):
        raise RuntimeError(
            f"checkpoint marker/state step mismatch under {checkpoint}"
        )
    return {
        "checkpoint": checkpoint,
        "marker": {**marker, "marker_sha256": recorded},
        "state": state,
    }


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"missing or invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must contain a mapping: {path}")
    return value


def _validate_hf_config_fp32(export: Path) -> dict[str, Any]:
    config = _read_json_mapping(export / "config.json")
    dtype_fields = {
        key: config[key]
        for key in ("dtype", "torch_dtype")
        if key in config
    }
    if not dtype_fields or any(
        value not in {"float32", "float", "torch.float32"}
        for value in dtype_fields.values()
    ):
        raise RuntimeError(
            "HF export config does not advertise FP32 persisted tensors: "
            f"{dtype_fields}"
        )
    return config


def write_hf_export_completion_marker(
    export: Path,
    *,
    global_step: int,
) -> dict[str, Any]:
    """Durably authenticate a complete FP32 Hugging Face export."""

    marker_path = export / HF_EXPORT_COMPLETE_FILE
    if marker_path.exists():
        raise FileExistsError(f"HF export completion marker exists: {marker_path}")
    state_path = export / "interleaved_training_state.json"
    state = _read_json_mapping(state_path)
    if int(state.get("global_step", -1)) != int(global_step):
        raise RuntimeError(
            "HF export state/global-step mismatch: "
            f"{state.get('global_step')!r} != {global_step}"
        )
    precision = _safetensors_fp32_identity(export)
    _validate_hf_config_fp32(export)
    _fsync_directory_tree(export)
    core = {
        "schema": HF_EXPORT_SCHEMA,
        "global_step": int(global_step),
        "trainer_state_sha256": sha256_file(state_path),
        "persisted_precision": precision,
        "export_identity": directory_file_identity(export),
    }
    marker = {
        **core,
        "marker_sha256": hashlib.sha256(canonical_json(core)).hexdigest(),
    }
    _write_durable_json(marker_path, marker)
    return marker


def validate_completed_hf_export(export: Path) -> dict[str, Any]:
    """Validate an immutable HF export, including actual tensor headers."""

    export = export.expanduser().resolve(strict=True)
    marker_path = export / HF_EXPORT_COMPLETE_FILE
    marker = _read_json_mapping(marker_path)
    recorded_hash = marker.pop("marker_sha256", None)
    if recorded_hash != hashlib.sha256(canonical_json(marker)).hexdigest():
        raise RuntimeError(f"HF export completion marker hash drifted: {marker_path}")
    if marker.get("schema") != HF_EXPORT_SCHEMA:
        raise RuntimeError(
            f"unsupported HF export completion schema: {marker.get('schema')!r}"
        )
    state_path = export / "interleaved_training_state.json"
    state = _read_json_mapping(state_path)
    if marker.get("trainer_state_sha256") != sha256_file(state_path):
        raise RuntimeError(f"HF export trainer state hash drifted: {state_path}")
    if int(marker.get("global_step", -1)) != int(state.get("global_step", -2)):
        raise RuntimeError(f"HF export marker/state step mismatch: {export}")
    observed_identity = directory_file_identity(
        export,
        excluded_relative_paths=frozenset({HF_EXPORT_COMPLETE_FILE}),
    )
    if marker.get("export_identity") != observed_identity:
        raise RuntimeError(f"HF export file identity drifted: {export}")
    precision = _safetensors_fp32_identity(export)
    if marker.get("persisted_precision") != precision:
        raise RuntimeError(f"HF export persisted precision drifted: {export}")
    _validate_hf_config_fp32(export)
    return {
        "export": export,
        "marker": {**marker, "marker_sha256": recorded_hash},
        "state": state,
    }


def publish_hf_export_directory(
    temporary: Path,
    final: Path,
    *,
    allow_serialized_fallback: bool = False,
) -> Path:
    """Durably publish one authenticated HF export without replacement."""

    if temporary.parent != final.parent:
        raise ValueError(
            f"HF export publication must use one filesystem: {temporary} -> {final}"
        )
    validate_completed_hf_export(temporary)
    _fsync_directory_tree(temporary)
    _fsync_directory(temporary.parent)
    _rename_directory_noreplace(
        temporary,
        final,
        allow_serialized_fallback=allow_serialized_fallback,
    )
    _fsync_directory(final.parent)
    return final


def write_diagnostic_snapshot_completion_marker(
    snapshot: Path,
    *,
    global_step: int,
    interval_unweighted_ce: Mapping[str, Any],
) -> dict[str, Any]:
    """Durably authenticate a paired resume checkpoint and HF export."""

    marker_path = snapshot / CHECKPOINT_COMPLETE_FILE
    if marker_path.exists():
        raise FileExistsError(f"diagnostic snapshot marker exists: {marker_path}")
    resume = validate_checkpoint_run_root(snapshot / "resume")
    resume_state = _read_json_mapping(resume / "trainer_state.json")
    hf = validate_completed_hf_export(snapshot / "hf")
    if resume_state != hf["state"]:
        raise RuntimeError("diagnostic resume and HF trainer state differ")
    if int(resume_state.get("global_step", -1)) != int(global_step):
        raise RuntimeError("diagnostic snapshot has the wrong global step")
    state_interval = resume_state.get("diagnostic_last_ce_interval")
    if not isinstance(state_interval, Mapping) or dict(state_interval) != dict(
        interval_unweighted_ce
    ):
        raise RuntimeError(
            "diagnostic interval evidence does not match authenticated trainer state"
        )
    if (
        int(state_interval.get("end_step", -1)) != int(global_step)
        or int(state_interval.get("pretrain_token_count", 0)) <= 0
        or int(state_interval.get("sft_token_count", 0)) <= 0
    ):
        raise RuntimeError("diagnostic interval lacks complete PT/SFT evidence")
    _fsync_directory_tree(snapshot, allow_symlinks=True)
    core = {
        "schema": DIAGNOSTIC_SNAPSHOT_SCHEMA,
        "global_step": int(global_step),
        "trainer_state_sha256": hashlib.sha256(
            canonical_json(resume_state)
        ).hexdigest(),
        "interval_unweighted_ce": dict(interval_unweighted_ce),
        "snapshot_identity": directory_file_identity(snapshot),
    }
    marker = {
        **core,
        "marker_sha256": hashlib.sha256(canonical_json(core)).hexdigest(),
    }
    _write_durable_json(marker_path, marker)
    return marker


def validate_completed_diagnostic_snapshot(snapshot: Path) -> dict[str, Any]:
    """Validate a paired diagnostic snapshot and all nested artifacts."""

    snapshot = snapshot.expanduser().resolve(strict=True)
    marker_path = snapshot / CHECKPOINT_COMPLETE_FILE
    marker = _read_json_mapping(marker_path)
    recorded_hash = marker.pop("marker_sha256", None)
    if recorded_hash != hashlib.sha256(canonical_json(marker)).hexdigest():
        raise RuntimeError(f"diagnostic snapshot marker hash drifted: {marker_path}")
    if marker.get("schema") != DIAGNOSTIC_SNAPSHOT_SCHEMA:
        raise RuntimeError(
            "unsupported diagnostic snapshot completion schema: "
            f"{marker.get('schema')!r}"
        )
    identity = directory_file_identity(
        snapshot,
        excluded_relative_paths=frozenset({CHECKPOINT_COMPLETE_FILE}),
    )
    if marker.get("snapshot_identity") != identity:
        raise RuntimeError(f"diagnostic snapshot file identity drifted: {snapshot}")
    resume = validate_checkpoint_run_root(snapshot / "resume")
    resume_state = _read_json_mapping(resume / "trainer_state.json")
    hf = validate_completed_hf_export(snapshot / "hf")
    if resume_state != hf["state"]:
        raise RuntimeError("diagnostic resume and HF trainer state differ")
    if marker.get("trainer_state_sha256") != hashlib.sha256(
        canonical_json(resume_state)
    ).hexdigest():
        raise RuntimeError("diagnostic snapshot trainer state hash drifted")
    if int(marker.get("global_step", -1)) != int(
        resume_state.get("global_step", -2)
    ):
        raise RuntimeError("diagnostic snapshot marker/state step mismatch")
    state_interval = resume_state.get("diagnostic_last_ce_interval")
    marker_interval = marker.get("interval_unweighted_ce")
    if not isinstance(state_interval, Mapping) or marker_interval != dict(
        state_interval
    ):
        raise RuntimeError(
            "diagnostic interval evidence drifted from authenticated trainer state"
        )
    if (
        int(state_interval.get("end_step", -1))
        != int(resume_state.get("global_step", -2))
        or int(state_interval.get("pretrain_token_count", 0)) <= 0
        or int(state_interval.get("sft_token_count", 0)) <= 0
    ):
        raise RuntimeError("diagnostic interval lacks complete PT/SFT evidence")
    return {
        "snapshot": snapshot,
        "marker": {**marker, "marker_sha256": recorded_hash},
        "state": resume_state,
    }


def publish_diagnostic_snapshot_directory(
    temporary: Path,
    final: Path,
    *,
    allow_serialized_fallback: bool = False,
) -> Path:
    """Durably publish one authenticated diagnostic snapshot."""

    if temporary.parent != final.parent:
        raise ValueError(
            f"snapshot publication must use one filesystem: {temporary} -> {final}"
        )
    validate_completed_diagnostic_snapshot(temporary)
    _fsync_directory_tree(temporary, allow_symlinks=True)
    _fsync_directory(temporary.parent)
    _rename_directory_noreplace(
        temporary,
        final,
        allow_serialized_fallback=allow_serialized_fallback,
    )
    _fsync_directory(final.parent)
    return final


def write_latest_checkpoint_pointer(root: Path, checkpoint: Path) -> dict[str, Any]:
    """Durably point a run root at a fully validated committed checkpoint."""

    root = root.resolve(strict=True)
    validated = validate_completed_checkpoint(checkpoint)
    checkpoint = validated["checkpoint"]
    if not checkpoint.is_relative_to(root):
        raise ValueError(f"checkpoint {checkpoint} is outside run root {root}")
    relative = checkpoint.relative_to(root).as_posix()
    marker_path = checkpoint / CHECKPOINT_COMPLETE_FILE
    core: dict[str, Any] = {
        "schema": CHECKPOINT_POINTER_SCHEMA,
        "global_step": int(validated["state"]["global_step"]),
        "checkpoint": relative,
        "completion_marker_sha256": sha256_file(marker_path),
    }
    pointer = {
        **core,
        "pointer_sha256": hashlib.sha256(canonical_json(core)).hexdigest(),
    }
    # Retain an atomic directory-style latest path for existing launchers and
    # upload tools.  It points only at a committed immutable directory.  Write
    # it before the JSON pointer, which remains the publication event watched
    # by the v4 launcher.
    latest_link = root / LATEST_CHECKPOINT_SYMLINK
    temporary_link = root / f".{LATEST_CHECKPOINT_SYMLINK}.tmp"
    if temporary_link.exists() or temporary_link.is_symlink():
        raise FileExistsError(f"stale temporary latest symlink: {temporary_link}")
    os.symlink(relative, temporary_link)
    try:
        if latest_link.exists() and not latest_link.is_symlink():
            raise RuntimeError(
                f"refusing to replace non-symlink latest checkpoint path: {latest_link}"
            )
        os.replace(temporary_link, latest_link)
        _fsync_directory(root)
    finally:
        if temporary_link.exists() or temporary_link.is_symlink():
            temporary_link.unlink()
            _fsync_directory(root)
    target = root / LATEST_CHECKPOINT_POINTER
    _write_durable_json(target, pointer)
    return pointer


def resolve_resume_checkpoint(path: Path) -> Path:
    """Resolve either a committed step directory or a run-root pointer."""

    path = path.expanduser().resolve(strict=True)
    if (path / CHECKPOINT_COMPLETE_FILE).is_file():
        return validate_completed_checkpoint(path)["checkpoint"]
    if (path / "trainer_state.json").is_file():
        raise RuntimeError(
            "resume checkpoint is not committed: missing authenticated "
            f"completion marker {path / CHECKPOINT_COMPLETE_FILE}"
        )

    pointer_path = path / LATEST_CHECKPOINT_POINTER
    if not pointer_path.is_file():
        raise RuntimeError(
            "resume path is neither a committed checkpoint nor a run root "
            f"with an atomic latest pointer: {path}"
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if not isinstance(pointer, dict):
        raise ValueError(f"invalid latest-checkpoint pointer: {pointer_path}")
    recorded = pointer.pop("pointer_sha256", None)
    if recorded != hashlib.sha256(canonical_json(pointer)).hexdigest():
        raise RuntimeError(f"latest-checkpoint pointer hash drifted: {pointer_path}")
    if pointer.get("schema") != CHECKPOINT_POINTER_SCHEMA:
        raise RuntimeError(
            f"unsupported latest-checkpoint pointer schema: {pointer.get('schema')!r}"
        )
    relative = Path(str(pointer.get("checkpoint", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe checkpoint path in {pointer_path}: {relative}")
    checkpoint = (path / relative).resolve(strict=True)
    if not checkpoint.is_relative_to(path):
        raise RuntimeError(f"latest checkpoint escapes run root: {checkpoint}")
    validated = validate_completed_checkpoint(checkpoint)
    marker_path = checkpoint / CHECKPOINT_COMPLETE_FILE
    if pointer.get("completion_marker_sha256") != sha256_file(marker_path):
        raise RuntimeError(f"latest pointer marker hash drifted: {pointer_path}")
    if int(pointer.get("global_step", -1)) != int(
        validated["state"].get("global_step", -2)
    ):
        raise RuntimeError(f"latest pointer step drifted: {pointer_path}")
    latest_link = path / LATEST_CHECKPOINT_SYMLINK
    if not latest_link.is_symlink():
        raise RuntimeError(f"latest checkpoint compatibility link is missing: {latest_link}")
    if latest_link.resolve(strict=True) != checkpoint:
        raise RuntimeError(
            f"latest checkpoint link/pointer mismatch under {path}: "
            f"{latest_link.resolve(strict=True)} != {checkpoint}"
        )
    return checkpoint


def validate_checkpoint_run_root(
    root: Path,
    *,
    allowed_root_directories: frozenset[str] = frozenset(),
) -> Path:
    """Reject unknown or incomplete contents and return the latest checkpoint."""

    root = root.expanduser().resolve(strict=True)
    allowed_files = {"config.yaml", "metrics.jsonl", LATEST_CHECKPOINT_POINTER}
    allowed_directories = {
        CHECKPOINTS_DIRECTORY,
        LATEST_CHECKPOINT_SYMLINK,
        *allowed_root_directories,
    }
    unknown = [
        child.name
        for child in root.iterdir()
        if (child.is_file() and child.name not in allowed_files)
        or (child.is_dir() and child.name not in allowed_directories)
        or (not child.is_file() and not child.is_dir())
    ]
    if unknown:
        raise RuntimeError(
            f"unauthenticated contents in checkpoint run root {root}: {sorted(unknown)}"
        )
    checkpoints_root = root / CHECKPOINTS_DIRECTORY
    if not checkpoints_root.is_dir():
        raise RuntimeError(f"checkpoint run root has no checkpoint directory: {root}")
    checkpoints = sorted(checkpoints_root.iterdir())
    if not checkpoints:
        raise RuntimeError(f"checkpoint run root contains no committed steps: {root}")
    checkpoint_steps: dict[Path, int] = {}
    for checkpoint in checkpoints:
        if not checkpoint.is_dir() or checkpoint.name.startswith("."):
            raise RuntimeError(
                f"incomplete or unknown checkpoint entry under {root}: {checkpoint.name}"
            )
        validated = validate_completed_checkpoint(checkpoint)
        step = int(validated["state"]["global_step"])
        if checkpoint.name != f"step_{step:08d}":
            raise RuntimeError(
                f"checkpoint directory/step mismatch under {root}: "
                f"{checkpoint.name} != step_{step:08d}"
            )
        checkpoint_steps[checkpoint.resolve(strict=True)] = step
    latest = resolve_resume_checkpoint(root)
    resolved_checkpoints = {checkpoint.resolve(strict=True) for checkpoint in checkpoints}
    if latest not in resolved_checkpoints:
        raise RuntimeError(f"latest pointer does not select a known checkpoint: {latest}")
    newest = max(checkpoint_steps, key=checkpoint_steps.__getitem__)
    if latest != newest:
        raise RuntimeError(
            "latest pointer does not select the newest committed checkpoint; "
            f"pointer={latest.name}, newest={newest.name}. Refusing to resume "
            "after an interrupted pointer publication."
        )
    return latest


__all__ = [
    "CHECKPOINTS_DIRECTORY",
    "CHECKPOINT_COMPLETE_FILE",
    "DIAGNOSTIC_SNAPSHOT_SCHEMA",
    "HF_EXPORT_COMPLETE_FILE",
    "HF_EXPORT_SCHEMA",
    "LATEST_CHECKPOINT_POINTER",
    "LATEST_CHECKPOINT_SYMLINK",
    "VOLUME_COMMIT_LOCK_SUFFIX",
    "checkpoint_volume_commit_lock",
    "checkpoint_directory",
    "directory_file_identity",
    "inspect_accelerator_checkpoint_fp32",
    "publish_checkpoint_directory",
    "publish_diagnostic_snapshot_directory",
    "publish_hf_export_directory",
    "resolve_resume_checkpoint",
    "sha256_file",
    "temporary_checkpoint_directory",
    "validate_checkpoint_run_root",
    "validate_completed_checkpoint",
    "validate_completed_diagnostic_snapshot",
    "validate_completed_hf_export",
    "write_diagnostic_snapshot_completion_marker",
    "write_completion_marker",
    "write_hf_export_completion_marker",
    "write_latest_checkpoint_pointer",
]
