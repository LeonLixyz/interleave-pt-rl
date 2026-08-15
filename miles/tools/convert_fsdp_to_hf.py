import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed.checkpoint as dist_cp
from transformers import AutoConfig, AutoModelForCausalLM
from typing_extensions import override


SOURCE_COMMIT_MARKER = "COMMITTED.json"
EXPORT_COMMIT_MARKER = "COMMITTED.json"
SOURCE_COMMIT_SCHEMA = "miles-fsdp-checkpoint-commit-v1"
EXPORT_COMMIT_SCHEMA = "miles-hf-export-commit-v1"


class UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


class WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: dist_cp.metadata.Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k:
                continue
            print(f"find {k} in torch_dist ckpt")
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Missing or invalid authenticated JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Authenticated JSON must contain an object: {path}")
    return value


def _file_manifest(root: Path, *, excluded_names: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise NotADirectoryError(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Authenticated artifacts may not contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded_names:
            continue
        size = path.stat().st_size
        if size <= 0:
            raise RuntimeError(f"Authenticated artifact file is empty: {path}")
        rows.append(
            {
                "path": relative,
                "bytes": size,
                "sha256": _sha256_file(path),
            }
        )
    if not rows:
        raise RuntimeError(f"Authenticated artifact contains no payload files: {root}")
    return rows


def _expected_source_payload(
    checkpoint_root: Path,
    *,
    optimizer_included: bool,
    rng_included: bool,
    rollout_state_included: bool,
    world_size: int,
) -> list[dict[str, Any]]:
    roots = [checkpoint_root / "model"]
    if optimizer_included:
        roots.extend(
            [
                checkpoint_root / "optimizer",
                checkpoint_root / "lr_scheduler",
            ]
        )
    files: list[Path] = []
    for root in roots:
        metadata = root / ".metadata"
        if not metadata.is_file():
            raise FileNotFoundError(
                f"Committed DCP metadata is missing: {metadata}"
            )
        files.extend(path for path in root.rglob("*") if path.is_file())

    meta_path = checkpoint_root / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Committed checkpoint metadata is missing: {meta_path}")
    files.append(meta_path)

    if rng_included:
        expected_rng = [
            checkpoint_root / f"rng_rank_{rank:05d}.pt"
            for rank in range(world_size)
        ]
        observed_rng = sorted(checkpoint_root.glob("rng_rank_*.pt"))
        if observed_rng != expected_rng:
            raise RuntimeError(
                "Committed checkpoint RNG inventory mismatch: "
                f"expected={expected_rng} actual={observed_rng}"
            )
        files.extend(expected_rng)
    rollout_state_path = checkpoint_root / "rollout_state.pt"
    if rollout_state_included:
        if not rollout_state_path.is_file():
            raise FileNotFoundError(
                f"Committed checkpoint rollout state is missing: {rollout_state_path}"
            )
        files.append(rollout_state_path)
    elif rollout_state_path.exists():
        raise RuntimeError(
            "Checkpoint contains unauthenticated rollout state while its "
            f"marker says rollout_state_included=false: {rollout_state_path}"
        )

    expected_files = set(files)
    allowed_files = set(expected_files)
    commit_marker = checkpoint_root / SOURCE_COMMIT_MARKER
    if commit_marker.exists():
        allowed_files.add(commit_marker)
    allowed_directories = {checkpoint_root}
    for path in allowed_files:
        parent = path.parent
        while parent != checkpoint_root:
            allowed_directories.add(parent)
            parent = parent.parent
    for path in checkpoint_root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(
                f"Committed checkpoint may not contain symlinks: {path}"
            )
        if path.is_file() and path not in allowed_files:
            raise RuntimeError(
                f"Committed checkpoint contains unauthenticated file: {path}"
            )
        if path.is_dir() and path not in allowed_directories:
            raise RuntimeError(
                f"Committed checkpoint contains unauthenticated directory: {path}"
            )

    rows: list[dict[str, Any]] = []
    for path in sorted(set(files)):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Committed checkpoint payload is not a regular file: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise RuntimeError(f"Committed checkpoint payload is empty: {path}")
        rows.append(
            {
                "path": path.relative_to(checkpoint_root).as_posix(),
                "bytes": size,
                "sha256": _sha256_file(path),
            }
        )
    return rows


def validate_committed_source(input_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate a Miles commit marker and every payload byte it binds."""

    supplied = Path(input_dir).expanduser().resolve(strict=True)
    checkpoint_root = supplied.parent if supplied.name == "model" else supplied
    marker_path = checkpoint_root / SOURCE_COMMIT_MARKER
    marker = _read_json_object(marker_path)
    core = {key: value for key, value in marker.items() if key != "commit_sha256"}
    if marker.get("commit_sha256") != _canonical_json_sha256(core):
        raise RuntimeError(f"Checkpoint commit marker hash mismatch: {marker_path}")
    if marker.get("schema") != SOURCE_COMMIT_SCHEMA:
        raise RuntimeError(
            f"Unsupported checkpoint commit schema at {marker_path}: {marker.get('schema')!r}"
        )

    iteration = marker.get("iteration")
    world_size = marker.get("world_size")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise RuntimeError(f"Invalid committed checkpoint iteration: {iteration!r}")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise RuntimeError(f"Invalid committed checkpoint world size: {world_size!r}")
    expected_name = f"iter_{iteration:07d}"
    if checkpoint_root.name != expected_name:
        raise RuntimeError(
            "Committed checkpoint directory/iteration mismatch: "
            f"{checkpoint_root.name!r} != {expected_name!r}"
        )
    for key in (
        "optimizer_included",
        "rng_included",
        "rollout_state_included",
    ):
        if not isinstance(marker.get(key), bool):
            raise RuntimeError(f"Invalid checkpoint marker field {key}: {marker.get(key)!r}")

    actual_payload = _expected_source_payload(
        checkpoint_root,
        optimizer_included=marker["optimizer_included"],
        rng_included=marker["rng_included"],
        rollout_state_included=marker["rollout_state_included"],
        world_size=world_size,
    )
    if marker.get("payload") != actual_payload:
        raise RuntimeError(
            "Checkpoint payload no longer matches its COMMITTED marker: "
            f"{checkpoint_root}"
        )

    metadata = _read_json_object(checkpoint_root / "meta.json")
    if metadata.get("iteration") != iteration or metadata.get("world_size") != world_size:
        raise RuntimeError(
            "Committed checkpoint metadata identity disagrees with its marker: "
            f"{checkpoint_root}"
        )
    model_dir = checkpoint_root / "model"
    if supplied not in {checkpoint_root, model_dir.resolve(strict=True)}:
        raise RuntimeError(
            "Input must be a committed checkpoint root or its model directory: "
            f"{supplied}"
        )
    return {
        "checkpoint_root": checkpoint_root,
        "model_dir": model_dir,
        "marker": marker,
        "marker_sha256": _sha256_file(marker_path),
    }


def _detect_model_dir(input_dir: str) -> str:
    return str(validate_committed_source(input_dir)["model_dir"])


def _load_fsdp_state_dict(input_dir: str) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=WrappedStorageReader(input_dir),
        planner=EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return state_dict


def load_committed_dcp_fp32_state_dict(
    input_dir: str | os.PathLike[str],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Authenticate and load a committed DCP model, rejecting non-FP32 state."""

    source = validate_committed_source(input_dir)
    loaded = _load_fsdp_state_dict(str(source["model_dir"]))
    non_tensors = sorted(
        name for name, value in loaded.items() if not isinstance(value, torch.Tensor)
    )
    if non_tensors:
        raise RuntimeError(
            "Committed DCP model state contains unsupported non-tensor values: "
            f"{non_tensors[:20]}"
        )
    tensor_items = {
        name: value for name, value in loaded.items() if isinstance(value, torch.Tensor)
    }
    _assert_fp32_floating_tensors(
        tensor_items,
        where="committed distributed FSDP checkpoint",
    )
    source_after_load = validate_committed_source(input_dir)
    if _source_identity(source) != _source_identity(source_after_load):
        raise RuntimeError(
            "Committed DCP source identity changed while it was being loaded"
        )
    return source, tensor_items


def inspect_committed_dcp_fp32(
    input_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Return authenticated FP32 DCP evidence without retaining loaded tensors."""

    source, state_dict = load_committed_dcp_fp32_state_dict(input_dir)
    floating = {
        name: tensor for name, tensor in state_dict.items() if tensor.is_floating_point()
    }
    return {
        "checkpoint_root": str(source["checkpoint_root"]),
        "iteration": source["marker"]["iteration"],
        "source_marker_sha256": source["marker_sha256"],
        "tensor_count": len(state_dict),
        "floating_tensor_count": len(floating),
        "floating_dtype": "float32",
    }


def _assert_fp32_floating_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    where: str,
) -> None:
    floating = [
        (name, tensor)
        for name, tensor in tensors.items()
        if tensor.is_floating_point()
    ]
    if not floating:
        raise RuntimeError(f"No floating tensors found at {where}")
    mismatches = [
        (name, tensor.dtype)
        for name, tensor in floating
        if tensor.dtype is not torch.float32
    ]
    if mismatches:
        examples = ", ".join(
            f"{name}={dtype}" for name, dtype in mismatches[:8]
        )
        raise RuntimeError(
            f"Expected every floating tensor to be FP32 at {where}; "
            f"found {len(mismatches)} mismatches: {examples}"
        )


def validate_safetensors_fp32(
    output_dir: str | os.PathLike[str],
) -> dict[str, object]:
    """Inspect safetensors headers and require FP32 for every floating tensor."""

    from safetensors import safe_open

    output_dir = Path(output_dir)
    files = sorted(output_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(
            f"No safetensors model shards were written under {output_dir}"
        )
    dtype_counts: dict[str, int] = {}
    tensor_count = 0
    tensor_shards: dict[str, str] = {}
    supported_nonfloating = {
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
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                dtype = str(handle.get_slice(key).get_dtype())
                if key in tensor_shards:
                    raise RuntimeError(
                        f"Safetensors key appears in multiple shards: {key}"
                    )
                tensor_shards[key] = path.name
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                tensor_count += 1
    unsupported = {
        dtype: count
        for dtype, count in dtype_counts.items()
        if dtype != "F32" and dtype not in supported_nonfloating
    }
    if unsupported:
        raise RuntimeError(
            "Exported safetensors contain non-FP32 floating weights: "
            f"{unsupported}"
        )
    if dtype_counts.get("F32", 0) == 0:
        raise RuntimeError("Exported safetensors contain no FP32 tensors")

    index_path = output_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = _read_json_object(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in weight_map.items()
        ):
            raise RuntimeError(f"Invalid safetensors weight map: {index_path}")
        if weight_map != tensor_shards:
            raise RuntimeError(
                f"Safetensors weight map does not match shard headers: {index_path}"
            )
        referenced_files = set(weight_map.values())
        actual_files = {path.name for path in files}
        if referenced_files != actual_files:
            raise RuntimeError(
                "Safetensors index shard inventory mismatch: "
                f"referenced={sorted(referenced_files)} actual={sorted(actual_files)}"
            )
    elif len(files) != 1 or files[0].name != "model.safetensors":
        raise RuntimeError(
            "Multiple or noncanonical safetensors shards require an authenticated index: "
            f"{[path.name for path in files]}"
        )
    return {
        "files": len(files),
        "tensors": tensor_count,
        "dtype_counts": dict(sorted(dtype_counts.items())),
    }


_assert_export_safetensors_fp32 = validate_safetensors_fp32


_APPROVED_ASSET_NAMES = frozenset(
    {
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer.py",
        "tokenizer_config.json",
        "vocab.json",
    }
)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise NotADirectoryError(root)
    directories = [root]
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Published HF exports may not contain symlinks: {path}")
        if path.is_file():
            _fsync_file(path)
        elif path.is_dir():
            directories.append(path)
    for directory in reversed(directories):
        _fsync_directory(directory)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale atomic JSON temporary file: {temporary}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


@contextlib.contextmanager
def _export_publication_lock(output: Path):
    """Serialize publication and recovery for one immutable export name."""

    lock_path = output.parent / f".{output.name}.export.lock"
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


def _quarantine_uncommitted_export(output: Path, *, reason: str) -> Path:
    """Preserve a marker-less final directory outside the committed namespace."""

    quarantine = output.parent / f".{output.name}.quarantine.{time.time_ns()}"
    os.replace(output, quarantine)
    reason_path = quarantine.with_name(f"{quarantine.name}.reason.txt")
    with reason_path.open("x", encoding="utf-8") as handle:
        handle.write(reason.rstrip() + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(output.parent)
    return quarantine


def _validate_saved_config_fp32(output_dir: Path) -> dict[str, Any]:
    config_path = output_dir / "config.json"
    config = _read_json_object(config_path)
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
            "Generated HF config does not advertise FP32 checkpoint tensors: "
            f"{dtype_fields}"
        )
    return config


def _source_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    marker = source["marker"]
    return {
        "iteration": marker["iteration"],
        "commit_sha256": marker["commit_sha256"],
        "marker_sha256": source["marker_sha256"],
        "payload_sha256": hashlib.sha256(
            json.dumps(
                marker["payload"],
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _write_export_commit_marker(
    output_dir: Path,
    *,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    marker_path = output_dir / EXPORT_COMMIT_MARKER
    if marker_path.exists():
        raise FileExistsError(f"HF export marker already exists: {marker_path}")
    precision = validate_safetensors_fp32(output_dir)
    _validate_saved_config_fp32(output_dir)

    # All generated files and copied assets become durable before the marker
    # can authenticate them.
    _fsync_tree(output_dir)
    core = {
        "schema": EXPORT_COMMIT_SCHEMA,
        "source_checkpoint": _source_identity(source),
        "precision": precision,
        "payload": _file_manifest(output_dir),
    }
    marker = {**core, "commit_sha256": _canonical_json_sha256(core)}
    _atomic_json(marker_path, marker)
    return marker


def validate_committed_hf_export(
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Validate the export marker, full file inventory, config, and tensor headers."""

    output = Path(output_dir).expanduser().resolve(strict=True)
    marker_path = output / EXPORT_COMMIT_MARKER
    marker = _read_json_object(marker_path)
    core = {key: value for key, value in marker.items() if key != "commit_sha256"}
    if marker.get("commit_sha256") != _canonical_json_sha256(core):
        raise RuntimeError(f"HF export commit marker hash mismatch: {marker_path}")
    if marker.get("schema") != EXPORT_COMMIT_SCHEMA:
        raise RuntimeError(
            f"Unsupported HF export commit schema at {marker_path}: {marker.get('schema')!r}"
        )
    actual_payload = _file_manifest(
        output,
        excluded_names=frozenset({EXPORT_COMMIT_MARKER}),
    )
    if marker.get("payload") != actual_payload:
        raise RuntimeError(
            f"HF export payload no longer matches its commit marker: {output}"
        )
    precision = validate_safetensors_fp32(output)
    if marker.get("precision") != precision:
        raise RuntimeError(f"HF export precision evidence drifted: {output}")
    _validate_saved_config_fp32(output)
    source = marker.get("source_checkpoint")
    expected_source_keys = {
        "iteration",
        "commit_sha256",
        "marker_sha256",
        "payload_sha256",
    }
    if not isinstance(source, dict) or set(source) != expected_source_keys:
        raise RuntimeError(f"HF export has invalid source identity: {marker_path}")
    iteration = source.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise RuntimeError(f"HF export has invalid source iteration: {marker_path}")
    for key in ("commit_sha256", "marker_sha256", "payload_sha256"):
        value = source.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(
                f"HF export has invalid source hash {key}: {marker_path}"
            )
    return marker


def _get_candidate_prefixes(keys: list[str]) -> list[str]:
    predefined = [
        "model_state.model.",
        "model_state.",
        "model.",
        "module.",
        "",
    ]

    detected: set[str] = set()
    for key in keys:
        for prefix in predefined:
            if prefix and key.startswith(prefix):
                detected.add(prefix)

    # Always keep empty string as a fall back option for exact match.
    detected.add("")
    # Preserve predefined order while keeping only detected prefixes.
    return [p for p in predefined if p in detected]


def _strip_best_prefix(keys: list[str], target_keys: set[str]) -> tuple[str, int]:
    best_prefix = ""
    best_match = -1

    for prefix in _get_candidate_prefixes(keys):
        mapped_keys = {k.removeprefix(prefix) for k in keys}
        match_count = len(mapped_keys & target_keys)
        if match_count > best_match:
            best_match = match_count
            best_prefix = prefix

    return best_prefix, best_match


def _convert_fsdp_to_hf(
    origin_hf_dir: str,
    input_dir: str,
    output_dir: str,
) -> dict[str, object]:
    print(f"loading FSDP model from {input_dir}")
    t = time.time()
    _source, tensor_items = load_committed_dcp_fp32_state_dict(input_dir)
    print(f"FSDP model loaded in {time.time()-t:.2f} sec.")

    config = AutoConfig.from_pretrained(origin_hf_dir, trust_remote_code=True)
    if hasattr(config, "dtype"):
        # Transformers 4.51 treats a model-specific ``dtype`` field as an
        # ordinary JSON value and does not stringify ``torch.dtype`` objects.
        # A canonical string works in both the 4.x and 5.x config APIs, while
        # the explicit model cast below remains the precision authority.
        config.dtype = "float32"
    else:
        config.torch_dtype = torch.float32
    hf_model = AutoModelForCausalLM.from_config(config).to(dtype=torch.float32)
    _assert_fp32_floating_tensors(
        hf_model.state_dict(),
        where="fresh Hugging Face destination model",
    )
    target_keys = set(hf_model.state_dict().keys())

    best_prefix, best_match = _strip_best_prefix(list(tensor_items.keys()), target_keys)
    total_keys = len(tensor_items)

    print(f"Using prefix '{best_prefix}' for key mapping. " f"Matched {best_match}/{total_keys} parameter keys.")

    model_state: dict[str, torch.Tensor] = {}
    source_keys: dict[str, str] = {}
    for source_key, tensor in tensor_items.items():
        target_key = source_key.removeprefix(best_prefix)
        if target_key in model_state:
            raise RuntimeError(
                "FSDP key-prefix mapping is ambiguous: "
                f"{source_keys[target_key]!r} and {source_key!r} both map to {target_key!r}"
            )
        model_state[target_key] = tensor
        source_keys[target_key] = source_key

    if not model_state:
        raise ValueError(
            "No model weights found in checkpoint. "
            "Please pass the checkpoint directory (e.g. iter_xxx or iter_xxx/model)."
        )

    missing = sorted(target_keys - set(model_state))
    unexpected = sorted(set(model_state) - target_keys)
    if missing or unexpected or best_match != len(target_keys):
        raise RuntimeError(
            "FSDP/Hugging Face state-dict keys disagree; conversion is fail-closed. "
            f"missing={missing[:20]} unexpected={unexpected[:20]} "
            f"matched={best_match} target={len(target_keys)}"
        )
    incompatible = hf_model.load_state_dict(model_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict FSDP/Hugging Face load unexpectedly returned incompatible keys: "
            f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
        )
    _assert_fp32_floating_tensors(
        hf_model.state_dict(),
        where="loaded Hugging Face model",
    )

    output = Path(output_dir)
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(
                f"HF conversion requires a fresh empty staging directory: {output}"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
    hf_model.save_pretrained(output, safe_serialization=True)
    export_precision = validate_safetensors_fp32(output)
    _validate_saved_config_fp32(output)
    print(f"Model weights saved to {output_dir}")
    return export_precision


def copy_assets(origin_hf_dir: str, output_dir: str) -> None:
    origin = Path(origin_hf_dir).expanduser().resolve(strict=True)
    output = Path(output_dir).expanduser().resolve(strict=True)
    for filename in sorted(os.listdir(origin)):
        if filename not in _APPROVED_ASSET_NAMES:
            continue
        origin_filename = origin / filename
        if origin_filename.is_symlink():
            raise RuntimeError(
                f"Approved HF assets must be regular files, not symlinks: {origin_filename}"
            )
        if not origin_filename.is_file():
            print(f"Skip {filename}, not a file.")
            continue
        destination = output / filename
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Approved asset would overwrite generated HF output: {destination}"
            )
        print(f"copy from {origin_filename} to {destination}")
        shutil.copy2(origin_filename, destination)


def convert_atomically(
    origin_hf_dir: str,
    input_dir: str,
    output_dir: str,
    *,
    force: bool = False,
) -> dict[str, object]:
    del force  # Retained for CLI compatibility; immutable exports are never replaced.
    source = validate_committed_source(input_dir)
    source_identity = _source_identity(source)
    output = Path(output_dir).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output = output.resolve(strict=False)
    with _export_publication_lock(output):
        if output.exists() or output.is_symlink():
            if output.is_symlink() or not output.is_dir():
                raise FileExistsError(
                    f"Refusing to replace immutable HF export: {output}"
                )
            if (output / EXPORT_COMMIT_MARKER).is_file():
                validate_committed_hf_export(output)
                raise FileExistsError(
                    f"Refusing to replace immutable HF export: {output}"
                )
            _quarantine_uncommitted_export(
                output,
                reason="HF export writer terminated before COMMITTED.json",
            )

        # Modal Volumes reject renameat2(RENAME_NOREPLACE) for directories.
        # Claim the final namespace directly and make it readable only by
        # writing the authenticated COMMITTED.json marker last.
        output.mkdir(parents=False, exist_ok=False)
        try:
            precision = _convert_fsdp_to_hf(
                origin_hf_dir,
                input_dir,
                str(output),
            )
            copy_assets(origin_hf_dir, str(output))
            source_after_conversion = validate_committed_source(input_dir)
            if _source_identity(source_after_conversion) != source_identity:
                raise RuntimeError(
                    "Committed DCP source identity changed during HF conversion"
                )
            marker = _write_export_commit_marker(
                output,
                source=source_after_conversion,
            )
            validate_committed_hf_export(output)
            _fsync_tree(output)
            _fsync_directory(output.parent)
            return {
                "precision": precision,
                "source_checkpoint": source_identity,
                "export_commit_sha256": marker["commit_sha256"],
            }
        except BaseException as exc:
            # A completed marker is immutable even if a later validation
            # detects corruption. Marker-less output is preserved outside the
            # committed namespace so a retry can safely claim the final name.
            if output.is_dir() and not (
                output / EXPORT_COMMIT_MARKER
            ).is_file():
                _quarantine_uncommitted_export(
                    output,
                    reason=f"HF export failed before commit: {type(exc).__name__}: {exc}",
                )
            raise


def validate_bf16_cuda_forward(
    output_dir: str | os.PathLike[str],
    *,
    input_ids: torch.Tensor | None = None,
    sequence_length: int = 4,
) -> dict[str, Any]:
    """Load the canonical FP32 export as BF16 on CUDA and run a finite forward."""

    output = Path(output_dir).expanduser().resolve(strict=True)
    export_marker = validate_committed_hf_export(output)
    if not torch.cuda.is_available():
        raise RuntimeError("BF16 export validation requires an available CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device does not support BF16 inference")
    if isinstance(sequence_length, bool) or sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length!r}")

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        output,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).to(device=device, dtype=torch.bfloat16)
    model.eval()
    floating_state = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if tensor.is_floating_point()
    }
    if not floating_state:
        raise RuntimeError("BF16 inference model contains no floating tensors")
    mismatches = {
        name: str(tensor.dtype)
        for name, tensor in floating_state.items()
        if tensor.dtype is not torch.bfloat16
    }
    if mismatches:
        raise RuntimeError(
            "Explicit BF16 inference load retained non-BF16 floating state: "
            f"{dict(list(mismatches.items())[:20])}"
        )

    if input_ids is None:
        vocab_size = int(getattr(model.config, "vocab_size", 0))
        if vocab_size <= 0:
            raise RuntimeError(f"Invalid model vocabulary size: {vocab_size}")
        input_ids = torch.arange(sequence_length, dtype=torch.long).remainder(vocab_size).unsqueeze(0)
    if input_ids.ndim != 2 or input_ids.numel() == 0:
        raise ValueError(
            f"BF16 inference input_ids must be a nonempty rank-2 tensor, got {tuple(input_ids.shape)}"
        )
    input_ids = input_ids.to(device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    logits = outputs.logits
    if logits.dtype is not torch.bfloat16:
        raise RuntimeError(
            f"BF16 inference forward produced {logits.dtype} logits instead of torch.bfloat16"
        )
    if not bool(torch.isfinite(logits).all().item()):
        raise FloatingPointError("BF16 inference forward produced non-finite logits")

    # Prove that the inference cast did not mutate the canonical on-disk FP32
    # artifact through any model-specific loading hook.
    validate_committed_hf_export(output)
    return {
        "export_commit_sha256": export_marker["commit_sha256"],
        "device": str(device),
        "in_memory_dtype": "bfloat16",
        "logits_dtype": "bfloat16",
        "input_shape": list(input_ids.shape),
        "logits_shape": list(logits.shape),
        "finite": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        required=True,
        help="The original Hugging Face model directory to load config/tokenizer assets.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Deprecated compatibility flag; committed output directories are never overwritten.",
    )
    args = parser.parse_args()

    convert_atomically(
        args.origin_hf_dir,
        args.input_dir,
        args.output_dir,
        force=args.force,
    )
