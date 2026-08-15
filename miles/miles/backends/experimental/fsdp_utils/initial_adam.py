from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)


CHECKPOINT_SCHEMA = "interleaved-accelerator-checkpoint-v1"
COMPLETION_MARKER = ".complete.json"
IMPORT_SCHEMA = "miles-initial-adam-import-v1"
MAPPING_RULE = "interleaved-hf-decay-then-bias-norm-v1"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return normalized


@dataclass(frozen=True)
class InitialAdamSpec:
    checkpoint: str
    completion_sha256: str
    source_tree_sha256: str
    step: int

    def identity(self) -> dict[str, Any]:
        return {
            "schema": IMPORT_SCHEMA,
            "checkpoint": self.checkpoint,
            "completion_sha256": self.completion_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "source_step": self.step,
            "mapping_rule": MAPPING_RULE,
        }


def initial_adam_spec_from_args(args: Any) -> InitialAdamSpec | None:
    raw = {
        "checkpoint": getattr(args, "initial_adam_checkpoint", None),
        "completion_sha256": getattr(args, "initial_adam_completion_sha256", None),
        "source_tree_sha256": getattr(args, "initial_adam_source_tree_sha256", None),
        "step": getattr(args, "initial_adam_step", None),
    }
    present = {
        key: value
        for key, value in raw.items()
        if value not in (None, "", 0)
    }
    if not present:
        return None
    if len(present) != len(raw):
        missing = sorted(set(raw) - set(present))
        raise ValueError(
            "initial Adam import requires checkpoint, completion SHA-256, "
            f"source-tree SHA-256, and step together; missing={missing}"
        )
    step = raw["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("initial Adam source step must be a positive integer")
    return InitialAdamSpec(
        checkpoint=str(raw["checkpoint"]),
        completion_sha256=_require_sha256(
            str(raw["completion_sha256"]),
            name="initial Adam completion SHA-256",
        ),
        source_tree_sha256=_require_sha256(
            str(raw["source_tree_sha256"]),
            name="initial Adam source-tree SHA-256",
        ),
        step=step,
    )


def _validate_checkpoint(spec: InitialAdamSpec, hf_checkpoint: Path) -> tuple[Path, dict[str, Any]]:
    checkpoint = Path(spec.checkpoint).expanduser().resolve(strict=True)
    marker_path = checkpoint / COMPLETION_MARKER
    if not marker_path.is_file() or marker_path.is_symlink():
        raise RuntimeError(
            f"initial Adam checkpoint is not committed: {marker_path}"
        )
    actual_marker_sha256 = _sha256_file(marker_path)
    if actual_marker_sha256 != spec.completion_sha256:
        raise RuntimeError(
            "initial Adam completion marker SHA-256 drifted: "
            f"expected={spec.completion_sha256} actual={actual_marker_sha256}"
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(marker, dict):
        raise RuntimeError("initial Adam completion marker must be an object")
    recorded_marker_sha256 = marker.pop("marker_sha256", None)
    if recorded_marker_sha256 != _canonical_sha256(marker):
        raise RuntimeError("initial Adam completion marker self-hash drifted")
    marker["marker_sha256"] = recorded_marker_sha256
    if marker.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError(
            f"unsupported initial Adam checkpoint schema: {marker.get('schema')!r}"
        )
    if marker.get("global_step") != spec.step:
        raise RuntimeError(
            "initial Adam checkpoint step drifted: "
            f"expected={spec.step} actual={marker.get('global_step')}"
        )

    identity = marker.get("checkpoint_identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("initial Adam checkpoint lacks a file identity")
    recorded_files = identity.get("files")
    if not isinstance(recorded_files, list) or not recorded_files:
        raise RuntimeError("initial Adam checkpoint file inventory is invalid")
    observed_files: list[dict[str, Any]] = []
    for row in recorded_files:
        if not isinstance(row, Mapping):
            raise RuntimeError("initial Adam checkpoint file row is invalid")
        relative = row.get("path")
        if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
            raise RuntimeError("initial Adam checkpoint file path is unsafe")
        path = checkpoint / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"initial Adam checkpoint payload is missing: {path}")
        observed_files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    expected_identity = {
        "file_count": len(observed_files),
        "files": observed_files,
        "manifest_sha256": _canonical_sha256(observed_files),
        "total_bytes": sum(int(row["bytes"]) for row in observed_files),
    }
    if dict(identity) != expected_identity:
        raise RuntimeError("initial Adam checkpoint file identity drifted")

    trainer_state_path = checkpoint / "trainer_state.json"
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    configured = trainer_state.get("configured_provenance")
    if not isinstance(configured, Mapping):
        raise RuntimeError("initial Adam trainer state lacks configured provenance")
    if configured.get("source_tree_sha256") != spec.source_tree_sha256:
        raise RuntimeError(
            "initial Adam source trainer tree drifted: "
            f"expected={spec.source_tree_sha256} "
            f"actual={configured.get('source_tree_sha256')}"
        )
    if trainer_state.get("global_step") != spec.step:
        raise RuntimeError("initial Adam trainer-state step drifted")
    if trainer_state.get("optimizer_resets_completed") not in ([], None):
        raise RuntimeError(
            "initial Adam checkpoint was produced after an optimizer-state reset"
        )
    precision = trainer_state.get("precision_contract")
    if not isinstance(precision, Mapping) or (
        precision.get("master_parameter_dtype") != "float32"
        or precision.get("optimizer_state_dtype") != "float32"
    ):
        raise RuntimeError("initial Adam source lacks FP32 model/optimizer evidence")

    persisted = marker.get("persisted_precision")
    if not isinstance(persisted, Mapping):
        raise RuntimeError("initial Adam marker lacks persisted precision evidence")
    model_rows = persisted.get("model", {}).get("shards") if isinstance(persisted.get("model"), Mapping) else None
    optimizer_rows = persisted.get("optimizer_files")
    if not isinstance(model_rows, list) or len(model_rows) != 1:
        raise RuntimeError("initial Adam import requires one authenticated model.safetensors")
    if not isinstance(optimizer_rows, list) or len(optimizer_rows) != 1:
        raise RuntimeError("initial Adam import requires one authenticated optimizer.bin")
    model_row = dict(model_rows[0])
    optimizer_row = dict(optimizer_rows[0])
    if model_row.get("path") != "model.safetensors" or optimizer_row.get("path") != "optimizer.bin":
        raise RuntimeError("initial Adam checkpoint payload names drifted")

    hf_model_path = hf_checkpoint / "model.safetensors"
    if not hf_model_path.is_file() or hf_model_path.is_symlink():
        raise RuntimeError(
            "initial Adam import requires the RL HF checkpoint to contain one model.safetensors"
        )
    hf_model_sha256 = _sha256_file(hf_model_path)
    if hf_model_sha256 != model_row.get("sha256"):
        raise RuntimeError(
            "RL model weights are not byte-identical to the optimizer source: "
            f"hf={hf_model_sha256} source={model_row.get('sha256')}"
        )
    return checkpoint, marker


def _source_parameter_order(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    decay: list[tuple[str, torch.nn.Parameter]] = []
    no_decay: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        normalized = name.lower()
        if (
            name.endswith(".bias")
            or "norm" in normalized
            or "ln" in normalized
            or "layernorm" in normalized
        ):
            no_decay.append((name, parameter))
        else:
            decay.append((name, parameter))
    ordered = [*decay, *no_decay]
    if not ordered:
        raise RuntimeError("initial Adam destination model has no trainable parameters")
    return ordered


def prepare_initial_adam_state(
    model: torch.nn.Module,
    *,
    args: Any,
) -> dict[str, Any] | None:
    """Authenticate an Accelerator AdamW checkpoint and map IDs to model names."""

    spec = initial_adam_spec_from_args(args)
    if spec is None:
        return None
    checkpoint, marker = _validate_checkpoint(
        spec,
        Path(args.hf_checkpoint).expanduser().resolve(strict=True),
    )
    optimizer_path = checkpoint / "optimizer.bin"
    optimizer_payload = torch.load(
        optimizer_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(optimizer_payload, Mapping):
        raise RuntimeError("initial Adam optimizer payload must be a mapping")
    source_state = optimizer_payload.get("state")
    source_groups = optimizer_payload.get("param_groups")
    if not isinstance(source_state, Mapping) or not isinstance(source_groups, list):
        raise RuntimeError("initial Adam optimizer payload is incomplete")
    if len(source_groups) != 2:
        raise RuntimeError("initial Adam source must contain decay and bias/norm groups")

    ordered_parameters = _source_parameter_order(model)
    decay_count = sum(
        1
        for name, _ in ordered_parameters
        if not (
            name.endswith(".bias")
            or "norm" in name.lower()
            or "ln" in name.lower()
            or "layernorm" in name.lower()
        )
    )
    group_ids: list[int] = []
    group_sizes: list[int] = []
    for group_index, group in enumerate(source_groups):
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise RuntimeError(f"initial Adam parameter group {group_index} is invalid")
        params = list(group["params"])
        if any(isinstance(item, bool) or not isinstance(item, int) for item in params):
            raise RuntimeError("initial Adam parameter IDs must be integers")
        group_ids.extend(params)
        group_sizes.append(len(params))
    if group_sizes != [decay_count, len(ordered_parameters) - decay_count]:
        raise RuntimeError(
            "initial Adam parameter groups do not match the authenticated trainer mapping rule"
        )
    if len(set(group_ids)) != len(group_ids) or set(group_ids) != set(source_state):
        raise RuntimeError("initial Adam state does not cover each parameter exactly once")
    if len(group_ids) != len(ordered_parameters):
        raise RuntimeError("initial Adam parameter count differs from the RL model")

    named_state: dict[str, dict[str, torch.Tensor]] = {}
    mapping_rows: list[dict[str, Any]] = []
    for parameter_id, (name, parameter) in zip(group_ids, ordered_parameters, strict=True):
        raw_state = source_state[parameter_id]
        if not isinstance(raw_state, Mapping):
            raise RuntimeError(f"initial Adam state for {name} is not a mapping")
        state: dict[str, torch.Tensor] = {}
        for field in ("step", "exp_avg", "exp_avg_sq"):
            value = raw_state.get(field)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"initial Adam state for {name} lacks {field}")
            if value.dtype is not torch.float32:
                raise RuntimeError(
                    f"initial Adam state for {name}.{field} must be FP32, got {value.dtype}"
                )
            state[field] = value.detach().cpu().clone()
        if tuple(state["exp_avg"].shape) != tuple(parameter.shape) or tuple(
            state["exp_avg_sq"].shape
        ) != tuple(parameter.shape):
            raise RuntimeError(f"initial Adam moment shape differs for {name}")
        if state["step"].numel() != 1 or not math.isfinite(float(state["step"].item())):
            raise RuntimeError(f"initial Adam step is invalid for {name}")
        if int(state["step"].item()) != spec.step:
            raise RuntimeError(
                f"initial Adam step differs for {name}: {state['step'].item()} != {spec.step}"
            )
        named_state[name] = state
        mapping_rows.append(
            {
                "parameter_id": parameter_id,
                "name": name,
                "shape": list(parameter.shape),
            }
        )

    source_hyperparameters = []
    for group in source_groups:
        source_hyperparameters.append(
            {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in group.items()
                if key != "params"
            }
        )
    evidence_core = {
        **spec.identity(),
        "completion_marker_self_sha256": marker["marker_sha256"],
        "checkpoint_manifest_sha256": marker["checkpoint_identity"]["manifest_sha256"],
        "model_sha256": marker["persisted_precision"]["model"]["shards"][0]["sha256"],
        "optimizer_sha256": marker["persisted_precision"]["optimizer_files"][0]["sha256"],
        "parameter_count": len(named_state),
        "adam_moment_tensor_count": 2 * len(named_state),
        "mapping_sha256": _canonical_sha256(mapping_rows),
        "source_param_groups": source_hyperparameters,
        "source_state_verified": True,
    }
    evidence = {
        **evidence_core,
        "evidence_sha256": _canonical_sha256(evidence_core),
    }
    return {"state": named_state, "evidence": evidence}


def _compare_imported_state(
    imported: Mapping[str, Any],
    expected: Mapping[str, Mapping[str, torch.Tensor]],
) -> None:
    observed_state = imported.get("state")
    if not isinstance(observed_state, Mapping) or set(observed_state) != set(expected):
        raise RuntimeError("installed Adam parameter-name inventory drifted")
    for name, expected_state in expected.items():
        observed = observed_state[name]
        if not isinstance(observed, Mapping):
            raise RuntimeError(f"installed Adam state for {name} is invalid")
        for field, expected_tensor in expected_state.items():
            value = observed.get(field)
            if not isinstance(value, torch.Tensor) or not torch.equal(
                value.detach().cpu(), expected_tensor
            ):
                raise RuntimeError(f"installed Adam state differs at {name}.{field}")


def install_initial_adam_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    prepared: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Install a rank-0 full named state and let DCP reshard it for FSDP2."""

    if prepared is None and (not dist.is_initialized() or dist.get_rank() == 0):
        return None
    rank = dist.get_rank() if dist.is_initialized() else 0
    distributed = dist.is_initialized()
    options = StateDictOptions(
        full_state_dict=distributed,
        cpu_offload=distributed,
        broadcast_from_rank0=distributed,
        strict=True,
    )
    destination = get_optimizer_state_dict(model, optimizer, options=options)
    if rank == 0:
        assert prepared is not None
        destination_names = [
            name
            for group in destination["param_groups"]
            for name in group["params"]
        ]
        if len(destination["param_groups"]) != 1:
            raise RuntimeError("Miles RL destination AdamW must have exactly one parameter group")
        if len(destination_names) != len(set(destination_names)) or set(destination_names) != set(
            prepared["state"]
        ):
            raise RuntimeError("Miles RL optimizer parameters differ from the imported Adam state")
        destination["state"] = prepared["state"]
    set_optimizer_state_dict(model, optimizer, destination, options=options)

    round_trip = get_optimizer_state_dict(model, optimizer, options=options)
    result: list[Any] = [None, None]
    if rank == 0:
        try:
            assert prepared is not None
            _compare_imported_state(round_trip, prepared["state"])
            destination_group = {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in dict(round_trip["param_groups"][0]).items()
            }
            destination_group.pop("params", None)
            core = {
                **prepared["evidence"],
                "destination_param_group": destination_group,
                "round_trip_full_state_verified": True,
            }
            core.pop("evidence_sha256", None)
            result[0] = {
                **core,
                "evidence_sha256": _canonical_sha256(core),
            }
        except Exception as exc:  # broadcast a deterministic failure to every rank
            result[1] = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        dist.broadcast_object_list(result, src=0)
    if result[1] is not None:
        raise RuntimeError(f"initial Adam import verification failed: {result[1]}")
    return result[0]


def validate_initial_adam_resume_evidence(
    args: Any,
    evidence: Any,
) -> dict[str, Any] | None:
    spec = initial_adam_spec_from_args(args)
    if spec is None:
        if evidence is not None:
            raise RuntimeError(
                "checkpoint contains initial Adam lineage but the launch omitted it"
            )
        return None
    if not isinstance(evidence, dict):
        raise RuntimeError("checkpoint lacks its required initial Adam import evidence")
    core = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    if evidence.get("evidence_sha256") != _canonical_sha256(core):
        raise RuntimeError("checkpoint initial Adam import evidence self-hash drifted")
    expected = spec.identity()
    mismatches = {
        key: {"expected": value, "actual": evidence.get(key)}
        for key, value in expected.items()
        if evidence.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"checkpoint initial Adam lineage drifted: {mismatches}")
    if evidence.get("round_trip_full_state_verified") is not True:
        raise RuntimeError("checkpoint initial Adam state was not round-trip verified")
    return evidence


def assert_initial_adam_step_progression(
    optimizer: torch.optim.Optimizer,
    evidence: dict[str, Any] | None,
    *,
    rl_global_step: int,
) -> dict[str, Any] | None:
    if evidence is None:
        return None
    expected = int(evidence["source_step"]) + int(rl_global_step)
    observed: list[int] = []
    for state in optimizer.state.values():
        step = state.get("step")
        if not isinstance(step, torch.Tensor) or step.numel() != 1:
            raise RuntimeError("imported Adam state contains an invalid step tensor")
        value = float(step.item())
        if not math.isfinite(value) or int(value) != value:
            raise RuntimeError("imported Adam state contains a non-integral step")
        observed.append(int(value))
    if len(observed) != int(evidence["parameter_count"]):
        raise RuntimeError(
            "imported Adam step inventory differs from the authenticated parameter count"
        )
    if set(observed) != {expected}:
        raise RuntimeError(
            "imported Adam steps are not continuous with RL updates: "
            f"expected={expected} observed={sorted(set(observed))}"
        )
    return {
        "source_step": int(evidence["source_step"]),
        "rl_global_step": int(rl_global_step),
        "expected_adam_step": expected,
        "parameter_count": len(observed),
        "all_parameter_steps_verified": True,
    }
