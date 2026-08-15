from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from safetensors.torch import save_file

from miles.backends.experimental.fsdp_utils.initial_adam import (
    assert_initial_adam_step_progression,
    install_initial_adam_state,
    prepare_initial_adam_state,
    validate_initial_adam_resume_evidence,
)


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model() -> torch.nn.Module:
    torch.manual_seed(7)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = torch.nn.Linear(3, 4)
            self.norm = torch.nn.LayerNorm(4)
            self.output = torch.nn.Linear(4, 2)

        def forward(self, value):
            return self.output(self.norm(self.input(value)))

    return TinyModel().float()


def _source_groups(model: torch.nn.Module) -> list[dict]:
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _fixture(tmp_path: Path, *, source_step: int = 7):
    source_model = _model()
    source_optimizer = torch.optim.AdamW(
        _source_groups(source_model),
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    source_model(torch.ones(2, 3)).sum().backward()
    source_optimizer.step()
    for state in source_optimizer.state.values():
        state["step"].fill_(source_step)

    checkpoint = tmp_path / "resume" / f"step_{source_step:08d}"
    checkpoint.mkdir(parents=True)
    save_file(source_model.state_dict(), checkpoint / "model.safetensors")
    torch.save(source_optimizer.state_dict(), checkpoint / "optimizer.bin")
    source_tree_sha256 = "a" * 64
    trainer_state = {
        "global_step": source_step,
        "optimizer_resets_completed": [],
        "precision_contract": {
            "master_parameter_dtype": "float32",
            "optimizer_state_dtype": "float32",
        },
        "configured_provenance": {
            "source_tree_sha256": source_tree_sha256,
        },
    }
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(trainer_state, sort_keys=True)
    )
    files = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in sorted(checkpoint.iterdir())
        if path.is_file()
    ]
    identity = {
        "file_count": len(files),
        "files": files,
        "manifest_sha256": hashlib.sha256(_canonical(files)).hexdigest(),
        "total_bytes": sum(row["bytes"] for row in files),
    }
    model_row = next(row for row in files if row["path"] == "model.safetensors")
    optimizer_row = next(row for row in files if row["path"] == "optimizer.bin")
    marker_core = {
        "schema": "interleaved-accelerator-checkpoint-v1",
        "global_step": source_step,
        "trainer_state_sha256": _sha(checkpoint / "trainer_state.json"),
        "checkpoint_identity": identity,
        "persisted_precision": {
            "model": {"shards": [model_row]},
            "optimizer_files": [optimizer_row],
        },
    }
    marker = {
        **marker_core,
        "marker_sha256": hashlib.sha256(_canonical(marker_core)).hexdigest(),
    }
    marker_path = checkpoint / ".complete.json"
    marker_path.write_text(json.dumps(marker, sort_keys=True))

    hf = tmp_path / "hf"
    hf.mkdir()
    save_file(source_model.state_dict(), hf / "model.safetensors")
    args = Namespace(
        hf_checkpoint=str(hf),
        initial_adam_checkpoint=str(checkpoint),
        initial_adam_completion_sha256=_sha(marker_path),
        initial_adam_source_tree_sha256=source_tree_sha256,
        initial_adam_step=source_step,
    )
    return source_model, source_optimizer, args


def test_imports_every_adam_tensor_and_preserves_rl_hyperparameters(tmp_path):
    source_model, source_optimizer, args = _fixture(tmp_path)
    prepared = prepare_initial_adam_state(source_model, args=args)
    destination = torch.optim.AdamW(
        source_model.parameters(),
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    evidence = install_initial_adam_state(source_model, destination, prepared)

    assert evidence is not None
    assert evidence["parameter_count"] == len(list(source_model.parameters()))
    assert evidence["round_trip_full_state_verified"] is True
    assert evidence["destination_param_group"]["lr"] == 1e-5
    assert evidence["destination_param_group"]["betas"] == [0.9, 0.999]
    assert evidence["destination_param_group"]["weight_decay"] == 0.01
    assert_initial_adam_step_progression(destination, evidence, rl_global_step=0)

    imported_state = get_optimizer_state_dict(source_model, destination)["state"]
    for name, source in prepared["state"].items():
        imported = imported_state[name]
        torch.testing.assert_close(source["exp_avg"], imported["exp_avg"], rtol=0, atol=0)
        torch.testing.assert_close(source["exp_avg_sq"], imported["exp_avg_sq"], rtol=0, atol=0)
        torch.testing.assert_close(source["step"], imported["step"], rtol=0, atol=0)

    destination.zero_grad(set_to_none=True)
    source_model(torch.ones(2, 3)).sum().backward()
    destination.step()
    assert_initial_adam_step_progression(destination, evidence, rl_global_step=1)
    assert validate_initial_adam_resume_evidence(args, evidence) == evidence


def test_import_rejects_marker_or_model_drift(tmp_path):
    model, _optimizer, args = _fixture(tmp_path)
    args.initial_adam_completion_sha256 = "0" * 64
    with pytest.raises(RuntimeError, match="completion marker SHA-256 drifted"):
        prepare_initial_adam_state(model, args=args)


def test_import_requires_complete_identity(tmp_path):
    model, _optimizer, args = _fixture(tmp_path)
    args.initial_adam_source_tree_sha256 = None
    with pytest.raises(ValueError, match="requires checkpoint"):
        prepare_initial_adam_state(model, args=args)
