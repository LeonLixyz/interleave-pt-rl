from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

from tools import convert_fsdp_to_hf as converter


def _tiny_config(*, dtype: torch.dtype = torch.bfloat16) -> GPT2Config:
    config = GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=8,
        n_positions=8,
        n_ctx=8,
        vocab_size=16,
        tie_word_embeddings=False,
    )
    if hasattr(config, "dtype"):
        config.dtype = dtype
    else:
        config.torch_dtype = dtype
    return config


def _write_origin(path: Path) -> None:
    path.mkdir()
    _tiny_config().save_pretrained(path)
    (path / "tokenizer_config.json").write_text(
        '{"model_max_length": 8}\n',
        encoding="utf-8",
    )
    (path / "tokenizer.py").write_text(
        "# authenticated custom tokenizer implementation\n",
        encoding="utf-8",
    )
    # This must not replace the config generated from the loaded model.
    (path / "generation_config.json").write_text(
        '{"source_only": true}\n',
        encoding="utf-8",
    )


def _write_committed_source(
    path: Path,
    *,
    dtype: torch.dtype = torch.float32,
    drop_key: str | None = None,
) -> tuple[Path, dict[str, torch.Tensor]]:
    checkpoint = path / "iter_0000001"
    model_dir = checkpoint / "model"
    model_dir.mkdir(parents=True)
    model = GPT2LMHeadModel(_tiny_config(dtype=torch.float32)).to(dtype=dtype)
    state = model.state_dict()
    if drop_key is not None:
        state.pop(drop_key)
    dcp.save(
        {"model_state": {"model": state}},
        checkpoint_id=str(model_dir),
    )
    (checkpoint / "meta.json").write_text(
        json.dumps({"iteration": 1, "world_size": 1}) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "schema": "miles-global-rollout-state-v1",
            "next_global_rollout_id": 1,
            "global_dataset_cursor": 256,
        },
        checkpoint / "rollout_state.pt",
    )
    core = {
        "schema": converter.SOURCE_COMMIT_SCHEMA,
        "iteration": 1,
        "optimizer_included": False,
        "rng_included": False,
        "rollout_state_included": True,
        "world_size": 1,
        "payload": converter._expected_source_payload(
            checkpoint,
            optimizer_included=False,
            rng_included=False,
            rollout_state_included=True,
            world_size=1,
        ),
    }
    marker = {
        **core,
        "commit_sha256": converter._canonical_json_sha256(core),
    }
    (checkpoint / converter.SOURCE_COMMIT_MARKER).write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checkpoint, state


def test_source_commit_marker_authenticates_every_dcp_payload_byte(tmp_path):
    source, _state = _write_committed_source(tmp_path)
    validated = converter.validate_committed_source(source)
    assert validated["checkpoint_root"] == source.resolve()
    assert validated["model_dir"] == source.resolve() / "model"

    shard = next((source / "model").glob("*.distcp"))
    payload = bytearray(shard.read_bytes())
    payload[-1] ^= 1
    shard.write_bytes(payload)
    with pytest.raises(RuntimeError, match="payload no longer matches"):
        converter.validate_committed_source(source)


def test_source_without_commit_marker_is_rejected(tmp_path):
    source = tmp_path / "iter_0000001"
    (source / "model").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Missing or invalid authenticated JSON"):
        converter.validate_committed_source(source)


def test_source_rejects_payload_not_bound_by_commit_marker(tmp_path):
    source, _state = _write_committed_source(tmp_path)
    (source / "uncommitted.bin").write_bytes(b"not in marker")
    with pytest.raises(RuntimeError, match="unauthenticated file"):
        converter.validate_committed_source(source)


def test_atomic_conversion_upcasts_bf16_origin_and_publishes_authenticated_fp32(
    tmp_path,
):
    origin = tmp_path / "origin"
    _write_origin(origin)
    source, expected_state = _write_committed_source(tmp_path / "source")
    output = tmp_path / "export"

    evidence = converter.convert_atomically(
        str(origin),
        str(source),
        str(output),
        force=True,
    )

    marker = converter.validate_committed_hf_export(output)
    assert marker["commit_sha256"] == evidence["export_commit_sha256"]
    assert marker["source_checkpoint"]["iteration"] == 1
    assert marker["precision"]["dtype_counts"]["F32"] > 0
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config.get("dtype", config.get("torch_dtype")) == "float32"
    assert (output / "tokenizer_config.json").is_file()
    assert (output / "tokenizer.py").read_text(encoding="utf-8") == (
        "# authenticated custom tokenizer implementation\n"
    )
    generation_config = json.loads(
        (output / "generation_config.json").read_text(encoding="utf-8")
    )
    assert "source_only" not in generation_config

    restored = AutoModelForCausalLM.from_pretrained(
        output,
        local_files_only=True,
        dtype=torch.float32,
    )
    assert all(
        tensor.dtype is torch.float32
        for tensor in restored.state_dict().values()
        if tensor.is_floating_point()
    )
    assert restored.state_dict().keys() == expected_state.keys()
    assert all(
        torch.equal(restored.state_dict()[key], expected_state[key])
        for key in expected_state
    )


def test_model_specific_dtype_field_is_json_serializable_after_conversion(monkeypatch, tmp_path):
    origin = tmp_path / "origin"
    _write_origin(origin)
    source, _state = _write_committed_source(tmp_path / "source")
    original_loader = converter.AutoConfig.from_pretrained

    def load_with_legacy_dtype(*args, **kwargs):
        config = original_loader(*args, **kwargs)
        config.dtype = torch.bfloat16
        return config

    monkeypatch.setattr(converter.AutoConfig, "from_pretrained", load_with_legacy_dtype)
    output = tmp_path / "export"
    converter.convert_atomically(str(origin), str(source), str(output))

    saved = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert saved["dtype"] == "float32"


def test_conversion_rejects_missing_model_key_and_keeps_final_absent(tmp_path):
    origin = tmp_path / "origin"
    _write_origin(origin)
    reference = GPT2LMHeadModel(_tiny_config(dtype=torch.float32)).state_dict()
    source, _state = _write_committed_source(
        tmp_path / "source",
        drop_key=next(iter(reference)),
    )
    output = tmp_path / "export"

    with pytest.raises(RuntimeError, match="state-dict keys disagree"):
        converter.convert_atomically(str(origin), str(source), str(output))
    assert not output.exists()
    quarantine = list(tmp_path.glob(".export.quarantine.*"))
    assert len([path for path in quarantine if path.is_dir()]) == 1
    assert len(
        [path for path in quarantine if path.name.endswith(".reason.txt")]
    ) == 1


def test_conversion_rejects_non_fp32_committed_dcp(tmp_path):
    origin = tmp_path / "origin"
    _write_origin(origin)
    source, _state = _write_committed_source(
        tmp_path / "source",
        dtype=torch.bfloat16,
    )
    with pytest.raises(RuntimeError, match="every floating tensor to be FP32"):
        converter.convert_atomically(
            str(origin),
            str(source),
            str(tmp_path / "export"),
        )


def test_uncommitted_final_is_quarantined_before_publication(tmp_path):
    origin = tmp_path / "origin"
    _write_origin(origin)
    source, _state = _write_committed_source(tmp_path / "source")
    output = tmp_path / "export"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")

    converter.convert_atomically(
        str(origin),
        str(source),
        str(output),
        force=True,
    )
    converter.validate_committed_hf_export(output)
    quarantined = [
        path
        for path in tmp_path.glob(".export.quarantine.*")
        if path.is_dir()
    ]
    assert len(quarantined) == 1
    assert (quarantined[0] / "sentinel").read_text(encoding="utf-8") == "keep"


def test_committed_final_is_never_replaced_even_with_force(tmp_path):
    origin = tmp_path / "origin"
    _write_origin(origin)
    source, _state = _write_committed_source(tmp_path / "source")
    output = tmp_path / "export"
    first = converter.convert_atomically(str(origin), str(source), str(output))

    with pytest.raises(FileExistsError, match="immutable HF export"):
        converter.convert_atomically(
            str(origin),
            str(source),
            str(output),
            force=True,
        )
    marker = converter.validate_committed_hf_export(output)
    assert marker["commit_sha256"] == first["export_commit_sha256"]


def test_safetensors_header_validation_rejects_bf16(tmp_path):
    save_file(
        {"weight": torch.ones(2, dtype=torch.bfloat16)},
        tmp_path / "model.safetensors",
    )
    with pytest.raises(RuntimeError, match="non-FP32 floating weights"):
        converter.validate_safetensors_fp32(tmp_path)


def test_export_marker_detects_same_size_payload_corruption(tmp_path):
    origin = tmp_path / "origin"
    _write_origin(origin)
    source, _state = _write_committed_source(tmp_path / "source")
    output = tmp_path / "export"
    converter.convert_atomically(str(origin), str(source), str(output))

    config_path = output / "config.json"
    payload = config_path.read_bytes()
    replacement = b" " if payload[-2:-1] != b" " else b"\t"
    config_path.write_bytes(payload[:-2] + replacement + payload[-1:])
    with pytest.raises(RuntimeError, match="payload no longer matches"):
        converter.validate_committed_hf_export(output)


def test_bf16_forward_helper_fails_closed_without_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(converter, "validate_committed_hf_export", lambda _path: {"commit_sha256": "abc"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires an available CUDA"):
        converter.validate_bf16_cuda_forward(tmp_path)
