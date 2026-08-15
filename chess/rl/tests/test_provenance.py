from __future__ import annotations

import json

import pytest

from chess_rl_miles.provenance import (
    directory_identity,
    source_tree_identity,
    write_run_provenance,
)


def test_source_tree_manifest_is_deterministic_and_ignores_caches(tmp_path):
    source = tmp_path / "source"
    (source / "pkg").mkdir(parents=True)
    (source / "pkg" / "a.py").write_text("A = 1\n")
    (source / "config.yaml").write_text("value: 2\n")
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "a.pyc").write_bytes(b"ignored")

    first = source_tree_identity(source)
    (source / ".pytest_cache").mkdir()
    (source / ".pytest_cache" / "state.json").write_text("{}")
    second = source_tree_identity(source)

    assert first == second
    assert first["file_count"] == 2
    (source / "pkg" / "a.py").write_text("A = 3\n")
    assert source_tree_identity(source)["manifest_sha256"] != first[
        "manifest_sha256"
    ]


def test_source_tree_hashes_all_mounted_assets_and_excludes_secrets(tmp_path):
    source = tmp_path / "source"
    (source / "pkg").mkdir(parents=True)
    (source / "pkg" / "kernel.cu").write_text("runtime kernel")
    (source / "pkg" / "template.jinja").write_text("runtime template")
    (source / ".env").write_text("SECRET=must-not-be-mounted")
    (source / "pkg" / "cached.pyc").write_bytes(b"executable-cache")

    first = source_tree_identity(source)
    assert first["file_count"] == 2
    (source / ".env").write_text("SECRET=changed")
    (source / "pkg" / "cached.pyc").write_bytes(b"changed-cache")
    assert source_tree_identity(source) == first
    (source / "pkg" / "template.jinja").write_text("changed runtime template")
    assert source_tree_identity(source)["manifest_sha256"] != first[
        "manifest_sha256"
    ]


def test_source_tree_excludes_launch_ledgers_and_recovery_tokens(tmp_path):
    source = tmp_path / "source"
    (source / "pkg").mkdir(parents=True)
    (source / "pkg" / "code.py").write_text("VALUE = 1\n")
    for directory in (
        ".modal-launch-records",
        ".modal-launch-recovery",
        ".launch-recovery",
    ):
        path = source / directory
        path.mkdir()
        (path / "record.json").write_text("secret launch material\n")

    identity = source_tree_identity(source)
    assert identity["file_count"] == 1
    for directory in (
        ".modal-launch-records",
        ".modal-launch-recovery",
        ".launch-recovery",
    ):
        (source / directory / "record.json").write_text("changed\n")
    assert source_tree_identity(source) == identity


def test_source_tree_manifest_supports_one_explicit_binding_exclusion(
    tmp_path,
):
    source = tmp_path / "source"
    (source / "pkg").mkdir(parents=True)
    (source / "pkg" / "code.py").write_text("VALUE = 1\n")
    binding = source / "pkg" / "contract_binding.py"
    binding.write_text('EXPECTED = "first"\n')

    first = source_tree_identity(
        source,
        excluded_relatives=("pkg/contract_binding.py",),
    )
    binding.write_text('EXPECTED = "second"\n')
    second = source_tree_identity(
        source,
        excluded_relatives=("pkg/contract_binding.py",),
    )

    assert first == second
    assert first["file_count"] == 1
    assert first["excluded_relatives"] == ["pkg/contract_binding.py"]

    with pytest.raises(ValueError, match="safe relative"):
        source_tree_identity(source, excluded_relatives=("../outside.py",))


def test_checkpoint_identity_hashes_exact_files(tmp_path):
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"qwen3"}')
    (checkpoint / "model.safetensors").write_bytes(b"weights")

    identity = directory_identity(checkpoint, logical_path="/models/origin")

    assert identity["logical_path"] == "/models/origin"
    assert identity["file_count"] == 2
    assert [item["path"] for item in identity["files"]] == [
        "config.json",
        "model.safetensors",
    ]


def test_run_provenance_is_idempotent_and_commands_are_append_only(tmp_path):
    run_root = tmp_path / "run"
    identity = {"run": {"name": "e1"}, "source": {"sha256": "abc"}}
    first = write_run_provenance(
        run_root=run_root,
        identity=identity,
        command=["train", "--step", "0"],
    )
    repeated = write_run_provenance(
        run_root=run_root,
        identity=identity,
        command=["train", "--step", "0"],
    )
    resumed = write_run_provenance(
        run_root=run_root,
        identity=identity,
        command=["train", "--load", "iter_40"],
    )

    assert first["identity_sha256"] == repeated["identity_sha256"]
    assert first["launch_manifest"] == repeated["launch_manifest"]
    assert resumed["launch_manifest"] != first["launch_manifest"]
    root = json.loads((run_root / "run_provenance.json").read_text())
    assert root["initial_command"] == ["train", "--step", "0"]
    assert len(list((run_root / "provenance").glob("launch_*.json"))) == 2

    with pytest.raises(RuntimeError, match="provenance mismatch"):
        write_run_provenance(
            run_root=run_root,
            identity={"run": {"name": "different"}},
            command=["train"],
        )
