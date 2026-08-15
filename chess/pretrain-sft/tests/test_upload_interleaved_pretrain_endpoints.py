from __future__ import annotations

from types import SimpleNamespace

import pytest

from modal_scripts import upload_interleaved_pretrain_endpoints as uploader


def test_registered_v1_specs_are_exactly_bound():
    for spec in uploader.V1_ENDPOINTS.values():
        observed = uploader._validate_bound_spec(spec, v2r2=False)
        assert observed["experiment_version"] == uploader.V1_VERSION
        assert str(observed["repo_id"]).startswith(
            "Pre-to-Post-2/pretrain_interleave_47m_v1_"
        )
        assert uploader.SHA256_RE.fullmatch(
            str(observed["expected_checkpoint_fingerprint"])
        )


def test_v2r2_bound_spec_rejects_version_repo_and_step_drift():
    spec = {
        "stage": "p1",
        "repo_id": "Pre-to-Post-2/pretrain_interleave_47m_v2r2_p1",
        "run_root": (
            "/checkpoints/interleave_50m/pretrain/"
            f"{uploader.V2R2_VERSION}/p1_shared"
        ),
        "experiment_version": uploader.V2R2_VERSION,
        "data_artifact_version": "clean-v2",
        "expected_step": 9_920,
        "expected_arc_steps": [9_920],
        "expected_manifest_hash": "a" * 64,
        "expected_config_sha256": "b" * 64,
        "expected_final_state_sha256": "c" * 64,
        "expected_checkpoint_fingerprint": "d" * 64,
        "source_tree_sha256": "e" * 64,
    }
    uploader._validate_bound_spec(spec, v2r2=True)

    drifted = dict(spec, expected_step=2_000)
    with pytest.raises(ValueError, match="step count"):
        uploader._validate_bound_spec(drifted, v2r2=True)

    drifted = dict(
        spec,
        repo_id="Pre-to-Post-2/pretrain_interleave_47m_v1_p1",
    )
    with pytest.raises(ValueError, match="repo"):
        uploader._validate_bound_spec(drifted, v2r2=True)


def test_repo_items_ignores_structural_folder_entries():
    class FakeApi:
        def list_repo_tree(self, **kwargs):
            assert kwargs["recursive"] is True
            assert kwargs["expand"] is True
            return [
                SimpleNamespace(path="evaluation", size=None),
                SimpleNamespace(path="evaluation/chess.json", size=10),
                SimpleNamespace(path="model.safetensors", size=100),
            ]

    assert set(uploader._repo_items(FakeApi(), "owner/repo")) == {
        "evaluation/chess.json",
        "model.safetensors",
    }


def test_final_inventory_is_exact_and_never_partial(tmp_path):
    final = tmp_path / "final"
    final.mkdir()
    for name in uploader.FINAL_FILES:
        (final / name).write_bytes(name.encode())
    records = uploader._final_records(final)
    assert {record["path"] for record in records} == uploader.FINAL_FILES

    (final / "model.safetensors").unlink()
    with pytest.raises(RuntimeError, match="inventory differs"):
        uploader._final_records(final)
