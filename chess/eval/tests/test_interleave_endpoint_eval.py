from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from Eval import interleave_endpoint_eval as endpoint


def _hashed(value: dict, field: str) -> dict:
    value[field] = endpoint.content_hash(value, field)
    return value


def _manifests() -> tuple[dict, dict]:
    shards = [
        {
            "shard_number": index,
            "relative_path": f"raw.{index}.npy",
            "num_tokens": 20_000,
            "dtype": "<u4",
            "byte_size": 80_128,
            "content_sha256": None,
        }
        for index in range(40)
    ]
    source = _hashed(
        {
            "schema": "interleaved-source-shards-v1",
            "schema_version": 1,
            "source_root": "/data/source",
            "pattern": "raw.*.npy",
            "content_hashes": False,
            "total_tokens": endpoint.SOURCE_TOTAL_TOKENS,
            "shards": shards,
        },
        "manifest_hash",
    )
    source["manifest_hash"] = endpoint.SOURCE_MANIFEST_HASH
    # Re-hash after installing the production identity. Tests monkeypatch the
    # expected constant because a tiny fixture cannot hash to its value.
    source["manifest_hash"] = endpoint.content_hash(source, "manifest_hash")
    selection = _hashed(
        {
            "schema": "interleaved-pretrain-selection-v1",
            "schema_version": 1,
            "algorithm": "python-random-shard-permutation-v1",
            "source_manifest_hash": source["manifest_hash"],
            "target_tokens": endpoint.TRAIN_SELECTION_TOKENS - 1,
            "source_tokens": endpoint.TRAIN_SELECTION_TOKENS,
            "seed": 42,
            "spans": [
                {
                    "shard_number": 0,
                    "relative_path": "raw.0.npy",
                    "start": 0,
                    "stop": 20_000,
                }
            ],
        },
        "selection_hash",
    )
    return source, selection


def _patch_fixture_identities(
    monkeypatch: pytest.MonkeyPatch, source: dict, selection: dict
) -> None:
    monkeypatch.setattr(endpoint, "SOURCE_MANIFEST_HASH", source["manifest_hash"])
    selection["source_manifest_hash"] = source["manifest_hash"]
    selection["selection_hash"] = endpoint.content_hash(selection, "selection_hash")
    monkeypatch.setattr(
        endpoint, "TRAIN_SELECTION_HASH", selection["selection_hash"]
    )


def test_pt_holdout_is_wholly_shard_disjoint_and_deterministic(monkeypatch):
    source, selection = _manifests()
    _patch_fixture_identities(monkeypatch, source, selection)
    monkeypatch.setattr(endpoint, "PT_RECORDS_PER_SHARD", 2)

    manifest = endpoint.build_pt_holdout_manifest(
        source,
        selection,
        num_records=8,
        shard_sha256=lambda name: hashlib.sha256(name.encode()).hexdigest(),
    )
    endpoint.validate_pt_holdout_manifest(manifest, source, selection)

    assert manifest["num_records"] == 8
    assert manifest["target_tokens"] == 8 * endpoint.SEQUENCE_LENGTH
    assert all(
        shard["relative_path"] != "raw.0.npy" for shard in manifest["shards"]
    )
    assert manifest == endpoint.build_pt_holdout_manifest(
        source,
        selection,
        num_records=8,
        shard_sha256=lambda name: hashlib.sha256(name.encode()).hexdigest(),
    )


def test_pt_holdout_rejects_train_overlap(monkeypatch):
    source, selection = _manifests()
    _patch_fixture_identities(monkeypatch, source, selection)
    manifest = endpoint.build_pt_holdout_manifest(
        source,
        selection,
        num_records=1,
        shard_sha256=lambda name: hashlib.sha256(name.encode()).hexdigest(),
    )
    manifest["records"][0]["relative_path"] = "raw.0.npy"
    manifest["holdout_hash"] = endpoint.content_hash(manifest, "holdout_hash")
    with pytest.raises(ValueError, match="record selection drifted"):
        endpoint.validate_pt_holdout_manifest(manifest, source, selection)


def test_sft_audit_proves_exact_union(monkeypatch):
    monkeypatch.setattr(endpoint, "SFT_ROWS", 5)
    monkeypatch.setattr(endpoint, "P1_SFT_ROWS", 2)
    monkeypatch.setattr(endpoint, "P2_SFT_ROWS", 3)
    manifest_set = {
        "manifest_set_hash": (
            "fc1721b9562b6b66048e3b2b085c9247e21006362ce46efe5b4714f4a8991ba8"
        ),
        "sft_cache_hash": endpoint.SFT_CACHE_HASH,
        "sft_repo": endpoint.SFT_REPO,
        "sft_revision": endpoint.SFT_REVISION,
        "sft_rows": 5,
    }
    cache = {
        "cache_hash": endpoint.SFT_CACHE_HASH,
        "rows_sha256": endpoint.SFT_ROWS_SHA256,
        "input_ids_sha256": endpoint.SFT_INPUT_IDS_SHA256,
        "labels_sha256": endpoint.SFT_LABELS_SHA256,
        "offsets_sha256": endpoint.SFT_OFFSETS_SHA256,
        "num_rows": 5,
    }
    p1_meta = {
        "leg": "p1",
        "order_sha256": endpoint.P1_ORDER_SHA256,
        "sft_records": 2,
    }
    p2_meta = {
        "leg": "p2",
        "order_sha256": endpoint.P2_ORDER_SHA256,
        "sft_records": 3,
    }
    p1 = np.asarray([-1, -3, endpoint.PAD_RECORD, 0], dtype=np.int64)
    p2 = np.asarray([-2, -4, -5, endpoint.PAD_RECORD], dtype=np.int64)
    audit = endpoint.build_sft_holdout_audit(
        manifest_set, cache, p1_meta, p2_meta, p1, p2
    )
    endpoint.validate_sft_holdout_audit(audit)
    assert audit["status"] == "unavailable_no_heldout"
    assert all(value is None for value in audit["metrics"].values())


def test_sft_audit_rejects_missing_or_duplicate_rows(monkeypatch):
    monkeypatch.setattr(endpoint, "SFT_ROWS", 4)
    monkeypatch.setattr(endpoint, "P1_SFT_ROWS", 2)
    monkeypatch.setattr(endpoint, "P2_SFT_ROWS", 2)
    manifest_set = {
        "manifest_set_hash": (
            "fc1721b9562b6b66048e3b2b085c9247e21006362ce46efe5b4714f4a8991ba8"
        ),
        "sft_cache_hash": endpoint.SFT_CACHE_HASH,
        "sft_repo": endpoint.SFT_REPO,
        "sft_revision": endpoint.SFT_REVISION,
        "sft_rows": 4,
    }
    cache = {
        "cache_hash": endpoint.SFT_CACHE_HASH,
        "rows_sha256": endpoint.SFT_ROWS_SHA256,
        "input_ids_sha256": endpoint.SFT_INPUT_IDS_SHA256,
        "labels_sha256": endpoint.SFT_LABELS_SHA256,
        "offsets_sha256": endpoint.SFT_OFFSETS_SHA256,
        "num_rows": 4,
    }
    p1_meta = {
        "leg": "p1",
        "order_sha256": endpoint.P1_ORDER_SHA256,
        "sft_records": 2,
    }
    p2_meta = {
        "leg": "p2",
        "order_sha256": endpoint.P2_ORDER_SHA256,
        "sft_records": 2,
    }
    with pytest.raises(ValueError, match="overlap"):
        endpoint.build_sft_holdout_audit(
            manifest_set,
            cache,
            p1_meta,
            p2_meta,
            np.asarray([-1, -2], dtype=np.int64),
            np.asarray([-2, -3], dtype=np.int64),
        )


def _write_endpoint(path: Path, *, step: int = 1) -> None:
    path.mkdir(parents=True)
    config = dict(endpoint.EXPECTED_MODEL_CONFIG)
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "tokenizer_config.json").write_text("{}")
    (path / "tokenizer.py").write_text("# pinned custom tokenizer\n")
    (path / "vocab.json").write_text("{}")
    (path / "interleaved_training_state.json").write_text(
        json.dumps({"global_step": step})
    )


def test_discover_fixed_endpoints(tmp_path):
    p1 = (
        tmp_path
        / "interleave_50m"
        / "pretrain"
        / endpoint.EXPERIMENT_VERSION
        / "p1_shared"
        / "final"
    )
    p2 = (
        tmp_path
        / "interleave_50m"
        / "pretrain"
        / endpoint.EXPERIMENT_VERSION
        / "p2"
        / "exp1-u-after-rl1500-from-abcdef123456"
        / "final"
    )
    _write_endpoint(p1)
    _write_endpoint(p2)
    found = endpoint.discover_endpoints(tmp_path)
    assert [item["endpoint_id"] for item in found] == ["e1-u-p2", "p1"]


def test_v1_and_v2_discovery_roots_are_strictly_separated(tmp_path):
    v1_p1 = (
        tmp_path
        / "interleave_50m"
        / "pretrain"
        / endpoint.EXPERIMENT_VERSION
        / "p1_shared"
        / "final"
    )
    v2_p1 = (
        tmp_path
        / "interleave_50m"
        / "pretrain"
        / endpoint.V2_EXPERIMENT_VERSION
        / "p1_shared"
        / "final"
    )
    v2_exp2 = (
        tmp_path
        / "interleave_50m"
        / "pretrain"
        / endpoint.V2_EXPERIMENT_VERSION
        / "exp2_monolithic"
        / "final"
    )
    v2_e3 = (
        tmp_path
        / "interleave_50m"
        / "pretrain"
        / endpoint.V2_EXPERIMENT_VERSION
        / "p2"
        / "exp3-two-cosine-control-from-p1-from-abcdef123456"
        / "final"
    )
    for path in (v1_p1, v2_p1, v2_exp2, v2_e3):
        _write_endpoint(path)

    v1 = endpoint.discover_endpoints(tmp_path)
    v2 = endpoint.discover_endpoints(
        tmp_path,
        experiment_version=endpoint.V2_EXPERIMENT_VERSION,
        include_exp4=False,
    )

    assert [item["endpoint_id"] for item in v1] == ["p1"]
    assert [item["endpoint_id"] for item in v2] == [
        "e2-final",
        "e3-p2",
        "p1",
    ]
    assert all(endpoint.EXPERIMENT_VERSION in item["checkpoint_path"] for item in v1)
    assert all(
        endpoint.V2_EXPERIMENT_VERSION in item["checkpoint_path"] for item in v2
    )


def test_discovery_rejects_unsafe_version(tmp_path):
    with pytest.raises(ValueError, match="unsafe experiment version"):
        endpoint.discover_endpoints(
            tmp_path,
            experiment_version="../../v1",
            include_exp4=False,
        )


def test_v2_sft_audit_binds_clean_cache_and_exact_row_union():
    p1_rows = endpoint.P1_SFT_ROWS
    p2_rows = endpoint.P2_SFT_ROWS
    p1_order = -(np.arange(p1_rows, dtype=np.int64) + 1)
    p2_order = -(
        np.arange(p1_rows, p1_rows + p2_rows, dtype=np.int64) + 1
    )
    p1_order_sha = "1" * 64
    p2_order_sha = "2" * 64
    p1_metadata_sha = "3" * 64
    p2_metadata_sha = "4" * 64

    cache = _hashed(
        {
            "schema": "interleaved-sft-cache-v1",
            "schema_version": 1,
            "num_rows": endpoint.SFT_ROWS,
            "total_positions": endpoint.V2_SFT_SUPERVISED_TARGETS + 1,
            "sequence_length": endpoint.SEQUENCE_LENGTH,
            "response_normalization": endpoint.V2_SFT_RESPONSE_NORMALIZATION,
            "supervised_unk_policy": endpoint.V2_SFT_SUPERVISED_UNK_POLICY,
            "strict_sft_audit_required": True,
            "supervised_targets": endpoint.V2_SFT_SUPERVISED_TARGETS,
            "supervised_unk_targets": 0,
            "supervised_delimiter_counts": {
                "</T>": endpoint.V2_END_THINKING_TARGETS,
                "<call_env>": endpoint.V2_CALL_ENV_TARGETS,
            },
            "strict_sft_audit": {
                "schema": endpoint.V2_SFT_STRICT_AUDIT_SCHEMA,
                "expected_supervised_targets": endpoint.V2_SFT_SUPERVISED_TARGETS,
                "t_end_rows_exactly_one": endpoint.SFT_ROWS,
                "call_env_rows_at_least_one": endpoint.SFT_ROWS,
            },
            "rows_sha256": "5" * 64,
            "input_ids_sha256": "6" * 64,
            "labels_sha256": "7" * 64,
            "offsets_sha256": "8" * 64,
        },
        "cache_hash",
    )

    def leg_metadata(
        leg: str,
        order_sha: str,
        rows: int,
        targets: int,
    ) -> dict:
        return _hashed(
            {
                "schema": "interleaved-leg-manifest-v1",
                "schema_version": 1,
                "leg": leg,
                "order_sha256": order_sha,
                "num_order_records": rows,
                "source_manifest_hash": endpoint.SOURCE_MANIFEST_HASH,
                "selection_hash": endpoint.TRAIN_SELECTION_HASH,
                "sft_cache_hash": cache["cache_hash"],
                "sft_records": rows,
                "sft_supervised_targets": targets,
            },
            "metadata_hash",
        )

    p1_metadata = leg_metadata(
        "p1",
        p1_order_sha,
        p1_rows,
        endpoint.V2_P1_SFT_SUPERVISED_TARGETS,
    )
    p2_metadata = leg_metadata(
        "p2",
        p2_order_sha,
        p2_rows,
        endpoint.V2_P2_SFT_SUPERVISED_TARGETS,
    )
    manifest_set = _hashed(
        {
            "schema": "interleaved-manifest-set-v1",
            "schema_version": 1,
            "experiment_version": endpoint.V2_DATA_ARTIFACT_VERSION,
            "source_repo": endpoint.SOURCE_REPO,
            "source_revision": endpoint.SOURCE_REVISION,
            "sft_repo": endpoint.SFT_REPO,
            "sft_revision": endpoint.SFT_REVISION,
            "pretrain_tokens": 10_000_000_000,
            "sft_rows": endpoint.SFT_ROWS,
            "source_manifest_hash": endpoint.SOURCE_MANIFEST_HASH,
            "selection_hash": endpoint.TRAIN_SELECTION_HASH,
            "sft_cache_hash": cache["cache_hash"],
            "manifests": {
                "p1": {
                    "path": "legs/p1/metadata.json",
                    "sha256": p1_metadata_sha,
                },
                "p2": {
                    "path": "legs/p2/metadata.json",
                    "sha256": p2_metadata_sha,
                },
            },
        },
        "manifest_set_hash",
    )
    audit = endpoint.build_v2_sft_holdout_audit(
        manifest_set,
        cache,
        p1_metadata,
        p2_metadata,
        p1_order,
        p2_order,
        p1_metadata_file_sha256=p1_metadata_sha,
        p2_metadata_file_sha256=p2_metadata_sha,
        p1_order_file_sha256=p1_order_sha,
        p2_order_file_sha256=p2_order_sha,
    )
    endpoint.validate_sft_holdout_audit(audit)
    assert audit["experiment_version"] == endpoint.V2_EXPERIMENT_VERSION
    assert audit["coverage_proof"]["union_rows"] == endpoint.SFT_ROWS
    assert audit["source"]["supervised_unk_targets"] == 0


def test_summarize_chess_metrics_requires_all_benchmarks():
    metrics = {}
    for index, benchmark in enumerate(("B1", "B2", "B3", "B4", "B5"), 1):
        metrics[f"val-core/test_{benchmark}/reward/mean@16"] = index / 10
    result = endpoint.summarize_chess_metrics(metrics)
    assert result["avg_reward"] == pytest.approx(0.3)
    assert result["pass_at_1"] == pytest.approx(0.3)
    assert result["b3_b4_avg"] == pytest.approx(0.35)
    assert result["pass_at_1_semantics"] == "binary_reward_mean@16_fallback"

    del metrics["val-core/test_B5/reward/mean@16"]
    with pytest.raises(ValueError, match="B5"):
        endpoint.summarize_chess_metrics(metrics)
