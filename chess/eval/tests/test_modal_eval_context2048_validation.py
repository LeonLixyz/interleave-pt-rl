from __future__ import annotations

import copy
import math

import numpy as np
import pytest

import modal_eval_context2048_validation as validation


def _hashed(value: dict, field: str) -> dict:
    value[field] = validation.content_hash(value, field)
    return value


def _contract(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_root.mkdir()
    shards = []
    for shard_number in range(5):
        path = source_root / f"raw.{shard_number:03d}.npy"
        values = np.arange(65, dtype=np.int64) + shard_number * 100
        np.save(path, values)
        shards.append(
            {
                "shard_number": shard_number,
                "relative_path": path.name,
                "num_tokens": len(values),
                "dtype": np.dtype(values.dtype).str,
                "byte_size": path.stat().st_size,
            }
        )
    source = _hashed(
        {
            "schema": "interleaved-source-shards-v1",
            "schema_version": 1,
            "sort": "numeric-final-npy-component",
            "content_hashes": False,
            "total_tokens": sum(item["num_tokens"] for item in shards),
            "shards": shards,
        },
        "manifest_hash",
    )
    selection = _hashed(
        {
            "schema": "interleaved-pretrain-selection-v1",
            "schema_version": 1,
            "algorithm": "test",
            "source_manifest_hash": source["manifest_hash"],
            "target_tokens": 31,
            "source_tokens": 32,
            "seed": 42,
            "spans": [
                {
                    "shard_number": 0,
                    "relative_path": "raw.000.npy",
                    "start": 4,
                    "stop": 36,
                }
            ],
        },
        "selection_hash",
    )
    monkeypatch.setattr(validation, "SOURCE_MANIFEST_HASH", source["manifest_hash"])
    monkeypatch.setattr(validation, "SELECTION_HASH", selection["selection_hash"])
    monkeypatch.setattr(validation, "SOURCE_TOTAL_TOKENS", 325)
    monkeypatch.setattr(validation, "TRAIN_TARGET_TOKENS", 31)
    monkeypatch.setattr(validation, "TRAIN_SOURCE_TOKENS", 32)
    return source_root, source, selection


def test_exact_four_checkpoint_identities_are_pinned():
    assert set(validation.CHECKPOINTS) == {
        "vocab81_then_sft3",
        "vocab85_then_sft3",
        "mixed_sft1",
        "mixed_sft3",
    }
    assert all(
        len(spec.expected_fingerprint) == 64
        for spec in validation.CHECKPOINTS.values()
    )
    assert validation.SEQUENCE_LENGTH == 2048
    assert validation.HOLDOUT_RECORDS == 4096
    assert validation.HOLDOUT_TARGET_TOKENS == 4096 * 2048
    assert validation.EVAL_BATCH_SIZE == 64


def test_hash_ranked_holdout_excludes_every_training_shard_and_is_repeatable(
    tmp_path, monkeypatch
):
    _, source, selection = _contract(tmp_path, monkeypatch)
    first = validation.plan_holdout(
        source,
        selection,
        num_records=8,
        sequence_length=8,
        records_per_shard=2,
    )
    second = validation.plan_holdout(
        source,
        selection,
        num_records=8,
        sequence_length=8,
        records_per_shard=2,
    )
    assert first == second
    selected_shards, records = first
    assert all(item["relative_path"] != "raw.000.npy" for item in selected_shards)
    assert [item["ordinal"] for item in records] == list(range(8))
    validation._validate_target_ranges(records)
    by_path = {}
    for record in records:
        by_path.setdefault(record["relative_path"], []).append(
            (record["target_start"], record["target_stop"])
        )
    for ranges in by_path.values():
        ordered = sorted(ranges)
        assert all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:]))


def test_training_contract_rejects_overlap_and_token_accounting(tmp_path, monkeypatch):
    _, source, selection = _contract(tmp_path, monkeypatch)
    bad = copy.deepcopy(selection)
    bad["spans"].append(
        {
            "shard_number": 0,
            "relative_path": "raw.000.npy",
            "start": 20,
            "stop": 24,
        }
    )
    bad["source_tokens"] = 36
    bad["selection_hash"] = validation.content_hash(bad, "selection_hash")
    monkeypatch.setattr(validation, "SELECTION_HASH", bad["selection_hash"])
    monkeypatch.setattr(validation, "TRAIN_SOURCE_TOKENS", 36)
    with pytest.raises(ValueError, match="overlap"):
        validation.validate_training_contract(source, bad)


def test_exact_token_content_hash_detects_source_mutation(tmp_path, monkeypatch):
    source_root, source, selection = _contract(tmp_path, monkeypatch)
    _, records = validation.plan_holdout(
        source,
        selection,
        num_records=2,
        sequence_length=8,
        records_per_shard=2,
    )
    before = validation.token_content_sha256(records, source_root)
    selected_path = source_root / records[0]["relative_path"]
    values = np.load(selected_path)
    values[records[0]["start"] + 1] += 1
    np.save(selected_path, values)
    after = validation.token_content_sha256(records, source_root)
    assert before != after


def test_success_marker_has_exact_metric_schema_and_rejects_tampering(monkeypatch):
    key = "vocab85_then_sft3"
    spec = validation.CHECKPOINTS[key]
    holdout_hash = "a" * 64
    correct = 123
    targets = validation.HOLDOUT_TARGET_TOKENS
    loss = 1.25
    value = {
        "schema": validation.RESULT_SCHEMA,
        "schema_version": 1,
        "version": validation.VERSION,
        "state": "complete",
        "checkpoint": key,
        "checkpoint_label": spec.label,
        "checkpoint_path": str(validation.checkpoint_path(spec)),
        "checkpoint_fingerprint": spec.expected_fingerprint,
        "holdout_hash": holdout_hash,
        "metrics": {
            "heldout_pretrain_loss": loss,
            "heldout_pretrain_perplexity": math.exp(loss),
            "heldout_pretrain_token_accuracy": correct / targets,
            "heldout_pretrain_correct_tokens": correct,
            "heldout_pretrain_target_tokens": targets,
        },
        "runtime": {},
        "started_at": "2026-08-13T00:00:00+00:00",
        "finished_at": "2026-08-13T00:01:00+00:00",
    }
    value["result_sha256"] = validation.content_hash(value, "result_sha256")
    validation.validate_result(value, key=key, holdout_hash=holdout_hash)
    tampered = copy.deepcopy(value)
    tampered["metrics"]["heldout_pretrain_loss"] += 0.01
    with pytest.raises(ValueError, match="result_sha256 mismatch"):
        validation.validate_result(tampered, key=key, holdout_hash=holdout_hash)


def test_perplexity_is_exp_of_global_token_mean_loss():
    assert math.isclose(validation.safe_perplexity(2.0), math.exp(2.0))
    with pytest.raises(ValueError, match="invalid cross-entropy"):
        validation.safe_perplexity(-0.1)


def test_complete_run_ledger_records_all_authenticated_results():
    ledger = {
        "schema": validation.RUN_SCHEMA,
        "schema_version": 1,
        "version": validation.VERSION,
        "state": "evaluating",
        "holdout_hash": "a" * 64,
        "started_at": "2026-08-13T00:00:00+00:00",
        "calls": {key: f"fc-{key}" for key in validation.CHECKPOINTS},
    }
    ledger["run_sha256"] = validation.content_hash(ledger, "run_sha256")
    results = {
        key: {
            "result_sha256": str(index) * 64,
            "metrics": {"heldout_pretrain_loss": float(index)},
        }
        for index, key in enumerate(validation.CHECKPOINTS, start=1)
    }
    completed = validation.complete_run_ledger(
        ledger,
        results,
        finished_at="2026-08-13T00:01:00+00:00",
    )
    assert completed["state"] == "complete"
    assert set(completed["results"]) == set(validation.CHECKPOINTS)
    assert completed["run_sha256"] == validation.content_hash(
        completed, "run_sha256"
    )
