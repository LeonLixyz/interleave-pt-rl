from __future__ import annotations

import copy
import hashlib
import io
import json

import numpy as np
import pytest

from training import v2r4_p2_sft_eval as gate


def _json_bytes(value: dict) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()


def _self_hashed(value: dict, field: str) -> dict:
    value = dict(value)
    value[field] = gate.content_hash(value, field)
    return value


def _npy_bytes(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, value, allow_pickle=False)
    return output.getvalue()


@pytest.fixture()
def artifact_fixture(monkeypatch: pytest.MonkeyPatch) -> dict:
    # Keep the production selection cardinality while making every source
    # artifact small enough for focused unit tests.
    rows = 5_000
    p1_rows = 500
    p2_rows = rows - p1_rows
    pretrain_records = 100
    order_records = 5_000
    p1_padding = order_records - p1_rows - pretrain_records
    p2_padding = order_records - p2_rows - pretrain_records

    p1_row_ids = np.arange(0, rows, 10, dtype=np.int64)
    p2_row_ids = np.asarray(
        sorted(set(range(rows)) - set(map(int, p1_row_ids))),
        dtype=np.int64,
    )
    assert len(p1_row_ids) == p1_rows
    assert len(p2_row_ids) == p2_rows
    p1_codes = -(p1_row_ids + 1)
    p2_codes = -(p2_row_ids + 1)
    pad = np.asarray([gate.PAD_RECORD], dtype=np.int64)
    p1_order = np.concatenate(
        [
            np.arange(pretrain_records, dtype=np.int64),
            p1_codes,
            np.repeat(pad, p1_padding),
        ]
    )
    p2_order = np.concatenate(
        [
            p2_codes,
            np.arange(pretrain_records, dtype=np.int64),
            np.repeat(pad, p2_padding),
        ]
    )
    p1_order_raw = _npy_bytes(p1_order)
    p2_order_raw = _npy_bytes(p2_order)
    p1_order_sha = hashlib.sha256(p1_order_raw).hexdigest()
    p2_order_sha = hashlib.sha256(p2_order_raw).hexdigest()

    positions_per_row = 3
    offsets = np.arange(
        0,
        rows * positions_per_row + 1,
        positions_per_row,
        dtype="<i8",
    )
    offsets_raw = _npy_bytes(offsets)
    labels = np.tile(
        np.asarray([-100, 1, 2], dtype="<i4"),
        rows,
    )
    labels_sha = hashlib.sha256(memoryview(labels).cast("B")).hexdigest()
    offsets_sha = hashlib.sha256(offsets_raw).hexdigest()
    total_positions = len(labels)
    supervised_targets = rows * 2
    p1_targets = p1_rows * 2
    p2_targets = p2_rows * 2
    cache = _self_hashed(
        {
            "schema": "interleaved-sft-cache-v1",
            "schema_version": 1,
            "num_rows": rows,
            "total_positions": total_positions,
            "sequence_length": gate.SEQUENCE_LENGTH,
            "cot_field": (
                "cot_by_method.trajectory_sep.cot_format_no_labels"
            ),
            "prompt_field": "pgn",
            "response_normalization": (
                "strip-numeric-verify-score-pairs-normalize-whitespace-v1"
            ),
            "supervised_unk_policy": "reject-supervised-unk-v1",
            "strict_sft_audit_required": True,
            "supervised_targets": supervised_targets,
            "supervised_unk_targets": 0,
            "masking": "multiturn-prompt-and-env-v1",
            "dtype": "<i4",
            "rows_sha256": "a" * 64,
            "input_ids_sha256": "b" * 64,
            "labels_sha256": labels_sha,
            "offsets_sha256": offsets_sha,
            "strict_sft_audit": {
                "schema": "interleaved-sft-strict-audit-v1",
                "expected_supervised_targets": supervised_targets,
                "t_end_rows_exactly_one": rows,
                "call_env_rows_at_least_one": rows,
            },
            "supervised_delimiter_counts": {
                "</T>": rows,
                "<call_env>": rows,
            },
        },
        "cache_hash",
    )
    cache_raw = _json_bytes(cache)

    def leg_metadata(
        leg: str,
        *,
        order_sha: str,
        sft_rows: int,
        sft_targets: int,
        padding_records: int,
    ) -> dict:
        return _self_hashed(
            {
                "schema": "interleaved-leg-manifest-v1",
                "schema_version": 1,
                "leg": leg,
                "num_order_records": order_records,
                "order_file": "order.npy",
                "order_sha256": order_sha,
                "order_encoding": {
                    "padding": gate.PAD_RECORD,
                    "pretrain": "nonnegative local packed-record index",
                    "sft": "-(global_sft_row_index+1)",
                },
                "sequence_length": gate.SEQUENCE_LENGTH,
                "source_manifest_hash": gate.SOURCE_MANIFEST_HASH,
                "selection_hash": gate.SELECTION_HASH,
                "sft_cache_hash": cache["cache_hash"],
                "sft_records": sft_rows,
                "sft_supervised_targets": sft_targets,
                "pretrain_records": pretrain_records,
                "padding_records": padding_records,
                "shuffle_seed": 42 if leg == "p1" else 43,
                "target_start": 0 if leg == "p1" else 5_000_000_000,
                "target_count": 5_000_000_000,
                "total_steps": 9_920,
                "physical_steps": 9_920,
                "world_size": 8,
                "local_batch_size": 21,
                "global_batch_size": 168,
            },
            "metadata_hash",
        )

    p1_metadata = leg_metadata(
        "p1",
        order_sha=p1_order_sha,
        sft_rows=p1_rows,
        sft_targets=p1_targets,
        padding_records=p1_padding,
    )
    p2_metadata = leg_metadata(
        "p2",
        order_sha=p2_order_sha,
        sft_rows=p2_rows,
        sft_targets=p2_targets,
        padding_records=p2_padding,
    )
    p1_metadata_raw = _json_bytes(p1_metadata)
    p2_metadata_raw = _json_bytes(p2_metadata)
    p1_metadata_file_sha = hashlib.sha256(p1_metadata_raw).hexdigest()
    p2_metadata_file_sha = hashlib.sha256(p2_metadata_raw).hexdigest()

    manifest = _self_hashed(
        {
            "schema": "interleaved-manifest-set-v1",
            "schema_version": 1,
            "experiment_version": gate.DATA_ARTIFACT_VERSION,
            "source_repo": gate.SOURCE_REPO,
            "source_revision": gate.SOURCE_REVISION,
            "source_manifest_hash": gate.SOURCE_MANIFEST_HASH,
            "selection_hash": gate.SELECTION_HASH,
            "pretrain_tokens": 10_000_000_000,
            "sft_repo": gate.SFT_REPO,
            "sft_revision": gate.SFT_REVISION,
            "sft_rows": rows,
            "sft_cache_hash": cache["cache_hash"],
            "manifests": {
                "p1": {
                    "path": "legs/p1/metadata.json",
                    "sha256": p1_metadata_file_sha,
                },
                "p2": {
                    "path": "legs/p2/metadata.json",
                    "sha256": p2_metadata_file_sha,
                },
            },
        },
        "manifest_set_hash",
    )
    manifest_raw = _json_bytes(manifest)

    patches = {
        "SFT_ROWS": rows,
        "P1_SFT_ROWS": p1_rows,
        "P2_SFT_ROWS": p2_rows,
        "SFT_TOTAL_POSITIONS": total_positions,
        "SFT_SUPERVISED_TARGETS": supervised_targets,
        "P1_SFT_SUPERVISED_TARGETS": p1_targets,
        "P2_SFT_SUPERVISED_TARGETS": p2_targets,
        "SFT_CALL_ENV_TARGETS": rows,
        "SFT_CACHE_HASH": cache["cache_hash"],
        "SFT_CACHE_METADATA_FILE_SHA256": hashlib.sha256(
            cache_raw
        ).hexdigest(),
        "SFT_ROWS_SHA256": cache["rows_sha256"],
        "SFT_INPUT_IDS_SHA256": cache["input_ids_sha256"],
        "SFT_LABELS_SHA256": labels_sha,
        "SFT_OFFSETS_SHA256": offsets_sha,
        "P1_METADATA_HASH": p1_metadata["metadata_hash"],
        "P2_METADATA_HASH": p2_metadata["metadata_hash"],
        "P1_METADATA_FILE_SHA256": p1_metadata_file_sha,
        "P2_METADATA_FILE_SHA256": p2_metadata_file_sha,
        "P1_ORDER_SHA256": p1_order_sha,
        "P2_ORDER_SHA256": p2_order_sha,
        "ORDER_RECORDS": order_records,
        "PRETRAIN_RECORDS_PER_LEG": pretrain_records,
        "P1_PADDING_RECORDS": p1_padding,
        "P2_PADDING_RECORDS": p2_padding,
        "MANIFEST_SET_HASH": manifest["manifest_set_hash"],
        "MANIFEST_SET_FILE_SHA256": hashlib.sha256(
            manifest_raw
        ).hexdigest(),
    }
    for key, value in patches.items():
        monkeypatch.setattr(gate, key, value)

    return {
        "raw": {
            "manifest_set_json": manifest_raw,
            "sft_cache_metadata_json": cache_raw,
            "p1_metadata_json": p1_metadata_raw,
            "p2_metadata_json": p2_metadata_raw,
            "p1_order_npy": p1_order_raw,
            "p2_order_npy": p2_order_raw,
        },
        "p1_codes": set(map(int, p1_codes)),
        "p2_codes": set(map(int, p2_codes)),
        "offsets_npy": offsets_raw,
        "labels": labels,
    }


def _candidate() -> dict:
    return {
        "candidate_id": "v2r3-w190-step6000",
        "checkpoint_sha256": "c" * 64,
        "checkpoint_step": 6_000,
        "sft_loss_weight": 190.189290837,
        "training_data_manifest_set_hash": gate.MANIFEST_SET_HASH,
        "training_leg": "p1",
        "has_consumed_p2": False,
    }


def _rehash(value: dict, field: str) -> dict:
    value[field] = gate.content_hash(value, field)
    return value


def test_production_contract_freezes_exact_cardinality_and_real_identities():
    assert gate.SELECTION_RECORDS == 4_096
    assert gate.SFT_ROWS == 77_717
    assert gate.P1_SFT_ROWS + gate.P2_SFT_ROWS == gate.SFT_ROWS
    assert gate.MANIFEST_SET_HASH == (
        "6f2cc9093b2515e0a6a3aedc56a0cfd597c6b0f76933c5dbcd69eefd22440a23"
    )
    assert gate.SFT_CACHE_HASH == (
        "d82378522d43d5db3e8333588c24b1f864bb9e8ecd46303e1d2cd2e31d31df98"
    )
    assert gate.P1_ORDER_SHA256 != gate.P2_ORDER_SHA256


def test_selector_is_exact_deterministic_p2_only_and_rebuildable(
    artifact_fixture: dict,
):
    selection = gate.build_p2_at_p1_sft_selection(
        **artifact_fixture["raw"]
    )
    again = gate.build_p2_at_p1_sft_selection(**artifact_fixture["raw"])
    assert selection == again
    codes = selection["selection"]["signed_sft_codes"]
    assert len(codes) == 4_096
    assert len(set(codes)) == 4_096
    assert set(codes).issubset(artifact_fixture["p2_codes"])
    assert set(codes).isdisjoint(artifact_fixture["p1_codes"])
    assert selection["evaluation_claim"] == "heldout_at_p1_only"
    assert selection["candidate_eligibility"][
        "candidate_must_not_have_consumed_p2"
    ]
    assert (
        gate.validate_p2_at_p1_sft_selection(
            selection,
            **artifact_fixture["raw"],
        )
        == selection
    )


def test_hash_ranking_is_input_order_independent(artifact_fixture: dict):
    codes = list(artifact_fixture["p2_codes"])
    assert gate.rank_p2_sft_codes(codes) == gate.rank_p2_sft_codes(
        reversed(codes)
    )
    with pytest.raises(ValueError, match="unique"):
        gate.rank_p2_sft_codes(codes + [codes[0]])


def test_selector_rejects_any_artifact_byte_drift(artifact_fixture: dict):
    raw = dict(artifact_fixture["raw"])
    raw["p2_order_npy"] = raw["p2_order_npy"] + b"\0"
    with pytest.raises(ValueError, match="byte SHA-256 drifted"):
        gate.build_p2_at_p1_sft_selection(**raw)

    raw = dict(artifact_fixture["raw"])
    cache = json.loads(raw["sft_cache_metadata_json"])
    cache["supervised_unk_targets"] = 1
    raw["sft_cache_metadata_json"] = _json_bytes(cache)
    with pytest.raises(ValueError, match="byte SHA-256 drifted"):
        gate.build_p2_at_p1_sft_selection(**raw)


def test_selection_validator_rejects_self_consistent_claim_drift(
    artifact_fixture: dict,
):
    selection = gate.build_p2_at_p1_sft_selection(
        **artifact_fixture["raw"]
    )
    tampered = copy.deepcopy(selection)
    tampered["evaluation_claim"] = "globally_heldout"
    _rehash(tampered, "selection_hash")
    with pytest.raises(ValueError, match="identity drifted"):
        gate.validate_p2_at_p1_sft_selection(
            tampered,
            **artifact_fixture["raw"],
        )


def test_cache_shape_authenticates_mask_and_exact_denominators(
    artifact_fixture: dict,
):
    selection = gate.build_p2_at_p1_sft_selection(
        **artifact_fixture["raw"]
    )
    shape = gate.build_selected_cache_shape(
        selection,
        offsets_npy=artifact_fixture["offsets_npy"],
        labels_i32=artifact_fixture["labels"],
    )
    assert shape["shape"]["num_records"] == 4_096
    assert shape["shape"]["total_aligned_positions"] == 4_096 * 3
    assert shape["shape"]["supervised_targets"] == 4_096 * 2
    assert shape["shape"]["ignored_positions"] == 4_096
    assert (
        gate.validate_selected_cache_shape(shape, selection)
        == shape
    )


def test_cache_shape_rejects_label_or_offset_drift(
    artifact_fixture: dict,
):
    selection = gate.build_p2_at_p1_sft_selection(
        **artifact_fixture["raw"]
    )
    labels = artifact_fixture["labels"].copy()
    labels[1] = 3
    with pytest.raises(ValueError, match="labels.i32 SHA-256 drifted"):
        gate.build_selected_cache_shape(
            selection,
            offsets_npy=artifact_fixture["offsets_npy"],
            labels_i32=labels,
        )
    with pytest.raises(ValueError, match="offsets.npy byte SHA-256 drifted"):
        gate.build_selected_cache_shape(
            selection,
            offsets_npy=artifact_fixture["offsets_npy"] + b"\0",
            labels_i32=artifact_fixture["labels"],
        )


def test_candidate_result_is_unweighted_response_masked_and_token_aggregated(
    artifact_fixture: dict,
):
    selection = gate.build_p2_at_p1_sft_selection(
        **artifact_fixture["raw"]
    )
    shape = gate.build_selected_cache_shape(
        selection,
        offsets_npy=artifact_fixture["offsets_npy"],
        labels_i32=artifact_fixture["labels"],
    )
    supervised = shape["shape"]["supervised_targets"]
    result = gate.build_candidate_sft_result(
        selection_manifest=selection,
        cache_shape=shape,
        candidate=_candidate(),
        negative_log_likelihood_sum=supervised * 0.75,
        correct_supervised_tokens=supervised // 2,
        batches=32,
    )
    assert result["objective"]["sft_loss_weight_applied"] is False
    assert result["metrics"]["masked_sft_unweighted_token_ce"] == 0.75
    assert result["metrics"]["masked_sft_token_accuracy"] == 0.5
    assert (
        gate.validate_candidate_sft_result(
            result,
            selection_manifest=selection,
            cache_shape=shape,
            expected_candidate=_candidate(),
        )
        == result
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["objective"].__setitem__(
                "sft_loss_weight_applied", True
            ),
            "not response-masked unweighted",
        ),
        (
            lambda value: value["aggregate"].__setitem__(
                "rows_evaluated", 4_095
            ),
            "denominator",
        ),
        (
            lambda value: value["metrics"].__setitem__(
                "masked_sft_unweighted_token_ce", 9.0
            ),
            "inconsistent",
        ),
        (
            lambda value: value["candidate"].__setitem__(
                "has_consumed_p2", True
            ),
            "not eligible",
        ),
    ],
)
def test_candidate_result_fails_closed_on_semantic_drift(
    artifact_fixture: dict,
    mutation,
    message: str,
):
    selection = gate.build_p2_at_p1_sft_selection(
        **artifact_fixture["raw"]
    )
    shape = gate.build_selected_cache_shape(
        selection,
        offsets_npy=artifact_fixture["offsets_npy"],
        labels_i32=artifact_fixture["labels"],
    )
    result = gate.build_candidate_sft_result(
        selection_manifest=selection,
        cache_shape=shape,
        candidate=_candidate(),
        negative_log_likelihood_sum=100.0,
        correct_supervised_tokens=100,
        batches=8,
    )
    tampered = copy.deepcopy(result)
    mutation(tampered)
    _rehash(tampered, "result_hash")
    with pytest.raises(ValueError, match=message):
        gate.validate_candidate_sft_result(
            tampered,
            selection_manifest=selection,
            cache_shape=shape,
        )


def test_candidate_builder_rejects_nonfinite_or_p2_consuming_candidate(
    artifact_fixture: dict,
):
    selection = gate.build_p2_at_p1_sft_selection(
        **artifact_fixture["raw"]
    )
    shape = gate.build_selected_cache_shape(
        selection,
        offsets_npy=artifact_fixture["offsets_npy"],
        labels_i32=artifact_fixture["labels"],
    )
    with pytest.raises(ValueError, match="finite"):
        gate.build_candidate_sft_result(
            selection_manifest=selection,
            cache_shape=shape,
            candidate=_candidate(),
            negative_log_likelihood_sum=float("nan"),
            correct_supervised_tokens=0,
            batches=1,
        )
    candidate = _candidate()
    candidate["has_consumed_p2"] = True
    with pytest.raises(ValueError, match="not eligible"):
        gate.build_candidate_sft_result(
            selection_manifest=selection,
            cache_shape=shape,
            candidate=candidate,
            negative_log_likelihood_sum=1.0,
            correct_supervised_tokens=0,
            batches=1,
        )
