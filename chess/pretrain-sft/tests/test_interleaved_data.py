from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from training.interleaved_data import (
    PAD_RECORD,
    SAMPLE_PAD,
    SAMPLE_PRETRAIN,
    SAMPLE_SFT,
    SFT_RESPONSE_NORMALIZATION_STRIP_VERIFY_V1,
    SFT_STRICT_AUDIT_SCHEMA_V1,
    SFT_SUPERVISED_DELIMITERS,
    SFT_SUPERVISED_UNK_POLICY_REJECT_V1,
    LegManifest,
    LogicalTokenSelection,
    PackedPretrainDataset,
    PretrainSelection,
    SFTCache,
    SFTCacheDataset,
    SourceShardManifest,
    build_leg_manifests,
    build_manifest_set,
    build_pretrain_selection,
    build_sft_cache,
    build_source_manifest,
    create_interleaved_dataloader,
    normalize_sft_response,
    tokenize_masked_sft_row,
)
from training.sft_data_utils import MultiTurnSFTDataset


class TinyTokenizer:
    def __init__(self):
        tokens = (
            "<bos>",
            "<eos>",
            "<pad>",
            "<unk>",
            "<T>",
            "</T>",
            "<sep>",
            "<call_env>",
            "=",
            "P",
            "B",
            "a1",
            "a2",
            "b1",
            "b2",
            "c1",
            "c2",
            "x",
        )
        self._vocab = {token: index for index, token in enumerate(tokens)}

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, text):
        return [
            self._vocab["<bos>"],
            *(
                self._vocab.get(token, self._vocab["<unk>"])
                for token in text.split()
            ),
            self._vocab["<eos>"],
        ]

    def bos_id(self):
        return self._vocab["<bos>"]

    def eos_id(self):
        return self._vocab["<eos>"]

    def pad_id(self):
        return self._vocab["<pad>"]

    def call_env_id(self):
        return self._vocab["<call_env>"]

    def env_token_ids(self):
        return {"<call_env>": self.call_env_id()}


def _sft_row(suffix: str = "c1"):
    return {
        "pgn": "P a1 a2",
        "cot_by_method": {
            "trajectory_sep": {
                "cot_format_no_labels": (
                    "<T> P a1 a2 <call_env> B b1 b2 "
                    f"P {suffix} c2 </T>"
                )
            }
        },
    }


def _write_rehashed_metadata(
    path: Path,
    value: dict,
    *,
    hash_field: str = "cache_hash",
) -> None:
    unhashed = {key: item for key, item in value.items() if key != hash_field}
    encoded = json.dumps(
        unhashed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    value[hash_field] = hashlib.sha256(encoded).hexdigest()
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class InterleavedDataTests(unittest.TestCase):
    def _source(self, root: Path):
        source_root = root / "source"
        source_root.mkdir()
        # Deliberately create out of lexical/numeric order.
        np.save(source_root / "raw.10.npy", np.arange(20, 30, dtype=np.int32))
        np.save(source_root / "raw.2.npy", np.arange(10, 20, dtype=np.int32))
        np.save(source_root / "raw.1.npy", np.arange(0, 10, dtype=np.int32))
        source_path = root / "source_manifest.json"
        source = build_source_manifest(source_root, source_path)
        selection_path = root / "selection.json"
        selection = build_pretrain_selection(
            source_path, selection_path, target_tokens=18, seed=7
        )
        return source_root, source_path, source, selection_path, selection

    def _sft(self, root: Path):
        tokenizer = TinyTokenizer()
        sft_path = root / "sft.json"
        sft_path.write_text(
            json.dumps({"results": [_sft_row("c1"), _sft_row("a1")]}),
            encoding="utf-8",
        )
        cache_dir = root / "sft_cache"
        cache = build_sft_cache(
            [sft_path],
            tokenizer,
            cache_dir,
            sequence_length=32,
            expected_rows=2,
        )
        return tokenizer, sft_path, cache_dir, cache

    def test_source_numeric_sort_hash_and_exact_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, source_path, source, selection_path, selection = self._source(root)
            self.assertEqual(
                [shard.shard_number for shard in source.shards], [1, 2, 10]
            )
            self.assertEqual(source.total_tokens, 30)
            self.assertEqual(selection.target_tokens, 18)
            self.assertEqual(selection.source_tokens, 19)
            self.assertEqual(
                sum(span.num_tokens for span in selection.spans), 19
            )
            self.assertEqual(
                SourceShardManifest.load(source_path).manifest_hash,
                source.manifest_hash,
            )

            second_path = root / "selection_again.json"
            second = build_pretrain_selection(
                source_path, second_path, target_tokens=18, seed=7
            )
            self.assertEqual(second.selection_hash, selection.selection_hash)
            self.assertEqual(
                [span.as_dict() for span in second.spans],
                [span.as_dict() for span in selection.spans],
            )
            self.assertEqual(
                PretrainSelection.load(selection_path).selection_hash,
                selection.selection_hash,
            )

    def test_trusted_source_manifest_infers_tokens_without_opening_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            source_root.mkdir()
            header_bytes = 128
            (source_root / "raw.1.npy").write_bytes(
                b"\0" * (header_bytes + 5 * 4)
            )
            (source_root / "raw.0.npy").write_bytes(
                b"\0" * (header_bytes + 3 * 4)
            )
            output = root / "manifest.json"

            with mock.patch(
                "training.interleaved_data.np.load",
                side_effect=AssertionError("trusted fast path opened a shard"),
            ):
                manifest = build_source_manifest(
                    source_root,
                    output,
                    trusted_npy_dtype="<u4",
                    trusted_npy_header_bytes=header_bytes,
                    expected_total_tokens=8,
                )

            self.assertEqual(manifest.total_tokens, 8)
            self.assertEqual(
                [shard.num_tokens for shard in manifest.shards], [3, 5]
            )
            self.assertTrue(
                all(shard.dtype == "<u4" for shard in manifest.shards)
            )

    def test_trusted_source_manifest_fails_closed_on_bad_sizes_or_total(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            source_root.mkdir()
            shard = source_root / "raw.0.npy"
            shard.write_bytes(b"\0" * (128 + 9))

            with self.assertRaisesRegex(ValueError, "not divisible"):
                build_source_manifest(
                    source_root,
                    root / "misaligned.json",
                    trusted_npy_dtype="<u4",
                    trusted_npy_header_bytes=128,
                    expected_total_tokens=2,
                )

            shard.write_bytes(b"\0" * (128 + 2 * 4))
            with self.assertRaisesRegex(ValueError, "token total mismatch"):
                build_source_manifest(
                    source_root,
                    root / "wrong-total.json",
                    trusted_npy_dtype="<u4",
                    trusted_npy_header_bytes=128,
                    expected_total_tokens=3,
                )
            with self.assertRaisesRegex(ValueError, "requires"):
                build_source_manifest(
                    source_root,
                    root / "partial-options.json",
                    trusted_npy_dtype="<u4",
                )

    def test_packed_legs_have_exact_targets_and_share_only_context_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, _, source, _, selection = self._source(root)
            logical = LogicalTokenSelection(source_root, source, selection)
            selected = logical.read(0, 19)
            p1 = PackedPretrainDataset(
                logical,
                target_start=0,
                target_count=9,
                bos_token_id=99,
                sequence_length=4,
            )
            p2 = PackedPretrainDataset(
                logical,
                target_start=9,
                target_count=9,
                bos_token_id=99,
                sequence_length=4,
            )
            self.assertEqual(len(p1), 3)
            self.assertEqual(len(p2), 3)
            self.assertEqual(p1[-1]["valid_targets"], 1)
            self.assertEqual(p2[-1]["valid_targets"], 1)

            p1_inputs = torch.cat([p1[i]["input_ids"] for i in range(len(p1))])
            p1_labels = torch.cat([p1[i]["labels"] for i in range(len(p1))])
            p2_inputs = torch.cat([p2[i]["input_ids"] for i in range(len(p2))])
            p2_labels = torch.cat([p2[i]["labels"] for i in range(len(p2))])
            np.testing.assert_array_equal(
                p1_inputs.numpy(),
                np.asarray([99, *selected[1:4], 99, *selected[5:8], 99]),
            )
            np.testing.assert_array_equal(p1_labels.numpy(), selected[1:10])
            np.testing.assert_array_equal(
                p2_inputs.numpy(),
                np.asarray([99, *selected[10:13], 99, *selected[14:17], 99]),
            )
            np.testing.assert_array_equal(p2_labels.numpy(), selected[10:19])
            self.assertEqual(int(p1_inputs[0]), 99)
            self.assertEqual(int(p2_inputs[0]), 99)

    def test_sft_masking_matches_existing_multiturn_dataset_and_context_boundary(self):
        tokenizer = TinyTokenizer()
        row = _sft_row()
        input_ids, labels = tokenize_masked_sft_row(
            row,
            tokenizer,
            cot_field="cot_by_method.trajectory_sep.cot_format_no_labels",
            sequence_length=32,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one.json"
            path.write_text(json.dumps({"results": [row]}), encoding="utf-8")
            legacy = MultiTurnSFTDataset(
                [str(path)],
                tokenizer,
                seq_len=32,
                mask_prompt=True,
                cot_field="cot_by_method.trajectory_sep.cot_format_no_labels",
                prompt_field="pgn",
            )
            legacy_input, legacy_labels, _ = legacy[0]
        np.testing.assert_array_equal(input_ids, legacy_input.numpy())
        np.testing.assert_array_equal(labels, legacy_labels.numpy())

        exact = _sft_row()
        exact["cot_by_method"]["trajectory_sep"]["cot_format_no_labels"] = (
            "<T> P P P P P P P"
        )
        exact_input, _ = tokenize_masked_sft_row(
            exact,
            tokenizer,
            cot_field="cot_by_method.trajectory_sep.cot_format_no_labels",
            sequence_length=12,
        )
        self.assertEqual(len(exact_input), 12)
        exact["cot_by_method"]["trajectory_sep"]["cot_format_no_labels"] += " P"
        with self.assertRaisesRegex(ValueError, "exceeds context"):
            tokenize_masked_sft_row(
                exact,
                tokenizer,
                cot_field="cot_by_method.trajectory_sep.cot_format_no_labels",
                sequence_length=12,
            )

    def test_sft_strips_all_verify_score_pairs_before_tokenization(self):
        tokenizer = TinyTokenizer()
        self.assertEqual(
            normalize_sft_response(
                " \n <T>   P a1 a2\t<verify>\n<-3>  "
                "<sep>\nP b1 b2 <verify><+0.5>  </T>\n"
            ),
            "<T> P a1 a2 <sep> P b1 b2 </T>",
        )
        clean = _sft_row()
        clean["cot_by_method"]["trajectory_sep"]["cot_format_no_labels"] = (
            "<T> P a1 a2 <sep> P b1 b2 </T>"
        )
        dirty = _sft_row()
        dirty["cot_by_method"]["trajectory_sep"]["cot_format_no_labels"] = (
            "<T> P a1 a2 <verify> <-3> <sep> "
            "P b1 b2 <verify><+0.5> </T>"
        )

        clean_arrays = tokenize_masked_sft_row(
            clean,
            tokenizer,
            cot_field="cot_by_method.trajectory_sep.cot_format_no_labels",
            sequence_length=32,
        )
        dirty_arrays = tokenize_masked_sft_row(
            dirty,
            tokenizer,
            cot_field="cot_by_method.trajectory_sep.cot_format_no_labels",
            sequence_length=32,
        )

        np.testing.assert_array_equal(dirty_arrays[0], clean_arrays[0])
        np.testing.assert_array_equal(dirty_arrays[1], clean_arrays[1])
        self.assertNotIn(
            tokenizer.get_vocab()["<unk>"],
            dirty_arrays[1][dirty_arrays[1] != -100],
        )

    def test_sft_rejects_residual_verify_and_supervised_unknown_targets(self):
        tokenizer = TinyTokenizer()
        residual_verify = _sft_row()
        residual_verify["cot_by_method"]["trajectory_sep"][
            "cot_format_no_labels"
        ] = "<T> P a1 a2 <verify> <sep> P b1 b2 </T>"
        with self.assertRaisesRegex(ValueError, "unpaired or non-numeric"):
            tokenize_masked_sft_row(
                residual_verify,
                tokenizer,
                cot_field="cot_by_method.trajectory_sep.cot_format_no_labels",
                sequence_length=32,
            )

        unknown = _sft_row()
        unknown["cot_by_method"]["trajectory_sep"][
            "cot_format_no_labels"
        ] = "<T> P a1 a2 <unexpected-label> </T>"
        with self.assertRaisesRegex(ValueError, "supervised <unk> target"):
            tokenize_masked_sft_row(
                unknown,
                tokenizer,
                cot_field="cot_by_method.trajectory_sep.cot_format_no_labels",
                sequence_length=32,
            )

        _, permissive_labels = tokenize_masked_sft_row(
            unknown,
            tokenizer,
            cot_field="cot_by_method.trajectory_sep.cot_format_no_labels",
            sequence_length=32,
            reject_supervised_unk=False,
        )
        self.assertIn(
            tokenizer.get_vocab()["<unk>"],
            permissive_labels[permissive_labels != -100],
        )

    def test_cache_is_mmap_friendly_hashed_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer, sft_path, cache_dir, cache = self._sft(root)
            self.assertEqual(cache.num_rows, 2)
            self.assertGreater(cache.total_positions, 0)
            self.assertEqual(SFTCache.load(cache_dir).cache_hash, cache.cache_hash)
            metadata = json.loads(
                (cache_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["response_normalization"],
                SFT_RESPONSE_NORMALIZATION_STRIP_VERIFY_V1,
            )
            self.assertEqual(
                metadata["supervised_unk_policy"],
                SFT_SUPERVISED_UNK_POLICY_REJECT_V1,
            )
            expected_targets = 0
            for row in (_sft_row("c1"), _sft_row("a1")):
                _, labels = tokenize_masked_sft_row(
                    row,
                    tokenizer,
                    cot_field=(
                        "cot_by_method.trajectory_sep.cot_format_no_labels"
                    ),
                    sequence_length=32,
                )
                expected_targets += int(np.count_nonzero(labels != -100))
            self.assertTrue(metadata["strict_sft_audit_required"])
            self.assertEqual(
                metadata["strict_sft_audit"]["schema"],
                SFT_STRICT_AUDIT_SCHEMA_V1,
            )
            self.assertIsNone(
                metadata["strict_sft_audit"]["expected_supervised_targets"]
            )
            self.assertEqual(
                metadata["supervised_targets"], expected_targets
            )
            self.assertEqual(
                set(metadata["supervised_delimiter_counts"]),
                set(SFT_SUPERVISED_DELIMITERS),
            )
            self.assertEqual(
                metadata["supervised_delimiter_counts"],
                {
                    "<T>": 0,
                    "</T>": 2,
                    "<sep>": 0,
                    "<call_env>": 2,
                    "<eos>": 2,
                },
            )
            self.assertEqual(metadata["supervised_unk_targets"], 0)
            again = build_sft_cache(
                [sft_path],
                tokenizer,
                cache_dir,
                sequence_length=32,
                expected_rows=2,
            )
            self.assertEqual(again.cache_hash, cache.cache_hash)
            with self.assertRaisesRegex(
                ValueError, "normalization/validation contract differs"
            ):
                build_sft_cache(
                    [sft_path],
                    tokenizer,
                    cache_dir,
                    sequence_length=32,
                    expected_rows=2,
                    strip_verify_scores=False,
                    strict_sft_audit=False,
                )

    def test_strict_cache_preregisters_target_total_and_row_delimiters(self):
        tokenizer = TinyTokenizer()
        rows = [_sft_row("c1"), _sft_row("a1")]
        expected_targets = sum(
            int(np.count_nonzero(labels != -100))
            for _, labels in (
                tokenize_masked_sft_row(
                    row,
                    tokenizer,
                    cot_field=(
                        "cot_by_method.trajectory_sep.cot_format_no_labels"
                    ),
                    sequence_length=32,
                )
                for row in rows
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sft.json"
            source.write_text(
                json.dumps({"results": rows}),
                encoding="utf-8",
            )
            cache_dir = root / "cache"
            cache = build_sft_cache(
                [source],
                tokenizer,
                cache_dir,
                sequence_length=32,
                expected_rows=2,
                expected_supervised_targets=expected_targets,
            )
            metadata = json.loads(
                cache.metadata_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["strict_sft_audit"][
                    "expected_supervised_targets"
                ],
                expected_targets,
            )
            self.assertEqual(
                metadata["supervised_targets"], expected_targets
            )

            wrong_cache = root / "wrong_cache"
            with self.assertRaisesRegex(
                ValueError, "supervised target count.*expected exactly"
            ):
                build_sft_cache(
                    [source],
                    tokenizer,
                    wrong_cache,
                    sequence_length=32,
                    expected_rows=2,
                    expected_supervised_targets=expected_targets + 1,
                )
            self.assertFalse(wrong_cache.exists())

            no_call_env = _sft_row()
            no_call_env["cot_by_method"]["trajectory_sep"][
                "cot_format_no_labels"
            ] = "<T> P a1 a2 </T>"
            no_call_env_source = root / "no_call_env.json"
            no_call_env_source.write_text(
                json.dumps({"results": [no_call_env]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "at least one supervised <call_env>"
            ):
                build_sft_cache(
                    [no_call_env_source],
                    tokenizer,
                    root / "no_call_env_cache",
                    sequence_length=32,
                    expected_rows=1,
                )

            two_t_end = _sft_row()
            two_t_end["cot_by_method"]["trajectory_sep"][
                "cot_format_no_labels"
            ] += " </T>"
            two_t_end_source = root / "two_t_end.json"
            two_t_end_source.write_text(
                json.dumps({"results": [two_t_end]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "exactly one supervised </T>"
            ):
                build_sft_cache(
                    [two_t_end_source],
                    tokenizer,
                    root / "two_t_end_cache",
                    sequence_length=32,
                    expected_rows=1,
                )

    def test_strict_cache_load_fails_closed_but_legacy_v1_stays_readable(self):
        mutations = (
            (
                "expected target mismatch",
                lambda value: value["strict_sft_audit"].__setitem__(
                    "expected_supervised_targets",
                    value["supervised_targets"] + 1,
                ),
                "preregistered expectation",
            ),
            (
                "delimiter mismatch",
                lambda value: value["supervised_delimiter_counts"].__setitem__(
                    "</T>", value["num_rows"] - 1
                ),
                "exactly one supervised </T>",
            ),
            (
                "missing required audit",
                lambda value: value.__setitem__("strict_sft_audit", None),
                "requires strict_sft_audit",
            ),
        )
        for label, mutate, message in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, _, cache_dir, _ = self._sft(root)
                metadata_path = cache_dir / "metadata.json"
                metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
                mutate(metadata)
                _write_rehashed_metadata(metadata_path, metadata)
                with self.assertRaisesRegex(ValueError, message):
                    SFTCache.load(cache_dir)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, cache_dir, cache = self._sft(root)
            metadata_path = cache_dir / "metadata.json"
            legacy = json.loads(metadata_path.read_text(encoding="utf-8"))
            for key in (
                "response_normalization",
                "supervised_unk_policy",
                "supervised_targets",
                "supervised_delimiter_counts",
                "supervised_unk_targets",
                "strict_sft_audit",
                "strict_sft_audit_required",
            ):
                legacy.pop(key, None)
            _write_rehashed_metadata(metadata_path, legacy)
            self.assertEqual(
                SFTCache.load(cache_dir).num_rows,
                cache.num_rows,
            )

    def test_leg_mix_padding_stream_cursor_composite_and_manifest_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                source_root,
                source_path,
                _,
                selection_path,
                _,
            ) = self._source(root)
            tokenizer, _, cache_dir, sft_cache = self._sft(root)
            sft_dataset = SFTCacheDataset(sft_cache)
            row_targets = np.asarray(
                [
                    sft_dataset[index]["valid_targets"]
                    for index in range(len(sft_dataset))
                ],
                dtype=np.int64,
            )
            split_order = np.random.Generator(
                np.random.PCG64(11)
            ).permutation(len(row_targets))
            split_at = len(row_targets) // 2
            expected_leg_targets = (
                int(row_targets[split_order[:split_at]].sum()),
                int(row_targets[split_order[split_at:]].sum()),
            )
            legs_root = root / "legs"
            paths = build_leg_manifests(
                selection_path,
                cache_dir,
                legs_root,
                source_manifest_path=source_path,
                sequence_length=32,
                world_size=1,
                local_batch_size=5,
                split_seed=11,
                p1_seed=12,
                p2_seed=13,
                canary_world_size=1,
                canary_local_batch_size=2,
                canary_total_steps=1,
                expected_sft_supervised_targets=expected_leg_targets,
            )
            p1 = LegManifest.load(paths["p1"])
            p2 = LegManifest.load(paths["p2"])
            canary_manifest = LegManifest.load(paths["canary"])
            # Legacy callers still emit the authenticated scalar shuffle seed;
            # the optional structured production provenance remains absent.
            self.assertIsNone(p1.order_provenance)
            self.assertIsNone(p2.order_provenance)
            self.assertIsNone(canary_manifest.order_provenance)
            self.assertEqual((p1.target_start, p2.target_start), (0, 9))
            self.assertEqual((p1.target_count, p2.target_count), (9, 9))
            self.assertEqual(p1.pretrain_records, 1)
            self.assertEqual(p1.sft_records, 1)
            self.assertEqual(
                (p1.sft_supervised_targets, p2.sft_supervised_targets),
                expected_leg_targets,
            )
            self.assertEqual(
                p1.sft_supervised_targets + p2.sft_supervised_targets,
                json.loads(
                    sft_cache.metadata_path.read_text(encoding="utf-8")
                )["supervised_targets"],
            )
            self.assertEqual(
                canary_manifest.sft_supervised_targets,
                int(row_targets[split_order[0]]),
            )
            self.assertEqual(p1.padding_records, 3)
            self.assertEqual(p1.physical_steps, 1)
            p1_order = np.load(p1.order_path, allow_pickle=False)
            p2_order = np.load(p2.order_path, allow_pickle=False)
            p1_sft = set(int(x) for x in p1_order if int(x) < 0 and x != PAD_RECORD)
            p2_sft = set(int(x) for x in p2_order if int(x) < 0 and x != PAD_RECORD)
            self.assertTrue(p1_sft)
            self.assertTrue(p2_sft)
            self.assertFalse(p1_sft & p2_sft)

            wrong_legs = root / "wrong_legs"
            with self.assertRaisesRegex(
                ValueError, "SFT supervised-target split.*expected exactly"
            ):
                build_leg_manifests(
                    selection_path,
                    cache_dir,
                    wrong_legs,
                    source_manifest_path=source_path,
                    sequence_length=32,
                    world_size=1,
                    local_batch_size=5,
                    split_seed=11,
                    p1_seed=12,
                    p2_seed=13,
                    canary_world_size=1,
                    canary_local_batch_size=2,
                    expected_sft_supervised_targets=(
                        expected_leg_targets[0] + 1,
                        expected_leg_targets[1] - 1,
                    ),
                )
            self.assertFalse(wrong_legs.exists())

            stream = create_interleaved_dataloader(
                source_root=source_root,
                source_manifest_path=source_path,
                selection_manifest_path=selection_path,
                sft_cache_dir=cache_dir,
                leg_manifest_path=paths["p1"],
                pad_token_id=tokenizer.pad_id(),
                bos_token_id=tokenizer.bos_id(),
                rank=0,
                world_size=1,
                local_batch_size=5,
            )
            batch = next(iter(stream))
            self.assertEqual(tuple(batch["input_ids"].shape), (5, 32))
            self.assertEqual(
                sorted(batch["sample_type"].tolist()),
                [SAMPLE_PAD] * 3
                + [SAMPLE_PRETRAIN]
                + [SAMPLE_SFT],
            )
            pad_rows = batch["sample_type"] == SAMPLE_PAD
            self.assertTrue(torch.all(batch["labels"][pad_rows] == -100))
            self.assertTrue(torch.all(batch["attention_mask"][pad_rows] == 1))
            self.assertEqual(batch["cursor_start"], 0)
            stream.commit_step()
            self.assertEqual(stream.cursor, 1)
            state = stream.state_dict()
            self.assertNotIn("rank", state)

            canary = create_interleaved_dataloader(
                source_root=source_root,
                source_manifest_path=source_path,
                selection_manifest_path=selection_path,
                sft_cache_dir=cache_dir,
                leg_manifest_path=paths["canary"],
                pad_token_id=tokenizer.pad_id(),
                bos_token_id=tokenizer.bos_id(),
                rank=0,
                world_size=1,
                local_batch_size=2,
            )
            canary_batch = next(iter(canary))
            self.assertEqual(
                canary_batch["sample_type"].tolist(),
                [SAMPLE_PRETRAIN, SAMPLE_SFT],
            )
            canary.commit_step()

            composite = create_interleaved_dataloader(
                source_root=source_root,
                source_manifest_path=source_path,
                selection_manifest_path=selection_path,
                sft_cache_dir=cache_dir,
                leg_manifest_path=paths["p1+p2"],
                pad_token_id=tokenizer.pad_id(),
                bos_token_id=tokenizer.bos_id(),
                rank=0,
                world_size=1,
                local_batch_size=5,
                start_cursor=1,
            )
            self.assertEqual(composite.total_steps, 2)
            resumed_batch = next(iter(composite))
            self.assertEqual(resumed_batch["cursor_start"], 1)
            composite.commit_step()
            self.assertEqual(composite.cursor, 2)

            manifest_set_path = root / "manifest_set.json"
            manifest_set = build_manifest_set(
                manifest_set_path,
                source_manifest_path=source_path,
                selection_manifest_path=selection_path,
                sft_cache_dir=cache_dir,
                legs_root=legs_root,
                experiment_version="test-v1",
                source_revision="source-rev",
                sft_revision="sft-rev",
                pretrain_tokens=18,
                sft_rows=2,
            )
            self.assertEqual(
                set(manifest_set["manifests"]),
                {"p1", "p2", "p1+p2", "canary"},
            )

            legacy_p1 = json.loads(
                p1.metadata_path.read_text(encoding="utf-8")
            )
            legacy_p1.pop("sft_supervised_targets")
            _write_rehashed_metadata(
                p1.metadata_path,
                legacy_p1,
                hash_field="metadata_hash",
            )
            self.assertIsNone(
                LegManifest.load(p1.metadata_path).sft_supervised_targets
            )

    def test_odd_sft_count_preserves_every_row_across_unequal_halves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, source_path, _, selection_path, _ = self._source(root)
            odd_rows = [
                _sft_row("a1"),
                _sft_row("b1"),
                _sft_row("c1"),
            ]
            odd_rows[1]["cot_by_method"]["trajectory_sep"][
                "cot_format_no_labels"
            ] = odd_rows[1]["cot_by_method"]["trajectory_sep"][
                "cot_format_no_labels"
            ].replace("</T>", "P a1 a2 </T>")
            odd_rows[2]["cot_by_method"]["trajectory_sep"][
                "cot_format_no_labels"
            ] = odd_rows[2]["cot_by_method"]["trajectory_sep"][
                "cot_format_no_labels"
            ].replace("</T>", "P a1 a2 P b1 b2 </T>")
            sft_path = root / "odd_sft.json"
            sft_path.write_text(
                json.dumps({"results": odd_rows}),
                encoding="utf-8",
            )
            cache_dir = root / "odd_cache"
            odd_cache = build_sft_cache(
                [sft_path],
                TinyTokenizer(),
                cache_dir,
                sequence_length=32,
                expected_rows=3,
            )
            odd_dataset = SFTCacheDataset(odd_cache)
            row_targets = np.asarray(
                [
                    odd_dataset[index]["valid_targets"]
                    for index in range(len(odd_dataset))
                ],
                dtype=np.int64,
            )
            split_order = np.random.Generator(
                np.random.PCG64(42)
            ).permutation(len(row_targets))
            split_at = len(row_targets) // 2
            expected_leg_targets = (
                int(row_targets[split_order[:split_at]].sum()),
                int(row_targets[split_order[split_at:]].sum()),
            )
            paths = build_leg_manifests(
                selection_path,
                cache_dir,
                root / "odd_legs",
                source_manifest_path=source_path,
                sequence_length=32,
                world_size=1,
                local_batch_size=4,
                canary_world_size=1,
                canary_local_batch_size=2,
                expected_sft_supervised_targets=expected_leg_targets,
            )
            p1 = LegManifest.load(paths["p1"])
            p2 = LegManifest.load(paths["p2"])
            self.assertEqual((p1.sft_records, p2.sft_records), (1, 2))
            self.assertEqual(
                (p1.sft_supervised_targets, p2.sft_supervised_targets),
                expected_leg_targets,
            )
            self.assertNotEqual(
                p1.sft_supervised_targets * p2.sft_records,
                p2.sft_supervised_targets * p1.sft_records,
            )
            observed = []
            for leg in (p1, p2):
                order = np.load(leg.order_path, allow_pickle=False)
                observed.extend(
                    -int(code) - 1
                    for code in order
                    if int(code) < 0 and int(code) != PAD_RECORD
                )
            self.assertEqual(sorted(observed), [0, 1, 2])

    def test_invalid_sft_rows_fail_instead_of_silently_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "bad.json"
            path.write_text(
                json.dumps({"results": [{"pgn": "P a1 a2"}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Invalid SFT row"):
                build_sft_cache(
                    [path],
                    TinyTokenizer(),
                    root / "cache",
                    sequence_length=32,
                    expected_rows=1,
                )


if __name__ == "__main__":
    unittest.main()
