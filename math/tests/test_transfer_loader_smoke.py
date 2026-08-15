from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from transfer_generate import sha256_file, write_jsonl_atomic, write_manifest_atomic
from transfer_loader_smoke import validate_matched_pair, validate_transfer_corpus


def _write_corpus(
    root: Path,
    arm: str,
    *,
    prompt_count: int = 3,
    assistant_count: int = 5,
    mask_byte: int = 1,
) -> dict:
    root.mkdir(parents=True)
    eos = 99
    tokens = np.asarray(
        list(range(10, 10 + prompt_count))
        + list(range(40, 40 + assistant_count - 1))
        + [eos],
        dtype=np.uint32,
    )
    mask = bytes([0] * prompt_count + [mask_byte] * assistant_count)
    token_path = root / "token_ids_00000.npy"
    mask_path = root / "labels_mask_00000.npy"
    tokens.tofile(token_path)
    mask_path.write_bytes(mask)
    row = {
        "schema_version": "math-transfer-corpus-v1",
        "arm": arm,
        "candidate_id": f"{arm}-candidate",
        "problem_uid": f"{arm}-problem",
        "dedup_group": f"{arm}-group",
        "difficulty_bin": "0",
        "token_offset_start": 0,
        "token_offset_end": len(tokens),
        "prompt_token_count": prompt_count,
        "assistant_token_count": assistant_count,
        "processed_token_count": len(tokens),
    }
    selected = write_jsonl_atomic(root / "selected.jsonl", [row])
    return write_manifest_atomic(
        root / "manifest.json",
        {
            "artifact_kind": "assistant_masked_transfer_corpus",
            "corpus_schema_version": "math-transfer-corpus-v1",
            "arm": arm,
            "document_count": 1,
            "processed_tokens": len(tokens),
            "loss_bearing_assistant_tokens": assistant_count,
            "masked_prompt_tokens": prompt_count,
            "eos_token_id": eos,
            "vocab_size": 128,
            "prompt_manifest_sha256": "prompt-sha",
            "model_bundle_sha256": [f"{arm}-model"],
            "tokenizer_sha256": ["tokenizer"],
            "files": [
                {
                    "path": token_path.name,
                    "format": "raw_headerless_uint32",
                    "bytes": token_path.stat().st_size,
                    "sha256": sha256_file(token_path),
                },
                {
                    "path": mask_path.name,
                    "format": "raw_headerless_bool",
                    "bytes": mask_path.stat().st_size,
                    "sha256": sha256_file(mask_path),
                },
                selected,
            ],
        },
    )


def _write_pair(root: Path, d0: dict, d1: dict, *, tolerance: float = 0.001) -> dict:
    d0_processed = int(d0["processed_tokens"])
    d1_processed = int(d1["processed_tokens"])
    delta = abs(d0_processed - d1_processed) / min(d0_processed, d1_processed)
    return write_manifest_atomic(
        root / "pair_manifest.json",
        {
            "artifact_kind": "matched_D0_D1_pair",
            "corpora": {"D0": d0["content_id"], "D1": d1["content_id"]},
            "match": {
                "schema_version": "math-transfer-match-v1",
                "arms": ["D0", "D1"],
                "assistant_tokens_per_arm": 5,
                "selected_documents": {"D0": 1, "D1": 1},
                "processed_tokens": {
                    "D0": d0_processed,
                    "D1": d1_processed,
                },
                "processed_token_relative_delta": delta,
                "max_processed_token_relative_delta": tolerance,
                "strata": {
                    "0": {
                        "document_count_per_arm": 1,
                        "assistant_tokens_per_arm": 5,
                        "processed_tokens_per_arm": d0_processed,
                    }
                },
            },
        },
    )


def _write_multi_stratum_corpus(
    root: Path, arm: str, specs: list[tuple[str, int, int]]
) -> dict:
    root.mkdir(parents=True)
    eos = 99
    all_tokens: list[int] = []
    all_masks: list[int] = []
    rows: list[dict] = []
    cursor = 0
    for index, (difficulty_bin, prompt_count, assistant_count) in enumerate(specs):
        document = (
            list(range(10 + index, 10 + index + prompt_count))
            + list(range(40 + index, 40 + index + assistant_count - 1))
            + [eos]
        )
        end = cursor + len(document)
        all_tokens.extend(document)
        all_masks.extend([0] * prompt_count + [1] * assistant_count)
        rows.append(
            {
                "schema_version": "math-transfer-corpus-v1",
                "arm": arm,
                "candidate_id": f"{arm}-candidate-{index}",
                "problem_uid": f"{arm}-problem-{index}",
                "dedup_group": f"{arm}-group-{index}",
                "difficulty_bin": difficulty_bin,
                "token_offset_start": cursor,
                "token_offset_end": end,
                "prompt_token_count": prompt_count,
                "assistant_token_count": assistant_count,
                "processed_token_count": len(document),
            }
        )
        cursor = end

    token_path = root / "token_ids_00000.npy"
    mask_path = root / "labels_mask_00000.npy"
    np.asarray(all_tokens, dtype=np.uint32).tofile(token_path)
    mask_path.write_bytes(bytes(all_masks))
    selected = write_jsonl_atomic(root / "selected.jsonl", rows)
    assistant_total = sum(spec[2] for spec in specs)
    return write_manifest_atomic(
        root / "manifest.json",
        {
            "artifact_kind": "assistant_masked_transfer_corpus",
            "corpus_schema_version": "math-transfer-corpus-v1",
            "arm": arm,
            "document_count": len(rows),
            "processed_tokens": len(all_tokens),
            "loss_bearing_assistant_tokens": assistant_total,
            "masked_prompt_tokens": len(all_tokens) - assistant_total,
            "eos_token_id": eos,
            "vocab_size": 128,
            "prompt_manifest_sha256": "prompt-sha",
            "model_bundle_sha256": [f"{arm}-model"],
            "tokenizer_sha256": ["tokenizer"],
            "files": [
                {
                    "path": token_path.name,
                    "format": "raw_headerless_uint32",
                    "bytes": token_path.stat().st_size,
                    "sha256": sha256_file(token_path),
                },
                {
                    "path": mask_path.name,
                    "format": "raw_headerless_bool",
                    "bytes": mask_path.stat().st_size,
                    "sha256": sha256_file(mask_path),
                },
                selected,
            ],
        },
    )


class TransferLoaderArtifactValidationTest(unittest.TestCase):
    def test_valid_pair_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            d0 = _write_corpus(root / "D0", "D0")
            d1 = _write_corpus(root / "D1", "D1")
            pair = _write_pair(root, d0, d1)

            result = validate_matched_pair(root)

            self.assertEqual(result["pair_content_id"], pair["content_id"])
            self.assertEqual(result["processed_token_relative_delta"], 0.0)
            self.assertEqual(result["corpora"]["D0"]["assistant_tokens"], 5)

    def test_global_token_match_allows_declared_per_stratum_imbalance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            d0 = _write_multi_stratum_corpus(
                root / "D0", "D0", [("0", 3, 4), ("1", 3, 6)]
            )
            d1 = _write_multi_stratum_corpus(
                root / "D1", "D1", [("0", 3, 5), ("1", 3, 5)]
            )
            write_manifest_atomic(
                root / "pair_manifest.json",
                {
                    "artifact_kind": "matched_D0_D1_pair",
                    "corpora": {"D0": d0["content_id"], "D1": d1["content_id"]},
                    "match": {
                        "schema_version": "math-transfer-match-v1",
                        "arms": ["D0", "D1"],
                        "assistant_tokens_per_arm": 10,
                        "selected_documents": {"D0": 2, "D1": 2},
                        "processed_tokens": {"D0": 16, "D1": 16},
                        "processed_token_relative_delta": 0.0,
                        "max_processed_token_relative_delta": 0.001,
                        "strata": {
                            "0": {
                                "status": "matched",
                                "document_count_per_arm": 1,
                                "assistant_tokens": {"D0": 4, "D1": 5},
                                "assistant_tokens_per_arm": None,
                                "processed_tokens": {"D0": 7, "D1": 8},
                                "processed_tokens_per_arm": None,
                            },
                            "1": {
                                "status": "matched",
                                "document_count_per_arm": 1,
                                "assistant_tokens": {"D0": 6, "D1": 5},
                                "assistant_tokens_per_arm": None,
                                "processed_tokens": {"D0": 9, "D1": 8},
                                "processed_tokens_per_arm": None,
                            },
                        },
                    },
                },
            )

            result = validate_matched_pair(root)

            self.assertEqual(result["processed_token_relative_delta"], 0.0)
            self.assertEqual(result["corpora"]["D0"]["assistant_tokens"], 10)
            self.assertEqual(result["corpora"]["D1"]["assistant_tokens"], 10)

    def test_checksum_corruption_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "D0"
            _write_corpus(root, "D0")
            token_path = root / "token_ids_00000.npy"
            payload = bytearray(token_path.read_bytes())
            payload[0] ^= 1
            token_path.write_bytes(payload)

            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                validate_transfer_corpus(root, expected_arm="D0")

    def test_noncanonical_bool_byte_fails_even_when_declared(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "D0"
            _write_corpus(root, "D0", mask_byte=2)

            with self.assertRaisesRegex(ValueError, "non-canonical bytes"):
                validate_transfer_corpus(root, expected_arm="D0")

    def test_loose_pair_tolerance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            d0 = _write_corpus(root / "D0", "D0")
            d1 = _write_corpus(root / "D1", "D1")
            _write_pair(root, d0, d1, tolerance=0.01)

            with self.assertRaisesRegex(ValueError, "looser processed-token tolerance"):
                validate_matched_pair(root)

    def test_prompt_mask_exposure_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "D0"
            manifest = _write_corpus(root, "D0")
            mask_path = root / "labels_mask_00000.npy"
            payload = bytearray(mask_path.read_bytes())
            payload[0] = 1
            mask_path.write_bytes(payload)

            # Refresh only the declared file identity and enclosing manifest so
            # validation reaches the semantic prompt-mask check.
            for descriptor in manifest["files"]:
                if descriptor["path"] == mask_path.name:
                    descriptor["sha256"] = sha256_file(mask_path)
            manifest.pop("content_id")
            write_manifest_atomic(root / "manifest.json", manifest)

            with self.assertRaisesRegex(ValueError, "exposes prompt tokens"):
                validate_transfer_corpus(root, expected_arm="D0")


if __name__ == "__main__":
    unittest.main()
