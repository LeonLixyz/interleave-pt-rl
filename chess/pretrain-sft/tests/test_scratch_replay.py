from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from training.interleaved_data import (
    PAD_RECORD,
    SAMPLE_PRETRAIN,
    SAMPLE_SFT,
    build_leg_manifests,
    build_pretrain_selection,
    build_sft_cache,
    build_source_manifest,
)
from training.interleaved_hf_trainer import ExactArcCosine, resolve_arc_steps
from training.positive_replay import canonical_json, token_ids_sha256
from training.scratch_replay import (
    REPLAY_CODE_BASE,
    SAMPLE_POSITIVE_REPLAY,
    ScratchReplayManifest,
    build_scratch_replay_manifest,
    create_scratch_replay_dataloader,
    replay_row_to_aligned_sample,
)


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


def _sft_row(suffix: str):
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


def _replay_row(index: int) -> dict:
    prompt = [1, 2]
    response = [3, 4, 5 + index]
    return {
        "schema_version": 1,
        "prompt": f"prompt-{index}",
        "response": f"response-{index}",
        "prompt_token_ids": prompt,
        "response_token_ids": response,
        "response_loss_mask": [1, 0, 1],
        "token_ids_sha256": token_ids_sha256(prompt, response),
        "prompt_response_sha256": str(index),
        "group_index": index,
        "sample_index": index,
    }


class ScratchReplayTests(unittest.TestCase):
    def _artifacts(self, root: Path):
        source_root = root / "source"
        source_root.mkdir()
        np.save(source_root / "raw.0.npy", np.arange(30, dtype=np.int32))
        source_manifest = root / "source.json"
        build_source_manifest(source_root, source_manifest)
        selection = root / "selection.json"
        build_pretrain_selection(
            source_manifest,
            selection,
            target_tokens=18,
            seed=7,
        )

        tokenizer = TinyTokenizer()
        sft_file = root / "sft.json"
        sft_file.write_text(
            json.dumps({"results": [_sft_row("c1"), _sft_row("a1")]}),
            encoding="utf-8",
        )
        sft_cache = root / "sft_cache"
        build_sft_cache(
            [sft_file],
            tokenizer,
            sft_cache,
            sequence_length=32,
            expected_rows=2,
        )
        legs = build_leg_manifests(
            selection,
            sft_cache,
            root / "legs",
            source_manifest_path=source_manifest,
            sequence_length=32,
            world_size=1,
            local_batch_size=4,
            split_seed=11,
            p1_seed=12,
            p2_seed=13,
            canary_world_size=1,
            canary_local_batch_size=2,
            canary_total_steps=1,
        )

        replay = root / "positive.jsonl"
        encoded = "".join(canonical_json(_replay_row(i)) + "\n" for i in range(6))
        replay.write_text(encoded, encoding="utf-8")
        replay_manifest = root / "positive.manifest.json"
        replay_manifest.write_text(
            json.dumps(
                {
                    "output": {
                        "sha256": hashlib.sha256(
                            encoded.encode("utf-8")
                        ).hexdigest(),
                        "rows": 6,
                    }
                }
            ),
            encoding="utf-8",
        )
        return {
            "source_root": source_root,
            "source_manifest": source_manifest,
            "selection": selection,
            "sft_cache": sft_cache,
            "p2": legs["p2"],
            "replay": replay,
            "replay_manifest": replay_manifest,
            "tokenizer": tokenizer,
        }

    def test_unified_order_keeps_every_record_and_extends_only_at_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = self._artifacts(root)
            manifest = build_scratch_replay_manifest(
                p2_manifest_path=artifacts["p2"],
                replay_path=artifacts["replay"],
                replay_manifest_path=artifacts["replay_manifest"],
                output_dir=root / "scratch",
                shuffle_seed=44,
                model_init_seed=123,
            )
            # P2 contains one packed PT + one SFT + two baseline sentinels:
            # one optimizer step. Adding six replay rows makes eight real
            # records, two optimizer steps, and a one-step LR floor tail.
            self.assertEqual(manifest.pretrain_records, 1)
            self.assertEqual(manifest.sft_records, 1)
            self.assertEqual(manifest.replay_records, 6)
            self.assertEqual(manifest.padding_records, 0)
            self.assertEqual(manifest.baseline_cosine_steps, 1)
            self.assertEqual(manifest.floor_tail_steps, 1)
            self.assertEqual(manifest.total_steps, 2)
            self.assertEqual(manifest.model_init_seed, 123)

            order = np.load(manifest.order_path, allow_pickle=False)
            self.assertEqual(len(order), 8)
            self.assertEqual(np.count_nonzero(order == PAD_RECORD), 0)
            self.assertEqual(np.count_nonzero(order >= 0), 1)
            self.assertEqual(
                np.count_nonzero(
                    (order < 0) & (order > REPLAY_CODE_BASE)
                ),
                1,
            )
            replay_indices = sorted(
                REPLAY_CODE_BASE - int(code)
                for code in order
                if int(code) <= REPLAY_CODE_BASE
                and int(code) != int(PAD_RECORD)
            )
            self.assertEqual(replay_indices, list(range(6)))

            # Idempotence authenticates all input hashes and returns the same
            # immutable artifact rather than reshuffling it.
            again = build_scratch_replay_manifest(
                p2_manifest_path=artifacts["p2"],
                replay_path=artifacts["replay"],
                replay_manifest_path=artifacts["replay_manifest"],
                output_dir=root / "scratch",
                shuffle_seed=44,
                model_init_seed=123,
            )
            self.assertEqual(again.metadata_hash, manifest.metadata_hash)
            self.assertEqual(
                ScratchReplayManifest.load(manifest.metadata_path).order_sha256,
                manifest.order_sha256,
            )

    def test_exact_replay_alignment_and_unified_stream_types(self):
        row = _replay_row(0)
        sample = replay_row_to_aligned_sample(row, record_id=9)
        self.assertEqual(sample["input_ids"].tolist(), [1, 2, 3, 4])
        self.assertEqual(sample["labels"].tolist(), [-100, 3, -100, 5])
        self.assertEqual(sample["valid_targets"], 2)
        self.assertEqual(sample["sample_type"], SAMPLE_POSITIVE_REPLAY)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = self._artifacts(root)
            manifest = build_scratch_replay_manifest(
                p2_manifest_path=artifacts["p2"],
                replay_path=artifacts["replay"],
                replay_manifest_path=artifacts["replay_manifest"],
                output_dir=root / "scratch",
            )
            stream = create_scratch_replay_dataloader(
                source_root=artifacts["source_root"],
                source_manifest_path=artifacts["source_manifest"],
                selection_manifest_path=artifacts["selection"],
                sft_cache_dir=artifacts["sft_cache"],
                leg_manifest_path=manifest.metadata_path,
                pad_token_id=artifacts["tokenizer"].pad_id(),
                bos_token_id=artifacts["tokenizer"].bos_id(),
                rank=0,
                world_size=1,
                local_batch_size=4,
            )
            seen_types = []
            for batch in stream:
                self.assertEqual(tuple(batch["input_ids"].shape), (4, 32))
                seen_types.extend(batch["sample_type"].tolist())
                stream.commit_step()
            self.assertEqual(seen_types.count(SAMPLE_PRETRAIN), 1)
            self.assertEqual(seen_types.count(SAMPLE_SFT), 1)
            self.assertEqual(seen_types.count(SAMPLE_POSITIVE_REPLAY), 6)
            self.assertEqual(stream.cursor, 2)

    def test_cursor_resume_is_strict_and_rank_agnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = self._artifacts(root)
            manifest = build_scratch_replay_manifest(
                p2_manifest_path=artifacts["p2"],
                replay_path=artifacts["replay"],
                replay_manifest_path=artifacts["replay_manifest"],
                output_dir=root / "scratch",
            )

            def open_stream(cursor):
                return create_scratch_replay_dataloader(
                    source_root=artifacts["source_root"],
                    source_manifest_path=artifacts["source_manifest"],
                    selection_manifest_path=artifacts["selection"],
                    sft_cache_dir=artifacts["sft_cache"],
                    leg_manifest_path=manifest.metadata_path,
                    pad_token_id=artifacts["tokenizer"].pad_id(),
                    bos_token_id=artifacts["tokenizer"].bos_id(),
                    rank=0,
                    world_size=1,
                    local_batch_size=4,
                    start_cursor=cursor,
                )

            first = open_stream(0)
            batch = next(iter(first))
            self.assertEqual(batch["cursor_start"], 0)
            with self.assertRaisesRegex(RuntimeError, "uncommitted"):
                first.state_dict()
            first.commit_step()
            state = first.state_dict()
            self.assertEqual(state["cursor"], 1)
            self.assertNotIn("rank", state)

            resumed = open_stream(1)
            resumed.load_state_dict(state)
            resumed_batch = next(iter(resumed))
            self.assertEqual(resumed_batch["cursor_start"], 1)
            resumed.commit_step()
            self.assertEqual(resumed.cursor, 2)

            wrong = dict(state)
            wrong["replay_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "replay_sha256"):
                open_stream(1).load_state_dict(wrong)

    def test_scheduler_preserves_cosine_and_uses_exact_floor_tail(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        scheduler = ExactArcCosine(
            optimizer,
            arc_steps=[4],
            peak_lr=1e-3,
            min_lr=1e-5,
            warmup_ratio=0.25,
            floor_tail_steps=2,
        )
        self.assertEqual(scheduler.cosine_steps, 4)
        self.assertEqual(scheduler.total_steps, 6)
        self.assertAlmostEqual(scheduler.lr_for_update(3), 1e-5)
        self.assertAlmostEqual(scheduler.lr_for_update(4), 1e-5)
        self.assertAlmostEqual(scheduler.lr_for_update(5), 1e-5)
        self.assertEqual(
            resolve_arc_steps(
                {
                    "arc_steps": [4],
                    "floor_tail_steps": 2,
                    "total_steps": 6,
                }
            ),
            (4,),
        )
        for _ in range(6):
            scheduler.step()
        with self.assertRaisesRegex(RuntimeError, "past"):
            scheduler.step()

        legacy = ExactArcCosine(
            torch.optim.AdamW(
                [torch.nn.Parameter(torch.tensor(2.0))],
                lr=1e-3,
            ),
            arc_steps=[4],
            peak_lr=1e-3,
            min_lr=1e-5,
            warmup_ratio=0.25,
        )
        legacy_state = legacy.state_dict()
        legacy_state.pop("floor_tail_steps")
        legacy.load_state_dict(legacy_state)


if __name__ == "__main__":
    unittest.main()
