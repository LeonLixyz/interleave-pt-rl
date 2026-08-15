from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from transfer_build import (
    cap_and_deduplicate_candidates,
    enforce_processed_token_tolerance,
    match_equal_assistant_tokens,
    prompt_map,
    require_math_verify_version,
    require_scipy_version,
    validate_generation_artifact_directory,
    validate_paired_generation_records,
    verify_completion_detailed,
    verify_generation_records,
    write_training_corpus,
)
from transfer_generate import (
    SamplingConfig,
    generate_records_pure,
    make_prompt_record,
    read_jsonl,
    write_jsonl_atomic,
    write_manifest_atomic,
)


def _prompt(index: int, token_ids: list[int]) -> dict:
    return make_prompt_record(
        dataset_sha256="b" * 64,
        split="train",
        dataset_index=index,
        data_source="toy",
        messages=[{"role": "user", "content": f"problem {index}"}],
        prompt_token_ids=token_ids,
        ground_truth=str(index),
        model_difficulty={"m": 0},
        difficulty_model="m",
    )


def _generate(prompts, arm: str, response_by_uid: dict[str, list[int]]):
    config = SamplingConfig(samples_per_prompt=1, base_seed=5, max_new_tokens=16)

    def sampler(prompt_ids, seed, _config):
        prompt = next(p for p in prompts if p["prompt_token_ids"] == list(prompt_ids))
        token_ids = response_by_uid[prompt["problem_uid"]]
        return {
            "token_ids": token_ids,
            "text": "correct",
            "finish_reason": "stop",
            "stop_reason": None,
        }

    return generate_records_pure(
        prompts,
        arm=arm,
        sampling_config=config,
        model_bundle_sha256=f"model-{arm}",
        tokenizer_sha256="shared-tokenizer",
        sampler=sampler,
    )


class TransferBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prompts = [_prompt(1, [10, 11]), _prompt(2, [12])]
        self.prompts_by_uid = prompt_map(self.prompts)

    def test_verifier_statuses_do_not_collapse_failures_into_incorrect(self) -> None:
        correct = verify_completion_detailed("x", "x", runner=lambda gold, out: 1.0)
        incorrect = verify_completion_detailed("x", "y", runner=lambda gold, out: 0.0)

        def timeout(_gold, _out):
            raise TimeoutError("slow")

        def broken(_gold, _out):
            raise ValueError("bad parse")

        timed_out = verify_completion_detailed("x", "x", runner=timeout)
        error = verify_completion_detailed("x", "x", runner=broken)
        self.assertEqual(correct["status"], "correct")
        self.assertEqual(incorrect["status"], "incorrect")
        self.assertEqual(timed_out["status"], "timeout")
        self.assertIsNone(timed_out["score"])
        self.assertEqual(error["status"], "error")
        self.assertEqual(error["error_type"], "ValueError")

    def test_math_verify_version_guard_fails_closed(self) -> None:
        self.assertEqual(
            require_math_verify_version(version_getter=lambda: "0.5.2"), "0.5.2"
        )
        with self.assertRaisesRegex(RuntimeError, "required '0.5.2', found '0.5.3'"):
            require_math_verify_version(version_getter=lambda: "0.5.3")
        with self.assertRaisesRegex(RuntimeError, "found None"):
            require_math_verify_version(version_getter=lambda: None)

    def test_milp_version_guard_fails_closed(self) -> None:
        self.assertEqual(
            require_scipy_version(version_getter=lambda: "1.17.1"), "1.17.1"
        )
        with self.assertRaisesRegex(RuntimeError, "required '1.17.1', found '1.18.0'"):
            require_scipy_version(version_getter=lambda: "1.18.0")

    def test_paired_provenance_rejects_seed_or_prompt_drift(self) -> None:
        responses = {prompt["problem_uid"]: [20] for prompt in self.prompts}
        d0 = _generate(self.prompts, "D0", responses)
        d1 = _generate(self.prompts, "D1", responses)
        summary = validate_paired_generation_records(d0, d1)
        self.assertEqual(summary["request_count_per_arm"], 2)
        d1[0]["sample_seed"] += 1
        with self.assertRaisesRegex(ValueError, "sample_seed"):
            validate_paired_generation_records(d0, d1)

    def test_generation_manifest_checks_every_part(self) -> None:
        responses = {prompt["problem_uid"]: [20] for prompt in self.prompts}
        records = _generate(self.prompts, "D0", responses)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            part = write_jsonl_atomic(root / "part-00000.jsonl", records)
            write_manifest_atomic(
                root / "manifest.json",
                {
                    "artifact_kind": "fixed_policy_generations",
                    "arm": "D0",
                    "prompt_manifest_sha256": "prompt-hash",
                    "generation_count": len(records),
                    "parts": [part],
                },
            )
            descriptor = validate_generation_artifact_directory(
                root,
                expected_arm="D0",
                expected_prompt_manifest_sha256="prompt-hash",
            )
            self.assertTrue(descriptor["manifest_verified"])
            self.assertEqual(descriptor["rows"], 2)
            with open(root / "part-00000.jsonl", "ab") as handle:
                handle.write(b" ")
            with self.assertRaisesRegex(ValueError, "byte-count"):
                validate_generation_artifact_directory(
                    root,
                    expected_arm="D0",
                    expected_prompt_manifest_sha256="prompt-hash",
                )

    def test_dedup_and_semantic_prompt_cap(self) -> None:
        prompt = self.prompts[0]
        config = SamplingConfig(samples_per_prompt=3, base_seed=4, max_new_tokens=8)

        def sampler(_prompt_ids, _seed, _config):
            return {"token_ids": [20], "text": "correct"}

        raw = generate_records_pure(
            [prompt],
            arm="D0",
            sampling_config=config,
            model_bundle_sha256="m",
            tokenizer_sha256="t",
            sampler=sampler,
        )
        verified = verify_generation_records(
            raw, self.prompts_by_uid, runner=lambda _gold, _output: 1.0
        )
        candidates, stats = cap_and_deduplicate_candidates(
            verified, self.prompts_by_uid, per_prompt_cap=1, eos_token_id=99
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(stats["duplicate_outputs"], 2)
        self.assertEqual(candidates[0]["assistant_token_count"], 2)

    def test_preregistered_default_retains_at_most_eight_per_semantic_prompt(self) -> None:
        prompt = self.prompts[0]
        config = SamplingConfig(samples_per_prompt=10, base_seed=8, max_new_tokens=8)

        def sampler(_prompt_ids, seed, _config):
            token = 20 + seed % 10_000
            return {"token_ids": [token], "text": f"correct-{token}"}

        raw = generate_records_pure(
            [prompt],
            arm="D0",
            sampling_config=config,
            model_bundle_sha256="m",
            tokenizer_sha256="t",
            sampler=sampler,
        )
        verified = verify_generation_records(
            raw, self.prompts_by_uid, runner=lambda _gold, _output: 1.0
        )
        candidates, stats = cap_and_deduplicate_candidates(
            verified, self.prompts_by_uid, eos_token_id=100_257
        )
        self.assertEqual(len(candidates), 8)
        self.assertEqual(stats["over_cap"], 2)

    def test_milp_matches_count_and_assistant_tokens(self) -> None:
        candidates = []
        for arm, lengths in (("D0", [2, 3]), ("D1", [1, 4])):
            for index, length in enumerate(lengths):
                candidates.append(
                    {
                        "arm": arm,
                        "candidate_id": f"{arm}-{index}",
                        "difficulty_bin": "0",
                        "dedup_group": f"{arm}-problem-{index}",
                        "assistant_token_count": length,
                        "processed_token_count": length + index + 1,
                    }
                )
        selected, summary = match_equal_assistant_tokens(candidates, max_states_per_arm_and_stratum=100)
        self.assertEqual(len(selected["D0"]), 2)
        self.assertEqual(len(selected["D1"]), 2)
        self.assertEqual(summary["assistant_tokens_per_arm"], 5)
        self.assertEqual(summary["strata"]["0"]["document_count_per_arm"], 2)
        self.assertEqual(summary["strata"]["0"]["processed_tokens_per_arm"], 8)
        self.assertEqual(summary["processed_token_relative_delta"], 0.0)
        self.assertEqual(summary["max_processed_token_relative_delta"], 0.001)

    def test_current_style_processed_token_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"D0=3,438, D1=3,461, relative_delta=0\.006690 > 0\.001000"
        ):
            enforce_processed_token_tolerance({"D0": 3438, "D1": 3461})
        self.assertAlmostEqual(
            enforce_processed_token_tolerance(
                {"D0": 3438, "D1": 3461}, max_relative_delta=0.01
            ),
            23 / 3438,
        )

        aggregate_candidates = [
            {
                "arm": "D0",
                "candidate_id": "D0-current",
                "difficulty_bin": "0",
                "dedup_group": "D0-current",
                "assistant_token_count": 2414,
                "processed_token_count": 3438,
            },
            {
                "arm": "D1",
                "candidate_id": "D1-current",
                "difficulty_bin": "0",
                "dedup_group": "D1-current",
                "assistant_token_count": 2414,
                "processed_token_count": 3461,
            },
        ]
        with self.assertRaisesRegex(ValueError, "selected no positive D0/D1 corpus"):
            match_equal_assistant_tokens(aggregate_candidates, max_states_per_arm_and_stratum=10)

    def test_milp_uses_global_exact_match_when_per_stratum_exact_is_impossible(self) -> None:
        candidates = [
            {
                "arm": "D0",
                "candidate_id": "D0-a",
                "difficulty_bin": "a",
                "dedup_group": "D0-a",
                "assistant_token_count": 10,
                "processed_token_count": 11,
            },
            {
                "arm": "D0",
                "candidate_id": "D0-b",
                "difficulty_bin": "b",
                "dedup_group": "D0-b",
                "assistant_token_count": 1,
                "processed_token_count": 2,
            },
            {
                "arm": "D1",
                "candidate_id": "D1-a",
                "difficulty_bin": "a",
                "dedup_group": "D1-a",
                "assistant_token_count": 9,
                "processed_token_count": 10,
            },
            {
                "arm": "D1",
                "candidate_id": "D1-b",
                "difficulty_bin": "b",
                "dedup_group": "D1-b",
                "assistant_token_count": 2,
                "processed_token_count": 3,
            },
        ]
        selected, summary = match_equal_assistant_tokens(candidates)
        self.assertEqual({arm: len(rows) for arm, rows in selected.items()}, {"D0": 2, "D1": 2})
        self.assertEqual(summary["assistant_tokens_per_arm"], 11)
        self.assertFalse(summary["exact_assistant_tokens_per_stratum"])
        self.assertEqual(summary["stratum_assistant_abs_delta"], 2)
        self.assertEqual(summary["strata"]["a"]["document_count_per_arm"], 1)
        self.assertEqual(summary["strata"]["b"]["document_count_per_arm"], 1)
        self.assertEqual(summary["processed_token_relative_delta"], 0.0)
        self.assertTrue(
            any(phase.get("proven_infeasible") for phase in summary["solver_phases"])
        )

    def test_milp_aggregation_and_seeded_selection_are_order_independent(self) -> None:
        candidates = []
        for arm in ("D0", "D1"):
            for index in range(200):
                candidates.append(
                    {
                        "arm": arm,
                        "candidate_id": f"{arm}-{index:04d}",
                        "difficulty_bin": "0",
                        "dedup_group": f"{arm}-problem-{index:04d}",
                        "assistant_token_count": 5,
                        "processed_token_count": 7,
                    }
                )
        selected_a, summary_a = match_equal_assistant_tokens(
            candidates, assistant_token_cap=50, selection_seed=123
        )
        selected_b, summary_b = match_equal_assistant_tokens(
            reversed(candidates), assistant_token_cap=50, selection_seed=123
        )
        ids_a = {
            arm: [row["candidate_id"] for row in rows]
            for arm, rows in selected_a.items()
        }
        ids_b = {
            arm: [row["candidate_id"] for row in rows]
            for arm, rows in selected_b.items()
        }
        self.assertEqual(ids_a, ids_b)
        self.assertEqual(summary_a["assistant_tokens_per_arm"], 50)
        self.assertEqual(summary_a["selected_documents"], {"D0": 10, "D1": 10})
        self.assertEqual(summary_a["candidate_count"], 400)
        self.assertEqual(summary_a["aggregated_bucket_count"], 2)
        self.assertEqual(summary_a["matching_input_sha256"], summary_b["matching_input_sha256"])
        self.assertEqual(
            summary_a["selected_candidate_ids_sha256"],
            summary_b["selected_candidate_ids_sha256"],
        )

    def test_raw_corpus_has_assistant_and_eos_only_masks(self) -> None:
        responses = {
            self.prompts[0]["problem_uid"]: [20, 21],
            self.prompts[1]["problem_uid"]: [22, 99],
        }
        raw = _generate(self.prompts, "D0", responses)
        verified = verify_generation_records(
            raw, self.prompts_by_uid, runner=lambda _gold, _output: 1.0
        )
        candidates, _ = cap_and_deduplicate_candidates(
            verified, self.prompts_by_uid, per_prompt_cap=1, eos_token_id=99
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first"
            second = Path(temp_dir) / "second"
            manifest_1 = write_training_corpus(
                candidates,
                self.prompts_by_uid,
                arm="D0",
                output_dir=first,
                eos_token_id=99,
                vocab_size=128,
                prompt_manifest_sha256="p",
            )
            manifest_2 = write_training_corpus(
                candidates,
                self.prompts_by_uid,
                arm="D0",
                output_dir=second,
                eos_token_id=99,
                vocab_size=128,
                prompt_manifest_sha256="p",
            )
            tokens = np.fromfile(first / "token_ids_00000.npy", dtype=np.uint32)
            masks = np.fromfile(first / "labels_mask_00000.npy", dtype=np.bool_)
            selected_rows = list(read_jsonl(first / "selected.jsonl"))
            expected_by_uid = {
                self.prompts[0]["problem_uid"]: (
                    [10, 11, 20, 21, 99],
                    [False, False, True, True, True],
                ),
                self.prompts[1]["problem_uid"]: (
                    [12, 22, 99],
                    [False, True, True],
                ),
            }
            for metadata in selected_rows:
                start, end = metadata["token_offset_start"], metadata["token_offset_end"]
                expected_tokens, expected_masks = expected_by_uid[metadata["problem_uid"]]
                self.assertEqual(tokens[start:end].tolist(), expected_tokens)
                self.assertEqual(masks[start:end].tolist(), expected_masks)
            self.assertEqual(manifest_1["loss_bearing_assistant_tokens"], 5)
            self.assertEqual(manifest_1["processed_tokens"], 8)
            self.assertEqual(manifest_1["content_id"], manifest_2["content_id"])
            self.assertNotEqual((first / "token_ids_00000.npy").read_bytes()[:6], b"\x93NUMPY")


if __name__ == "__main__":
    unittest.main()
