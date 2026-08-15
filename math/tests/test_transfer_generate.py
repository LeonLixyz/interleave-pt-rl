from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from transfer_generate import (
    SamplingConfig,
    derive_sample_seed,
    freeze_prompt_rows,
    generate_records_pure,
    make_prompt_record,
    normalize_problem_text,
    prompt_dedup_group,
    read_jsonl,
    select_stratified_smoke_prompts,
    sha256_file,
    split_semantic_prompt_pool,
    validate_generation_record,
    write_jsonl_atomic,
    write_split_prompt_artifacts,
)


class TransferGenerateTests(unittest.TestCase):
    def test_preregistered_sampling_default_is_sixteen(self) -> None:
        self.assertEqual(SamplingConfig().samples_per_prompt, 16)

    def test_seed_derivation_is_stable_and_request_specific(self) -> None:
        first = derive_sample_seed(17, "problem-a", 0)
        self.assertEqual(first, derive_sample_seed(17, "problem-a", 0))
        self.assertNotEqual(first, derive_sample_seed(17, "problem-a", 1))
        self.assertNotEqual(first, derive_sample_seed(18, "problem-a", 0))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**31 - 1)

    def test_problem_normalization_and_dedup_group(self) -> None:
        left = [{"role": "user", "content": "  Solve\n  x + 1 = 2  "}]
        right = [{"role": "user", "content": "Solve x + 1 = 2"}]
        self.assertEqual(normalize_problem_text(left[0]["content"]), right[0]["content"])
        self.assertEqual(prompt_dedup_group(left), prompt_dedup_group(right))

    def test_freeze_prompt_rows_filters_overlong_and_sorts(self) -> None:
        rows = [
            {
                "data_source": "source-b",
                "prompt": [{"role": "user", "content": "bb"}],
                "reward_model": {"ground_truth": "2"},
                "extra_info": {"index": 9, "model_difficulty": {"m": 2}},
            },
            {
                "data_source": "source-a",
                "prompt": [{"role": "user", "content": "a"}],
                "reward_model": {"ground_truth": "1"},
                "extra_info": {"index": 3, "model_difficulty": {"m": 0}},
            },
            {
                "data_source": "source-c",
                "prompt": [{"role": "user", "content": "too-long"}],
                "reward_model": {"ground_truth": "3"},
                "extra_info": {"index": 11, "model_difficulty": None},
            },
        ]

        def tokenize(messages):
            return list(range(len(messages[0]["content"])))

        frozen, stats = freeze_prompt_rows(
            rows,
            dataset_sha256="d" * 64,
            split="train",
            tokenize_prompt=tokenize,
            max_prompt_tokens=3,
            difficulty_model="m",
        )
        self.assertEqual([row["dataset_index"] for row in frozen], [3, 9])
        self.assertEqual([row["difficulty_bin"] for row in frozen], ["0", "2-3"])
        self.assertEqual(
            stats, {"input_rows": 3, "kept_rows": 2, "overlong_rows": 1}
        )

    def test_fake_generation_persists_exact_ids_and_is_byte_reproducible(self) -> None:
        prompt = make_prompt_record(
            dataset_sha256="a" * 64,
            split="train",
            dataset_index=7,
            data_source="toy",
            messages=[{"role": "user", "content": "1+1?"}],
            prompt_token_ids=[10, 11, 12],
            ground_truth="2",
            model_difficulty={"m": 0},
            difficulty_model="m",
        )
        config = SamplingConfig(samples_per_prompt=2, base_seed=123, max_new_tokens=8)

        def sampler(prompt_ids, seed, _config):
            self.assertEqual(list(prompt_ids), [10, 11, 12])
            return {
                "token_ids": [20, seed % 97],
                "text": f"seed={seed}",
                "finish_reason": "stop",
                "stop_reason": 99,
            }

        records = generate_records_pure(
            [prompt],
            arm="D0",
            sampling_config=config,
            model_bundle_sha256="model-hash",
            tokenizer_sha256="tokenizer-hash",
            sampler=sampler,
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["prompt_token_ids"], [10, 11, 12])
        self.assertEqual(records[0]["response_token_ids"][0], 20)
        for record in records:
            validate_generation_record(record)

        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.jsonl"
            second = Path(temp_dir) / "second.jsonl"
            write_jsonl_atomic(first, records)
            write_jsonl_atomic(second, records)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(sha256_file(first), sha256_file(second))
            self.assertEqual(list(read_jsonl(first)), records)

    def test_semantic_pool_split_is_disjoint_deduplicated_and_deterministic(self) -> None:
        prompts = []
        for group_index in range(12):
            copies = 2 if group_index % 2 == 0 else 1
            for copy_index in range(copies):
                prompts.append(
                    make_prompt_record(
                        dataset_sha256="c" * 64,
                        split="train",
                        dataset_index=group_index * 10 + copy_index,
                        data_source="toy",
                        messages=[
                            {"role": "user", "content": f"semantic problem {group_index}"}
                        ],
                        prompt_token_ids=[group_index + 1, copy_index + 50],
                        ground_truth=str(group_index),
                    )
                )
        pools_a, summary_a = split_semantic_prompt_pool(
            prompts, split_seed=20260710, generation_groups=7, dev_groups=3
        )
        pools_b, summary_b = split_semantic_prompt_pool(
            reversed(prompts), split_seed=20260710, generation_groups=7, dev_groups=3
        )
        self.assertEqual(summary_a, summary_b)
        self.assertEqual(summary_a["counts"], {"generation": 7, "dev": 3, "reserve": 2})
        self.assertEqual(summary_a["unique_semantic_groups"], 12)
        self.assertEqual(summary_a["dropped_duplicate_rows"], 6)
        for name in pools_a:
            self.assertEqual(
                [row["problem_uid"] for row in pools_a[name]],
                [row["problem_uid"] for row in pools_b[name]],
            )
            self.assertTrue(all(row["pool_split"] == name for row in pools_a[name]))
        group_sets = {
            name: {row["dedup_group"] for row in rows} for name, rows in pools_a.items()
        }
        self.assertFalse(group_sets["generation"] & group_sets["dev"])
        self.assertFalse(group_sets["generation"] & group_sets["reserve"])
        self.assertFalse(group_sets["dev"] & group_sets["reserve"])
        self.assertEqual(sum(len(groups) for groups in group_sets.values()), 12)

        pools_other_seed, _ = split_semantic_prompt_pool(
            prompts, split_seed=20260711, generation_groups=7, dev_groups=3
        )
        ranks_a = {
            row["dedup_group"]: row["pool_split_rank_sha256"]
            for rows in pools_a.values()
            for row in rows
        }
        ranks_other = {
            row["dedup_group"]: row["pool_split_rank_sha256"]
            for rows in pools_other_seed.values()
            for row in rows
        }
        self.assertTrue(all(ranks_a[group] != ranks_other[group] for group in ranks_a))

    def test_semantic_pool_split_rejects_infeasible_request(self) -> None:
        prompt = make_prompt_record(
            dataset_sha256="d" * 64,
            split="train",
            dataset_index=1,
            data_source="toy",
            messages=[{"role": "user", "content": "only group"}],
            prompt_token_ids=[1],
            ground_truth="1",
        )
        with self.assertRaisesRegex(ValueError, "infeasible"):
            split_semantic_prompt_pool(
                [prompt], split_seed=1, generation_groups=1, dev_groups=1
            )

    def test_split_artifact_writer_emits_three_checked_pools(self) -> None:
        prompts = [
            make_prompt_record(
                dataset_sha256="e" * 64,
                split="train",
                dataset_index=index,
                data_source="toy",
                messages=[{"role": "user", "content": f"problem {index}"}],
                prompt_token_ids=[index + 1],
                ground_truth=str(index),
            )
            for index in range(5)
        ]
        pools, summary = split_semantic_prompt_pool(
            prompts, split_seed=42, generation_groups=2, dev_groups=2
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = write_split_prompt_artifacts(
                temp_dir,
                pools,
                split_summary=summary,
                provenance={"dataset_sha256": "e" * 64, "tokenizer_sha256": "t"},
            )
            self.assertEqual(
                set(manifest["artifacts"]), {"generation", "dev", "reserve"}
            )
            self.assertEqual(manifest["split_summary"]["counts"]["reserve"], 1)
            for split_name, expected_rows in (("generation", 2), ("dev", 2), ("reserve", 1)):
                path = Path(temp_dir) / f"{split_name}.jsonl"
                self.assertTrue(path.is_file())
                self.assertEqual(len(list(read_jsonl(path))), expected_rows)
                self.assertEqual(manifest["artifacts"][split_name]["sha256"], sha256_file(path))

    def test_smoke_selector_balances_known_bins_and_is_deterministic(self) -> None:
        difficulty_values = {"0": 0, "1": 1, "2-3": 2, "4+": 4}
        prompts = []
        dataset_index = 0
        for bin_name, value in difficulty_values.items():
            for item_index in range(10):
                row = make_prompt_record(
                    dataset_sha256="f" * 64,
                    split="train",
                    dataset_index=dataset_index,
                    data_source="toy",
                    messages=[
                        {"role": "user", "content": f"{bin_name} problem {item_index}"}
                    ],
                    prompt_token_ids=[dataset_index + 1],
                    ground_truth=str(item_index),
                    model_difficulty={"m": value},
                    difficulty_model="m",
                )
                row["pool_split"] = "dev"
                prompts.append(row)
                dataset_index += 1
        for item_index in range(3):
            row = make_prompt_record(
                dataset_sha256="f" * 64,
                split="train",
                dataset_index=dataset_index,
                data_source="toy",
                messages=[{"role": "user", "content": f"unknown {item_index}"}],
                prompt_token_ids=[dataset_index + 1],
                ground_truth=str(item_index),
            )
            row["pool_split"] = "dev"
            prompts.append(row)
            dataset_index += 1

        selected_a, summary_a = select_stratified_smoke_prompts(
            prompts, count=32, selection_seed=20260710
        )
        selected_b, summary_b = select_stratified_smoke_prompts(
            reversed(prompts), count=32, selection_seed=20260710
        )
        self.assertEqual(summary_a, summary_b)
        self.assertEqual(
            summary_a["selected_bin_counts"], {"0": 8, "1": 8, "2-3": 8, "4+": 8}
        )
        self.assertEqual(summary_a["excluded_unknown_difficulty"], 3)
        self.assertEqual(
            [row["problem_uid"] for row in selected_a],
            [row["problem_uid"] for row in selected_b],
        )
        self.assertTrue(
            all(row["difficulty_bin"] in {"0", "1", "2-3", "4+"} for row in selected_a)
        )

    def test_smoke_selector_redistributes_a_capacity_limited_bin_evenly(self) -> None:
        capacities = {"0": 1, "1": 5, "2-3": 5, "4+": 5}
        values = {"0": 0, "1": 1, "2-3": 2, "4+": 4}
        prompts = []
        dataset_index = 0
        for bin_name, capacity in capacities.items():
            for item_index in range(capacity):
                row = make_prompt_record(
                    dataset_sha256="1" * 64,
                    split="train",
                    dataset_index=dataset_index,
                    data_source="toy",
                    messages=[
                        {"role": "user", "content": f"{bin_name} scarce {item_index}"}
                    ],
                    prompt_token_ids=[dataset_index + 1],
                    ground_truth=str(item_index),
                    model_difficulty={"m": values[bin_name]},
                    difficulty_model="m",
                )
                row["pool_split"] = "dev"
                prompts.append(row)
                dataset_index += 1
        selected, summary = select_stratified_smoke_prompts(
            prompts, count=10, selection_seed=9
        )
        self.assertEqual(len(selected), 10)
        self.assertEqual(
            summary["selected_bin_counts"], {"0": 1, "1": 3, "2-3": 3, "4+": 3}
        )

    def test_smoke_selector_refuses_infeasible_or_non_dev_input(self) -> None:
        row = make_prompt_record(
            dataset_sha256="2" * 64,
            split="train",
            dataset_index=1,
            data_source="toy",
            messages=[{"role": "user", "content": "known"}],
            prompt_token_ids=[1],
            ground_truth="1",
            model_difficulty={"m": 0},
            difficulty_model="m",
        )
        row["pool_split"] = "dev"
        with self.assertRaisesRegex(ValueError, "infeasible"):
            select_stratified_smoke_prompts([row], count=2, selection_seed=1)
        non_dev = dict(row)
        non_dev["pool_split"] = "generation"
        with self.assertRaisesRegex(ValueError, "requires a frozen dev pool"):
            select_stratified_smoke_prompts([non_dev], count=1, selection_seed=1)
        duplicate = dict(row)
        duplicate["problem_uid"] = "different-row-same-semantic-group"
        with self.assertRaisesRegex(ValueError, "duplicate semantic group"):
            select_stratified_smoke_prompts([row, duplicate], count=1, selection_seed=1)


if __name__ == "__main__":
    unittest.main()
