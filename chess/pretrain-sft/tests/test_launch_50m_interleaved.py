from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from modal_scripts import launch_50m_interleaved as launcher


class InterleavedLauncherTests(unittest.TestCase):
    def test_full_corpus_identity_is_pinned(self):
        self.assertEqual(launcher.SOURCE_SHARDS, 47_090)
        self.assertEqual(launcher.SOURCE_LAST_SHARD, 47_089)
        self.assertEqual(launcher.SOURCE_TOKENS, 53_970_293_905)
        self.assertEqual(
            launcher.SOURCE_REVISION,
            "07dd1b7090ca5f0fb05ef624c26b20bff19483c8",
        )
        self.assertEqual(
            launcher.SOURCE_FLAT_MANIFEST_SHA256,
            "07ae91cded540a00e9b6554d1d54ed46310715b7fd68e3520a64b7f5967f99aa",
        )
        self.assertEqual(launcher.SFT_JSON_FILES, 180)
        self.assertEqual(launcher.SFT_ROWS, 77_717)
        self.assertEqual(launcher.SFT_BYTES, 8_416_392_280)

    def test_name_size_manifest_digest_is_order_independent(self):
        entries = [("raw.0001.npy", 20), ("raw.0000.npy", 10)]
        reverse = list(reversed(entries))
        self.assertEqual(
            launcher._canonical_name_size_digest(entries),
            launcher._canonical_name_size_digest(reverse),
        )
        with self.assertRaises(ValueError):
            launcher._canonical_name_size_digest([("bad\nname", 1)])
        with self.assertRaises(ValueError):
            launcher._canonical_name_size_digest([("bad", -1)])

    def test_sft_inventory_excludes_completion_and_hf_cache_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "shard.json").write_text("[]\n", encoding="utf-8")
            marker = root / ".complete.json"
            marker.write_text("{}\n", encoding="utf-8")
            cache = root / ".cache" / "huggingface"
            cache.mkdir(parents=True)
            (cache / "metadata.json").write_text("{}\n", encoding="utf-8")
            with (
                mock.patch.object(launcher, "SFT_SNAPSHOT_DIR", root),
                mock.patch.object(launcher, "SFT_SNAPSHOT_MARKER", marker),
            ):
                self.assertEqual(
                    launcher._sft_snapshot_entries(),
                    [("shard.json", 3)],
                )

    def test_trusted_source_fast_path_requires_exact_verified_inventory(self):
        metadata = {
            "repo": launcher.SOURCE_REPO,
            "revision": launcher.SOURCE_REVISION,
            "path": str(launcher.SOURCE_DIR),
            "shards": launcher.SOURCE_SHARDS,
            "tokens": launcher.SOURCE_TOKENS,
            "bytes": launcher.SOURCE_BYTES,
            "flat_manifest_sha256": launcher.SOURCE_FLAT_MANIFEST_SHA256,
            "npy_dtype": launcher.SOURCE_NPY_DTYPE,
            "npy_header_bytes": launcher.SOURCE_NPY_HEADER_BYTES,
            "header_check_shards": list(
                launcher.SOURCE_HEADER_CHECK_SHARDS
            ),
        }
        self.assertEqual(
            launcher._verified_source_fast_path(metadata),
            {
                "trusted_npy_dtype": "<u4",
                "trusted_npy_header_bytes": 128,
                "expected_total_tokens": launcher.SOURCE_TOKENS,
            },
        )

        wrong = dict(metadata)
        wrong["flat_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "Refusing trusted"):
            launcher._verified_source_fast_path(wrong)

    def test_production_plans_are_8xh200_without_accumulation(self):
        self.assertEqual(launcher.PRODUCTION_GPU_TYPE, "H200")
        self.assertEqual(launcher.PRODUCTION_GPUS, 8)
        self.assertEqual(launcher.PRODUCTION_LOCAL_BATCH, 21)
        self.assertEqual(launcher.PRODUCTION_GRADIENT_ACCUMULATION, 1)
        self.assertEqual(launcher.PRODUCTION_ATTENTION_BACKEND, "sdpa")
        self.assertEqual(launcher.PRODUCTION_TORCH_COMPILE_MODE, "none")
        self.assertEqual(launcher.PRODUCTION_DATA_WORKERS, 8)
        self.assertAlmostEqual(
            launcher.P1_SFT_LOSS_WEIGHT, 190.189290837
        )
        self.assertAlmostEqual(
            launcher.P2_SFT_LOSS_WEIGHT, 190.889566377
        )
        self.assertAlmostEqual(
            launcher.MONOLITHIC_SFT_LOSS_WEIGHT, 190.538785189
        )

        p1 = launcher._fixed_plan("p1")
        exp2 = launcher._fixed_plan("exp2")
        self.assertEqual(p1.total_steps, 9_920)
        self.assertEqual(p1.arc_steps, (9_920,))
        self.assertEqual(p1.manifest_leg, "p1")
        self.assertEqual(exp2.total_steps, 19_840)
        self.assertEqual(exp2.arc_steps, (19_840,))
        self.assertEqual(exp2.manifest_leg, "p1+p2")
        self.assertEqual(
            exp2.manifest_metadata,
            str(launcher.EXP2_METADATA_PATH),
        )
        self.assertNotEqual(p1.output_dir, exp2.output_dir)
        self.assertEqual(
            p1.sft_loss_weight, launcher.P1_SFT_LOSS_WEIGHT
        )
        self.assertEqual(
            exp2.sft_loss_weight,
            launcher.MONOLITHIC_SFT_LOSS_WEIGHT,
        )

    def test_v2r2_plans_are_version_scoped_and_gate_selected(self):
        p1 = launcher._v2r2_plan("v2r2-p1", selected_weight=32)
        canary = launcher._v2r2_plan(
            "v2r2-exp2-monolithic-canary", selected_weight=32
        )
        exp2 = launcher._v2r2_plan("v2r2-exp2", selected_weight=32)
        self.assertTrue(
            p1.output_dir.startswith(str(launcher.V2R2_CHECKPOINT_ROOT))
        )
        self.assertNotIn(launcher.EXPERIMENT_VERSION, p1.output_dir)
        self.assertEqual(
            p1.experiment_version, launcher.V2R2_EXPERIMENT_VERSION
        )
        self.assertEqual(p1.sft_loss_weight, 32)
        self.assertEqual(p1.manifest_leg, "p1")
        self.assertEqual(canary.manifest_leg, "p1+p2")
        self.assertEqual(canary.total_steps, 19_840)
        self.assertEqual(canary.arc_steps, (19_840,))
        self.assertEqual(canary.max_steps, 2_000)
        self.assertTrue(canary.structure_canary)
        self.assertEqual(exp2.total_steps, 19_840)
        self.assertIsNone(exp2.max_steps)
        with self.assertRaisesRegex(ValueError, "ineligible"):
            launcher._v2r2_plan("v2r2-p1", selected_weight=64)

    def test_v2r2_plan_overrides_bind_new_version_and_current_source(self):
        plan = launcher._v2r2_plan("v2r2-p1", selected_weight=96)
        overrides = launcher._plan_overrides(plan, "a" * 64)
        self.assertIn(
            "provenance.experiment_version="
            f"{launcher.V2R2_EXPERIMENT_VERSION}",
            overrides,
        )
        self.assertIn(
            f"provenance.source_tree_sha256={launcher.SOURCE_TREE_SHA256}",
            overrides,
        )
        self.assertIn("training.sft_loss_weight=96.0", overrides)

    def test_v2r3_exact_continuous_trajectory_inventory(self):
        expected = {
            190.189290837: (
                9_920,
                (1_000, 2_000, 4_000, 6_000, 8_000, 9_920),
            ),
            256.0: (2_000, (1_000, 2_000)),
            384.0: (2_000, (1_000, 2_000)),
            768.0: (2_000, (1_000, 2_000)),
        }
        self.assertEqual(launcher.V2R3_SNAPSHOT_COUNT, 12)
        for weight, (max_steps, snapshot_steps) in expected.items():
            plan = launcher._v2r3_plan(weight)
            self.assertTrue(plan.diagnostic_only)
            self.assertFalse(plan.structure_canary)
            self.assertEqual(plan.manifest_leg, "p1")
            self.assertEqual(plan.total_steps, 9_920)
            self.assertEqual(plan.arc_steps, (9_920,))
            self.assertEqual(plan.max_steps, max_steps)
            self.assertEqual(plan.snapshot_steps, snapshot_steps)
            self.assertEqual(plan.sft_loss_weight, weight)
            self.assertEqual(
                plan.experiment_version,
                launcher.V2R3_EXPERIMENT_VERSION,
            )
            overrides = launcher._plan_overrides(plan, "a" * 64)
            self.assertIn("training.seed=42", overrides)
            self.assertIn("training.optimizer.lr=1e-3", overrides)
            self.assertIn(
                "training.scheduler.warmup_ratio=0.05", overrides
            )
            self.assertIn(
                "training.scheduler.eta_min=1e-5", overrides
            )
            self.assertIn("training.save_interval=0", overrides)
            self.assertIn("training.export_interval=0", overrides)
            self.assertIn(
                f"training.snapshot_steps={list(snapshot_steps)}",
                overrides,
            )
        with self.assertRaisesRegex(ValueError, "one of"):
            launcher._v2r3_plan(32.0)

    def test_v2r3_rollout_names_bind_weight_step_and_seed(self):
        first = launcher._v2r3_rollout_run_name(
            190.189290837, 1_000
        )
        second = launcher._v2r3_rollout_run_name(
            190.189290837, 2_000
        )
        self.assertNotEqual(first, second)
        self.assertTrue(first.endswith("-s1000-seed42-rollout"))
        with self.assertRaisesRegex(ValueError, "not declared"):
            launcher._v2r3_rollout_run_name(256.0, 4_000)

    def test_v2r3_forbids_even_seemingly_valid_mutable_latest(self):
        plan = launcher._v2r3_plan(256.0)
        manifest_hash = "e" * 64
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            latest = output / "latest"
            latest.mkdir(parents=True)
            (latest / "trainer_state.json").write_text(
                json.dumps(
                    {
                        "manifest_hash": manifest_hash,
                        "arc_steps": [9_920],
                        "local_batch_size": 21,
                        "world_size": 8,
                        "gradient_accumulation_steps": 1,
                        "sft_loss_weight": 256.0,
                        "attention_backend": "sdpa",
                        "torch_compile_mode": "none",
                        "snapshot_steps": [1_000, 2_000],
                        "global_step": 200,
                        "manifest_cursor": 200,
                    }
                ),
                encoding="utf-8",
            )
            plan = replace(plan, output_dir=str(output))
            with (
                mock.patch.object(
                    launcher,
                    "_validate_existing_run_identity",
                    return_value={},
                ),
                self.assertRaisesRegex(RuntimeError, "forbid.*latest"),
            ):
                launcher._resolve_existing_run(
                    plan, manifest_hash=manifest_hash
                )

    def test_v2r3_snapshot_prefix_rejects_self_consistent_delta_tamper(self):
        manifest_hash = "e" * 64
        identity = {
            "file_count": 1,
            "total_bytes": 1,
            "files": [{"path": "mock", "bytes": 1, "sha256": "f" * 64}],
            "manifest_sha256": "a" * 64,
        }

        def cumulative(step, loss_sum, token_count):
            return {
                "schema": "interleaved-diagnostic-ce-cumulative-v1",
                "through_step": step,
                "pretrain_loss_sum": float(loss_sum),
                "pretrain_token_count": token_count,
                "pretrain_contributing_steps": step,
                "sft_loss_sum": float(loss_sum * 2),
                "sft_token_count": token_count * 2,
                "sft_contributing_steps": step,
            }

        def interval(base, current):
            start = int(base["through_step"]) + 1
            end = int(current["through_step"])
            result = {
                "schema": "interleaved-diagnostic-ce-interval-v1",
                "measurement_semantics": (
                    "token_weighted_training_stream_pre_update_batch_logits"
                ),
                "held_out": False,
                "endpoint_checkpoint_evaluation": False,
                "start_step": start,
                "end_step": end,
                "optimizer_steps": end - start + 1,
            }
            for prefix in ("pretrain", "sft"):
                loss = (
                    current[f"{prefix}_loss_sum"]
                    - base[f"{prefix}_loss_sum"]
                )
                count = (
                    current[f"{prefix}_token_count"]
                    - base[f"{prefix}_token_count"]
                )
                contributing = (
                    current[f"{prefix}_contributing_steps"]
                    - base[f"{prefix}_contributing_steps"]
                )
                result[f"{prefix}_loss_sum"] = loss
                result[f"{prefix}_token_count"] = count
                result[f"{prefix}_contributing_steps"] = contributing
                result[f"{prefix}_token_ce"] = loss / count
            return result

        with tempfile.TemporaryDirectory() as temporary:
            plan = replace(
                launcher._v2r3_plan(256.0),
                output_dir=str(Path(temporary) / "run"),
            )
            zero = launcher._initial_v2r3_ce_cumulative()
            first = cumulative(1_000, 1_000, 1_000)
            second = cumulative(2_000, 2_000, 2_000)

            def write_snapshot(step, base, current):
                root = (
                    Path(plan.output_dir)
                    / "snapshots"
                    / f"step_{step}"
                )
                resume = root / "resume"
                hf = root / "hf"
                resume.mkdir(parents=True, exist_ok=True)
                hf.mkdir(exist_ok=True)
                metric = interval(base, current)
                state = {
                    "global_step": step,
                    "manifest_cursor": step,
                    "diagnostic_ce_cumulative": current,
                    "diagnostic_ce_interval_base": current,
                    "diagnostic_last_ce_interval_base": base,
                    "diagnostic_last_ce_interval": metric,
                }
                (resume / "trainer_state.json").write_text(
                    json.dumps(state), encoding="utf-8"
                )
                for name in (
                    "config.json",
                    "generation_config.json",
                    "tokenizer.py",
                    "tokenizer_config.json",
                    "vocab.json",
                ):
                    (hf / name).write_text("{}\n", encoding="utf-8")
                (hf / "model.safetensors").write_bytes(b"x")
                (hf / "interleaved_training_state.json").write_text(
                    json.dumps(state), encoding="utf-8"
                )
                marker_core = {
                    "schema": "interleaved-diagnostic-snapshot-v1",
                    "global_step": step,
                    "trainer_state_sha256": (
                        launcher._canonical_mapping_sha256(state)
                    ),
                    "interval_unweighted_ce": metric,
                    "resume_identity": identity,
                    "hf_identity": identity,
                }
                marker = {
                    **marker_core,
                    "marker_sha256": launcher._canonical_mapping_sha256(
                        marker_core
                    ),
                }
                (root / ".complete.json").write_text(
                    json.dumps(marker), encoding="utf-8"
                )

            write_snapshot(1_000, zero, first)
            write_snapshot(2_000, first, second)

            def read_resume_state(_plan, *, manifest_hash, state_dir):
                self.assertEqual(manifest_hash, "e" * 64)
                return json.loads(
                    (state_dir / "trainer_state.json").read_text(
                        encoding="utf-8"
                    )
                )

            patches = (
                mock.patch.object(
                    launcher,
                    "_directory_file_identity",
                    return_value=identity,
                ),
                mock.patch.object(
                    launcher,
                    "_directory_manifest_sha256",
                    return_value="b" * 64,
                ),
                mock.patch.object(
                    launcher,
                    "_validate_resume_checkpoint_identity",
                    side_effect=read_resume_state,
                ),
            )
            with patches[0], patches[1], patches[2]:
                valid = launcher._validate_v2r3_snapshot_prefix(
                    plan, manifest_hash=manifest_hash
                )
                self.assertEqual([row["step"] for row in valid], [1_000, 2_000])

                # Rebuild every local self-hash around a forged second-interval
                # base. Only continuity with snapshot 1000 exposes the tamper.
                forged_base = {**first, "pretrain_loss_sum": 900.0}
                write_snapshot(2_000, forged_base, second)
                with self.assertRaisesRegex(
                    RuntimeError, "interval/state boundary mismatch"
                ):
                    launcher._validate_v2r3_snapshot_prefix(
                        plan, manifest_hash=manifest_hash
                    )

    def test_v2r1_gate_candidate_authenticates_frozen_training_source(self):
        plan = launcher._v2r1_candidate_plan(
            {"weight": 32, "run_id": "diag-w032-s2000-r1"}
        )
        self.assertEqual(
            launcher._plan_source_tree_sha256(plan),
            launcher.V2R1_AUDITED_SOURCE_TREE_SHA256,
        )
        self.assertEqual(plan.experiment_version, launcher.EXPERIMENT_VERSION)
        self.assertEqual(plan.max_steps, 2_000)

    def test_training_command_uses_new_cli_and_one_override_group(self):
        plan = launcher._fixed_plan("p1")
        command = launcher._build_training_command(
            plan,
            manifest_hash="a" * 64,
            main_process_port=29651,
        )
        self.assertIn(launcher.TRAIN_CLI, command)
        self.assertEqual(command.count("--override"), 1)
        overrides = command[command.index("--override") + 1 :]
        self.assertIn("training.local_batch_size=21", overrides)
        self.assertIn("training.gradient_accumulation_steps=1", overrides)
        self.assertIn("training.scheduler.eta_min=1e-5", overrides)
        self.assertIn("training.optimizer.lr=1e-3", overrides)
        self.assertIn("model.attn_implementation=sdpa", overrides)
        self.assertIn("training.torch_compile=none", overrides)
        self.assertIn(
            f"training.sft_loss_weight={launcher.P1_SFT_LOSS_WEIGHT}",
            overrides,
        )
        self.assertIn("data.num_workers=8", overrides)
        self.assertIn("provenance.attention_backend=sdpa", overrides)
        self.assertIn("provenance.torch_compile_mode=none", overrides)
        self.assertIn("provenance.data_num_workers=8", overrides)
        self.assertIn("logging.backend=none", overrides)
        self.assertIn(
            "provenance.metrics_format=local-jsonl-v1",
            overrides,
        )
        self.assertIn(
            f"data.leg_manifest_path={launcher.P1_METADATA_PATH}",
            overrides,
        )
        self.assertNotIn("--weights-only", command)

    def test_p2_is_weights_only_and_content_namespaced(self):
        plan = launcher._p2_plan(
            run_id="exp1-u-after-rl1500",
            init_checkpoint=Path(
                "/checkpoints/interleave_50m/rl_hf/exp1-u-rl1500"
            ),
            init_fingerprint="b" * 64,
        )
        self.assertIn("from-bbbbbbbbbbbb", plan.output_dir)
        self.assertEqual(plan.manifest_leg, "p2")
        self.assertEqual(plan.total_steps, 9_920)
        command = launcher._build_training_command(
            plan,
            manifest_hash="c" * 64,
            main_process_port=29651,
        )
        self.assertEqual(command[-2], "--weights-only")
        self.assertEqual(command[-1], plan.weights_only)

    def test_canary_is_one_step_and_upload_free(self):
        plan = launcher._canary_plan("canary1")
        command = launcher._build_training_command(
            plan,
            manifest_hash="d" * 64,
            main_process_port=29641,
        )
        overrides = command[command.index("--override") + 1 :]
        self.assertEqual(plan.total_steps, 9_920)
        self.assertEqual(plan.arc_steps, (9_920,))
        self.assertEqual(plan.max_steps, 1)
        self.assertIn("training.max_steps=1", overrides)
        self.assertIn("training.allow_topology_override=true", overrides)
        self.assertIn("data.num_workers=0", overrides)
        self.assertNotIn("training.num_workers=0", overrides)
        self.assertIn("logging.backend=none", overrides)
        if plan.num_gpus == 1:
            self.assertNotIn("--multi_gpu", command)
        self.assertFalse(
            any(value.startswith("training.hf_upload_repo=") for value in overrides)
        )

    def test_production_canary_uses_real_p1_batch_on_8xh200(self):
        plan = launcher._production_canary_plan("topology1")
        command = launcher._build_training_command(
            plan,
            manifest_hash="f" * 64,
            main_process_port=29651,
        )
        overrides = command[command.index("--override") + 1 :]
        self.assertEqual(plan.manifest_leg, "p1")
        self.assertEqual(plan.manifest_metadata, str(launcher.P1_METADATA_PATH))
        self.assertEqual(plan.num_gpus, 8)
        self.assertEqual(plan.local_batch_size, 21)
        self.assertEqual(plan.max_steps, 20)
        self.assertIn("--multi_gpu", command)
        self.assertIn("training.max_steps=20", overrides)
        self.assertIn("training.local_batch_size=21", overrides)
        dry_run = launcher._dry_run_plan(
            "production-canary",
            run_id="topology1",
            init_checkpoint="",
        )
        self.assertEqual(dry_run["gpus"], "H200:8")

    def test_sft_weight_canary_is_versioned_exact_topology_and_bounded(self):
        plan = launcher._sft_weight_canary_plan(
            "gate-a",
            sft_loss_weight=190.189290837,
            max_steps=500,
        )
        command = launcher._build_training_command(
            plan,
            manifest_hash="7" * 64,
            main_process_port=29671,
        )
        overrides = command[command.index("--override") + 1 :]
        self.assertEqual(plan.manifest_leg, "p1")
        self.assertEqual(plan.num_gpus, 8)
        self.assertEqual(plan.local_batch_size, 21)
        self.assertEqual(plan.max_steps, 500)
        self.assertTrue(plan.structure_canary)
        self.assertIn("training.max_steps=500", overrides)
        self.assertIn(
            "training.sft_loss_weight=190.189290837", overrides
        )
        self.assertIn("training.save_interval=0", overrides)
        self.assertNotIn(
            "training.allow_topology_override=true", overrides
        )
        self.assertIn(
            f"w{launcher._weight_slug(190.189290837)}-s500",
            plan.output_dir,
        )
        dry_run = launcher._dry_run_plan(
            "sft-weight-canary",
            run_id="gate-a",
            init_checkpoint="",
            sft_loss_weight=190.189290837,
            max_steps=500,
        )
        self.assertEqual(dry_run["gpus"], "H200:8")
        self.assertEqual(dry_run["sft_loss_weight"], 190.189290837)
        self.assertTrue(dry_run["structure_canary"])

        for invalid in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                launcher._sft_weight_canary_plan(
                    "gate-b",
                    sft_loss_weight=invalid,
                    max_steps=500,
                )
        for invalid_steps in (0, launcher.SFT_WEIGHT_CANARY_MAX_STEPS + 1):
            with self.assertRaises(ValueError):
                launcher._sft_weight_canary_plan(
                    "gate-c",
                    sft_loss_weight=64.0,
                    max_steps=invalid_steps,
                )

    def test_production_gate_metrics_and_dry_run_fail_closed(self):
        passing = {
            "rollout_rows": 2_048,
            "prompt_groups": 256,
            "samples_per_group": 8,
            "outputs_with_end_thinking": 1,
            "outputs_with_call_env": 1,
            "rows_with_parsed_moves": 1,
            "positive_samples": 1,
            "nonzero_variance_groups": 1,
        }
        launcher._validate_gate_metrics(passing)
        for key in (
            "outputs_with_end_thinking",
            "outputs_with_call_env",
            "rows_with_parsed_moves",
            "positive_samples",
            "nonzero_variance_groups",
        ):
            failing = {**passing, key: 0}
            with self.assertRaises(RuntimeError):
                launcher._validate_gate_metrics(failing)
        dry_run = launcher._dry_run_plan(
            "approve-gate",
            run_id="exact-weight-gate",
            init_checkpoint="",
            candidate_call_id="fc-CANDIDATE1",
            rollout_run_name="exact-weight-gate-rollout",
            rollout_call_id="fc-ROLLOUT1",
        )
        self.assertEqual(
            dry_run["candidate_sft_loss_weight"],
            launcher.P1_SFT_LOSS_WEIGHT,
        )
        self.assertEqual(
            dry_run["candidate_steps"],
            launcher.SFT_WEIGHT_CANARY_DEFAULT_STEPS,
        )
        self.assertEqual(
            dry_run["gate_path"], str(launcher.PRODUCTION_GATE_PATH)
        )

    def test_v2r2_dry_runs_expose_gate_chain_without_launching(self):
        primary = {
            "seed": 42,
            "run_name": "primary",
            "call_id": "fc-PRIMARY",
        }
        spec = json.dumps(
            [
                {
                    "weight": 32,
                    "run_id": "candidate",
                    "call_id": "fc-CANDIDATE",
                    "primary": primary,
                    "confirmations": [],
                }
            ]
        )
        approval = launcher._dry_run_plan(
            "v2r2-approve-p1",
            run_id="",
            init_checkpoint="",
            gate_spec_json=spec,
        )
        self.assertEqual(approval["candidate_weights"], [32.0])
        self.assertEqual(
            approval["contract_plan_sha256"],
            launcher.V2R2_CONTRACT_PLAN_SHA256,
        )
        monolithic = launcher._dry_run_plan(
            "v2r2-monolithic-canary",
            run_id="",
            init_checkpoint="",
        )
        self.assertEqual(
            monolithic["required_gate"],
            str(launcher.V2R2_P1_GATE_PATH),
        )
        exp2 = launcher._dry_run_plan(
            "v2r2-exp2", run_id="", init_checkpoint=""
        )
        self.assertEqual(
            exp2["required_gate"], str(launcher.V2R2_EXP2_GATE_PATH)
        )

    def test_v2r2_marker_reader_recomputes_fingerprint_and_stage_evidence(self):
        manifest_set = {"manifest_set_hash": "a" * 64}
        stable_core = {
            "contract_schema": launcher.V2R2_CONTRACT_SCHEMA,
            "contract_version": launcher.V2R2_EXPERIMENT_VERSION,
            "contract_plan_sha256": launcher.V2R2_CONTRACT_PLAN_SHA256,
            "approved": True,
            "manifest_set_hash": "a" * 64,
            "data_artifact_version": launcher.DATA_ARTIFACT_VERSION,
            "production_source_tree_sha256": launcher.SOURCE_TREE_SHA256,
            "gate_stage": "p1_protocol",
            "selected_sft_loss_weight": 32.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            marker_path = Path(temporary) / "gate.json"
            valid = launcher._self_hash_v2r2_marker(
                {
                    **stable_core,
                    "approval_fingerprint": launcher._v2r2_json_sha256(
                        stable_core
                    ),
                    "approved_at": "2026-07-30T12:00:00+00:00",
                }
            )
            marker_path.write_text(json.dumps(valid), encoding="utf-8")
            with mock.patch.object(
                launcher, "_validate_v2r2_p1_marker_evidence"
            ) as evidence:
                observed = launcher._validate_v2r2_gate_marker(
                    marker_path,
                    gate_stage="p1_protocol",
                    manifest_set=manifest_set,
                )
            self.assertEqual(observed, valid)
            evidence.assert_called_once()

            bad = launcher._self_hash_v2r2_marker(
                {
                    **stable_core,
                    "approval_fingerprint": "0" * 64,
                    "approved_at": "2026-07-30T12:00:00+00:00",
                }
            )
            marker_path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError, "approval fingerprint mismatch"
            ):
                launcher._validate_v2r2_gate_marker(
                    marker_path,
                    gate_stage="p1_protocol",
                    manifest_set=manifest_set,
                )
            wrong_contract_core = {
                **stable_core,
                "contract_version": "wrong-contract",
            }
            wrong_contract = launcher._self_hash_v2r2_marker(
                {
                    **wrong_contract_core,
                    "approval_fingerprint": launcher._v2r2_json_sha256(
                        wrong_contract_core
                    ),
                    "approved_at": "2026-07-30T12:00:00+00:00",
                }
            )
            marker_path.write_text(
                json.dumps(wrong_contract), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "contract version mismatch"
            ):
                launcher._validate_v2r2_gate_marker(
                    marker_path,
                    gate_stage="p1_protocol",
                    manifest_set=manifest_set,
                )

    def test_v2r2_rejection_reader_binds_exact_decision_source(self):
        manifest_set = {"manifest_set_hash": "a" * 64}
        rejected = [
            {
                "weight": weight,
                "status": "protocol_rejected",
            }
            for weight in launcher.V2R2_ELIGIBLE_SFT_LOSS_WEIGHTS
        ]
        stable_core = {
            "contract_schema": launcher.V2R2_CONTRACT_SCHEMA,
            "contract_version": launcher.V2R2_EXPERIMENT_VERSION,
            "contract_plan_sha256": launcher.V2R2_CONTRACT_PLAN_SHA256,
            "approved": False,
            "gate_stage": "p1_protocol",
            "decision": "rejected_all_eligible_weights",
            "manifest_set_hash": "a" * 64,
            "data_artifact_version": launcher.DATA_ARTIFACT_VERSION,
            "decision_source_tree_sha256": launcher.SOURCE_TREE_SHA256,
            "eligible_sft_loss_weights": list(
                launcher.V2R2_ELIGIBLE_SFT_LOSS_WEIGHTS
            ),
            "rejected_candidates": rejected,
            "full_p1_authorized": False,
            "full_exp2_authorized": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rejection.json"

            def write(core):
                value = launcher._self_hash_v2r2_marker(
                    {
                        **core,
                        "decision_fingerprint": (
                            launcher._v2r2_json_sha256(core)
                        ),
                        "decided_at": "2026-07-30T12:00:00+00:00",
                    }
                )
                path.write_text(json.dumps(value), encoding="utf-8")

            write(stable_core)
            with mock.patch.object(
                launcher, "_validate_v2r2_p1_candidate_record"
            ) as candidate_validator:
                launcher._validate_v2r2_p1_rejection(
                    path, manifest_set=manifest_set
                )
            self.assertEqual(candidate_validator.call_count, 3)

            write(
                {
                    **stable_core,
                    "decision_source_tree_sha256": "0" * 64,
                }
            )
            with self.assertRaisesRegex(
                RuntimeError, "decision_source_tree_sha256 drifted"
            ):
                launcher._validate_v2r2_p1_rejection(
                    path, manifest_set=manifest_set
                )

    def test_v2r2_rollout_call_result_must_match_artifact_run(self):
        call_result = {
            "run_name": "unrelated-run",
            "checkpoint_root": "/rl-checkpoints/unrelated-run",
            "num_rollout": 1,
            "dynamic_filter": False,
            "rollout_seed": 42,
            "provenance": {"identity_sha256": "b" * 64},
        }
        with (
            mock.patch.object(
                launcher,
                "_require_successful_modal_call",
                return_value=call_result,
            ),
            mock.patch.object(
                launcher,
                "_inspect_v2r2_rollout_audit",
                return_value=(
                    {"seed": 42},
                    {"identity_sha256": "b" * 64},
                    {},
                ),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "call-result drift"):
                launcher._inspect_v2r2_rollout_spec(
                    candidate_final=Path("/checkpoints/candidate"),
                    spec={
                        "seed": 42,
                        "run_name": "expected-run",
                        "call_id": "fc-SUCCESS",
                    },
                    expected_seed=42,
                    label="primary",
                )

    def test_unweighted_ce_requires_exact_final_metric_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = replace(
                launcher._v2r2_plan("v2r2-p1", selected_weight=32),
                output_dir=temporary,
                max_steps=2_000,
            )
            runtime_provenance = {
                "attention_backend": plan.attention_backend,
                "torch_compile_mode": plan.torch_compile_mode,
                "data_num_workers": plan.data_workers,
                "sft_loss_weight": plan.sft_loss_weight,
                "configured_provenance": {
                    "experiment_version": plan.experiment_version,
                    "data_artifact_version": (
                        launcher.DATA_ARTIFACT_VERSION
                    ),
                    "source_repo": launcher.SOURCE_REPO,
                    "source_revision": launcher.SOURCE_REVISION,
                    "source_flat_manifest_sha256": (
                        launcher.SOURCE_FLAT_MANIFEST_SHA256
                    ),
                    "sft_repo": launcher.SFT_REPO,
                    "sft_revision": launcher.SFT_REVISION,
                    "attention_backend": plan.attention_backend,
                    "torch_compile_mode": plan.torch_compile_mode,
                    "data_num_workers": plan.data_workers,
                    "sft_loss_weight": plan.sft_loss_weight,
                    "sft_response_normalization": (
                        launcher.SFT_RESPONSE_NORMALIZATION
                    ),
                    "sft_supervised_unk_policy": (
                        launcher.SFT_SUPERVISED_UNK_POLICY
                    ),
                    "metrics_format": (
                        launcher.PRODUCTION_METRICS_FORMAT
                    ),
                    "source_tree_sha256": (
                        launcher._plan_source_tree_sha256(plan)
                    ),
                },
            }
            records = [
                {
                    "schema": "interleaved-local-metrics-v1",
                    "step": 1_990,
                    "manifest_hash": "e" * 64,
                    "runtime_provenance": runtime_provenance,
                    "metrics": {
                        "train/pretrain_token_loss": 1.2,
                        "train/sft_token_loss": 0.8,
                    },
                },
                {
                    "schema": "interleaved-local-metrics-v1",
                    "step": 2_000,
                    "manifest_hash": "e" * 64,
                    "runtime_provenance": runtime_provenance,
                    "metrics": {
                        "train/pretrain_token_loss": 1.1,
                        "train/sft_token_loss": 0.7,
                    },
                },
            ]
            (Path(temporary) / "metrics.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            final = launcher._last_unweighted_ce(
                plan, manifest_hash="e" * 64
            )
            self.assertEqual(final["step"], 2_000)
            self.assertEqual(final["train/pretrain_token_loss"], 1.1)
            self.assertEqual(final["train/sft_token_loss"], 0.7)
            earlier = launcher._unweighted_ce_at_step(
                plan,
                expected_step=1_990,
                manifest_hash="e" * 64,
            )
            self.assertEqual(earlier["step"], 1_990)
            self.assertEqual(earlier["train/pretrain_token_loss"], 1.2)
            self.assertEqual(earlier["train/sft_token_loss"], 0.8)
            with self.assertRaisesRegex(RuntimeError, "exact candidate"):
                launcher._unweighted_ce_at_step(
                    plan,
                    expected_step=1_000,
                    manifest_hash="e" * 64,
                )
            reversed_records = list(reversed(records))
            (Path(temporary) / "metrics.jsonl").write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in reversed_records
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError, "strictly increasing"
            ):
                launcher._unweighted_ce_at_step(
                    plan,
                    expected_step=2_000,
                    manifest_hash="e" * 64,
                )
            invalid_step = dict(records[0])
            invalid_step["step"] = 1990.0
            (Path(temporary) / "metrics.jsonl").write_text(
                json.dumps(invalid_step) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "exact integer"):
                launcher._unweighted_ce_at_step(
                    plan,
                    expected_step=1_990,
                    manifest_hash="e" * 64,
                )
            records.pop()
            (Path(temporary) / "metrics.jsonl").write_text(
                json.dumps(records[0]) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "exact candidate"):
                launcher._last_unweighted_ce(
                    plan, manifest_hash="e" * 64
                )

    def test_rollout_gate_authenticates_fixed_batch_and_variance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            candidate.mkdir()
            (candidate / "model.safetensors").write_bytes(b"weights")
            run_name = "rollout-gate"
            run_root = root / "rl" / run_name
            training = run_root / "rollouts" / "training"
            positive = (
                run_root / "rollouts" / "all_attempts_positive"
            )
            training.mkdir(parents=True)
            positive.mkdir(parents=True)

            identity = {
                "run": {
                    "run_name": run_name,
                    "app_name": "chess-interleave-rl",
                    "model_id": "interleave_47m_qwen3",
                    "num_rollout": 1,
                    "dynamic_filter": False,
                    "rollout_seed": 42,
                    "save_interval": 0,
                    "eval_interval": 0,
                    "canary": True,
                },
                "policy_update_profile": {
                    "name": "small-model-h200",
                    "max_tokens_per_gpu": 131_072,
                    "gradient_checkpointing": False,
                    "actor_num_nodes": 1,
                    "actor_num_gpus_per_node": 8,
                    "gpu_type": "H200",
                    "host_memory_gb": 192,
                    "sglang_server_concurrency": 128,
                },
                "fixed_rl_semantics": {
                    "rollout_batch_size": 256,
                    "samples_per_prompt": 8,
                    "global_batch_size": 2_048,
                    "policy_loss_agg_mode": "token-mean",
                    "advantage_estimator": "grpo",
                    "cispo": False,
                    "lr": 1e-5,
                    "rollout_max_prompt_len": 512,
                    "rollout_max_response_len": 2_560,
                    "rollout_max_context_len": 3_072,
                },
                "balanced_data": {
                    "logical_path": (
                        "/data/chess-rl-data/"
                        "train_v4_dataset_balanced_multi_turn.parquet"
                    ),
                    "sha256": launcher.RL_GATE_BALANCED_SHA256,
                },
                "origin_hf": {
                    "logical_path": str(candidate),
                    "manifest_sha256": (
                        launcher._directory_manifest_sha256(candidate)
                    ),
                },
                "sources": {
                    "chess_rl_miles": {
                        "manifest_sha256": (
                            launcher.RL_GATE_CHESS_SOURCE_SHA256
                        )
                    },
                    "miles": {
                        "manifest_sha256": (
                            launcher.RL_GATE_MILES_SOURCE_SHA256
                        )
                    },
                },
            }
            initial_command = ["python", "train.py", "--save-interval", "0"]
            provenance = {
                "identity_sha256": (
                    launcher._canonical_mapping_sha256(identity)
                ),
                "identity": identity,
                "initial_command": initial_command,
                "initial_command_sha256": hashlib.sha256(
                    json.dumps(
                        initial_command, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest(),
            }
            (run_root / "run_provenance.json").write_text(
                json.dumps(provenance),
                encoding="utf-8",
            )
            rows = []
            for group_index in range(256):
                for sample_index in range(8):
                    is_positive = group_index == 0 and sample_index == 0
                    rows.append(
                        {
                            "status": "completed",
                            "output": (
                                "</T> e2e4 <call_env>"
                                if is_positive
                                else "move soup"
                            ),
                            "extracted_moves": (
                                "e2e4" if is_positive else ""
                            ),
                            "score": 1.0 if is_positive else 0.0,
                            "group_index": group_index,
                        }
                    )
            (training / "rollout_0.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            (positive / "rollout_0.summary.json").write_text(
                json.dumps(
                    {
                        "attempted_groups": 256,
                        "attempted_samples": 2_048,
                        "completed_samples": 2_048,
                        "positive_completed_samples": 1,
                    }
                ),
                encoding="utf-8",
            )
            (positive / "rollout_0.jsonl").write_text(
                json.dumps(rows[0]) + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(launcher, "RL_GATE_ROOT", root / "rl"):
                metrics, observed, artifact_hashes = (
                    launcher._inspect_rollout_gate(
                        candidate_final=candidate,
                        rollout_run_name=run_name,
                    )
                )
            self.assertEqual(metrics["positive_samples"], 1)
            self.assertEqual(metrics["nonzero_variance_groups"], 1)
            self.assertEqual(metrics["rows_with_parsed_moves"], 1)
            self.assertEqual(observed, provenance)
            self.assertEqual(
                set(artifact_hashes),
                {
                    "run_provenance.json",
                    "rollout_0.jsonl",
                    "all_attempts_positive_rollout_0.jsonl",
                    "all_attempts_positive_rollout_0.summary.json",
                },
            )
            self.assertTrue(
                all(
                    re.fullmatch(r"[0-9a-f]{64}", value)
                    for value in artifact_hashes.values()
                )
            )

    def test_benchmark_is_ephemeral_exact_topology_and_checkpoint_free(self):
        plan = launcher._benchmark_plan(
            "sdpa-r1",
            attention_backend="sdpa",
            compile_mode="none",
        )
        command = launcher._build_training_command(
            plan,
            manifest_hash="9" * 64,
            main_process_port=29661,
        )
        overrides = command[command.index("--override") + 1 :]
        self.assertTrue(plan.benchmark_only)
        self.assertEqual(plan.num_gpus, 8)
        self.assertEqual(plan.local_batch_size, 21)
        self.assertEqual(plan.max_steps, launcher.BENCHMARK_STEPS)
        self.assertTrue(
            Path(plan.output_dir).is_relative_to(
                launcher.BENCHMARK_OUTPUT_ROOT
            )
        )
        self.assertIn("model.attn_implementation=sdpa", overrides)
        self.assertIn("training.torch_compile=none", overrides)
        self.assertIn("training.benchmark_only=true", overrides)
        self.assertIn("data.num_workers=8", overrides)
        self.assertIn("training.save_interval=0", overrides)
        self.assertIn("training.export_interval=0", overrides)
        self.assertIn("logging.backend=none", overrides)
        self.assertNotIn("training.allow_topology_override=true", overrides)

    def test_benchmark_backend_and_compile_mode_fail_closed(self):
        with self.assertRaises(ValueError):
            launcher._benchmark_plan(
                "bad-backend",
                attention_backend="eager",
                compile_mode="none",
            )
        with self.assertRaises(ValueError):
            launcher._benchmark_plan(
                "bad-compile",
                attention_backend="sdpa",
                compile_mode="magic",
            )

    def test_same_leg_retry_resumes_and_complete_run_is_idempotent(self):
        base = launcher._fixed_plan("p1")
        manifest_hash = "e" * 64

        def write_config(output: Path, plan) -> None:
            (output / "config.yaml").write_text(
                json.dumps(
                    {
                        "training": {
                            "output_dir": plan.output_dir,
                            "run_name": plan.run_name,
                            "total_steps": plan.total_steps,
                            "arc_steps": list(plan.arc_steps),
                            "local_batch_size": plan.local_batch_size,
                            "gradient_accumulation_steps": 1,
                            "sft_loss_weight": plan.sft_loss_weight,
                        },
                        "data": {
                            "expected_manifest_hash": manifest_hash,
                        },
                        "provenance": {
                            "experiment_version": launcher.EXPERIMENT_VERSION,
                            "data_artifact_version": (
                                launcher.DATA_ARTIFACT_VERSION
                            ),
                            "source_tree_sha256": (
                                launcher.SOURCE_TREE_SHA256
                            ),
                            "sft_response_normalization": (
                                launcher.SFT_RESPONSE_NORMALIZATION
                            ),
                            "sft_supervised_unk_policy": (
                                launcher.SFT_SUPERVISED_UNK_POLICY
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )

        def write_state(output: Path, plan, step: int) -> Path:
            latest = output / "latest"
            latest.mkdir(parents=True, exist_ok=True)
            (latest / "trainer_state.json").write_text(
                json.dumps(
                    {
                        "manifest_hash": manifest_hash,
                        "arc_steps": list(plan.arc_steps),
                        "local_batch_size": plan.local_batch_size,
                        "world_size": plan.num_gpus,
                        "gradient_accumulation_steps": 1,
                        "sft_loss_weight": plan.sft_loss_weight,
                        "attention_backend": plan.attention_backend,
                        "torch_compile_mode": plan.torch_compile_mode,
                        "global_step": step,
                        "manifest_cursor": step,
                    }
                ),
                encoding="utf-8",
            )
            return latest

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            plan = replace(base, output_dir=str(output), run_name="run")
            self.assertEqual(
                launcher._resolve_existing_run(plan),
                ("fresh", None),
            )
            output.mkdir(parents=True)
            (output / "metrics.jsonl").write_text(
                '{"schema":"interleaved-local-metrics-v1","step":1}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                launcher._resolve_existing_run(plan)
            write_config(output, plan)
            self.assertEqual(
                launcher._resolve_existing_run(
                    plan, manifest_hash=manifest_hash
                ),
                ("fresh", None),
            )

            latest = write_state(output, plan, 1)
            self.assertEqual(
                launcher._resolve_existing_run(
                    plan, manifest_hash=manifest_hash
                ),
                ("resume", str(latest)),
            )
            command = launcher._build_training_command(
                plan,
                manifest_hash="e" * 64,
                main_process_port=29651,
                resume=str(latest),
            )
            self.assertIn("--resume", command)
            self.assertNotIn("--weights-only", command)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            final = output / "final"
            final.mkdir(parents=True)
            (final / "config.json").write_text("{}\n", encoding="utf-8")
            (final / "model.safetensors").write_bytes(b"weights")
            plan = replace(base, output_dir=str(output), run_name="run")
            write_config(output, plan)
            latest = write_state(output, plan, plan.total_steps)
            with self.assertRaisesRegex(RuntimeError, "incomplete final"):
                launcher._resolve_existing_run(
                    plan, manifest_hash=manifest_hash
                )
            (final / "generation_config.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (final / "tokenizer.py").write_text(
                "# tokenizer\n", encoding="utf-8"
            )
            (final / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "auto_map": {
                            "AutoTokenizer": [
                                "tokenizer.HFTokenizerWrapper",
                                None,
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            (final / "vocab.json").write_text("{}\n", encoding="utf-8")
            (final / "interleaved_training_state.json").write_text(
                (latest / "trainer_state.json").read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                launcher._resolve_existing_run(
                    plan, manifest_hash=manifest_hash
                ),
                ("complete", str(final)),
            )

    def test_run_ids_and_actions_fail_closed(self):
        self.assertEqual(
            launcher._validate_run_id("exp1-u-after-rl1500"),
            "exp1-u-after-rl1500",
        )
        for invalid in ("", "../escape", "UPPER", "contains space"):
            with self.assertRaises(ValueError):
                launcher._validate_run_id(invalid)
        with self.assertRaises(ValueError):
            launcher._validate_action("all")

    def test_source_tree_digest_changes_with_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "training").mkdir()
            target = root / "training" / "module.py"
            target.write_text("one\n", encoding="utf-8")
            first = launcher._source_tree_digest(root)
            target.write_text("two\n", encoding="utf-8")
            second = launcher._source_tree_digest(root)
            self.assertNotEqual(first, second)
        self.assertEqual(
            launcher._effective_source_tree_digest("a" * 64, "b" * 64),
            "b" * 64,
        )
        with self.assertRaises(RuntimeError):
            launcher._effective_source_tree_digest("a" * 64, "not-a-digest")


if __name__ == "__main__":
    unittest.main()
