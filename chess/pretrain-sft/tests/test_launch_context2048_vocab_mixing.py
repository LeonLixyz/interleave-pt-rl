from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import torch
import numpy as np
from safetensors.torch import save_file

from modal_scripts import launch_context2048_vocab_mixing as launcher
from training.immutable_checkpoint import (
    checkpoint_volume_commit_lock,
    checkpoint_directory,
    publish_checkpoint_directory,
    temporary_checkpoint_directory,
    write_completion_marker,
    write_hf_export_completion_marker,
    write_latest_checkpoint_pointer,
)


def _write_test_accelerator_payload(checkpoint: Path, *, step: int) -> None:
    checkpoint.mkdir(parents=True)
    save_file(
        {"weight": torch.ones(2, dtype=torch.float32)},
        checkpoint / "model.safetensors",
    )
    torch.save(
        {
            "state": {
                0: {
                    "step": torch.tensor(float(step), dtype=torch.float32),
                    "exp_avg": torch.zeros(2, dtype=torch.float32),
                    "exp_avg_sq": torch.ones(2, dtype=torch.float32),
                }
            },
            "param_groups": [{"params": [0]}],
        },
        checkpoint / "optimizer.bin",
    )
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}) + "\n",
        encoding="utf-8",
    )


class Context2048VocabMixingLauncherTests(unittest.TestCase):
    def setUp(self):
        self.runtime_identity = mock.patch.object(
            launcher,
            "_modal_runtime_identity",
            return_value={
                "modal_app_name": launcher.APP_NAME,
                "modal_app_id": "ap-unit-test",
                "modal_image_id": "im-unit-test",
                "modal_base_image": launcher.CUDA_BASE_IMAGE,
                "modal_client_version": "1.4.2",
                "runtime_package_versions": launcher.PINNED_RUNTIME_PACKAGE_VERSIONS,
                "runtime_distribution_count": 42,
                "runtime_distribution_inventory_sha256": "d" * 64,
                "python_version": "3.11.0",
            },
        )
        self.runtime_identity.start()

    def tearDown(self):
        self.runtime_identity.stop()

    def test_shared_batch_and_exposure_contract(self):
        self.assertEqual(launcher.CONTEXT_LENGTH, 2_048)
        self.assertEqual(launcher.WORLD_SIZE, 8)
        self.assertEqual(launcher.PT_LOCAL_BATCH_SIZE, 16)
        self.assertEqual(launcher.PT_GLOBAL_SEQUENCES, 128)
        self.assertEqual(launcher.PT_GLOBAL_TOKEN_BATCH, 262_144)
        self.assertEqual(launcher.PT_TARGET_TOKENS, 9_181_735_000)
        self.assertEqual(launcher.PT_RECORDS, 4_483_270)
        self.assertEqual(launcher.PT_STEPS, 35_026)

    def test_interleaved_data_module_docstring_is_context_and_layout_neutral(self):
        import training.interleaved_data as interleaved_data

        doc = interleaved_data.__doc__ or ""
        self.assertNotIn("3072-token", doc)
        self.assertNotIn("production contract is two legs", doc.lower())
        self.assertIn("configurable context lengths", doc)

    def test_exact_four_experiments(self):
        self.assertEqual(
            set(launcher.EXPERIMENTS),
            {
                "vocab81_then_sft3",
                "vocab85_then_sft3",
                "mixed_sft1",
                "mixed_sft3",
            },
        )
        self.assertEqual(launcher.MIXED_STEPS, {1: 35_633, 3: 36_848})

    def test_real_tokenizer_variants_have_exact_ids(self):
        from llm_tokens.chess.tokenizer_factory import init_tokenizer

        common = {
            "include_move_numbers": False,
            "include_black_tripledots": False,
            "bos": "<bos>",
            "eos": "<eos>",
            "unk": "<unk>",
            "pad": "<bos>",
            "keep_result": False,
            "include_reward_tokens": False,
        }
        tokenizer_81 = init_tokenizer(
            "LanTokenizer",
            {**common, "include_env_tokens": False},
        )
        tokenizer_85 = init_tokenizer(
            "LanTokenizerSFT",
            {**common, "include_env_tokens": True},
        )
        self.assertEqual(len(tokenizer_81.get_vocab()), 81)
        self.assertEqual(len(tokenizer_85.get_vocab()), 85)
        self.assertEqual(tokenizer_81.get_vocab(), launcher.EXPECTED_VOCAB_81)
        self.assertEqual(tokenizer_85.get_vocab(), launcher.EXPECTED_VOCAB_85)
        self.assertEqual(
            {
                token: tokenizer_81.get_vocab().get(token)
                for token in launcher.EXPECTED_TOKEN_IDS_81
            },
            launcher.EXPECTED_TOKEN_IDS_81,
        )
        self.assertEqual(
            {
                token: tokenizer_85.get_vocab().get(token)
                for token in launcher.EXPECTED_TOKEN_IDS_85
            },
            launcher.EXPECTED_TOKEN_IDS_85,
        )

    def test_staged_sft_recipe_is_historical(self):
        experiment = launcher.EXPERIMENTS["vocab81_then_sft3"]
        pt = launcher._stage_spec(experiment, "pt")
        sft = launcher._stage_spec(experiment, "sft")
        self.assertEqual(pt["vocab_size"], 81)
        self.assertEqual(pt["peak_lr"], 1e-3)
        self.assertEqual(pt["eta_min"], 1e-4)
        self.assertEqual(sft["vocab_size"], 85)
        self.assertTrue(sft["allow_vocab_expansion"])
        self.assertEqual(sft["peak_lr"], 3e-4)
        self.assertEqual(sft["eta_min"], 1e-5)
        self.assertEqual(sft["weight_decay"], 0.01)
        self.assertEqual(sft["mixed_precision"], "bf16")
        self.assertEqual(sft["steps"], 911)
        self.assertEqual(
            int(sft["steps"] * sft["warmup_ratio"]),
            launcher.SFT_WARMUP_STEPS,
        )

    def test_v7_uses_fresh_precision_scoped_identities(self):
        self.assertEqual(
            launcher.EXPERIMENT_VERSION,
            "context2048_vocab_mixing_fp32_master_v13_20260813",
        )
        self.assertIn("fp32-master-v13", launcher.APP_NAME)
        self.assertIn("fp32-master-v13", launcher.WANDB_PROJECT)
        for experiment in launcher.EXPERIMENTS.values():
            for stage in experiment.stages:
                self.assertIn(
                    "fp32-master-v13",
                    launcher._run_name(experiment, stage, canary=False),
                )
                self.assertEqual(
                    launcher._stage_spec(experiment, stage)["mixed_precision"],
                    "bf16",
                )
        self.assertEqual(
            launcher.SOURCE_TREE_SHA256,
            launcher.COMPUTED_SOURCE_TREE_SHA256,
        )
        self.assertEqual(
            launcher.REPOSITORY_DOTENV,
            launcher.REPOSITORY_ROOT / ".env",
        )
        self.assertFalse(
            launcher.REPOSITORY_DOTENV.is_relative_to(launcher.REPO_DIR)
        )
        self.assertEqual(
            launcher._repository_root(Path("/root")),
            Path("/root"),
        )
        self.assertEqual(
            launcher._repository_root(
                Path("/workspace/interleave-pt-rl/chess/pretrain-sft")
            ),
            Path("/workspace/interleave-pt-rl"),
        )
        self.assertEqual(launcher.WANDB_SECRET, "wandb-interleave-pt-rl")
        self.assertEqual(
            launcher._wandb_secret_sync_command(),
            [
                "modal",
                "secret",
                "create",
                "--force",
                "--from-dotenv",
                str(launcher.REPOSITORY_DOTENV),
                launcher.WANDB_SECRET,
            ],
        )
        dry_run_secret = launcher._dry_run_payload()["wandb_secret"]
        self.assertFalse(dry_run_secret["dotenv_uploaded"])
        self.assertFalse(dry_run_secret["dotenv_hashed"])

    def test_modal_image_recipe_is_digest_and_exact_version_pinned(self):
        registry, digest = launcher.CUDA_BASE_IMAGE.split("@sha256:", 1)
        self.assertEqual(registry, "nvidia/cuda:12.8.0-devel-ubuntu22.04")
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(launcher.PYTHON_VERSION, "3.11")
        self.assertIn(
            "typing-extensions==4.16.0",
            launcher.PINNED_PIP_PACKAGES,
        )
        self.assertIn("metadata_path.parent != site_root", launcher.DISTRIBUTION_METADATA_CLEANUP_COMMAND)
        self.assertIn("duplicate distribution survived cleanup", launcher.DISTRIBUTION_METADATA_CLEANUP_COMMAND)
        self.assertTrue(Path(launcher.RUNTIME_SITE_PACKAGES).is_absolute())
        self.assertEqual(Path(launcher.RUNTIME_SITE_PACKAGES).name, "site-packages")
        names = []
        for requirement in launcher.PINNED_PIP_PACKAGES:
            self.assertEqual(requirement.count("=="), 1)
            self.assertNotIn(">=", requirement)
            self.assertNotIn("~=", requirement)
            name, version = requirement.split("==", 1)
            self.assertTrue(name)
            self.assertRegex(version, r"^[0-9]+(?:\.[0-9]+)+(?:[A-Za-z0-9.+-]*)$")
            names.append(name)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            launcher.PINNED_RUNTIME_PACKAGE_VERSIONS,
            {
                requirement.split("==", 1)[0]: requirement.split("==", 1)[1]
                for requirement in launcher.PINNED_PIP_PACKAGES
            },
        )
        dry_run = launcher._dry_run_payload()
        self.assertEqual(
            dry_run["runtime_identity"]["cuda_base_image"],
            launcher.CUDA_BASE_IMAGE,
        )
        self.assertEqual(
            dry_run["runtime_identity"]["pip_packages"],
            list(launcher.PINNED_PIP_PACKAGES),
        )
        self.assertEqual(
            dry_run["deployment"]["required_command"],
            launcher._deployment_command(),
        )

    def test_source_tree_hash_is_independent_of_modal_mount_paths(self):
        with TemporaryDirectory() as first_dir, TemporaryDirectory() as second_dir:
            roots = [Path(first_dir) / "local", Path(second_dir) / "root" / "chess"]
            launchers = [
                Path(first_dir) / "modal_scripts" / "launch_context2048_vocab_mixing.py",
                Path(second_dir) / "root" / "launch_context2048_vocab_mixing.py",
            ]
            for root, launcher_path in zip(roots, launchers, strict=True):
                for relative in ("config", "llm_tokens", "scripts", "training"):
                    path = root / relative / "contract.txt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"{relative}-bytes\n", encoding="utf-8")
                launcher_path.parent.mkdir(parents=True, exist_ok=True)
                launcher_path.write_text("launcher-bytes\n", encoding="utf-8")
            self.assertEqual(
                launcher._source_tree_sha256(
                    repo_dir=roots[0], launcher_path=launchers[0]
                ),
                launcher._source_tree_sha256(
                    repo_dir=roots[1], launcher_path=launchers[1]
                ),
            )

    def test_modal_client_version_prefers_injected_module_version(self):
        with mock.patch.object(launcher.modal, "__version__", "1.4.2", create=True):
            with mock.patch.object(
                launcher.importlib.metadata,
                "version",
                side_effect=AssertionError("metadata fallback must not run"),
            ):
                self.assertEqual(launcher._modal_client_version(), "1.4.2")

    def test_modal_source_upload_excludes_local_generated_artifacts(self):
        excluded = (
            Path(".env"),
            Path(".venv/bin/python"),
            Path("training/__pycache__/trainer.cpython-313.pyc"),
            Path("training/.pytest_cache/state"),
            Path("training/.ruff_cache/state"),
            Path("wandb/run-1/history.json"),
        )
        for path in excluded:
            with self.subTest(path=path):
                self.assertTrue(launcher._ignore_local_upload_artifact(path))
        self.assertFalse(
            launcher._ignore_local_upload_artifact(
                Path("training/interleaved_hf_trainer.py")
            )
        )

    def test_non_dry_run_controller_requires_matching_stable_deployment(self):
        runtime = {
            "modal_app_name": launcher.APP_NAME,
            "modal_app_id": "ap-unit-test",
            "modal_image_id": "im-unit-test",
            "modal_base_image": launcher.CUDA_BASE_IMAGE,
            "modal_client_version": "1.4.2",
            "runtime_package_versions": launcher.PINNED_RUNTIME_PACKAGE_VERSIONS,
            "runtime_distribution_count": 42,
            "runtime_distribution_inventory_sha256": "d" * 64,
            "python_version": "3.11.0",
        }
        identity = {
            "schema": "context2048-modal-deployment-identity-v1",
            "experiment_version": launcher.EXPERIMENT_VERSION,
            "source_tree_sha256": launcher.SOURCE_TREE_SHA256,
            "runtime_identity": runtime,
        }
        deployed = SimpleNamespace(remote=mock.Mock(return_value=identity))
        with mock.patch.object(
            launcher,
            "_deployed_function",
            return_value=deployed,
        ) as lookup:
            self.assertEqual(launcher._require_matching_deployment(), identity)
        lookup.assert_called_once_with("deployment_identity")

        prep = SimpleNamespace(remote=mock.Mock(return_value="prepared"))
        with mock.patch.object(
            launcher,
            "_require_matching_deployment",
            return_value=identity,
        ) as require_deployment, mock.patch.object(
            launcher,
            "_require_repo_wandb_api_key",
        ), mock.patch.object(
            launcher,
            "_deployed_function",
            return_value=prep,
        ) as lookup, mock.patch("builtins.print"):
            launcher.main(action="prep")
        require_deployment.assert_called_once_with()
        lookup.assert_called_once_with("prepare_data")
        prep.remote.assert_called_once_with()

    def test_separate_canary_orders_exercise_required_sample_types_per_rank(self):
        targets = np.arange(
            1,
            launcher.CANARY_TOTAL_STEPS * launcher.SFT_GLOBAL_SEQUENCES + 1,
            dtype=np.int64,
        )
        orders = launcher._build_canary_orders(targets)
        self.assertEqual(
            set(orders),
            {"pt", "sft3", "mixed_sft1", "mixed_sft3"},
        )
        pt_order = orders["pt"][0]
        sft_order = orders["sft3"][0]
        self.assertEqual(
            len(pt_order),
            launcher.CANARY_TOTAL_STEPS * launcher.PT_GLOBAL_SEQUENCES,
        )
        self.assertTrue(bool((pt_order >= 0).all()))
        self.assertEqual(
            len(sft_order),
            launcher.CANARY_TOTAL_STEPS * launcher.SFT_GLOBAL_SEQUENCES,
        )
        self.assertEqual(orders["sft3"][2], len(sft_order))
        self.assertEqual(orders["sft3"][3], int(targets.sum()))
        self.assertTrue(bool((sft_order < 0).all()))
        for name in ("mixed_sft1", "mixed_sft3"):
            mixed, pt_rows, sft_rows, sft_targets = orders[name]
            expected_type_rows = (
                launcher.CANARY_TOTAL_STEPS
                * launcher.PT_GLOBAL_SEQUENCES
                // 2
            )
            self.assertEqual(pt_rows, expected_type_rows)
            self.assertEqual(sft_rows, expected_type_rows)
            self.assertEqual(sft_targets, int(targets[:sft_rows].sum()))
            for update in range(launcher.CANARY_TOTAL_STEPS):
                update_start = update * launcher.PT_GLOBAL_SEQUENCES
                for rank in range(launcher.WORLD_SIZE):
                    start = (
                        update_start
                        + rank * launcher.PT_LOCAL_BATCH_SIZE
                    )
                    local = mixed[start : start + launcher.PT_LOCAL_BATCH_SIZE]
                    self.assertTrue(bool((local >= 0).any()))
                    self.assertTrue(bool((local < 0).any()))

    def test_canary_schedule_matches_two_update_manifest(self):
        manifest = {
            "metadata_path": "/data/manifest/metadata.json",
            "metadata_sha256": "a" * 64,
            "total_steps": launcher.CANARY_TOTAL_STEPS,
        }
        for key, experiment in launcher.EXPERIMENTS.items():
            for stage in experiment.stages:
                with self.subTest(experiment=key, stage=stage):
                    overrides = launcher._overrides(
                        experiment,
                        stage,
                        manifest,
                        canary=True,
                        initialization_identity=(
                            {
                                "schema": "unit-test-parent-v1",
                                "mode": "weights-only",
                            }
                            if stage == "sft"
                            else None
                        ),
                    )
                    self.assertIn(
                        f"training.total_steps={launcher.CANARY_TOTAL_STEPS}",
                        overrides,
                    )
                    self.assertIn(
                        f"training.arc_steps=[{launcher.CANARY_TOTAL_STEPS}]",
                        overrides,
                    )
                    self.assertIn("training.max_steps=1", overrides)

    def test_canary_metric_record_is_bound_to_sample_evidence(self):
        evidence = {
            "contract": "mixed-pt-sft",
            "global_pretrain_rows": 64,
            "global_sft_rows": 64,
            "global_pretrain_supervised_tokens": 128,
            "global_sft_supervised_tokens": 32,
            "pt_leading_bos_validated": True,
            "sft_bos_and_mask_validated": True,
        }
        launcher._validate_canary_sample_evidence(evidence, stage="mixed")
        with TemporaryDirectory() as directory:
            output = Path(directory)
            record = {
                "schema": "interleaved-local-metrics-v1",
                "step": 1,
                "runtime_provenance": {
                    "canary_sample_evidence": evidence,
                },
                "metrics": {
                    "train/loss": 2.5,
                    "train/global_pretrain_valid_tokens": 128,
                    "train/global_sft_valid_tokens": 32,
                    "train/global_valid_tokens": 160,
                    "train/global_weighted_valid_tokens": 160.0,
                    "train/sft_loss_weight": 1.0,
                    "train/effective_sft_loss_mass_share": 0.2,
                    "train/manifest_cursor": 1,
                    "train/pretrain_token_loss": 2.0,
                    "train/sft_token_loss": 4.5,
                },
            }
            metrics = output / "metrics.jsonl"
            metrics.write_text(json.dumps(record) + "\n", encoding="utf-8")
            validated = launcher._validate_canary_metrics(
                output,
                stage="mixed",
                sample_evidence=evidence,
            )
            self.assertEqual(validated["global_valid_tokens"], 160)
            self.assertRegex(validated["metrics_file_sha256"], r"^[0-9a-f]{64}$")

            record["metrics"]["train/global_sft_valid_tokens"] = 31
            metrics.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "global_sft_valid_tokens"):
                launcher._validate_canary_metrics(
                    output,
                    stage="mixed",
                    sample_evidence=evidence,
                )

    def test_canary_metric_accepts_expected_fp32_share_rounding(self):
        evidence = {
            "contract": "mixed-pt-sft",
            "global_pretrain_rows": 64,
            "global_sft_rows": 64,
            "global_pretrain_supervised_tokens": 131_072,
            "global_sft_supervised_tokens": 39_506,
            "pt_leading_bos_validated": True,
            "sft_bos_and_mask_validated": True,
        }
        total = 131_072 + 39_506
        fp32_share = float(np.float32(39_506) / np.float32(total))
        self.assertNotEqual(fp32_share, 39_506 / total)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            record = {
                "schema": "interleaved-local-metrics-v1",
                "step": 1,
                "runtime_provenance": {
                    "canary_sample_evidence": evidence,
                },
                "metrics": {
                    "train/loss": 2.5,
                    "train/global_pretrain_valid_tokens": 131_072,
                    "train/global_sft_valid_tokens": 39_506,
                    "train/global_valid_tokens": total,
                    "train/global_weighted_valid_tokens": float(total),
                    "train/sft_loss_weight": 1.0,
                    "train/effective_sft_loss_mass_share": fp32_share,
                    "train/manifest_cursor": 1,
                    "train/pretrain_token_loss": 2.0,
                    "train/sft_token_loss": 4.5,
                },
            }
            (output / "metrics.jsonl").write_text(
                json.dumps(record) + "\n",
                encoding="utf-8",
            )
            validated = launcher._validate_canary_metrics(
                output,
                stage="mixed",
                sample_evidence=evidence,
            )
        self.assertEqual(
            validated["effective_sft_loss_mass_share"],
            fp32_share,
        )

    def test_repo_dotenv_wandb_key_check_fails_closed_without_exposing_value(self):
        with TemporaryDirectory() as directory:
            dotenv = Path(directory) / ".env"
            with self.assertRaisesRegex(FileNotFoundError, "credential file"):
                launcher._require_repo_wandb_api_key(dotenv)
            dotenv.write_text("OTHER=value\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "WANDB_API_KEY") as error:
                launcher._require_repo_wandb_api_key(dotenv)
            self.assertNotIn("value", str(error.exception))
            dotenv.write_text(
                "export WANDB_API_KEY=unit-test-secret\n",
                encoding="utf-8",
            )
            launcher._require_repo_wandb_api_key(dotenv)

    def test_authenticated_v3_data_identities_are_pinned(self):
        manifests = {
            key: {
                "order_sha256": value,
                "order_provenance": copy.deepcopy(
                    launcher.EXPECTED_ORDER_PROVENANCE[key]
                ),
            }
            for key, value in launcher.EXPECTED_ORDER_SHA256.items()
        }
        launcher._validate_data_identities(
            {
                "source_manifest_hash": launcher.EXPECTED_SOURCE_MANIFEST_HASH,
                "selection_hash": launcher.EXPECTED_SELECTION_HASH,
                "sft_cache_hash": launcher.EXPECTED_CONTEXT2048_SFT_CACHE_HASH,
                "manifests": manifests,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "selection_hash"):
            launcher._validate_data_identities(
                {
                    "source_manifest_hash": launcher.EXPECTED_SOURCE_MANIFEST_HASH,
                    "selection_hash": "wrong",
                    "sft_cache_hash": launcher.EXPECTED_CONTEXT2048_SFT_CACHE_HASH,
                    "manifests": manifests,
                }
            )
        drifted = copy.deepcopy(manifests)
        drifted["mixed_sft1"]["order_provenance"]["composition"][1][
            "shuffle"
        ]["seed"] = 20_260_812
        with self.assertRaisesRegex(RuntimeError, "order provenance drifted"):
            launcher._validate_data_identities(
                {
                    "source_manifest_hash": launcher.EXPECTED_SOURCE_MANIFEST_HASH,
                    "selection_hash": launcher.EXPECTED_SELECTION_HASH,
                    "sft_cache_hash": launcher.EXPECTED_CONTEXT2048_SFT_CACHE_HASH,
                    "manifests": drifted,
                }
            )

    def test_order_provenance_records_every_actual_rng_and_composition(self):
        expected_hashes = {
            "pt": "4d2d43999220b1abd54c3fb775937323e9cb94ad3150b5467acbc02095d733dd",
            "sft3": "cf86894c99b23475237d741a1a02b527e56dd0bdab9664e38115fb681f108446",
            "mixed_sft1": "8c4dd690c8d9795f041ce8cdb642916b575546fe01fb668e3f0c983cc3de1ec3",
            "mixed_sft3": "49e805ea4296cff7fa919e5bb4793a865354039bc9a97ea91ad710ae4c6e2f99",
        }
        self.assertEqual(launcher.EXPECTED_ORDER_SHA256, expected_hashes)
        self.assertEqual(launcher.PT_ORDER_SEED, 42)
        self.assertEqual(launcher.SFT_EPOCH_ORDER_SEEDS, (43, 44, 45))
        self.assertEqual(
            launcher.MIXED_PLACEMENT_SEEDS,
            {1: 20_260_813, 3: 20_260_815},
        )

        pt = launcher.EXPECTED_ORDER_PROVENANCE["pt"]
        self.assertEqual(pt["components"][0]["name"], "pretrain")
        self.assertEqual(pt["components"][0]["permutation"]["seed"], 42)
        self.assertEqual(
            pt["components"][0]["permutation"]["api"],
            "numpy.random.Generator.permutation",
        )

        sft = launcher.EXPECTED_ORDER_PROVENANCE["sft3"]
        self.assertEqual(
            [component["permutation"]["seed"] for component in sft["components"]],
            [43, 44, 45],
        )
        self.assertEqual(
            sft["composition"][0]["inputs"],
            ["sft_epoch_1", "sft_epoch_2", "sft_epoch_3"],
        )

        for name, copies, placement_seed in (
            ("mixed_sft1", 1, 20_260_813),
            ("mixed_sft3", 3, 20_260_815),
        ):
            with self.subTest(name=name):
                provenance = launcher.EXPECTED_ORDER_PROVENANCE[name]
                stable_mix = provenance["composition"][1]
                self.assertEqual(
                    stable_mix["algorithm"],
                    "stable-binary-placement-from-shuffled-flags",
                )
                self.assertEqual(stable_mix["shuffle"]["seed"], placement_seed)
                self.assertEqual(
                    stable_mix["shuffle"]["api"],
                    "numpy.random.Generator.shuffle",
                )
                self.assertTrue(stable_mix["preserves_each_input_relative_order"])
                self.assertEqual(
                    len(
                        [
                            component
                            for component in provenance["components"]
                            if component["record_type"] == "sft"
                        ]
                    ),
                    copies,
                )

    def test_structured_provenance_reconstructs_pinned_production_order_hashes(self):
        pt_order = np.random.Generator(
            np.random.PCG64(launcher.PT_ORDER_SEED)
        ).permutation(launcher.PT_RECORDS).astype("<i8")
        sft_epoch_orders = [
            -(
                np.random.Generator(np.random.PCG64(seed)).permutation(
                    launcher.SFT_ROWS
                ).astype("<i8")
                + 1
            )
            for seed in launcher.SFT_EPOCH_ORDER_SEEDS
        ]

        with TemporaryDirectory() as directory:
            root = Path(directory)

            def order_sha(name, order, global_batch_size):
                padded, _ = launcher._pad_order(order, global_batch_size)
                path = root / f"{name}.npy"
                np.save(path, padded, allow_pickle=False)
                return hashlib.sha256(path.read_bytes()).hexdigest()

            observed = {
                "pt": order_sha(
                    "pt", pt_order, launcher.PT_GLOBAL_SEQUENCES
                ),
                "sft3": order_sha(
                    "sft3",
                    np.concatenate(sft_epoch_orders),
                    launcher.SFT_GLOBAL_SEQUENCES,
                ),
            }
            for copies in (1, 3):
                name = f"mixed_sft{copies}"
                observed[name] = order_sha(
                    name,
                    launcher._stable_mixed_order(
                        pt_order,
                        np.concatenate(sft_epoch_orders[:copies]),
                        seed=launcher.MIXED_PLACEMENT_SEEDS[copies],
                    ),
                    launcher.PT_GLOBAL_SEQUENCES,
                )
        self.assertEqual(observed, launcher.EXPECTED_ORDER_SHA256)

    def test_process_boundary_resume_canary_command_is_bf16_compute(self):
        manifest = {
            "metadata_path": "/data/manifest/metadata.json",
            "metadata_sha256": "a" * 64,
        }
        output = launcher.PRECISION_RESUME_ROOT / "resumed"
        resume = output / "latest"
        command = launcher._precision_resume_command(
            manifest=manifest,
            output_dir=output,
            run_name="resume-test-fp32-master-v13",
            max_steps=2,
            resume=resume,
        )
        self.assertEqual(
            command[command.index("--mixed_precision") + 1],
            "bf16",
        )
        self.assertIn(f"training.output_dir={output}", command)
        self.assertIn(
            f"training.total_steps={launcher.CANARY_TOTAL_STEPS}",
            command,
        )
        self.assertIn(
            f"training.arc_steps=[{launcher.CANARY_TOTAL_STEPS}]",
            command,
        )
        self.assertIn("training.max_steps=2", command)
        self.assertIn("training.mixed_precision=bf16", command)
        self.assertEqual(command[-2:], ["--resume", str(resume)])

    def test_staged_sft_resume_commands_cover_both_parent_tokenizer_paths(self):
        manifest = {
            "metadata_path": "/data/manifest/sft3/metadata.json",
            "metadata_sha256": "a" * 64,
        }
        for experiment_key, transition in (
            ("vocab81_then_sft3", "81-to-85"),
            ("vocab85_then_sft3", "identity"),
        ):
            with self.subTest(experiment=experiment_key):
                experiment = launcher.EXPERIMENTS[experiment_key]
                parent = Path(f"/checkpoints/{experiment_key}/pt/final")
                initialization = {
                    "schema": "interleaved-authenticated-parent-v1",
                    "mode": "weights-only",
                    "source_marker_sha256": "b" * 64,
                    "source_export_manifest_sha256": "c" * 64,
                    "source_tokenizer_contract": {"mapping_sha256": "d" * 64},
                    "tokenizer_transition": {"transition": transition},
                    "destination_seed": 42,
                }
                first_root = launcher.STAGED_SFT_RESUME_ROOT / experiment_key / "first"
                first = launcher._staged_sft_resume_command(
                    experiment=experiment,
                    manifest=manifest,
                    initialization_identity=initialization,
                    output_dir=first_root,
                    run_name="first",
                    max_steps=1,
                    weights_only=parent,
                )
                self.assertIn("training.total_steps=2", first)
                self.assertIn("training.arc_steps=[2]", first)
                self.assertIn("training.max_steps=1", first)
                self.assertEqual(first[-2:], ["--weights-only", str(parent)])
                self.assertNotIn("--resume", first)

                checkpoint = first_root / "resume_checkpoints" / "step_00000001"
                resumed = launcher._staged_sft_resume_command(
                    experiment=experiment,
                    manifest=manifest,
                    initialization_identity=initialization,
                    output_dir=launcher.STAGED_SFT_RESUME_ROOT
                    / experiment_key
                    / "resumed",
                    run_name="resumed",
                    max_steps=2,
                    resume=checkpoint,
                )
                self.assertIn("training.max_steps=2", resumed)
                self.assertEqual(resumed[-2:], ["--resume", str(checkpoint)])
                self.assertNotIn("--weights-only", resumed)

                initial_identity = launcher._initial_launch_command_identity(first)
                self.assertEqual(
                    launcher._validate_launch_command_identity(
                        initial_identity,
                        label="unit-test",
                    ),
                    first,
                )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            launcher._staged_sft_resume_command(
                experiment=launcher.EXPERIMENTS["vocab81_then_sft3"],
                manifest=manifest,
                initialization_identity={"mode": "weights-only"},
                output_dir=Path("/tmp/output"),
                run_name="invalid",
                max_steps=1,
                weights_only=Path("/tmp/parent"),
                resume=Path("/tmp/checkpoint"),
            )

    def test_persisted_hf_export_must_contain_actual_fp32_tensors(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"weight": torch.ones(2, dtype=torch.float32)}, root / "model.safetensors")
            (root / "config.json").write_text('{"dtype":"float32"}\n')
            (root / "interleaved_training_state.json").write_text(
                '{"global_step":1}\n'
            )
            write_hf_export_completion_marker(root, global_step=1)
            launcher._validate_fp32_hf_export(root)
            payload = bytearray((root / "model.safetensors").read_bytes())
            payload[-1] ^= 1
            (root / "model.safetensors").write_bytes(payload)
            with self.assertRaisesRegex(RuntimeError, "identity drifted"):
                launcher._validate_fp32_hf_export(root)

    def test_same_fp32_export_is_loaded_as_bf16_for_real_inference(self):
        class TinyInferenceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(7, 4).to(torch.bfloat16)
                self.head = torch.nn.Linear(4, 7).to(torch.bfloat16)
                self.config = SimpleNamespace(bos_token_id=1)

            def forward(self, input_ids, attention_mask=None, use_cache=False):
                del attention_mask, use_cache
                return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

        model = TinyInferenceModel()
        tokenizer_contract = {
            "vocab_size": 85,
            "token_ids": dict(launcher.EXPECTED_TOKEN_IDS_85),
            "model_max_length": 2_048,
        }
        with mock.patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ) as load, mock.patch.object(
            launcher,
            "_validate_hf_tokenizer_contract",
            return_value=tokenizer_contract,
        ):
            evidence = launcher._validate_bf16_inference_from_fp32_hf_export(
                Path("/canonical/fp32/export"),
                device="cpu",
            )
        self.assertEqual(load.call_args.kwargs["dtype"], torch.bfloat16)
        self.assertEqual(evidence["in_memory_parameter_dtypes"], ["bfloat16"])
        self.assertEqual(evidence["forward_logits_dtype"], "bfloat16")
        self.assertTrue(evidence["forward_logits_finite"])
        self.assertEqual(evidence["tokenizer_contract"], tokenizer_contract)

    def test_hf_handoff_requires_exact_85_token_mapping(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            vocab = dict(launcher.EXPECTED_VOCAB_85)
            (root / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
            (root / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "bos_token": "<bos>",
                        "eos_token": "<eos>",
                        "unk_token": "<unk>",
                        "pad_token": "<bos>",
                        "model_max_length": 2_048,
                    }
                ),
                encoding="utf-8",
            )
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "vocab_size": 85,
                        "bos_token_id": 0,
                        "eos_token_id": 1,
                        "pad_token_id": 0,
                        "max_position_embeddings": 2_048,
                    }
                ),
                encoding="utf-8",
            )
            evidence = launcher._validate_hf_tokenizer_contract(
                root,
                expected_vocab_size=85,
            )
            self.assertEqual(evidence["token_ids"], launcher.EXPECTED_TOKEN_IDS_85)
            vocab["a1"], vocab["a2"] = vocab["a2"], vocab["a1"]
            (root / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError,
                "complete tokenizer mapping drifted",
            ):
                launcher._validate_hf_tokenizer_contract(
                    root,
                    expected_vocab_size=85,
                )

    def test_nonempty_unauthenticated_stage_root_is_rejected(self):
        experiment = launcher.EXPERIMENTS["mixed_sft1"]
        manifest = {
            "metadata_path": "/data/manifest/metadata.json",
            "metadata_sha256": "a" * 64,
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.yaml").write_text("partial: true\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "no checkpoint directory"):
                launcher._authenticate_stage_output_root(
                    root,
                    experiment=experiment,
                    stage="mixed",
                    manifest=manifest,
                    expected_step=1,
                    initialization_identity={
                        "schema": "interleaved-random-initialization-v1",
                        "mode": "random",
                        "destination_seed": 42,
                    },
                    initial_launch_command={
                        "schema": "interleaved-initial-launch-command-v1",
                        "argv": ["train"],
                        "sha256": "a" * 64,
                    },
                )

    def test_running_process_commits_each_authenticated_pointer_and_final_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "run"

            class FakeProcess:
                def __init__(self):
                    self.wait_calls = 0

                def wait(self, timeout):
                    self.wait_calls += 1
                    if self.wait_calls == 1:
                        temporary = temporary_checkpoint_directory(root, 1)
                        _write_test_accelerator_payload(temporary, step=1)
                        write_completion_marker(temporary, step=1)
                        final = checkpoint_directory(root, 1)
                        publish_checkpoint_directory(temporary, final)
                        write_latest_checkpoint_pointer(root, final)
                        raise subprocess.TimeoutExpired("train", timeout)
                    return 0

                def poll(self):
                    return 0

            fake_process = FakeProcess()
            with mock.patch.object(
                launcher.subprocess,
                "Popen",
                return_value=fake_process,
            ), mock.patch.object(
                launcher.checkpoint_volume,
                "commit",
            ) as commit:
                returncode = launcher._run_process_with_incremental_checkpoint_commits(
                    ["train"],
                    label="unit-test",
                    output_dir=root,
                    poll_seconds=0.001,
                )
        self.assertEqual(returncode, 0)
        # This loop commits only the authenticated checkpoint.  Its caller
        # validates the final HF export before performing the final commit.
        self.assertEqual(commit.call_count, 1)

    def test_failed_process_does_not_commit_partial_final_artifacts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "run"

            class FailedProcess:
                def __init__(self):
                    self.wait_calls = 0

                def wait(self, timeout):
                    self.wait_calls += 1
                    if self.wait_calls == 1:
                        temporary = temporary_checkpoint_directory(root, 1)
                        _write_test_accelerator_payload(temporary, step=1)
                        write_completion_marker(temporary, step=1)
                        final = checkpoint_directory(root, 1)
                        publish_checkpoint_directory(temporary, final)
                        write_latest_checkpoint_pointer(root, final)
                        (root / "final").mkdir()
                        (root / "final" / "model.safetensors").write_bytes(
                            b"partial"
                        )
                        raise subprocess.TimeoutExpired("train", timeout)
                    return 9

                def poll(self):
                    return 9

            with mock.patch.object(
                launcher.subprocess,
                "Popen",
                return_value=FailedProcess(),
            ), mock.patch.object(
                launcher.checkpoint_volume,
                "commit",
            ) as commit:
                returncode = launcher._run_process_with_incremental_checkpoint_commits(
                    ["train"],
                    label="unit-test-failure",
                    output_dir=root,
                    poll_seconds=0.001,
                )
        self.assertEqual(returncode, 9)
        commit.assert_not_called()

    def test_incremental_commit_waits_for_final_export_lock(self):
        """A Volume commit cannot race with `.final.tmp` publication."""

        with TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            temporary = temporary_checkpoint_directory(root, 1)
            _write_test_accelerator_payload(temporary, step=1)
            write_completion_marker(temporary, step=1)
            checkpoint = checkpoint_directory(root, 1)
            publish_checkpoint_directory(temporary, checkpoint)
            write_latest_checkpoint_pointer(root, checkpoint)
            writer_holds_lock = threading.Event()
            release_writer = threading.Event()

            def publish_final() -> None:
                with checkpoint_volume_commit_lock(root):
                    staging = root / ".final.tmp"
                    staging.mkdir()
                    writer_holds_lock.set()
                    self.assertTrue(release_writer.wait(timeout=5.0))
                    staging.rename(root / "final")

            with mock.patch.object(
                launcher.checkpoint_volume,
                "commit",
            ) as commit, ThreadPoolExecutor(max_workers=2) as pool:
                writer = pool.submit(publish_final)
                self.assertTrue(writer_holds_lock.wait(timeout=5.0))
                poller = pool.submit(
                    launcher._commit_new_authenticated_checkpoint,
                    root,
                    None,
                    label="lock-race-test",
                )
                time.sleep(0.05)
                self.assertFalse(poller.done())
                commit.assert_not_called()
                release_writer.set()
                writer.result(timeout=5.0)
                self.assertIsNone(poller.result(timeout=5.0))
                commit.assert_not_called()

    def test_incremental_supervisor_failure_kills_accelerate_process_group(self):
        class Process:
            pid = 4242

            def __init__(self):
                self.wait_calls = 0

            def wait(self, timeout):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    raise subprocess.TimeoutExpired("train", timeout)
                self.assert_timeout = timeout
                return -launcher.signal.SIGTERM

            def poll(self):
                return None if self.wait_calls == 1 else -launcher.signal.SIGTERM

        process = Process()
        with mock.patch.object(
            launcher.subprocess,
            "Popen",
            return_value=process,
        ) as popen, mock.patch.object(
            launcher,
            "_commit_new_authenticated_checkpoint",
            side_effect=RuntimeError("checkpoint poll failed"),
        ), mock.patch.object(launcher.os, "killpg") as killpg:
            with self.assertRaisesRegex(RuntimeError, "checkpoint poll failed"):
                launcher._run_process_with_incremental_checkpoint_commits(
                    ["train"],
                    label="supervisor-failure-test",
                    output_dir=Path("/tmp/supervisor-failure-test"),
                    poll_seconds=0.001,
                )
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        killpg.assert_called_once_with(4242, launcher.signal.SIGTERM)
        self.assertEqual(process.assert_timeout, 30.0)

    def test_resume_process_receives_stable_initial_command_identity(self):
        class FinishedProcess:
            def wait(self, timeout=None):
                del timeout
                return 0

            def poll(self):
                return 0

        initial_command = ["train", "--weights-only", "/pt/final"]
        identity = launcher._initial_launch_command_identity(initial_command)
        with TemporaryDirectory() as directory, mock.patch.object(
            launcher.subprocess,
            "Popen",
            return_value=FinishedProcess(),
        ) as popen:
            returncode = launcher._run_process_with_incremental_checkpoint_commits(
                ["train", "--resume", "/sft/latest"],
                label="stable-command-test",
                output_dir=Path(directory),
                initial_launch_command=identity,
            )
        self.assertEqual(returncode, 0)
        self.assertEqual(
            popen.call_args.args[0],
            ["train", "--resume", "/sft/latest"],
        )
        environment = popen.call_args.kwargs["env"]
        self.assertEqual(
            json.loads(environment[launcher.INITIAL_LAUNCH_COMMAND_ENV]),
            initial_command,
        )
        self.assertEqual(
            environment[launcher.INITIAL_LAUNCH_COMMAND_SHA256_ENV],
            identity["sha256"],
        )

    def test_mixed_recipe_uses_one_continuous_schedule(self):
        for key, copies in (("mixed_sft1", 1), ("mixed_sft3", 3)):
            experiment = launcher.EXPERIMENTS[key]
            self.assertEqual(experiment.sft_copies, copies)
            spec = launcher._stage_spec(experiment, "mixed")
            self.assertEqual(spec["vocab_size"], 85)
            self.assertEqual(spec["peak_lr"], 1e-3)
            self.assertEqual(spec["eta_min"], 1e-5)
            self.assertEqual(spec["weight_decay"], 0.1)

    def test_overrides_pin_unpacked_sft_and_wandb(self):
        experiment = launcher.EXPERIMENTS["mixed_sft1"]
        manifest = {
            "metadata_path": "/data/manifest/metadata.json",
            "metadata_sha256": "a" * 64,
        }
        overrides = launcher._overrides(
            experiment, "mixed", manifest, canary=False
        )
        self.assertIn("model.block_size=2048", overrides)
        self.assertIn("data.sequence_length=2048", overrides)
        self.assertIn("logging.backend=wandb", overrides)
        self.assertIn(
            "provenance.sft_packing=one-row-per-sequence-right-padded",
            overrides,
        )

    def test_production_staged_gate_binds_resume_marker_but_mixed_does_not(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_set_path = root / "manifest_set.json"
            manifest_set_path.write_text(
                json.dumps({"set_hash": "m" * 64}), encoding="utf-8"
            )

            def write_experiment_gate(experiment, staged_sha):
                samples = {stage: {"stage": stage} for stage in experiment.stages}
                metrics = {stage: {"stage": stage} for stage in experiment.stages}
                gate = {
                    "schema": "context2048-vocab-mixing-canary-gate-v1",
                    "decision": "pass",
                    "experiment_version": launcher.EXPERIMENT_VERSION,
                    "experiment": experiment.key,
                    "source_tree_sha256": launcher.SOURCE_TREE_SHA256,
                    "manifest_set_hash": "m" * 64,
                    "runtime_identity": {"runtime": "unit-test"},
                    "sample_evidence": samples,
                    "metric_evidence": metrics,
                    "staged_sft_resume_gate_sha256": staged_sha,
                    "created_at": "2026-08-13T00:00:00+00:00",
                }
                gate["gate_sha256"] = hashlib.sha256(
                    launcher._canonical_json(gate)
                ).hexdigest()
                path = root / f"{experiment.key}.json"
                path.write_text(json.dumps(gate), encoding="utf-8")
                return path, gate

            staged = launcher.EXPERIMENTS["vocab81_then_sft3"]
            staged_sha = "s" * 64
            staged_path, staged_gate = write_experiment_gate(staged, staged_sha)
            with mock.patch.object(launcher, "GATE_ROOT", root), mock.patch.object(
                launcher, "MANIFEST_SET_PATH", manifest_set_path
            ), mock.patch.object(
                launcher, "_validate_precision_resume_gate"
            ), mock.patch.object(
                launcher,
                "_validate_staged_sft_resume_gate",
                return_value={"gate_sha256": staged_sha},
            ) as staged_validator, mock.patch.object(
                launcher, "_validate_recorded_runtime_identity"
            ), mock.patch.object(
                launcher, "_validate_canary_sample_evidence"
            ), mock.patch.object(
                launcher,
                "_validate_canary_metrics",
                side_effect=lambda output, *, stage, sample_evidence: {
                    "stage": stage
                },
            ):
                launcher._validate_gate(staged)
                staged_validator.assert_called_once_with()
                changed = copy.deepcopy(staged_gate)
                changed["staged_sft_resume_gate_sha256"] = "0" * 64
                changed.pop("gate_sha256")
                changed["gate_sha256"] = hashlib.sha256(
                    launcher._canonical_json(changed)
                ).hexdigest()
                staged_path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "resume binding"):
                    launcher._validate_gate(staged)

            mixed = launcher.EXPERIMENTS["mixed_sft1"]
            write_experiment_gate(mixed, None)
            with mock.patch.object(launcher, "GATE_ROOT", root), mock.patch.object(
                launcher, "MANIFEST_SET_PATH", manifest_set_path
            ), mock.patch.object(
                launcher, "_validate_precision_resume_gate"
            ), mock.patch.object(
                launcher,
                "_validate_staged_sft_resume_gate",
                side_effect=AssertionError("mixed gate must not require staged SFT"),
            ) as staged_validator, mock.patch.object(
                launcher, "_validate_recorded_runtime_identity"
            ), mock.patch.object(
                launcher, "_validate_canary_sample_evidence"
            ), mock.patch.object(
                launcher,
                "_validate_canary_metrics",
                side_effect=lambda output, *, stage, sample_evidence: {
                    "stage": stage
                },
            ):
                launcher._validate_gate(mixed)
                staged_validator.assert_not_called()

    def test_staged_sft_resume_gate_recomputes_artifacts_and_rejects_tampering(self):
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            gate_path = temporary / "gate.json"
            staged_root = temporary / "staged"
            checkpoint_root = temporary / "checkpoints"
            manifest = {
                "metadata_path": str(temporary / "sft3" / "metadata.json"),
                "metadata_sha256": "a" * 64,
            }
            manifest_set = {"set_hash": "b" * 64}
            sample = {
                "contract": "sft-only",
                "global_pretrain_rows": 0,
                "global_sft_rows": launcher.SFT_GLOBAL_SEQUENCES,
                "global_pretrain_supervised_tokens": 0,
                "global_sft_supervised_tokens": 4096,
                "sft_bos_and_mask_validated": True,
            }
            precision = {
                "schema": "interleaved-accelerator-persisted-fp32-v1",
                "model": {
                    "tensor_count": 3,
                    "dtype_counts": {"F32": 3},
                },
                "optimizer_files": [
                    {
                        "parameter_state_count": 3,
                        "parameter_group_count": 2,
                    }
                ],
                "optimizer_tensor_count": 9,
                "model_floating_dtype": "float32",
                "adam_moment_dtype": "float32",
                "adam_moment_tensor_count": 6,
            }
            identities = {
                key: {
                    "schema": "interleaved-authenticated-parent-v1",
                    "mode": "weights-only",
                    "source_export_path": str(
                        checkpoint_root / f"{key}_canary" / "pt" / "final"
                    ),
                    "source_marker_sha256": hashlib.sha256(
                        f"{key}:marker".encode()
                    ).hexdigest(),
                    "source_trainer_state_sha256": hashlib.sha256(
                        f"{key}:state".encode()
                    ).hexdigest(),
                    "source_export_manifest_sha256": hashlib.sha256(
                        f"{key}:manifest".encode()
                    ).hexdigest(),
                    "source_tokenizer_contract": {
                        "mapping_sha256": hashlib.sha256(key.encode()).hexdigest(),
                        "vocab_size": 81 if key.startswith("vocab81") else 85,
                    },
                    "tokenizer_transition": {
                        "transition": "81-to-85"
                        if key.startswith("vocab81")
                        else "identity",
                    },
                    "destination_seed": 42,
                }
                for key in launcher.STAGED_SFT_RESUME_VARIANTS
            }

            def initialization_identity(experiment, stage, **kwargs):
                self.assertEqual(stage, "sft")
                self.assertTrue(kwargs["canary"])
                return copy.deepcopy(identities[experiment.key])

            def checkpoint_for(output_root, **kwargs):
                del kwargs
                return Path(output_root) / "checkpoint"

            def path_sha256(path):
                return hashlib.sha256(str(path).encode()).hexdigest()

            def metric_evidence(output_root, **kwargs):
                del kwargs
                return {
                    "metrics_file_sha256": path_sha256(
                        Path(output_root) / "metrics.jsonl"
                    ),
                    "step": 1,
                    "global_sft_valid_tokens": 4096,
                }

            def lr_trace(path):
                path = Path(path)
                if "reference" in path.parts:
                    return [
                        {"step": 1, "lr": 1e-4},
                        {"step": 2, "lr": 2e-4},
                    ]
                if "resumed" in path.parts:
                    return [{"step": 2, "lr": 2e-4}]
                return [{"step": 1, "lr": 1e-4}]

            def model_sha(left, right):
                self.assertIn("resumed", Path(left).parts)
                self.assertIn("reference", Path(right).parts)
                key = next(
                    key
                    for key in launcher.STAGED_SFT_RESUME_VARIANTS
                    if key in Path(left).parts
                )
                return hashlib.sha256(f"{key}:model".encode()).hexdigest()

            final_fp32 = {
                "floating_dtype": "float32",
                "floating_tensor_count": 3,
            }
            tokenizer_contract = {
                "vocab_size": 85,
                "mapping_sha256": "e" * 64,
            }

            patches = (
                mock.patch.object(
                    launcher, "STAGED_SFT_RESUME_GATE_PATH", gate_path
                ),
                mock.patch.object(
                    launcher, "STAGED_SFT_RESUME_ROOT", staged_root
                ),
                mock.patch.object(launcher, "CHECKPOINT_ROOT", checkpoint_root),
                mock.patch.object(
                    launcher,
                    "_manifest",
                    return_value=(manifest_set, manifest),
                ),
                mock.patch.object(
                    launcher,
                    "_stage_initialization_identity",
                    side_effect=initialization_identity,
                ),
                mock.patch.object(
                    launcher,
                    "validate_checkpoint_run_root",
                    side_effect=checkpoint_for,
                ),
                mock.patch.object(
                    launcher,
                    "inspect_accelerator_checkpoint_fp32",
                    return_value=precision,
                ),
                mock.patch.object(launcher, "_validate_stage_resume_state"),
                mock.patch.object(launcher, "_validate_final"),
                mock.patch.object(launcher, "_validate_canary_sample_evidence"),
                mock.patch.object(
                    launcher,
                    "_validate_canary_metrics",
                    side_effect=metric_evidence,
                ),
                mock.patch.object(
                    launcher, "_load_metric_trace", side_effect=lr_trace
                ),
                mock.patch.object(
                    launcher,
                    "_assert_exact_fp32_models_equal",
                    side_effect=model_sha,
                ),
                mock.patch.object(
                    launcher,
                    "_validate_fp32_hf_export",
                    return_value=final_fp32,
                ),
                mock.patch.object(
                    launcher,
                    "_validate_hf_tokenizer_contract",
                    return_value=tokenizer_contract,
                ),
                mock.patch.object(launcher, "_validate_recorded_runtime_identity"),
                mock.patch.object(
                    launcher,
                    "_checkpoint_sha256_file",
                    side_effect=path_sha256,
                ),
            )
            entered = []
            try:
                for patcher in patches:
                    entered.append(patcher)
                    patcher.start()

                variants = {}
                for key in launcher.STAGED_SFT_RESUME_VARIANTS:
                    experiment = launcher.EXPERIMENTS[key]
                    identity = identities[key]
                    parent = checkpoint_root / f"{key}_canary" / "pt" / "final"
                    variant_root = staged_root / key
                    first_root = variant_root / "first_update"
                    resumed_root = variant_root / "resumed"
                    reference_root = variant_root / "reference"
                    first_checkpoint = first_root / "checkpoint"
                    resumed_checkpoint = resumed_root / "checkpoint"
                    reference_checkpoint = reference_root / "checkpoint"
                    first_command = launcher._staged_sft_resume_command(
                        experiment=experiment,
                        manifest=manifest,
                        initialization_identity=identity,
                        output_dir=first_root,
                        run_name=launcher._staged_sft_gate_run_name(
                            experiment, "first-update"
                        ),
                        max_steps=1,
                        weights_only=parent,
                    )
                    initial = launcher._initial_launch_command_identity(
                        first_command
                    )
                    resume_command = launcher._staged_sft_resume_command(
                        experiment=experiment,
                        manifest=manifest,
                        initialization_identity=identity,
                        output_dir=resumed_root,
                        run_name=launcher._staged_sft_gate_run_name(
                            experiment, "resumed"
                        ),
                        max_steps=2,
                        resume=first_checkpoint,
                    )
                    reference_command = launcher._staged_sft_resume_command(
                        experiment=experiment,
                        manifest=manifest,
                        initialization_identity=identity,
                        output_dir=reference_root,
                        run_name=launcher._staged_sft_gate_run_name(
                            experiment, "reference"
                        ),
                        max_steps=2,
                        weights_only=parent,
                    )
                    reference_initial = launcher._initial_launch_command_identity(
                        reference_command
                    )

                    def runtime(command):
                        argv = launcher._expected_training_process_argv(command)
                        return {
                            "canary_sample_evidence": sample,
                            "process_argv": argv,
                            "process_argv_sha256": hashlib.sha256(
                                json.dumps(
                                    argv,
                                    ensure_ascii=True,
                                    separators=(",", ":"),
                                ).encode()
                            ).hexdigest(),
                        }

                    states = {
                        first_checkpoint: {
                            "global_step": 1,
                            "manifest_cursor": 1,
                            "runtime_provenance": runtime(first_command),
                        },
                        resumed_checkpoint: {
                            "global_step": 2,
                            "manifest_cursor": 2,
                            "runtime_provenance": runtime(resume_command),
                        },
                        reference_checkpoint: {
                            "global_step": 2,
                            "manifest_cursor": 2,
                            "runtime_provenance": runtime(reference_command),
                        },
                    }
                    for checkpoint, state in states.items():
                        checkpoint.mkdir(parents=True)
                        (checkpoint / "trainer_state.json").write_text(
                            json.dumps(state), encoding="utf-8"
                        )
                    first_process = launcher._validate_training_process_argv(
                        states[first_checkpoint],
                        command=first_command,
                        label="first",
                    )
                    resume_process = launcher._validate_training_process_argv(
                        states[resumed_checkpoint],
                        command=resume_command,
                        label="resumed",
                    )
                    reference_process = launcher._validate_training_process_argv(
                        states[reference_checkpoint],
                        command=reference_command,
                        label="reference",
                    )
                    resumed_trace = lr_trace(first_root / "metrics.jsonl")
                    resumed_trace.extend(lr_trace(resumed_root / "metrics.jsonl"))
                    reference_trace = lr_trace(
                        reference_root / "metrics.jsonl"
                    )
                    variants[key] = {
                        "parent_final": str(parent),
                        "initialization_identity": identity,
                        "initial_launch_command": initial,
                        "reference_initial_launch_command": reference_initial,
                        "first_process_command": first_process,
                        "resume_process_command": resume_process,
                        "reference_process_command": reference_process,
                        "checkpoint_1_path": str(first_checkpoint),
                        "checkpoint_1_completion_marker_sha256": path_sha256(
                            first_checkpoint / ".complete.json"
                        ),
                        "checkpoint_1_precision": precision,
                        "resumed_checkpoint_precision": precision,
                        "reference_checkpoint_precision": precision,
                        "sample_evidence": sample,
                        "metric_evidence": metric_evidence(first_root),
                        "resumed_step": 2,
                        "resumed_manifest_cursor": 2,
                        "resumed_lr_trace": resumed_trace,
                        "reference_lr_trace": reference_trace,
                        "model_sha256": model_sha(
                            resumed_root / "final", reference_root / "final"
                        ),
                        "final_fp32_evidence": final_fp32,
                        "final_tokenizer_contract": tokenizer_contract,
                        "runtime_identity": states[first_checkpoint][
                            "runtime_provenance"
                        ],
                    }

                baseline = {
                    "schema": "context2048-staged-sft-resume-canary-v1",
                    "decision": "pass",
                    "experiment_version": launcher.EXPERIMENT_VERSION,
                    "source_tree_sha256": launcher.SOURCE_TREE_SHA256,
                    "manifest_set_hash": manifest_set["set_hash"],
                    "variant_keys": list(launcher.STAGED_SFT_RESUME_VARIANTS),
                    "variants": variants,
                    "created_at": "2026-08-13T00:00:00+00:00",
                }

                def write_gate(value, *, valid_self_hash=True):
                    payload = copy.deepcopy(value)
                    payload["gate_sha256"] = hashlib.sha256(
                        launcher._canonical_json(payload)
                    ).hexdigest()
                    if not valid_self_hash:
                        payload["gate_sha256"] = "0" * 64
                    gate_path.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(RuntimeError, "Missing staged-SFT"):
                    launcher._validate_staged_sft_resume_gate()
                write_gate(baseline)
                validated = launcher._validate_staged_sft_resume_gate()
                self.assertEqual(validated["decision"], "pass")

                mutations = []

                def mutate(name, regex, callback):
                    mutations.append((name, regex, callback))

                mutate(
                    "variant inventory",
                    "variant inventory",
                    lambda value: value["variants"].pop("vocab85_then_sft3"),
                )
                mutate(
                    "source tree",
                    "source_tree_sha256",
                    lambda value: value.__setitem__("source_tree_sha256", "wrong"),
                )
                mutate(
                    "manifest set",
                    "manifest_set_hash",
                    lambda value: value.__setitem__("manifest_set_hash", "wrong"),
                )
                mutate(
                    "parent marker",
                    "full parent identity",
                    lambda value: value["variants"]["vocab81_then_sft3"][
                        "initialization_identity"
                    ].__setitem__("source_marker_sha256", "0" * 64),
                )
                mutate(
                    "parent export",
                    "full parent identity",
                    lambda value: value["variants"]["vocab85_then_sft3"][
                        "initialization_identity"
                    ].__setitem__("source_export_manifest_sha256", "0" * 64),
                )
                mutate(
                    "tokenizer transition",
                    "full parent identity",
                    lambda value: value["variants"]["vocab81_then_sft3"][
                        "initialization_identity"
                    ]["tokenizer_transition"].__setitem__(
                        "transition", "identity"
                    ),
                )
                mutate(
                    "initial command argv",
                    "command identity SHA-256",
                    lambda value: value["variants"]["vocab81_then_sft3"][
                        "initial_launch_command"
                    ]["argv"].append("--tampered"),
                )
                mutate(
                    "resume command",
                    "resume process command",
                    lambda value: value["variants"]["vocab81_then_sft3"][
                        "resume_process_command"
                    ]["argv"].append("--weights-only"),
                )
                mutate(
                    "optimizer evidence",
                    "checkpoint-1 precision",
                    lambda value: value["variants"]["vocab81_then_sft3"][
                        "checkpoint_1_precision"
                    ].__setitem__("adam_moment_tensor_count", 0),
                )
                mutate(
                    "sample evidence",
                    "sample evidence",
                    lambda value: value["variants"]["vocab81_then_sft3"][
                        "sample_evidence"
                    ].__setitem__("sft_bos_and_mask_validated", False),
                )
                mutate(
                    "step",
                    "step/cursor",
                    lambda value: value["variants"]["vocab81_then_sft3"].__setitem__(
                        "resumed_step", 1
                    ),
                )
                mutate(
                    "lr trace",
                    "LR trace",
                    lambda value: value["variants"]["vocab81_then_sft3"][
                        "resumed_lr_trace"
                    ].append({"step": 3, "lr": 0.0}),
                )
                mutate(
                    "model",
                    "model equality",
                    lambda value: value["variants"]["vocab81_then_sft3"].__setitem__(
                        "model_sha256", "0" * 64
                    ),
                )
                mutate(
                    "final tokenizer",
                    "tokenizer evidence",
                    lambda value: value["variants"]["vocab81_then_sft3"][
                        "final_tokenizer_contract"
                    ].__setitem__("mapping_sha256", "0" * 64),
                )

                for name, regex, callback in mutations:
                    with self.subTest(tamper=name):
                        changed = copy.deepcopy(baseline)
                        callback(changed)
                        write_gate(changed)
                        with self.assertRaisesRegex(RuntimeError, regex):
                            launcher._validate_staged_sft_resume_gate()
                write_gate(baseline, valid_self_hash=False)
                with self.assertRaisesRegex(RuntimeError, "self hash"):
                    launcher._validate_staged_sft_resume_gate()
            finally:
                for patcher in reversed(entered):
                    patcher.stop()


if __name__ == "__main__":
    unittest.main()
