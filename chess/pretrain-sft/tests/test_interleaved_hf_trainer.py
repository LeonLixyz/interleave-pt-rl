from __future__ import annotations

import errno
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from llm_tokens.chess.tokenizer_factory import init_tokenizer
from scripts.train.train_interleaved_hf import parse_args
from training.interleaved_hf_trainer import (
    COMPUTE_DTYPE,
    DETERMINISM_CONTRACT,
    EXPECTED_CONTEXT_LENGTH,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_TOKEN_POSITIONS_PER_UPDATE,
    IGNORE_INDEX,
    MASTER_PARAMETER_DTYPE,
    PRECISION_CONTRACT,
    SAMPLE_PRETRAIN,
    SAMPLE_SFT,
    STATE_SCHEMA_VERSION,
    ExactArcCosine,
    InterleavedHFTrainer,
    add_diagnostic_ce_step,
    authenticated_weights_only_identity,
    assert_finite_gradient_norm,
    assert_fp32_gradients,
    assert_fp32_master_parameters,
    assert_fp32_optimizer,
    assert_fp32_state_dict,
    build_interleaved_qwen_model,
    causal_ce_sum,
    configure_deterministic_training,
    diagnostic_ce_interval,
    globally_normalized_backward_loss,
    load_weights_only,
    new_diagnostic_ce_cumulative,
    normalize_attention_backend,
    normalize_compile_mode,
    normalize_sft_loss_weight,
    register_bf16_output_head_assertion,
    resolve_arc_steps,
    RUNTIME_SITE_PACKAGES,
    runtime_distribution_identity,
    validate_diagnostic_ce_resume_state,
    validate_runtime_package_versions,
    validate_resume_state,
    validate_topology,
    weighted_causal_ce_sum,
)
from training.hf_tokenizer_utils import save_hf_tokenizer
from training.tokenizer_contract import (
    EXPECTED_VOCAB_81,
    EXPECTED_VOCAB_85,
    validate_vocab_transition,
)
import training.immutable_checkpoint as immutable_checkpoint
from training.immutable_checkpoint import (
    CHECKPOINT_COMPLETE_FILE,
    LATEST_CHECKPOINT_POINTER,
    LATEST_CHECKPOINT_SYMLINK,
    checkpoint_directory,
    inspect_accelerator_checkpoint_fp32,
    publish_checkpoint_directory,
    publish_diagnostic_snapshot_directory,
    publish_hf_export_directory,
    resolve_resume_checkpoint,
    temporary_checkpoint_directory,
    validate_checkpoint_run_root,
    validate_completed_checkpoint,
    validate_completed_diagnostic_snapshot,
    validate_completed_hf_export,
    write_completion_marker,
    write_diagnostic_snapshot_completion_marker,
    write_hf_export_completion_marker,
    write_latest_checkpoint_pointer,
)


def _write_test_accelerator_payload(
    checkpoint: Path,
    *,
    step: int,
    model_dtype: torch.dtype = torch.float32,
    adam_dtype: torch.dtype = torch.float32,
    trainer_state: dict | None = None,
) -> None:
    checkpoint.mkdir(parents=True, exist_ok=True)
    save_file(
        {"weight": torch.arange(4, dtype=model_dtype)},
        checkpoint / "model.safetensors",
    )
    torch.save(
        {
            "state": {
                0: {
                    "step": torch.tensor(float(step), dtype=torch.float32),
                    "exp_avg": torch.zeros(4, dtype=adam_dtype),
                    "exp_avg_sq": torch.ones(4, dtype=adam_dtype),
                }
            },
            "param_groups": [{"params": [0]}],
        },
        checkpoint / "optimizer.bin",
    )
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(trainer_state or {"global_step": step}),
        encoding="utf-8",
    )


def _authenticate_test_hf_export(
    export: Path,
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
    step: int = 0,
    trainer_state: dict | None = None,
    vocab: dict[str, int] | None = None,
    context_length: int = 2_048,
) -> None:
    export.mkdir(parents=True, exist_ok=True)
    if state_dict is not None:
        save_file(state_dict, export / "model.safetensors")
    config_path = export / "config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else {}
    )
    vocab = dict(vocab or EXPECTED_VOCAB_81)
    config.update(
        {
            "dtype": "float32",
            "vocab_size": len(vocab),
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 0,
            "max_position_embeddings": context_length,
            "interleaved_model_init_seed": 42,
        }
    )
    config.pop("torch_dtype", None)
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (export / "vocab.json").write_text(
        json.dumps(vocab, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (export / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
                "pad_token": "<bos>",
                "model_max_length": context_length,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (export / "interleaved_training_state.json").write_text(
        json.dumps(
            trainer_state
            or {
                "global_step": step,
                "model_init_seed": 42,
                "configured_provenance": {
                    "experiment": "unit-test-source",
                    "stage": "pt",
                    "experiment_version": "unit-test-v1",
                    "seed": 42,
                },
            }
        ),
        encoding="utf-8",
    )
    write_hf_export_completion_marker(export, global_step=step)


class InterleavedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = init_tokenizer(
            "LanTokenizerSFT",
            {
                "include_move_numbers": False,
                "include_black_tripledots": False,
                "bos": "<bos>",
                "eos": "<eos>",
                "unk": "<unk>",
                "keep_result": False,
                "include_env_tokens": True,
                "include_reward_tokens": False,
            },
        )
        cls.model = build_interleaved_qwen_model(cls.tokenizer)

    def test_exact_qwen_shape_vocab_and_native_context(self):
        self.assertEqual(len(self.tokenizer.get_vocab()), 85)
        self.assertEqual(
            sum(parameter.numel() for parameter in self.model.parameters()),
            EXPECTED_PARAMETER_COUNT,
        )
        config = self.model.config
        self.assertEqual(config.max_position_embeddings, EXPECTED_CONTEXT_LENGTH)

        self.assertEqual(config.num_hidden_layers, 12)
        self.assertEqual(config.hidden_size, 512)
        self.assertEqual(config.intermediate_size, 1536)
        self.assertEqual(config.num_attention_heads, 8)
        self.assertEqual(config.num_key_value_heads, 4)
        self.assertEqual(config.head_dim, 128)
        rope_config = config.rope_scaling
        if rope_config is not None:
            self.assertEqual(rope_config["rope_type"], "default")
        self.assertTrue(
            any("q_norm" in name for name, _ in self.model.named_parameters())
        )
        self.assertEqual(
            self.model.get_input_embeddings().weight.data_ptr(),
            self.model.get_output_embeddings().weight.data_ptr(),
        )
        self.assertEqual(MASTER_PARAMETER_DTYPE, torch.float32)
        self.assertEqual(COMPUTE_DTYPE, torch.bfloat16)
        self.assertEqual(
            {parameter.dtype for parameter in self.model.parameters()},
            {torch.float32},
        )
        self.assertEqual(self.model.config.torch_dtype, torch.float32)
        assert_fp32_master_parameters(self.model, where="unit test")

    def test_deterministic_training_contract_is_applied_and_fail_closed(self):
        with mock.patch.dict(
            "os.environ",
            {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        ):
            observed = configure_deterministic_training(
                42,
                process_index=3,
            )
        self.assertEqual(observed, DETERMINISM_CONTRACT)
        with mock.patch.dict(
            "os.environ",
            {"CUBLAS_WORKSPACE_CONFIG": ":16:8"},
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "CUBLAS_WORKSPACE_CONFIG drifted",
            ):
                configure_deterministic_training(42)

    def test_precision_contract_rejects_bf16_master_weights_and_export(self):
        model = torch.nn.Linear(3, 2).to(dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "FP32 master weights"):
            assert_fp32_master_parameters(model, where="unit test")
        with self.assertRaisesRegex(RuntimeError, "actual FP32"):
            assert_fp32_state_dict(model.state_dict(), where="unit test")

    def test_precision_contract_accepts_fp32_adam_state_and_gradients(self):
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        model(torch.ones(1, 3)).sum().backward()
        assert_fp32_gradients(model, where="unit test")
        assert_fp32_optimizer(
            optimizer,
            where="before update",
            require_initialized_state=False,
        )
        optimizer.step()
        assert_fp32_optimizer(
            optimizer,
            where="after update",
            require_initialized_state=True,
        )
        assert_fp32_state_dict(model.state_dict(), where="unit test")

    def test_precision_contract_rejects_bf16_adam_state(self):
        model = torch.nn.Linear(3, 2).to(dtype=torch.bfloat16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        model(torch.ones(1, 3, dtype=torch.bfloat16)).sum().backward()
        optimizer.step()
        with self.assertRaisesRegex(RuntimeError, "non-FP32 parameters"):
            assert_fp32_optimizer(
                optimizer,
                where="unit test",
                require_initialized_state=True,
            )

    def test_nonfinite_gradient_norm_fails_before_optimizer_step(self):
        self.assertEqual(
            assert_finite_gradient_norm(torch.tensor(3.5), where="unit test"),
            3.5,
        )
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(
                FloatingPointError,
                "gradient norm is non-finite",
            ):
                assert_finite_gradient_norm(value, where="unit test")

    def test_forward_has_85_logits_at_native_context_contract(self):
        input_ids = torch.randint(0, 85, (2, 17))
        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits
        self.assertEqual(tuple(logits.shape), (2, 17, 85))
        self.assertEqual(
            self.model.config._attn_implementation,
            "sdpa",
        )
        self.assertEqual(
            self.model.config.interleaved_attention_backend,
            "sdpa",
        )

    def test_one_shot_output_head_hook_observes_inner_bf16(self):
        class TinyHeadModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(4, 3)

            def get_output_embeddings(self):
                return self.head

            def forward(self, inputs):
                return self.head(inputs)

        from accelerate import Accelerator

        accelerator = Accelerator(mixed_precision="bf16")
        model = TinyHeadModel()
        evidence, _handle = register_bf16_output_head_assertion(model)
        model = accelerator.prepare(model)
        with accelerator.autocast():
            caller_output = model(torch.ones(2, 4, device=accelerator.device))
        self.assertTrue(evidence["validated"])
        self.assertEqual(evidence["output_dtype"], "bfloat16")
        # Accelerate deliberately converts the outer model result back to FP32;
        # this is why the assertion must live on the inner output head.
        self.assertEqual(caller_output.dtype, torch.float32)

    def test_attention_and_compile_backends_fail_closed(self):
        self.assertEqual(normalize_attention_backend("SDPA"), "sdpa")
        self.assertEqual(
            normalize_attention_backend("flash_attention_2"),
            "flash_attention_2",
        )
        with self.assertRaises(ValueError):
            normalize_attention_backend("eager")
        self.assertEqual(normalize_compile_mode(False), "none")
        self.assertEqual(normalize_compile_mode(True), "default")
        self.assertEqual(
            normalize_compile_mode("max-autotune"),
            "max-autotune",
        )
        with self.assertRaises(ValueError):
            normalize_compile_mode("mystery")

    def test_weights_only_requires_authenticated_fp32_hf_export(self):
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            export = Path(directory) / "export"
            _authenticate_test_hf_export(
                export,
                state_dict=source.state_dict(),
            )
            load_weights_only(
                target,
                export,
                destination_vocab=EXPECTED_VOCAB_81,
                context_length=2_048,
            )
        for source_value, target_value in zip(
            source.parameters(), target.parameters()
        ):
            self.assertTrue(torch.equal(source_value, target_value))

        with tempfile.TemporaryDirectory() as directory:
            unauthenticated = Path(directory) / "export"
            unauthenticated.mkdir()
            save_file(source.state_dict(), unauthenticated / "model.safetensors")
            with self.assertRaisesRegex(RuntimeError, "missing or invalid JSON"):
                load_weights_only(
                    target,
                    unauthenticated,
                    destination_vocab=EXPECTED_VOCAB_81,
                    context_length=2_048,
                )

    def test_weights_only_accepts_authenticated_indexed_fp32_shards(self):
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            export = Path(directory) / "export"
            export.mkdir()
            first = "model-00001-of-00002.safetensors"
            second = "model-00002-of-00002.safetensors"
            save_file({"weight": source.weight.detach()}, export / first)
            save_file({"bias": source.bias.detach()}, export / second)
            (export / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "weight_map": {"bias": second, "weight": first},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            _authenticate_test_hf_export(export)
            load_weights_only(
                target,
                export,
                destination_vocab=EXPECTED_VOCAB_81,
                context_length=2_048,
            )
        self.assertTrue(torch.equal(source.weight, target.weight))
        self.assertTrue(torch.equal(source.bias, target.bias))

    def test_weights_only_accepts_clean_hf_tied_embedding_export(self):
        with tempfile.TemporaryDirectory() as directory:
            self.model.save_pretrained(directory, safe_serialization=True)
            _authenticate_test_hf_export(
                Path(directory),
                vocab=EXPECTED_VOCAB_85,
                context_length=3_072,
            )
            target = build_interleaved_qwen_model(self.tokenizer)
            load_weights_only(
                target,
                directory,
                destination_vocab=EXPECTED_VOCAB_85,
                context_length=3_072,
            )
        self.assertTrue(
            torch.equal(
                self.model.get_input_embeddings().weight,
                target.get_input_embeddings().weight,
            )
        )
        self.assertEqual(
            target.get_input_embeddings().weight.data_ptr(),
            target.get_output_embeddings().weight.data_ptr(),
        )

    def test_context2048_vocab81_model_and_controlled_expansion(self):
        base_tokenizer = init_tokenizer(
            "LanTokenizer",
            {
                "include_move_numbers": False,
                "include_black_tripledots": False,
                "bos": "<bos>",
                "eos": "<eos>",
                "unk": "<unk>",
                "keep_result": False,
            },
        )
        base_model = build_interleaved_qwen_model(
            base_tokenizer,
            context_length=2_048,
        )
        self.assertEqual(len(base_tokenizer.get_vocab()), 81)
        self.assertEqual(
            sum(parameter.numel() for parameter in base_model.parameters()),
            47_243_264,
        )
        self.assertEqual(base_model.config.max_position_embeddings, 2_048)

        with tempfile.TemporaryDirectory() as directory:
            base_model.save_pretrained(directory, safe_serialization=True)
            _authenticate_test_hf_export(
                Path(directory),
                vocab=EXPECTED_VOCAB_81,
                context_length=2_048,
            )
            expanded = build_interleaved_qwen_model(
                self.tokenizer,
                context_length=2_048,
            )
            extra_before = (
                expanded.get_input_embeddings().weight[81:].detach().clone()
            )
            expanded_from = load_weights_only(
                expanded,
                directory,
                allow_vocab_expansion=True,
                destination_vocab=EXPECTED_VOCAB_85,
                context_length=2_048,
            )
        self.assertEqual(expanded_from, 81)
        self.assertTrue(
            torch.equal(
                expanded.get_input_embeddings().weight[:81],
                base_model.get_input_embeddings().weight,
            )
        )
        self.assertTrue(
            torch.equal(
                expanded.get_input_embeddings().weight[81:], extra_before
            )
        )
        self.assertEqual(
            expanded.get_input_embeddings().weight.data_ptr(),
            expanded.get_output_embeddings().weight.data_ptr(),
        )

    def test_weights_only_rejects_authenticated_ordinary_token_swap(self):
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        swapped = dict(EXPECTED_VOCAB_81)
        swapped["a1"], swapped["a2"] = swapped["a2"], swapped["a1"]
        with tempfile.TemporaryDirectory() as directory:
            export = Path(directory) / "export"
            _authenticate_test_hf_export(
                export,
                state_dict=source.state_dict(),
                vocab=swapped,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "complete tokenizer mapping drifted",
            ):
                load_weights_only(
                    target,
                    export,
                    destination_vocab=EXPECTED_VOCAB_81,
                    context_length=2_048,
                )

    def test_vocab85_ids_zero_through_eighty_equal_vocab81_exactly(self):
        transition = validate_vocab_transition(
            EXPECTED_VOCAB_81,
            EXPECTED_VOCAB_85,
            allow_vocab_expansion=True,
        )
        self.assertEqual(transition["transition"], "81-to-85")
        self.assertEqual(
            transition["source_vocab_mapping"],
            {
                token: token_id
                for token, token_id in transition[
                    "destination_vocab_mapping"
                ].items()
                if token_id < 81
            },
        )
        swapped = dict(EXPECTED_VOCAB_85)
        swapped["a1"], swapped["a2"] = swapped["a2"], swapped["a1"]
        with self.assertRaisesRegex(
            RuntimeError,
            "complete tokenizer mapping drifted",
        ):
            validate_vocab_transition(
                EXPECTED_VOCAB_81,
                swapped,
                allow_vocab_expansion=True,
            )

    def test_parent_identity_changes_for_wrong_same_shaped_export(self):
        source = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            _authenticate_test_hf_export(
                first,
                state_dict=source.state_dict(),
            )
            changed = {
                key: value.detach().clone()
                for key, value in source.state_dict().items()
            }
            changed["weight"][0, 0] += 1.0
            _authenticate_test_hf_export(second, state_dict=changed)
            first_identity = authenticated_weights_only_identity(
                first,
                destination_vocab=EXPECTED_VOCAB_81,
                allow_vocab_expansion=False,
                context_length=2_048,
            )
            second_identity = authenticated_weights_only_identity(
                second,
                destination_vocab=EXPECTED_VOCAB_81,
                allow_vocab_expansion=False,
                context_length=2_048,
            )
        self.assertNotEqual(
            first_identity["source_marker_sha256"],
            second_identity["source_marker_sha256"],
        )
        self.assertNotEqual(
            first_identity["source_export_manifest_sha256"],
            second_identity["source_export_manifest_sha256"],
        )

    def test_vocab81_and_vocab85_share_one_bitwise_scratch_initialization(self):
        base_tokenizer = init_tokenizer(
            "LanTokenizer",
            {
                "include_move_numbers": False,
                "include_black_tripledots": False,
                "bos": "<bos>",
                "eos": "<eos>",
                "unk": "<unk>",
                "keep_result": False,
            },
        )
        init_seed = 9_173

        # Ambient RNG use and builder call order must not affect either model,
        # and constructing a model must not advance the caller's RNG state.
        torch.manual_seed(123_456)
        torch.rand(37)
        caller_rng_before = torch.get_rng_state().clone()
        model81 = build_interleaved_qwen_model(
            base_tokenizer,
            context_length=2_048,
            model_init_seed=init_seed,
        )
        self.assertTrue(torch.equal(torch.get_rng_state(), caller_rng_before))
        torch.rand(113)
        caller_rng_before = torch.get_rng_state().clone()
        model85 = build_interleaved_qwen_model(
            self.tokenizer,
            context_length=2_048,
            model_init_seed=init_seed,
        )
        self.assertTrue(torch.equal(torch.get_rng_state(), caller_rng_before))

        parameters81 = dict(model81.named_parameters())
        parameters85 = dict(model85.named_parameters())
        self.assertEqual(set(parameters81), set(parameters85))
        input81 = model81.get_input_embeddings().weight
        input85 = model85.get_input_embeddings().weight
        embedding_name = next(
            name
            for name, parameter in parameters81.items()
            if parameter.data_ptr() == input81.data_ptr()
        )
        self.assertEqual(tuple(input81.shape), (81, 512))
        self.assertEqual(tuple(input85.shape), (85, 512))
        self.assertTrue(torch.equal(input81, input85[:81]))

        compared_non_vocab = 0
        for name, parameter81 in parameters81.items():
            if name == embedding_name:
                continue
            parameter85 = parameters85[name]
            self.assertEqual(
                tuple(parameter81.shape),
                tuple(parameter85.shape),
                msg=name,
            )
            self.assertTrue(
                torch.equal(parameter81, parameter85),
                msg=f"shared parameter differs: {name}",
            )
            compared_non_vocab += 1
        self.assertEqual(compared_non_vocab, len(parameters81) - 1)
        self.assertGreater(compared_non_vocab, 100)
        self.assertEqual(
            model81.get_input_embeddings().weight.data_ptr(),
            model81.get_output_embeddings().weight.data_ptr(),
        )
        self.assertEqual(
            model85.get_input_embeddings().weight.data_ptr(),
            model85.get_output_embeddings().weight.data_ptr(),
        )
        self.assertEqual(model81.config.vocab_size, 81)
        self.assertEqual(model85.config.vocab_size, 85)
        self.assertEqual(model81.config.interleaved_model_init_seed, init_seed)
        self.assertEqual(model85.config.interleaved_model_init_seed, init_seed)
        self.assertEqual(
            model81.config.interleaved_initialization_vocab_size,
            85,
        )
        self.assertEqual(
            model85.config.interleaved_initialization_vocab_size,
            85,
        )
        self.assertEqual(
            {parameter.dtype for parameter in model81.parameters()},
            {torch.float32},
        )
        self.assertEqual(
            {parameter.dtype for parameter in model85.parameters()},
            {torch.float32},
        )

        extra_rows = input85[81:].detach().clone()
        repeated85 = build_interleaved_qwen_model(
            self.tokenizer,
            context_length=2_048,
            model_init_seed=init_seed,
        )
        self.assertTrue(
            torch.equal(
                extra_rows,
                repeated85.get_input_embeddings().weight[81:],
            )
        )
        self.assertTrue(
            torch.equal(
                parameters85["model.layers.0.self_attn.q_proj.weight"],
                dict(repeated85.named_parameters())[
                    "model.layers.0.self_attn.q_proj.weight"
                ],
            )
        )

    def test_scratch_initialization_seed_changes_model(self):
        first = build_interleaved_qwen_model(
            self.tokenizer,
            context_length=2_048,
            model_init_seed=1,
        )
        second = build_interleaved_qwen_model(
            self.tokenizer,
            context_length=2_048,
            model_init_seed=2,
        )
        self.assertFalse(
            torch.equal(
                dict(first.named_parameters())[
                    "model.layers.0.self_attn.q_proj.weight"
                ],
                dict(second.named_parameters())[
                    "model.layers.0.self_attn.q_proj.weight"
                ],
            )
        )

    def test_exported_tokenizer_round_trip_has_explicit_env_token(self):
        tokenizer_cfg = {
            "include_move_numbers": False,
            "include_black_tripledots": False,
            "bos": "<bos>",
            "eos": "<eos>",
            "unk": "<unk>",
            "keep_result": False,
            "include_env_tokens": True,
            "include_reward_tokens": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            save_hf_tokenizer(
                tokenizer=self.tokenizer,
                tokcfg=tokenizer_cfg,
                save_directory=directory,
                model_max_length=EXPECTED_CONTEXT_LENGTH,
                env_id=self.tokenizer.call_env_id(),
            )
            loaded = AutoTokenizer.from_pretrained(
                directory,
                trust_remote_code=True,
            )
        self.assertEqual(loaded.model_max_length, EXPECTED_CONTEXT_LENGTH)
        self.assertEqual(loaded.env_token, "<call_env>")
        self.assertEqual(
            loaded.convert_tokens_to_ids("<call_env>"),
            self.tokenizer.call_env_id(),
        )


class InterleavedScheduleTests(unittest.TestCase):
    def _optimizer(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        return torch.optim.AdamW([parameter], lr=1e-3)

    def test_real_arc_has_exact_warmup_and_final_floor(self):
        scheduler = ExactArcCosine(
            self._optimizer(),
            arc_steps=[9_956],
            peak_lr=1e-3,
            min_lr=1e-5,
            warmup_ratio=0.05,
        )
        self.assertEqual(scheduler.position(0).warmup_steps, 497)
        self.assertAlmostEqual(scheduler.lr_for_update(496), 1e-3)
        self.assertAlmostEqual(scheduler.lr_for_update(9_955), 1e-5)

    def test_two_arcs_restart_lr_and_serialize_exactly(self):
        optimizer = self._optimizer()
        scheduler = ExactArcCosine(
            optimizer,
            arc_steps=[10, 10],
            peak_lr=1e-3,
            min_lr=1e-5,
            warmup_ratio=0.2,
        )
        self.assertEqual(scheduler.boundaries, (10,))
        self.assertAlmostEqual(scheduler.lr_for_update(0), 5e-4)
        self.assertAlmostEqual(scheduler.lr_for_update(1), 1e-3)
        self.assertAlmostEqual(scheduler.lr_for_update(9), 1e-5)
        self.assertAlmostEqual(scheduler.lr_for_update(10), 5e-4)
        for _ in range(13):
            scheduler.step()

        restored = ExactArcCosine(
            self._optimizer(),
            arc_steps=[10, 10],
            peak_lr=1e-3,
            min_lr=1e-5,
            warmup_ratio=0.2,
        )
        restored.load_state_dict(scheduler.state_dict())
        self.assertEqual(restored.completed_steps, 13)
        self.assertEqual(restored.get_last_lr(), scheduler.get_last_lr())

    def test_arc_resolution_rejects_ambiguous_compute(self):
        self.assertEqual(
            resolve_arc_steps({"arc_steps": [9_956, 9_956]}),
            (9_956, 9_956),
        )
        self.assertEqual(
            resolve_arc_steps({"total_steps": 19_912}),
            (19_912,),
        )
        with self.assertRaisesRegex(ValueError, "does not equal"):
            resolve_arc_steps(
                {"arc_steps": [9_956, 9_956], "total_steps": 9_956}
            )


class GlobalTokenLossTests(unittest.TestCase):
    def test_aligned_causal_sum_honors_label_and_attention_masks(self):
        logits = torch.tensor(
            [
                [
                    [3.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 3.0],
                    [3.0, 0.0, 0.0],
                ]
            ],
            requires_grad=True,
        )
        labels = torch.tensor([[0, 0, 1, 2]])
        attention_mask = torch.tensor([[1, 1, 1, 0]])
        loss_sum, valid = causal_ce_sum(logits, labels, attention_mask)
        expected = torch.nn.functional.cross_entropy(
            logits[:, :3, :].reshape(-1, 3),
            torch.tensor([0, 0, 1]),
            reduction="sum",
        )
        self.assertEqual(valid.item(), 3)
        self.assertTrue(torch.allclose(loss_sum, expected))

    def test_full_pt_record_contributes_every_aligned_target(self):
        logits = torch.zeros(1, EXPECTED_CONTEXT_LENGTH, 2)
        labels = torch.zeros(1, EXPECTED_CONTEXT_LENGTH, dtype=torch.long)
        attention_mask = torch.ones_like(labels)
        _, valid = causal_ce_sum(logits, labels, attention_mask)
        self.assertEqual(valid.item(), EXPECTED_CONTEXT_LENGTH)

    def test_unit_sft_weight_matches_original_raw_token_objective(self):
        torch.manual_seed(7)
        logits = torch.randn(3, 5, 7, requires_grad=True)
        labels = torch.tensor(
            [
                [0, 1, 2, 3, 4],
                [-100, -100, 3, 2, 1],
                [-100, -100, -100, -100, -100],
            ]
        )
        attention_mask = torch.ones_like(labels)
        sample_type = torch.tensor([1, 2, 0])
        original_sum, original_count = causal_ce_sum(
            logits, labels, attention_mask
        )
        weighted = weighted_causal_ce_sum(
            logits,
            labels,
            sample_type,
            sft_loss_weight=1.0,
            attention_mask=attention_mask,
        )
        self.assertTrue(torch.allclose(weighted[0], original_sum))
        self.assertEqual(weighted[1].item(), original_count.item())
        self.assertEqual(weighted[2].item(), 8)
        self.assertEqual(weighted[3].item(), 5)
        self.assertEqual(weighted[4].item(), 3)

    def test_sft_weight_scales_numerator_and_denominator_by_row_type(self):
        logits = torch.tensor(
            [
                [[2.0, 0.0], [0.0, 2.0]],
                [[0.0, 2.0], [2.0, 0.0]],
            ],
            requires_grad=True,
        )
        labels = torch.tensor([[0, 1], [0, 1]])
        sample_type = torch.tensor([1, 2])
        (
            weighted_sum,
            weighted_count,
            raw_count,
            pretrain_count,
            sft_count,
            pretrain_sum,
            sft_sum,
        ) = weighted_causal_ce_sum(
            logits,
            labels,
            sample_type,
            sft_loss_weight=3.0,
        )
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 2),
            labels.reshape(-1),
            reduction="none",
        ).reshape(2, 2)
        expected = losses[0].sum() + 3.0 * losses[1].sum()
        self.assertTrue(torch.allclose(weighted_sum, expected))
        self.assertEqual(weighted_count.item(), 8.0)
        self.assertEqual(raw_count.item(), 4)
        self.assertEqual(pretrain_count.item(), 2)
        self.assertEqual(sft_count.item(), 2)
        self.assertTrue(torch.allclose(pretrain_sum, losses[0].sum()))
        self.assertTrue(torch.allclose(sft_sum, losses[1].sum()))
        (weighted_sum / weighted_count).backward()
        self.assertIsNotNone(logits.grad)

    def test_sft_weight_and_sample_types_fail_closed(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                normalize_sft_loss_weight(value)
        logits = torch.zeros(1, 1, 2)
        labels = torch.zeros(1, 1, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "unsupported sample_type"):
            weighted_causal_ce_sum(
                logits,
                labels,
                torch.tensor([99]),
                sft_loss_weight=1.0,
            )
        with self.assertRaisesRegex(ValueError, "padding records"):
            weighted_causal_ce_sum(
                logits,
                labels,
                torch.tensor([0]),
                sft_loss_weight=1.0,
            )

    def test_ddp_scaling_produces_global_valid_token_mean_gradient(self):
        rank_zero = torch.tensor(2.0, requires_grad=True)
        rank_one = torch.tensor(5.0, requires_grad=True)
        loss_zero = globally_normalized_backward_loss(
            3.0 * rank_zero,
            global_valid_tokens=8,
            world_size=2,
        )
        loss_one = globally_normalized_backward_loss(
            7.0 * rank_one,
            global_valid_tokens=8,
            world_size=2,
        )
        loss_zero.backward()
        loss_one.backward()
        ddp_averaged_gradient = (rank_zero.grad + rank_one.grad) / 2
        self.assertAlmostEqual(ddp_averaged_gradient.item(), (3.0 + 7.0) / 8.0)


class InterleavedStateAndTopologyTests(unittest.TestCase):
    def test_runtime_identity_hashes_complete_canonical_distribution_inventory(self):
        distributions = [
            SimpleNamespace(metadata={"Name": "Zeta_Pkg"}, version="2.0"),
            SimpleNamespace(metadata={"Name": "alpha.pkg"}, version="1.3"),
        ]
        with mock.patch(
            "training.interleaved_hf_trainer.importlib.metadata.distributions",
            return_value=distributions,
        ) as inventory:
            first = runtime_distribution_identity()
        inventory.assert_called_once_with(path=[RUNTIME_SITE_PACKAGES])
        with mock.patch(
            "training.interleaved_hf_trainer.importlib.metadata.distributions",
            return_value=list(reversed(distributions)),
        ):
            second = runtime_distribution_identity()
        self.assertEqual(first, second)
        self.assertEqual(first["distribution_count"], 2)
        self.assertRegex(first["inventory_sha256"], r"^[0-9a-f]{64}$")

        versions = {"alpha": "1.0", "zeta": "2.0"}
        with mock.patch(
            "training.interleaved_hf_trainer.importlib.metadata.version",
            side_effect=lambda name: versions[name],
        ):
            self.assertEqual(validate_runtime_package_versions(versions), versions)
            with self.assertRaisesRegex(RuntimeError, "runtime package drift"):
                validate_runtime_package_versions({"alpha": "9.9"})

    def test_existing_export_and_snapshot_reconciliation_have_symmetric_barriers(self):
        class FakeAccelerator:
            def __init__(self, *, is_main_process: bool):
                self.is_main_process = is_main_process
                self.wait_calls = 0

            def wait_for_everyone(self):
                self.wait_calls += 1

            def get_state_dict(self, model):
                return model.state_dict()

            def print(self, *_args, **_kwargs):
                return None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / "final"
            export.mkdir()
            snapshot = root / "run" / "snapshots" / "step_1"
            snapshot.mkdir(parents=True)
            state = {"global_step": 1}
            for is_main_process in (False, True):
                export_trainer = object.__new__(InterleavedHFTrainer)
                export_trainer.acc = FakeAccelerator(
                    is_main_process=is_main_process
                )
                export_trainer.model = torch.nn.Linear(2, 2)
                export_trainer._trainer_state = lambda: state
                with mock.patch(
                    "training.interleaved_hf_trainer.validate_completed_hf_export",
                    return_value={"state": state},
                ) as validate_export:
                    export_trainer.export_hf(export)
                validate_export.assert_called_once_with(export)
                self.assertEqual(export_trainer.acc.wait_calls, 2)

                snapshot_trainer = object.__new__(InterleavedHFTrainer)
                snapshot_trainer.acc = FakeAccelerator(
                    is_main_process=is_main_process
                )
                snapshot_trainer.global_step = 1
                snapshot_trainer.output_dir = root / "run"
                with mock.patch(
                    "training.interleaved_hf_trainer."
                    "validate_completed_diagnostic_snapshot"
                ) as validate_snapshot:
                    snapshot_trainer.save_diagnostic_snapshot()
                validate_snapshot.assert_called_once_with(snapshot)
                self.assertEqual(snapshot_trainer.acc.wait_calls, 1)

    def test_non_main_rank_accepts_rank_zero_atomic_staging_directories(self):
        class FakeNonMainAccelerator:
            is_main_process = False
            num_processes = 2

            def __init__(self):
                self.wait_calls = 0
                self.saved_state_path = None

            def wait_for_everyone(self):
                self.wait_calls += 1

            def save_state(self, path):
                self.saved_state_path = path

            def get_state_dict(self, model):
                return model.state_dict()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = {"global_step": 1}

            checkpoint_trainer = object.__new__(InterleavedHFTrainer)
            checkpoint_trainer.acc = FakeNonMainAccelerator()
            checkpoint_trainer.model = torch.nn.Linear(2, 2)
            checkpoint_trainer.optimizer = torch.optim.AdamW(
                checkpoint_trainer.model.parameters()
            )
            checkpoint_trainer.global_step = 1
            checkpoint_trainer._committed_checkpoint_paths = {}
            checkpoint_trainer._trainer_state = lambda: state
            checkpoint_root = root / "checkpoint-run"
            checkpoint_temporary = temporary_checkpoint_directory(
                checkpoint_root, 1
            )
            checkpoint_temporary.mkdir(parents=True)
            with (
                mock.patch(
                    "training.interleaved_hf_trainer."
                    "assert_fp32_optimizer"
                ),
                mock.patch(
                    "training.interleaved_hf_trainer."
                    "validate_completed_checkpoint",
                    return_value={"state": state},
                ),
            ):
                checkpoint = (
                    checkpoint_trainer._save_resume_checkpoint_unlocked(
                        checkpoint_root
                    )
                )
            self.assertEqual(
                checkpoint_trainer.acc.saved_state_path,
                str(checkpoint_temporary),
            )
            self.assertEqual(checkpoint, checkpoint_directory(checkpoint_root, 1))

            snapshot_trainer = object.__new__(InterleavedHFTrainer)
            snapshot_trainer.acc = FakeNonMainAccelerator()
            snapshot_trainer.global_step = 1
            snapshot_trainer.output_dir = root / "snapshot-run"
            snapshot_temporary = (
                snapshot_trainer.output_dir / "snapshots" / ".step_1.tmp"
            )
            snapshot_temporary.mkdir(parents=True)
            snapshot_trainer.save_resume_checkpoint = mock.Mock(
                return_value=snapshot_temporary / "resume" / "checkpoint"
            )
            snapshot_trainer.export_hf = mock.Mock()
            with mock.patch(
                "training.interleaved_hf_trainer."
                "validate_completed_diagnostic_snapshot"
            ):
                snapshot_trainer._save_diagnostic_snapshot_unlocked()
            snapshot_trainer.save_resume_checkpoint.assert_called_once()
            snapshot_trainer.export_hf.assert_called_once_with(
                snapshot_temporary / "hf"
            )

            export_trainer = object.__new__(InterleavedHFTrainer)
            export_trainer.acc = FakeNonMainAccelerator()
            export_trainer.model = torch.nn.Linear(2, 2)
            export_trainer._trainer_state = lambda: state
            export = root / "export-run" / "final"
            export_temporary = export.with_name(".final.tmp")
            export_temporary.mkdir(parents=True)
            with mock.patch(
                "training.interleaved_hf_trainer.validate_completed_hf_export",
                return_value={"state": state},
            ):
                export_trainer._export_hf_unlocked(export)


    def test_mixed_canary_proves_pt_and_masked_bos_sft_in_same_update(self):
        trainer = object.__new__(InterleavedHFTrainer)
        trainer.configured_provenance = {
            "canary_sample_contract": "mixed-pt-sft"
        }
        trainer.runtime_provenance = {}
        trainer.tokenizer = SimpleNamespace(bos_id=lambda: 0)
        trainer.acc = SimpleNamespace(device=torch.device("cpu"))
        input_ids = torch.tensor(
            [
                [0, 4, 5, 6],
                [0, 10, 11, 0],
            ],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [
                [4, 5, 6, 7],
                [IGNORE_INDEX, 11, 12, IGNORE_INDEX],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [[1, 1, 1, 1], [1, 1, 1, 0]],
            dtype=torch.long,
        )
        sample_type = torch.tensor(
            [SAMPLE_PRETRAIN, SAMPLE_SFT],
            dtype=torch.long,
        )

        trainer._validate_canary_sample_contract(
            input_ids,
            labels,
            attention_mask,
            sample_type,
        )
        self.assertEqual(
            trainer.runtime_provenance["canary_sample_evidence"],
            {
                "contract": "mixed-pt-sft",
                "global_pretrain_rows": 1,
                "global_sft_rows": 1,
                "global_pretrain_supervised_tokens": 4,
                "global_sft_supervised_tokens": 2,
                "pt_leading_bos_validated": True,
                "sft_bos_and_mask_validated": True,
            },
        )

        unmasked_bos = labels.clone()
        unmasked_bos[1, 0] = 10
        with self.assertRaisesRegex(RuntimeError, "prompt/BOS target is not masked"):
            trainer._validate_canary_sample_contract(
                input_ids,
                unmasked_bos,
                attention_mask,
                sample_type,
            )

        missing_pt_bos = input_ids.clone()
        missing_pt_bos[0, 0] = 3
        with self.assertRaisesRegex(RuntimeError, "PT row must start"):
            trainer._validate_canary_sample_contract(
                missing_pt_bos,
                labels,
                attention_mask,
                sample_type,
            )

        masked_pt_first_target = labels.clone()
        masked_pt_first_target[0, 0] = IGNORE_INDEX
        with self.assertRaisesRegex(RuntimeError, "does not predict an active target"):
            trainer._validate_canary_sample_contract(
                input_ids,
                masked_pt_first_target,
                attention_mask,
                sample_type,
            )

    def test_diagnostic_ce_interval_is_exact_token_weighted_delta(self):
        zero = new_diagnostic_ce_cumulative()
        cumulative = add_diagnostic_ce_step(
            zero,
            step=1,
            pretrain_loss_sum=4.0,
            pretrain_token_count=2,
            sft_loss_sum=9.0,
            sft_token_count=3,
        )
        cumulative = add_diagnostic_ce_step(
            cumulative,
            step=2,
            pretrain_loss_sum=6.0,
            pretrain_token_count=3,
            sft_loss_sum=5.0,
            sft_token_count=2,
        )
        first = diagnostic_ce_interval(zero, cumulative)
        self.assertEqual(first["start_step"], 1)
        self.assertEqual(first["end_step"], 2)
        self.assertEqual(first["optimizer_steps"], 2)
        self.assertEqual(first["pretrain_loss_sum"], 10.0)
        self.assertEqual(first["pretrain_token_count"], 5)
        self.assertEqual(first["pretrain_contributing_steps"], 2)
        self.assertEqual(first["pretrain_token_ce"], 2.0)
        self.assertEqual(first["sft_loss_sum"], 14.0)
        self.assertEqual(first["sft_token_count"], 5)
        self.assertEqual(first["sft_contributing_steps"], 2)
        self.assertEqual(first["sft_token_ce"], 2.8)
        self.assertEqual(
            first["measurement_semantics"],
            "token_weighted_training_stream_pre_update_batch_logits",
        )
        self.assertFalse(first["held_out"])
        self.assertFalse(first["endpoint_checkpoint_evaluation"])

        next_cumulative = add_diagnostic_ce_step(
            cumulative,
            step=3,
            pretrain_loss_sum=8.0,
            pretrain_token_count=4,
            sft_loss_sum=12.0,
            sft_token_count=6,
        )
        second = diagnostic_ce_interval(cumulative, next_cumulative)
        self.assertEqual(second["start_step"], 3)
        self.assertEqual(second["end_step"], 3)
        self.assertEqual(second["pretrain_loss_sum"], 8.0)
        self.assertEqual(second["sft_loss_sum"], 12.0)

    def test_diagnostic_ce_interval_fails_closed_on_gap_or_zero_mass(self):
        zero = new_diagnostic_ce_cumulative()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            add_diagnostic_ce_step(
                zero,
                step=2,
                pretrain_loss_sum=1.0,
                pretrain_token_count=1,
                sft_loss_sum=1.0,
                sft_token_count=1,
            )
        pretrain_only = add_diagnostic_ce_step(
            zero,
            step=1,
            pretrain_loss_sum=2.0,
            pretrain_token_count=1,
            sft_loss_sum=0.0,
            sft_token_count=0,
        )
        with self.assertRaisesRegex(ValueError, "no valid sft mass"):
            diagnostic_ce_interval(zero, pretrain_only)

    def test_diagnostic_ce_resume_boundary_tamper_fails_closed(self):
        zero = new_diagnostic_ce_cumulative()
        cumulative = add_diagnostic_ce_step(
            zero,
            step=1,
            pretrain_loss_sum=2.0,
            pretrain_token_count=1,
            sft_loss_sum=3.0,
            sft_token_count=1,
        )
        interval = diagnostic_ce_interval(zero, cumulative)
        state = {
            "diagnostic_ce_cumulative": cumulative,
            "diagnostic_ce_interval_base": cumulative,
            "diagnostic_last_ce_interval_base": zero,
            "diagnostic_last_ce_interval": interval,
        }
        validated = validate_diagnostic_ce_resume_state(
            state, global_step=1
        )
        self.assertEqual(validated[0], cumulative)

        tampered_boundary = {
            **state,
            "diagnostic_last_ce_interval_base": {
                **zero,
                "through_step": 1,
            },
        }
        with self.assertRaisesRegex(
            ValueError, "interval delta mismatch|empty or reversed"
        ):
            validate_diagnostic_ce_resume_state(
                tampered_boundary, global_step=1
            )

        tampered_current_base = {
            **state,
            "diagnostic_ce_interval_base": zero,
        }
        with self.assertRaisesRegex(ValueError, "boundary/state mismatch"):
            validate_diagnostic_ce_resume_state(
                tampered_current_base, global_step=1
            )

    def test_no_accumulation_topology_keeps_old_token_batch_scale(self):
        validate_topology(
            {
                "local_batch_size": 21,
                "gradient_accumulation_steps": 1,
                "mixed_precision": "bf16",
            },
            world_size=8,
        )
        self.assertEqual(EXPECTED_TOKEN_POSITIONS_PER_UPDATE, 516_096)
        with self.assertRaisesRegex(ValueError, "does not use"):
            validate_topology(
                {
                    "local_batch_size": 21,
                    "gradient_accumulation_steps": 2,
                    "mixed_precision": "bf16",
                },
                world_size=8,
            )
        validate_topology(
            {
                "local_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "mixed_precision": "bf16",
            },
            world_size=1,
            allow_topology_override=True,
        )
        for precision in ("fp32", "no"):
            with self.assertRaisesRegex(
                ValueError, "accuracy-first BF16 mixed precision"
            ):
                validate_topology(
                    {
                        "local_batch_size": 2,
                        "gradient_accumulation_steps": 1,
                        "mixed_precision": precision,
                    },
                    world_size=1,
                    allow_topology_override=True,
                )

    def test_resume_fails_closed_on_manifest_or_arc_change(self):
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "global_step": 123,
            "manifest_hash": "abc",
            "manifest_cursor": 123,
            "arc_steps": [9_956],
            "local_batch_size": 21,
            "world_size": 8,
            "gradient_accumulation_steps": 1,
            "precision_contract": dict(PRECISION_CONTRACT),
        }
        validate_resume_state(
            state,
            manifest_hash="abc",
            arc_steps=[9_956],
            local_batch_size=21,
            world_size=8,
        )
        with self.assertRaisesRegex(ValueError, "manifest_hash"):
            validate_resume_state(
                state,
                manifest_hash="different",
                arc_steps=[9_956],
                local_batch_size=21,
                world_size=8,
            )
        with self.assertRaisesRegex(ValueError, "arc_steps"):
            validate_resume_state(
                state,
                manifest_hash="abc",
                arc_steps=[19_912],
                local_batch_size=21,
                world_size=8,
            )
        # Old v1 states omitted the field and therefore bind to weight 1.0.
        validate_resume_state(
            state,
            manifest_hash="abc",
            arc_steps=[9_956],
            local_batch_size=21,
            world_size=8,
            sft_loss_weight=1.0,
        )
        with self.assertRaisesRegex(ValueError, "sft_loss_weight"):
            validate_resume_state(
                state,
                manifest_hash="abc",
                arc_steps=[9_956],
                local_batch_size=21,
                world_size=8,
                sft_loss_weight=171.0,
            )
        weighted_state = {**state, "sft_loss_weight": 171.0}
        validate_resume_state(
            weighted_state,
            manifest_hash="abc",
            arc_steps=[9_956],
            local_batch_size=21,
            world_size=8,
            sft_loss_weight=171.0,
        )

    def test_resume_binds_parent_seed_and_stable_initial_command(self):
        parent = {
            "mode": "weights-only",
            "source_marker_sha256": "a" * 64,
            "source_export_manifest_sha256": "b" * 64,
        }
        launch = {
            "schema": "interleaved-initial-launch-command-v1",
            "argv": ["accelerate", "launch", "train.py", "--weights-only", "pt"],
            "sha256": "c" * 64,
        }
        provenance = {
            "seed": 42,
            "initialization_identity": parent,
            "initial_launch_command": launch,
        }
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "global_step": 1,
            "manifest_hash": "manifest",
            "manifest_cursor": 1,
            "arc_steps": [2],
            "local_batch_size": 32,
            "world_size": 8,
            "gradient_accumulation_steps": 1,
            "precision_contract": dict(PRECISION_CONTRACT),
            "configured_provenance": provenance,
        }
        validate_resume_state(
            state,
            manifest_hash="manifest",
            arc_steps=[2],
            local_batch_size=32,
            world_size=8,
            configured_provenance=provenance,
        )
        for key, changed in (
            ("seed", 43),
            (
                "initialization_identity",
                {**parent, "source_marker_sha256": "d" * 64},
            ),
            (
                "initial_launch_command",
                {**launch, "sha256": "e" * 64},
            ),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError,
                "configured_provenance",
            ):
                validate_resume_state(
                    state,
                    manifest_hash="manifest",
                    arc_steps=[2],
                    local_batch_size=32,
                    world_size=8,
                    configured_provenance={**provenance, key: changed},
                )

    def test_staged_sft_resume_contract_survives_process_boundary(self):
        provenance = {
            "seed": 42,
            "initialization_identity": {
                "mode": "weights-only",
                "source_marker_sha256": "a" * 64,
                "source_export_manifest_sha256": "b" * 64,
            },
            "initial_launch_command": {
                "schema": "interleaved-initial-launch-command-v1",
                "argv": ["train", "--weights-only", "/pt/final"],
                "sha256": "c" * 64,
            },
        }
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "global_step": 1,
            "manifest_hash": "sft-manifest",
            "manifest_cursor": 1,
            "arc_steps": [2],
            "local_batch_size": 32,
            "world_size": 8,
            "gradient_accumulation_steps": 1,
            "precision_contract": dict(PRECISION_CONTRACT),
            "configured_provenance": provenance,
        }
        child = """
import json, sys
from training.interleaved_hf_trainer import validate_resume_state
state = json.loads(sys.argv[1])
provenance = json.loads(sys.argv[2])
validate_resume_state(
    state,
    manifest_hash='sft-manifest',
    arc_steps=[2],
    local_batch_size=32,
    world_size=8,
    configured_provenance=provenance,
)
"""
        accepted = subprocess.run(
            [
                sys.executable,
                "-c",
                child,
                json.dumps(state, sort_keys=True),
                json.dumps(provenance, sort_keys=True),
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        wrong_parent = {
            **provenance,
            "initialization_identity": {
                **provenance["initialization_identity"],
                "source_marker_sha256": "d" * 64,
            },
        }
        rejected = subprocess.run(
            [
                sys.executable,
                "-c",
                child,
                json.dumps(state, sort_keys=True),
                json.dumps(wrong_parent, sort_keys=True),
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("configured_provenance", rejected.stderr)

    def test_cli_makes_full_resume_and_weights_only_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--config",
                    "experiment.yaml",
                    "--resume",
                    "latest",
                    "--weights-only",
                    "final",
                ]
            )

    def test_immutable_checkpoint_requires_authenticated_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            incomplete = temporary_checkpoint_directory(root, 1)
            _write_test_accelerator_payload(incomplete, step=1)
            with self.assertRaisesRegex(
                RuntimeError,
                "not committed|neither a committed checkpoint",
            ):
                resolve_resume_checkpoint(incomplete)

            write_completion_marker(incomplete, step=1)
            final = checkpoint_directory(root, 1)
            publish_checkpoint_directory(incomplete, final)
            write_latest_checkpoint_pointer(root, final)
            self.assertEqual(resolve_resume_checkpoint(root), final.resolve())
            self.assertTrue((root / LATEST_CHECKPOINT_SYMLINK).is_symlink())
            self.assertEqual(
                (root / LATEST_CHECKPOINT_SYMLINK).resolve(),
                final.resolve(),
            )
            self.assertEqual(
                validate_checkpoint_run_root(root),
                final.resolve(),
            )

            (final / "model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "identity drifted"):
                validate_completed_checkpoint(final)

    def test_persisted_checkpoint_inspection_rejects_bf16_model_or_adam(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bf16_model = root / "bf16-model"
            _write_test_accelerator_payload(
                bf16_model,
                step=1,
                model_dtype=torch.bfloat16,
            )
            with self.assertRaisesRegex(RuntimeError, "non-FP32 floating"):
                inspect_accelerator_checkpoint_fp32(bf16_model)

            bf16_adam = root / "bf16-adam"
            _write_test_accelerator_payload(
                bf16_adam,
                step=1,
                adam_dtype=torch.bfloat16,
            )
            with self.assertRaisesRegex(RuntimeError, "Adam tensor is not FP32"):
                inspect_accelerator_checkpoint_fp32(bf16_adam)

    def test_hf_export_publication_is_authenticated_atomic_and_no_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            temporary = parent / ".final.tmp"
            final = parent / "final"
            _authenticate_test_hf_export(
                temporary,
                state_dict={"weight": torch.ones(2, dtype=torch.float32)},
                step=1,
            )
            publish_hf_export_directory(temporary, final)
            validated = validate_completed_hf_export(final)
            self.assertEqual(validated["state"]["global_step"], 1)

            second = parent / ".second.tmp"
            _authenticate_test_hf_export(
                second,
                state_dict={"weight": torch.zeros(2, dtype=torch.float32)},
                step=1,
            )
            with self.assertRaises(FileExistsError):
                publish_hf_export_directory(second, final)
            self.assertTrue(second.is_dir())
            self.assertTrue(
                torch.equal(
                    torch.ones(2),
                    load_file(str(final / "model.safetensors"))["weight"],
                )
            )

    def test_unsupported_noreplace_requires_explicit_serialized_fallback(self):
        class UnsupportedRenameAt2:
            argtypes = None
            restype = None

            def __call__(self, *_args):
                return -1

        fake_libc = SimpleNamespace(renameat2=UnsupportedRenameAt2())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temporary = root / ".step.tmp"
            final = root / "step"
            temporary.mkdir()
            with (
                mock.patch.object(
                    immutable_checkpoint.sys,
                    "platform",
                    "linux",
                ),
                mock.patch.object(
                    immutable_checkpoint.ctypes,
                    "CDLL",
                    return_value=fake_libc,
                ),
                mock.patch.object(
                    immutable_checkpoint.ctypes,
                    "get_errno",
                    return_value=errno.EINVAL,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "checkpoint_volume_commit_lock is held",
                ):
                    immutable_checkpoint._rename_directory_noreplace(
                        temporary,
                        final,
                    )
                immutable_checkpoint._rename_directory_noreplace(
                    temporary,
                    final,
                    allow_serialized_fallback=True,
                )
            self.assertTrue(final.is_dir())

            second = root / ".second.tmp"
            second.mkdir()
            with (
                mock.patch.object(
                    immutable_checkpoint.sys,
                    "platform",
                    "linux",
                ),
                mock.patch.object(
                    immutable_checkpoint.ctypes,
                    "CDLL",
                    return_value=fake_libc,
                ),
                mock.patch.object(
                    immutable_checkpoint.ctypes,
                    "get_errno",
                    return_value=errno.EINVAL,
                ),
            ):
                with self.assertRaises(FileExistsError):
                    immutable_checkpoint._rename_directory_noreplace(
                        second,
                        final,
                        allow_serialized_fallback=True,
                    )
            self.assertTrue(second.is_dir())

    def test_diagnostic_snapshot_authenticates_nested_resume_hf_and_interval(self):
        interval = {
            "end_step": 1,
            "pretrain_token_count": 8,
            "sft_token_count": 5,
        }
        state = {
            "global_step": 1,
            "diagnostic_last_ce_interval": interval,
        }
        with tempfile.TemporaryDirectory() as directory:
            snapshots = Path(directory) / "snapshots"
            temporary = snapshots / ".step_1.tmp"
            final = snapshots / "step_1"
            resume_root = temporary / "resume"
            checkpoint_temporary = temporary_checkpoint_directory(resume_root, 1)
            _write_test_accelerator_payload(
                checkpoint_temporary,
                step=1,
                trainer_state=state,
            )
            write_completion_marker(checkpoint_temporary, step=1)
            checkpoint_final = checkpoint_directory(resume_root, 1)
            publish_checkpoint_directory(checkpoint_temporary, checkpoint_final)
            write_latest_checkpoint_pointer(resume_root, checkpoint_final)
            _authenticate_test_hf_export(
                temporary / "hf",
                state_dict={"weight": torch.ones(2, dtype=torch.float32)},
                step=1,
                trainer_state=state,
            )
            write_diagnostic_snapshot_completion_marker(
                temporary,
                global_step=1,
                interval_unweighted_ce=interval,
            )
            publish_diagnostic_snapshot_directory(temporary, final)
            validated = validate_completed_diagnostic_snapshot(final)
            self.assertEqual(validated["state"], state)

            (final / "hf" / "config.json").write_text(
                '{"dtype":"float32","tampered":true}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "identity drifted"):
                validate_completed_diagnostic_snapshot(final)

    def test_checkpoint_publication_fsyncs_each_durability_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            temporary = temporary_checkpoint_directory(root, 1)
            _write_test_accelerator_payload(temporary, step=1)
            final = checkpoint_directory(root, 1)

            events: list[tuple[str, str, str | None]] = []
            real_replace = immutable_checkpoint.os.replace
            real_symlink = immutable_checkpoint.os.symlink
            real_publish = immutable_checkpoint._rename_directory_noreplace

            def record_file(path):
                events.append(("file", Path(path).name, None))

            def record_directory(path):
                events.append(("directory", Path(path).name, None))

            def record_replace(source, target):
                events.append(
                    ("replace", Path(source).name, Path(target).name)
                )
                return real_replace(source, target)

            def record_symlink(target, link_name):
                events.append(
                    ("symlink", str(target), Path(link_name).name)
                )
                return real_symlink(target, link_name)

            def record_publish(source, target, **kwargs):
                events.append(
                    ("publish", Path(source).name, Path(target).name)
                )
                return real_publish(source, target, **kwargs)

            with mock.patch.object(
                immutable_checkpoint,
                "_fsync_file",
                side_effect=record_file,
            ), mock.patch.object(
                immutable_checkpoint,
                "_fsync_directory",
                side_effect=record_directory,
            ), mock.patch.object(
                immutable_checkpoint.os,
                "replace",
                side_effect=record_replace,
            ), mock.patch.object(
                immutable_checkpoint.os,
                "symlink",
                side_effect=record_symlink,
            ), mock.patch.object(
                immutable_checkpoint,
                "_rename_directory_noreplace",
                side_effect=record_publish,
            ):
                write_completion_marker(temporary, step=1)
                publish_checkpoint_directory(temporary, final)
                write_latest_checkpoint_pointer(root, final)

            def event_index(event, *, after=-1):
                return next(
                    index
                    for index, observed in enumerate(events)
                    if index > after and observed == event
                )

            marker_file = event_index(("file", ".complete.json.tmp", None))
            self.assertLess(
                event_index(("file", "model.safetensors", None)),
                marker_file,
            )
            self.assertLess(
                event_index(("file", "trainer_state.json", None)),
                marker_file,
            )
            marker_replace = event_index(
                ("replace", ".complete.json.tmp", ".complete.json")
            )
            self.assertLess(marker_file, marker_replace)
            marker_directory_sync = event_index(
                ("directory", temporary.name, None),
                after=marker_replace,
            )
            checkpoint_replace = event_index(
                ("publish", temporary.name, final.name),
                after=marker_directory_sync,
            )
            checkpoint_parent_sync = event_index(
                ("directory", final.parent.name, None),
                after=checkpoint_replace,
            )
            self.assertLess(checkpoint_replace, checkpoint_parent_sync)

            symlink_create = event_index(
                ("symlink", final.relative_to(root).as_posix(), ".latest.tmp"),
                after=checkpoint_parent_sync,
            )
            symlink_replace = event_index(
                ("replace", ".latest.tmp", LATEST_CHECKPOINT_SYMLINK),
                after=symlink_create,
            )
            symlink_parent_sync = event_index(
                ("directory", root.name, None),
                after=symlink_replace,
            )
            pointer_file = event_index(
                ("file", f"{LATEST_CHECKPOINT_POINTER}.tmp", None),
                after=symlink_parent_sync,
            )
            pointer_replace = event_index(
                (
                    "replace",
                    f"{LATEST_CHECKPOINT_POINTER}.tmp",
                    LATEST_CHECKPOINT_POINTER,
                ),
                after=pointer_file,
            )
            pointer_parent_sync = event_index(
                ("directory", root.name, None),
                after=pointer_replace,
            )
            self.assertLess(pointer_replace, pointer_parent_sync)

    def test_nonempty_unauthenticated_checkpoint_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir()
            (root / "config.yaml").write_text("partial: true\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "no checkpoint directory"):
                validate_checkpoint_run_root(root)

    def test_resume_output_root_rejects_unauthenticated_final_and_snapshot_dirs(self):
        def committed_root(parent: Path) -> tuple[Path, Path, dict]:
            root = parent / "run"
            temporary = temporary_checkpoint_directory(root, 1)
            state = {"global_step": 1}
            _write_test_accelerator_payload(
                temporary,
                step=1,
                trainer_state=state,
            )
            write_completion_marker(temporary, step=1)
            final = checkpoint_directory(root, 1)
            publish_checkpoint_directory(temporary, final)
            write_latest_checkpoint_pointer(root, final)
            return root, final, state

        with tempfile.TemporaryDirectory() as directory:
            root, checkpoint, state = committed_root(Path(directory))
            (root / "final").mkdir()
            trainer = object.__new__(InterleavedHFTrainer)
            trainer.output_dir = root
            trainer.resume_path = str(checkpoint)
            trainer.resume_state = state
            with self.assertRaisesRegex(RuntimeError, "missing or invalid JSON"):
                trainer._validate_output_root_before_write()

        with tempfile.TemporaryDirectory() as directory:
            root, checkpoint, state = committed_root(Path(directory))
            (root / "snapshots" / ".step_1.tmp").mkdir(parents=True)
            trainer = object.__new__(InterleavedHFTrainer)
            trainer.output_dir = root
            trainer.resume_path = str(checkpoint)
            trainer.resume_state = state
            with self.assertRaisesRegex(RuntimeError, "incomplete or unknown"):
                trainer._validate_output_root_before_write()

    def test_pointer_to_older_step_rejects_newer_committed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            committed = []
            for step in (1, 2):
                temporary = temporary_checkpoint_directory(root, step)
                _write_test_accelerator_payload(temporary, step=step)
                write_completion_marker(temporary, step=step)
                final = checkpoint_directory(root, step)
                publish_checkpoint_directory(temporary, final)
                committed.append(final)
            # Simulate a kill after step 2's temp->final rename but before its
            # latest-pointer publication.
            write_latest_checkpoint_pointer(root, committed[0])
            with self.assertRaisesRegex(RuntimeError, "newest committed"):
                validate_checkpoint_run_root(root)


class _TinyTokenizer:
    def __init__(self):
        self._vocab = {f"token_{index}": index for index in range(85)}

    def get_vocab(self):
        return dict(self._vocab)

    def bos_id(self):
        return 0

    def eos_id(self):
        return 1

    def pad_id(self):
        return 0


class _TinyCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(85, 4)
        self.head = torch.nn.Linear(4, 85)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    def get_output_embeddings(self):
        return self.head


class _FakeInterleavedStream:
    def __init__(self, *, manifest_hash: str, start_cursor: int, total_steps: int):
        self.manifest_hash = manifest_hash
        self.source_manifest_hash = "source"
        self.selection_hash = "selection"
        self.sft_cache_hash = "sft"
        self.cursor = start_cursor
        self.total_steps = total_steps
        self.remaining_steps = total_steps - start_cursor
        self._offered_cursor = None

    def __iter__(self):
        for cursor in range(self.cursor, self.total_steps):
            self._offered_cursor = cursor
            input_ids = torch.full(
                (21, EXPECTED_CONTEXT_LENGTH),
                cursor % 85,
                dtype=torch.long,
            )
            input_ids[:, 0] = 0
            yield {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
                "attention_mask": torch.ones_like(input_ids),
                "sample_type": torch.ones(21, dtype=torch.long),
                "manifest_hash": self.manifest_hash,
                "cursor_start": cursor,
                "cursor_end": cursor + 1,
            }

    def commit_step(self):
        if self._offered_cursor != self.cursor:
            raise RuntimeError("commit without the next offered step")
        self.cursor += 1
        self.remaining_steps = self.total_steps - self.cursor
        self._offered_cursor = None

    def state_dict(self):
        return {
            "manifest_hash": self.manifest_hash,
            "source_manifest_hash": self.source_manifest_hash,
            "selection_hash": self.selection_hash,
            "sft_cache_hash": self.sft_cache_hash,
            "cursor": self.cursor,
            "world_size": 1,
            "local_batch_size": 21,
        }

    def load_state_dict(self, state):
        if state["manifest_hash"] != self.manifest_hash:
            raise ValueError("manifest mismatch")
        if int(state["cursor"]) != self.cursor:
            raise ValueError("cursor mismatch")


class InterleavedCheckpointIntegrationTests(unittest.TestCase):
    def _config(self, root: Path, manifest: Path, *, max_steps: int):
        return OmegaConf.create(
            {
                "model": {},
                "tokenizer": {
                    "name": "LanTokenizerSFT",
                    "include_env_tokens": True,
                },
                "data": {"leg_manifest_path": str(manifest)},
                "training": {
                    "run_name": "integration",
                    "output_dir": str(root / "integration"),
                    "seed": 7_771,
                    "local_batch_size": 21,
                    "gradient_accumulation_steps": 1,
                    "mixed_precision": "bf16",
                    "allow_topology_override": True,
                    "total_steps": 20,
                    "max_steps": max_steps,
                    "save_interval": 1,
                    "export_interval": 0,
                    "log_interval": 1,
                    "optimizer": {
                        "name": "adamw",
                        "lr": 1e-3,
                        "weight_decay": 0.1,
                        "betas": [0.9, 0.95],
                    },
                    "scheduler": {
                        "eta_min": 1e-5,
                        "warmup_ratio": 0.05,
                    },
                },
                "logging": {"backend": "none"},
            }
        )

    def test_one_step_checkpoint_and_full_resume_continue_exact_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "leg.jsonl"
            manifest.write_text('{"record":"test"}\n', encoding="utf-8")

            def stream_factory(
                data_cfg,
                tokenizer,
                *,
                rank,
                world_size,
                local_batch_size,
                start_cursor,
            ):
                del tokenizer, rank, world_size, local_batch_size
                from training.interleaved_hf_trainer import sha256_file

                return _FakeInterleavedStream(
                    manifest_hash=sha256_file(data_cfg.leg_manifest_path),
                    start_cursor=start_cursor,
                    total_steps=20,
                )

            patches = (
                mock.patch(
                    "training.interleaved_hf_trainer.init_tokenizer",
                    return_value=_TinyTokenizer(),
                ),
                mock.patch(
                    "training.interleaved_hf_trainer.build_interleaved_qwen_model",
                    side_effect=lambda tokenizer, **_kwargs: _TinyCausalModel(),
                ),
                mock.patch(
                    "training.interleaved_hf_trainer._make_interleaved_stream",
                    side_effect=stream_factory,
                ),
                mock.patch.object(InterleavedHFTrainer, "export_hf"),
            )
            with (
                patches[0],
                patches[1] as model_builder,
                patches[2],
                patches[3],
            ):
                first = InterleavedHFTrainer(
                    self._config(root, manifest, max_steps=1)
                )
                first.train()
                run_root = root / "integration"
                latest = resolve_resume_checkpoint(run_root)
                self.assertTrue((latest / CHECKPOINT_COMPLETE_FILE).is_file())
                self.assertTrue((run_root / LATEST_CHECKPOINT_POINTER).is_file())
                validate_completed_checkpoint(latest)
                state = (
                    latest / "trainer_state.json"
                ).read_text(encoding="utf-8")
                self.assertIn('"global_step": 1', state)
                self.assertIn('"manifest_cursor": 1', state)

                resumed_cfg = self._config(root, manifest, max_steps=2)
                resumed_cfg.training.resume = str(latest)
                resumed = InterleavedHFTrainer(resumed_cfg)
                resumed.train()
                self.assertEqual(resumed.global_step, 2)
                metric_lines = (
                    root / "integration" / "metrics.jsonl"
                ).read_text(encoding="utf-8").splitlines()
                metric_records = [json.loads(line) for line in metric_lines]
                self.assertEqual(
                    [record["step"] for record in metric_records],
                    [1, 2],
                )
                self.assertTrue(
                    all(
                        record["schema"]
                        == "interleaved-local-metrics-v1"
                        for record in metric_records
                    )
                )
                self.assertEqual(resumed.manifest_cursor, 2)
                self.assertEqual(model_builder.call_count, 2)
                self.assertTrue(
                    all(
                        call.kwargs["model_init_seed"] == 7_771
                        for call in model_builder.call_args_list
                    )
                )


if __name__ == "__main__":
    unittest.main()
