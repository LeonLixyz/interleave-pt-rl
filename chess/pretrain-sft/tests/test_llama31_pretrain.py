from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen3Config

from modal_scripts import launch_20m_llama31_sweep as launcher
from training.trainer_hf import (
    _add_legacy_llama_rope_aliases,
    _build_scratch_hf_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOKENS = {
    "0.050": 2_641_541_000,
    "0.100": 5_284_229_000,
    "0.200": 10_569_605_000,
    "0.400": 21_139_210_000,
    "0.750": 39_636_879_000,
}


class Llama31ConfigTests(unittest.TestCase):
    def _load(self, alpha: str):
        return OmegaConf.load(
            REPO_ROOT
            / "config"
            / "configs"
            / "6p5e18_llama31"
            / f"20m_alpha{alpha}.yaml"
        )

    def test_all_five_configs_are_isolated_and_keep_token_budgets(self):
        self.assertEqual(tuple(EXPECTED_TOKENS), launcher.ALPHAS)
        run_names = set()
        hub_repos = set()
        for alpha, tokens in EXPECTED_TOKENS.items():
            cfg = self._load(alpha)
            self.assertEqual(cfg.data.pretrain_tokens, tokens)
            self.assertEqual(cfg.model.model_family, "llama")
            self.assertEqual(cfg.model.block_size, 1024)
            self.assertEqual(cfg.model.max_position_embeddings, 131072)
            self.assertEqual(cfg.model.n_layer, 6)
            self.assertEqual(cfg.model.n_embed, 512)
            self.assertEqual(cfg.model.n_head, 4)
            self.assertEqual(cfg.model.num_key_value_heads, 1)
            self.assertEqual(cfg.model.head_dim, 128)
            self.assertEqual(cfg.model.intermediate_size, 1792)
            self.assertEqual(cfg.model.rope_scaling.rope_type, "llama3")
            self.assertEqual(cfg.model.rope_scaling.factor, 8.0)
            self.assertEqual(
                cfg.logging.project,
                "chess-scaling-C_6p5e18-llama31",
            )
            self.assertIn("llama31", cfg.training.experiment_name)
            self.assertIn("llama31", cfg.training.hf_upload_repo)
            run_names.add(cfg.training.experiment_name)
            hub_repos.add(cfg.training.hf_upload_repo)

        self.assertEqual(len(run_names), len(EXPECTED_TOKENS))
        self.assertEqual(len(hub_repos), len(EXPECTED_TOKENS))

    def test_llama_model_uses_official_config_and_expected_parameter_count(self):
        cfg = self._load("0.050")
        hf_config = _build_scratch_hf_config(
            cfg.model,
            vocab_size=81,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
        )
        self.assertIsInstance(hf_config, LlamaConfig)
        self.assertEqual(hf_config.model_type, "llama")
        config_dict = hf_config.to_dict()
        rope_config = config_dict.get("rope_parameters") or config_dict.get(
            "rope_scaling"
        )
        rope_theta = config_dict.get("rope_theta") or rope_config.get("rope_theta")
        self.assertEqual(rope_theta, 500000.0)
        self.assertEqual(rope_config["rope_type"], "llama3")
        self.assertEqual(hf_config.rms_norm_eps, 1e-5)

        model = AutoModelForCausalLM.from_config(hf_config)
        self.assertEqual(type(model).__name__, "LlamaForCausalLM")
        self.assertEqual(sum(p.numel() for p in model.parameters()), 20_495_360)
        self.assertFalse(
            any(
                "q_norm" in name or "k_norm" in name
                for name, _ in model.named_parameters()
            )
        )
        self.assertEqual(
            model.get_input_embeddings().weight.data_ptr(),
            model.get_output_embeddings().weight.data_ptr(),
        )

        input_ids = torch.randint(0, 81, (1, 8))
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
        self.assertEqual(tuple(logits.shape), (1, 8, 81))

    def test_llama_config_accepts_tokenizer_without_pad_token(self):
        cfg = self._load("0.050")
        hf_config = _build_scratch_hf_config(
            cfg.model,
            vocab_size=81,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=None,
        )
        self.assertIsNone(hf_config.pad_token_id)

    def test_transformers5_config_includes_transformers4_rope_aliases(self):
        cfg = self._load("0.050")
        hf_config = _build_scratch_hf_config(
            cfg.model,
            vocab_size=81,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=None,
        )
        config_dict = _add_legacy_llama_rope_aliases(hf_config.to_dict())
        self.assertEqual(config_dict["rope_theta"], 500000.0)
        self.assertEqual(config_dict["rope_scaling"]["rope_type"], "llama3")
        self.assertEqual(config_dict["rope_scaling"]["factor"], 8.0)
        self.assertNotIn("rope_theta", config_dict["rope_scaling"])
        self.assertEqual(
            config_dict["rope_parameters"]["rope_theta"],
            config_dict["rope_theta"],
        )

    def test_existing_qwen_shape_keeps_original_parameter_count(self):
        cfg = OmegaConf.load(
            REPO_ROOT
            / "config"
            / "configs"
            / "6p5e18_small"
            / "20m_alpha0.050.yaml"
        )
        hf_config = _build_scratch_hf_config(
            cfg.model,
            vocab_size=81,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
        )
        self.assertIsInstance(hf_config, Qwen3Config)
        model = AutoModelForCausalLM.from_config(hf_config)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 20_496_896)
        self.assertTrue(any("q_norm" in name for name, _ in model.named_parameters()))

    def test_unknown_scratch_family_fails_closed(self):
        cfg = OmegaConf.create(
            {
                "architecture": "mistral/unknown",
                "n_embed": 64,
                "intermediate_size": 128,
                "n_layer": 1,
                "n_head": 1,
                "block_size": 8,
            }
        )
        with self.assertRaisesRegex(ValueError, "Unsupported from-scratch model family"):
            _build_scratch_hf_config(
                cfg,
                vocab_size=81,
                bos_token_id=0,
                eos_token_id=1,
                pad_token_id=2,
            )


class Llama31LauncherTests(unittest.TestCase):
    def test_repair_adds_matching_legacy_rope_aliases_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model_type": "llama",
                        "rope_parameters": {
                            "rope_type": "llama3",
                            "factor": 8.0,
                            "low_freq_factor": 1.0,
                            "high_freq_factor": 4.0,
                            "original_max_position_embeddings": 8192,
                            "rope_theta": 500000.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(launcher._patch_rope_config_file(config_path))
            repaired = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(repaired["rope_theta"], 500000.0)
            self.assertEqual(repaired["rope_scaling"]["rope_type"], "llama3")
            self.assertNotIn("rope_theta", repaired["rope_scaling"])
            self.assertFalse(launcher._patch_rope_config_file(config_path))
            self.assertEqual(list(Path(temp_dir).iterdir()), [config_path])

    def test_full_command_has_one_dotlist_override_flag(self):
        overrides = launcher._build_overrides(
            "0.050",
            smoke=False,
            num_gpus=8,
        )
        command = launcher._build_training_command(
            config=launcher.JOBS["0.050"]["config"],
            output_root=launcher.OUTPUT_ROOT,
            overrides=overrides,
            num_gpus=8,
        )
        self.assertEqual(command.count("--override"), 1)
        override_index = command.index("--override")
        self.assertEqual(command[override_index + 1 :], overrides)
        self.assertIn(
            "training.hf_upload_repo="
            "Pre-to-Post-2/pretrain_20m_llama31_C_6p5e18_alpha0.050",
            overrides,
        )

    def test_smoke_is_one_step_and_disables_hub_upload(self):
        overrides = launcher._build_overrides(
            "0.200",
            smoke=True,
            num_gpus=2,
        )
        self.assertIn("training.batch_size=1", overrides)
        self.assertIn("training.gradient_accumulation_steps=1", overrides)
        self.assertIn("data.pretrain_tokens=2048", overrides)
        self.assertIn("training.hf_upload_repo=null", overrides)
        self.assertIn(
            "logging.project=chess-scaling-C_6p5e18-llama31-smoke",
            overrides,
        )
        self.assertTrue(launcher._experiment_name("0.200", smoke=True).endswith("_smoke1"))

    def test_alpha_selection_fails_closed(self):
        self.assertEqual(launcher._choose_alphas("all"), list(launcher.ALPHAS))
        self.assertEqual(
            launcher._choose_alphas("0.050,0.750"),
            ["0.050", "0.750"],
        )
        with self.assertRaises(ValueError):
            launcher._choose_alphas("0.123")
        with self.assertRaises(ValueError):
            launcher._choose_alphas("0.050,0.050")


if __name__ == "__main__":
    unittest.main()
