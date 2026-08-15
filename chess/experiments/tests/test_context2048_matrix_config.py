from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import shutil

import numpy as np

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "chess/experiments"))

from context2048_matrix_config import load_experiment_config


CONFIG_ROOT = ROOT / "experiments/context2048_pt_sft_trace_rl_v1/5b"


def _paths() -> list[Path]:
    return sorted(CONFIG_ROOT.glob("*.yaml"))


def test_exactly_seven_5b_configs_resolve_with_unique_identities() -> None:
    paths = _paths()
    assert len(paths) == 7
    configs = [load_experiment_config(path) for path in paths]
    assert len({config["experiment"]["key"] for config in configs}) == 7
    assert len({config["resolved_config_sha256"] for config in configs}) == 7
    assert all(config["experiment"]["total_pt_target_tokens"] == 5_000_000_000 for config in configs)


def test_reset_restores_complete_pre_rl_pt_plus_sft_checkpoint() -> None:
    for filename in (
        "03_full_pt_shuffled_trace_rl1500x2.yaml",
        "04_full_pt_chronological_trace_rl1500x2.yaml",
    ):
        config = load_experiment_config(CONFIG_ROOT / filename)
        stages = {stage["id"]: stage for stage in config["stages"]}
        assert stages["pt_full"]["sft_exposures"] == 233_151
        assert stages["trace_train"]["model_from"] == "pt_full"
        assert stages["trace_train"]["traces_from"] == "rl_first"


def test_ordinary_sft_is_always_inside_pt_and_never_a_standalone_stage() -> None:
    for path in _paths():
        config = load_experiment_config(path)
        stages = config["stages"]
        assert sum(
            int(stage.get("sft_exposures", 0)) for stage in stages
        ) == 233_151
        assert all(stage["kind"] != "sft" for stage in stages)
        assert all(
            int(stage.get("sft_exposures", 0)) == 0
            for stage in stages
            if stage["kind"] not in {"pt_sft_mixed", "pt_sft_trace_mixed"}
        )


def test_all_learning_rates_match_the_sealed_decision() -> None:
    for path in _paths():
        config = load_experiment_config(path)
        for stage in config["stages"]:
            if stage["kind"] in {"pt_sft_mixed", "pt_sft_trace_mixed"}:
                assert stage["learning_rate"] == {
                    "schedule": "linear_warmup_then_cosine",
                    "warmup_ratio": 0.05,
                    "peak": 1e-3,
                    "minimum": 1e-5,
                }
            else:
                assert stage["learning_rate"] == 1e-5


def test_controlled_arms_share_bit_identical_initial_pt_data_orders() -> None:
    configs = {
        path.name: load_experiment_config(path) for path in _paths()
    }
    full = [
        configs[name]["stages"][0]
        for name in (
            "01_one_pt_decay_rl3000.yaml",
            "03_full_pt_shuffled_trace_rl1500x2.yaml",
            "04_full_pt_chronological_trace_rl1500x2.yaml",
        )
    ]
    assert {(stage["pt_target_tokens"], stage["sft_exposures"], stage["placement_seed"]) for stage in full} == {
        (5_000_000_000, 233_151, 520_100)
    }
    first_halves = [
        configs[name]["stages"][0]
        for name in (
            "02_two_pt_decays_rl3000.yaml",
            "05_split_pt_shuffled_trace_rl1500x2.yaml",
            "06_split_pt_chronological_trace_rl1500x2.yaml",
            "07_split_pt_trace_mixed_rl1500x2.yaml",
        )
    ]
    assert {(stage["pt_target_tokens"], stage["sft_exposures"], stage["placement_seed"]) for stage in first_halves} == {
        (2_500_000_000, 116_575, 520_200)
    }


def test_launcher_partitions_all_seven_configs_between_two_initial_parents() -> None:
    launcher_path = (
        ROOT
        / "chess/pretrain-sft/modal_scripts/modal_context2048_pt_matrix.py"
    )
    spec = importlib.util.spec_from_file_location(
        "modal_context2048_pt_matrix_test", launcher_path
    )
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    expected = {f"5b/{path.name}" for path in _paths()}
    groups = [
        set(stage["dependent_config_files"])
        for stage in launcher.INITIAL_STAGES.values()
    ]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0] | groups[1] == expected
    assert set(launcher._load_resolved_configs(CONFIG_ROOT)) == expected
    assert launcher.CUBLAS_WORKSPACE_CONFIG == ":4096:8"
    assert launcher._random_initialization_identity() == {
        "schema": "interleaved-random-initialization-v1",
        "mode": "random",
        "destination_seed": 42,
    }


def test_launcher_canary_contains_pt_and_sft_on_every_rank_and_update() -> None:
    launcher_path = (
        ROOT
        / "chess/pretrain-sft/modal_scripts/modal_context2048_pt_matrix.py"
    )
    spec = importlib.util.spec_from_file_location(
        "modal_context2048_pt_matrix_canary_test", launcher_path
    )
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    order = launcher._build_canary_mixed_order()
    assert order.dtype == np.dtype("<i8")
    assert len(order) == launcher.CANARY_TOTAL_STEPS * launcher.GLOBAL_SEQUENCES
    for update in range(launcher.CANARY_TOTAL_STEPS):
        update_start = update * launcher.GLOBAL_SEQUENCES
        for rank in range(launcher.WORLD_SIZE):
            rank_start = update_start + rank * launcher.LOCAL_BATCH_SIZE
            local = order[rank_start : rank_start + launcher.LOCAL_BATCH_SIZE]
            assert bool((local >= 0).any())
            assert bool((local < 0).any())


def test_config_fails_closed_if_budget_is_changed(tmp_path: Path) -> None:
    source = CONFIG_ROOT / "01_one_pt_decay_rl3000.yaml"
    text = source.read_text().replace("pt_target_tokens: 5000000000", "pt_target_tokens: 4999999999")
    target = tmp_path / source.name
    target.write_text(text)
    shared = CONFIG_ROOT.parent / "shared.yaml"
    (tmp_path / "shared.yaml").write_bytes(shared.read_bytes())
    target.write_text(target.read_text().replace("../shared.yaml", "shared.yaml"))
    with pytest.raises(ValueError, match="5B matrix|PT target total"):
        load_experiment_config(target)


def test_resolved_hash_is_independent_of_checkout_path(tmp_path: Path) -> None:
    copied_root = tmp_path / "copied_contract"
    shutil.copytree(CONFIG_ROOT.parent, copied_root)
    original = load_experiment_config(CONFIG_ROOT / "01_one_pt_decay_rl3000.yaml")
    copied = load_experiment_config(copied_root / "5b/01_one_pt_decay_rl3000.yaml")
    assert copied["resolved_config_sha256"] == original["resolved_config_sha256"]
