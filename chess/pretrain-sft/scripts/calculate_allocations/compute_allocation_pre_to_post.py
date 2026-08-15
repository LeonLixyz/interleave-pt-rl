
import math
import json
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# User-tunable config
# =========================

TOKENS_PER_SHARD = 1_147_000
TOKENS_PER_SFT_SAMPLE = 2048
SFT_EPOCHS = 3
SFT_MAX_SAMPLES_PER_EPOCH = 60_000

PRETRAIN_COEFF = 6.0
SFT_COEFF = 6.0
TOKENS_PER_STEP_APPROX = 131_072

MODEL_SPECS = [
    {"name": "50M", "params": 50e6},
    {"name": "100M", "params": 100e6},
    {"name": "200M", "params": 200e6},
    {"name": "300M", "params": 300e6},
    {"name": "500M", "params": 500e6},
]


MAX_SHARDS_300M = 5800

# You can change these two lists directly.
C_TOTAL_FRACS = [0.10, 0.25, 0.50, 0.75, 1.00]
NUM_BETA_BINS = 5
NUM_ALPHA_BINS = 5

# RL settings copied from the earlier code path
RL_PROMPT_TOKENS = 512
RL_RESPONSE_TOKENS = 1536
RL_SAMPLES_PER_PROMPT = 8
RL_PPO_EPOCHS = 1


# =========================
# FLOP / token helpers
# =========================

def pretrain_flops(params, tokens):
    return PRETRAIN_COEFF * params * tokens

def pretrain_flops_from_shards(params, shards):
    return pretrain_flops(params, shards * TOKENS_PER_SHARD)

def sft_tokens_from_flops(flops, params):
    return flops / (SFT_COEFF * params)

def sft_num_samples_from_tokens(tokens):
    return tokens / TOKENS_PER_SFT_SAMPLE

def sft_samples_per_epoch(C_total, beta, params):
    total_tokens = sft_tokens_from_flops(beta * C_total, params)
    total_samples = sft_num_samples_from_tokens(total_tokens)
    return total_samples / SFT_EPOCHS

def sft_total_samples(C_total, beta, params):
    total_tokens = sft_tokens_from_flops(beta * C_total, params)
    return sft_num_samples_from_tokens(total_tokens)

def sft_beta_max(C_total, params):
    max_sft_flops = SFT_COEFF * params * TOKENS_PER_SFT_SAMPLE * SFT_EPOCHS * SFT_MAX_SAMPLES_PER_EPOCH
    return min(1.0, max_sft_flops / C_total)

def rl_flops_per_prompt(params):
    L = RL_PROMPT_TOKENS + RL_RESPONSE_TOKENS
    g = RL_SAMPLES_PER_PROMPT
    E = RL_PPO_EPOCHS
    return params * g * L * (4.0 + 6.0 * E)

def rl_stats_from_budget(rl_flops_budget, params):
    L = RL_PROMPT_TOKENS + RL_RESPONSE_TOKENS
    g = RL_SAMPLES_PER_PROMPT
    cost_per_prompt = rl_flops_per_prompt(params)

    rl_num_prompts = rl_flops_budget / cost_per_prompt if cost_per_prompt > 0 else 0.0
    rl_num_rollouts = rl_num_prompts * g
    rl_generated_response_tokens = rl_num_rollouts * RL_RESPONSE_TOKENS
    rl_rollout_total_tokens = rl_num_rollouts * L
    rl_optimizer_total_tokens = rl_rollout_total_tokens * RL_PPO_EPOCHS
    rl_total_processed_tokens = rl_rollout_total_tokens + rl_optimizer_total_tokens
    rl_estimated_gradient_steps = rl_num_prompts / 256

    return {
        "rl_flops_per_prompt": cost_per_prompt,
        "rl_num_prompts": rl_num_prompts,
        "rl_num_rollouts": rl_num_rollouts,
        "rl_generated_response_tokens": rl_generated_response_tokens,
        "rl_rollout_total_tokens": rl_rollout_total_tokens,
        # "rl_optimizer_total_tokens": rl_optimizer_total_tokens,
        "rl_total_processed_tokens": rl_total_processed_tokens,
        "rl_estimated_gradient_steps": rl_estimated_gradient_steps,
    }

def pretrain_stats_from_budget(pt_flops_budget, params):
    target_tokens = pt_flops_budget / (PRETRAIN_COEFF * params) if params > 0 else 0.0
    shards = math.floor(target_tokens / TOKENS_PER_SHARD)
    shards = max(shards, 0)

    actual_tokens = shards * TOKENS_PER_SHARD
    actual_flops = pretrain_flops(params, actual_tokens)
    steps = math.floor(actual_tokens / TOKENS_PER_STEP_APPROX)

    return {
        "pretrain_tokens_target": target_tokens,
        "pretrain_shards": shards,
        "pretrain_tokens_actual": actual_tokens,
        "pretrain_flops_actual": actual_flops,
        "pretrain_steps_approx": steps,
    }


# =========================
# Table builders
# =========================

def build_beta_sft_table_for_ctotal(C_total):
    beta_max_values = {
        m["name"]: sft_beta_max(C_total, m["params"])
        for m in MODEL_SPECS
    }

    beta_min = min(beta_max_values.values())
    beta_max = max(beta_max_values.values())

    if NUM_BETA_BINS == 1:
        beta_bins = np.array([beta_min])
    else:
        beta_bins = np.linspace(beta_min, beta_max, NUM_BETA_BINS)

    rows = []
    for beta_idx, beta in enumerate(beta_bins, start=1):
        for m in MODEL_SPECS:
            params = m["params"]
            model_beta_max = beta_max_values[m["name"]]
            total_samples = sft_total_samples(C_total, beta, params)
            per_epoch_samples = total_samples / SFT_EPOCHS
            total_tokens = sft_tokens_from_flops(beta * C_total, params)

            rows.append({
                "C_total": C_total,
                "beta_bin_idx": beta_idx,
                "beta": beta,
                "model": m["name"],
                "params": int(params),
                "beta_max_under_80k": model_beta_max,
                "sft_total_tokens": total_tokens,
                "sft_total_samples_all_epochs": total_samples,
                "sft_samples_per_epoch": per_epoch_samples,
                "sft_feasible_under_80k_per_epoch": per_epoch_samples <= SFT_MAX_SAMPLES_PER_EPOCH + 1e-9,
            })

    beta_df = pd.DataFrame(rows).sort_values(["beta_bin_idx", "params"]).reset_index(drop=True)
    beta_meta = {
        "beta_min_across_models": beta_min,
        "beta_max_across_models": beta_max,
        "beta_bins": beta_bins.tolist(),
        "beta_max_per_model": beta_max_values,
    }
    return beta_df, beta_meta

def build_allocation_table_for_ctotal(C_total, beta_df):
    rows = []
    unique_betas = (
        beta_df[["beta_bin_idx", "beta"]]
        .drop_duplicates()
        .sort_values("beta_bin_idx")
        .itertuples(index=False)
    )

    for beta_bin_idx, beta in unique_betas:
        remaining = max(0.0, 1.0 - beta)
        if NUM_ALPHA_BINS == 1:
            alpha_fracs = np.array([0.5])
        else:
            alpha_fracs = np.linspace(0.05, 1.0, NUM_ALPHA_BINS)

        for alpha_bin_idx, alpha_frac in enumerate(alpha_fracs, start=1):
            alpha = remaining * alpha_frac
            gamma = remaining - alpha

            for m in MODEL_SPECS:
                params = m["params"]
                model_beta_max = sft_beta_max(C_total, params)
                sft_total_samples_all_epochs = sft_total_samples(C_total, beta, params)
                sft_samples_epoch = sft_total_samples_all_epochs / SFT_EPOCHS
                sft_tokens = sft_tokens_from_flops(beta * C_total, params)

                pt_flops_budget = alpha * C_total
                rl_flops_budget = gamma * C_total

                pt_stats = pretrain_stats_from_budget(pt_flops_budget, params)
                rl_stats = rl_stats_from_budget(rl_flops_budget, params)

                rows.append({
                    "C_total": C_total,
                    "beta_bin_idx": beta_bin_idx,
                    "beta": beta,
                    "alpha_bin_idx": alpha_bin_idx,
                    "alpha_frac_within_remaining": alpha_frac,
                    "alpha": alpha,
                    "gamma": gamma,
                    "model": m["name"],
                    "params": int(params),

                    "sft_beta_max_under_80k": model_beta_max,
                    "sft_feasible_under_80k_per_epoch": sft_samples_epoch <= SFT_MAX_SAMPLES_PER_EPOCH + 1e-9,
                    "sft_total_tokens": sft_tokens,
                    "sft_total_samples_all_epochs": sft_total_samples_all_epochs,
                    "sft_samples_per_epoch": sft_samples_epoch,

                    "pretrain_flops_budget": pt_flops_budget,
                    **pt_stats,

                    "rl_flops_budget": rl_flops_budget,
                    **rl_stats,
                })

    alloc_df = pd.DataFrame(rows).sort_values(
        ["beta_bin_idx", "alpha_bin_idx", "params"]
    ).reset_index(drop=True)
    return alloc_df


# =========================
# Export
# =========================

def sanitize_float_for_name(x):
    s = f"{x:.3e}"
    return s.replace("+", "").replace(".", "p")

def export_tables(output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    C_max = pretrain_flops_from_shards(300e6, MAX_SHARDS_300M)
    c_totals = [frac * C_max for frac in C_TOTAL_FRACS]

    manifest_rows = []

    for c_idx, C_total in enumerate(c_totals, start=1):
        folder_name = f"C_total_{c_idx:02d}_{sanitize_float_for_name(C_total)}"
        folder = output_root / folder_name
        folder.mkdir(parents=True, exist_ok=True)

        beta_df, beta_meta = build_beta_sft_table_for_ctotal(C_total)
        alloc_df = build_allocation_table_for_ctotal(C_total, beta_df)

        beta_path = folder / "01_beta_sft_table.csv"
        alloc_path = folder / "02_allocation_data_table.csv"
        meta_path = folder / "meta.json"

        beta_df.to_csv(beta_path, index=False)
        alloc_df.to_csv(alloc_path, index=False)

        with open(meta_path, "w") as f:
            json.dump(
                {
                    "C_total": C_total,
                    "C_total_fraction_of_C_max": C_TOTAL_FRACS[c_idx - 1],
                    "C_max": C_max,
                    "beta_selection_rule": "shared beta bins linearly spaced between min(beta_max) and max(beta_max) across selected model sizes for this C_total",
                    "alpha_selection_rule": "alpha bins linearly spaced over [0, 1-beta]",
                    "num_beta_bins": NUM_BETA_BINS,
                    "num_alpha_bins": NUM_ALPHA_BINS,
                    "models": MODEL_SPECS,
                    "rl_config": {
                        "prompt_tokens": RL_PROMPT_TOKENS,
                        "response_tokens": RL_RESPONSE_TOKENS,
                        "samples_per_prompt": RL_SAMPLES_PER_PROMPT,
                        "ppo_epochs": RL_PPO_EPOCHS,
                    },
                    "beta_meta": beta_meta,
                },
                f,
                indent=2,
            )

        manifest_rows.append(
            {
                "folder": folder_name,
                "C_total": C_total,
                "C_total_fraction_of_C_max": C_TOTAL_FRACS[c_idx - 1],
                "beta_table_rows": len(beta_df),
                "allocation_table_rows": len(alloc_df),
                "beta_table_file": str(beta_path.name),
                "allocation_table_file": str(alloc_path.name),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_root / "manifest.csv", index=False)

    readme = output_root / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "Each C_total folder contains:",
                "  01_beta_sft_table.csv",
                "      For shared beta bins, shows required SFT data by model size.",
                "  02_allocation_data_table.csv",
                "      For each beta bin and alpha bin, shows pretraining and RL data amounts by model size.",
                "",
                "Conventions:",
                f"  SFT max samples per epoch = {SFT_MAX_SAMPLES_PER_EPOCH}",
                f"  SFT epochs = {SFT_EPOCHS}",
                f"  RL prompt tokens = {RL_PROMPT_TOKENS}",
                f"  RL response tokens = {RL_RESPONSE_TOKENS}",
                f"  RL samples per prompt = {RL_SAMPLES_PER_PROMPT}",
                f"  RL PPO epochs = {RL_PPO_EPOCHS}",
                "",
                "Important assumption:",
                "  Beta bins are shared within a C_total folder and are chosen from",
                "  [min(beta_max across models), max(beta_max across models)].",
                "  Rows that exceed a model's 80k/epoch SFT cap are marked as infeasible.",
            ]
        )
    )

    return output_root


if __name__ == "__main__":
    export_tables("compute_allocation_tables")
