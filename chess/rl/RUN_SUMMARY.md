# Chess-RL Miles Run Summary

Last checked: 2026-06-29.

## 2026-07-04 50M Launch Check

Briefly launched three Miles/SGLang GRPO AdamW 50M jobs, then stopped them from
the CLI before training produced checkpoints. The intended corrected batch
accounting was:

```text
--rollout-batch-size 256
--n-samples-per-prompt 8
--global-batch-size 2048
```

Planned runs:

- `C6p5e18_50m_alpha0.750_beta0.023_miles`: fresh Miles run to 3000
- `C6p5e18_50m_alpha1.000_beta0.023_miles`: fresh Miles run to 3000
- `C6p5e19_50m_alpha0.180_beta0.002_miles`: resumed from `iter_0002000` to 3000

Stopped Modal apps:

- `ap-nccGOuxelzkBlFnCKhl7zB`: the two fresh `6p5e18 50M` runs
- `ap-LFyIKJa57HqMIQnZqy9C3f`: resumed `6p5e19 50M alpha0.180`

Status after stop:

- `C6p5e18_50m_alpha1.000_beta0.023_miles` has empty Miles artifact directories
  only; no `iter_*` checkpoints were created.
- `C6p5e19_50m_alpha0.180_beta0.002_miles` remains at `iter_0002000` locally
  and in `Pre-to-Post-2`; it still needs resume to 3000.

## 2026-07-04 Active 50M Launches

Launched exactly two jobs under Modal app name `train rl`:

- `ap-XHSpyXKJMOKAujotCNOFgH`: `C6p5e18_50m_alpha1.000_beta0.023_miles`,
  fresh Miles run to 2000
- `ap-4NgXskM4xMJmAWklAS6bm8`: `C6p5e19_50m_alpha0.180_beta0.002_miles`,
  resumed from `iter_0002000` to 3000

Both use the corrected Miles/SGLang GRPO AdamW config:

```text
--rollout-batch-size 256
--n-samples-per-prompt 8
--global-batch-size 2048
--no-cispo
--optim-tag adamw
--adam-beta2 0.999
--adam-eps 1e-8
--weight-decay 0.01
```

Shared hparam tag:

```text
multi_turn_lr1e-5_bs2048_kl0.001_res2560_adamw_grpo_miles_sglang_grpo_adamw_sgl64_cvd_mrouter_ctx16fix
```

The `6p5e18|50m|0.750|0.023` spec was added to
`chess_rl_miles/scripts/run_chess_miles.py` because the SFT exists on HF but the
local Miles spec allowlist did not include it yet.

## 2026-07-06 50M Partial Uploads

Uploaded the current saved progress for the two 50M Miles runs to
`Pre-to-Post-2/rl_*` under:

```text
miles_sglang_grpo_adamw_bs2048_sgl64_cvd_mrouter_ctx16fix
```

Final HF audit after upload:

| Model | Converted HF max step | Converted HF count | Raw checkpoint max step | Rollout archive |
| --- | ---: | ---: | ---: | --- |
| `C6p5e18_50m_alpha1.000_beta0.023` | 1200 | 60 | 1200 | uploaded |
| `C6p5e19_50m_alpha0.180_beta0.002` | 3000 | 150 | 3000 | uploaded |

The raw checkpoints include optimizer state at the latest saved step for each
run.

## Completed Upload Status From 2026-06-29

The latest tracked Miles/SGLang GRPO runs are finished and uploaded.

| Model | Target rollouts | Converted HF max step | Converted HF count | Raw checkpoint max step | Rollout archive |
| --- | ---: | ---: | ---: | ---: | --- |
| `C6p5e18_20m_alpha1.000_beta0.008` | 5000 | 5000 | 250 | 5000 | uploaded |
| `C6p5e19_680m_alpha3.000_beta0.030` | 2000 | 2000 | 100 | 2000 | uploaded |

All Modal `rl training` apps were stopped at the final check. There were no active
training or upload workers left.

## Hugging Face Uploads

The final uploads are under the `Pre-to-Post-2` model namespace.

Path prefix:

```text
miles_sglang_grpo_adamw_bs2048_sgl64_cvd_mrouter_ctx16fix
```

20m run:

```text
https://huggingface.co/Pre-to-Post-2/rl_C6p5e18_20m_alpha1.000_beta0.008
```

Alpha-3 680m run:

```text
https://huggingface.co/Pre-to-Post-2/rl_C6p5e19_680m_alpha3.000_beta0.030
```

Each repo contains:

- converted HF checkpoints at `global_step_<step>/`
- final raw Miles checkpoint with optimizer state under `raw_checkpoints/`
- rollout archive named `<model_id>_<hparam_tag>_rollout.tar.gz`

The hparam tag for these runs is:

```text
multi_turn_lr1e-5_bs2048_kl0.001_res2560_adamw_grpo_miles_sglang_grpo_adamw_sgl64_cvd_mrouter_ctx16fix
```

## Training Configuration

These final runs use the corrected prompt/trajectory accounting:

```text
--rollout-batch-size 256      # prompts
--n-samples-per-prompt 8      # trajectories per prompt
--global-batch-size 2048      # trajectories per optimizer step
```

Core training settings:

- backend: Miles FSDP training with SGLang rollout generation
- optimizer: normal Chess-RL-style AdamW
- Adam settings: `beta1=0.9`, `beta2=0.999`, `eps=1e-8`, `weight_decay=0.01`
- algorithm: GRPO, no CISPO for these final jobs
- KL coefficient: `0.001`
- rollout temperature/top-p: `1.0` / `1.0`
- response length: `2560`
- context length: `3072`
- W&B project: `jingyanshen-new-york-university/chess_rl_6p5e18`
- W&B run suffix: `miles`

The current code defaults in `chess_rl_miles/scripts/run_chess_miles.py` match
the corrected data-count setup above.

## Chess-RL Matching Notes

The Miles adapter is intended to mirror the Chess-RL/veRL data and I/O where
possible:

- data repo: `chess-pre-to-post/chess-rl-data`
- train file: `train_thinking/train_v4_easy_skewed_multi_turn.parquet`
- eval file: `puzzles/test_multi_turn_final.parquet`
- SFT repo: `chess-pre-to-post/sft_trajectory_no_labels`
- COT type: `trajectory_sep_no_labels`
- reward path: `chess_rl_miles.reward.reward_func`
- rollout path: `chess_rl_miles.rollout.generate`
- custom rollout logs: `chess_rl_miles.io.log_rollout_data`
- custom eval logs: `chess_rl_miles.io.log_eval_rollout_data`

Implementation details to preserve:

- old policy logprobs are recomputed after rollout by default; rollout logprobs
  are not used unless `--use-rollout-logprobs` is explicitly enabled
- `--balance-data` is enabled by default to match Chess-RL/veRL
  `trainer.balance_batch=True`
- the batched rollout path sorts groups by env-reply count before dispatch
- W&B random suffixes are disabled when a W&B key is available

## Useful Audit Commands

Check Modal app state:

```bash
modal app list | rg 'rl training'
```

Audit HF checkpoint coverage:

```bash
python - <<'PY'
from huggingface_hub import HfApi
import re

api = HfApi()
repo_prefix = "Pre-to-Post-2/rl_"
path_prefix = "miles_sglang_grpo_adamw_bs2048_sgl64_cvd_mrouter_ctx16fix"
targets = [
    ("C6p5e18_20m_alpha1.000_beta0.008", 5000),
    ("C6p5e19_680m_alpha3.000_beta0.030", 2000),
]

for model, target in targets:
    repo = repo_prefix + model
    files = api.list_repo_files(repo, repo_type="model")
    steps = sorted({
        int(m.group(1))
        for f in files
        for m in [re.search(rf"^{re.escape(path_prefix)}/global_step_(\d+)/", f)]
        if m
    })
    raw_steps = sorted({
        int(m.group(1))
        for f in files
        for m in [re.search(rf"^{re.escape(path_prefix)}/raw_checkpoints/.*steps_(\d+)(?:/|$)", f)]
        if m
    })
    rollout = any("rollout" in f.lower() for f in files)
    print(
        model,
        f"converted_max={steps[-1] if steps else None}/{target}",
        f"converted_count={len(steps)}",
        f"raw_max={raw_steps[-1] if raw_steps else None}/{target}",
        f"rollout={rollout}",
    )
PY
```
