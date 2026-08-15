# chess-rl-miles

Chess-RL adapters for running the existing chess RL data and models on Miles/SGLang.

This checkout is meant to mirror the important Chess-RL training I/O while using
Miles for FSDP training and SGLang for faster multi-turn rollout generation.

## What Is Included

- Async multi-turn rollout support for `<call_env>` using `extra_info.env_replies`.
- Chess reward parsing that matches the veRL single-turn and multi-turn move scoring.
- CISPO in the sibling `miles` checkout with direct ratio clipping `[0.0, 5.0]`.
- MiniMax-style Adam settings for RL stability:
  - `adam_beta1=0.9`
  - `adam_beta2=0.95`
  - `adam_eps=1e-15`
  - `weight_decay=0.0`
- W&B logging to `jingyanshen-new-york-university/chess_rl_6p5e18`.
- Checkpoints, rollout JSONL, Miles `dump_details`, eval outputs, local logs, and mlflow directories.

## Fast Rollout Defaults

Chess rollout now uses three defaults validated together on Modal: prompt-group
sticky routing, 128 requests of per-engine concurrency, and token-ID-only
batched SGLang requests. The token-ID path avoids incremental decode through the
model's slow custom Python tokenizer; the local Miles tokenizer still decodes
the completed samples for reward and logging. Modal training pins the tested
Miles image digest; set `CHESS_RL_MILES_IMAGE` only when intentionally testing a
newer image.

For compatibility debugging, restore the legacy request path with:

```bash
--no-batched-rollout --no-sglang-token-id-only --sglang-server-concurrency 64
```

The 2026-07-22 Modal canary used 128 prompts × 8 samples on 8×H200. On the
matched warm step, the legacy path took 69.62 s at 1,706 effective
tokens/GPU/s; sticky routing alone took 53.13 s at 2,236; the complete default
path took 10.21 s at 11,523. Mean model response lengths were 928, 928, and 919
tokens respectively. Treat any nonzero
`rollout/chess_batch_generate/fallbacks` or `unsupported` metric as a failed
fast-path run.

## Current Fresh 2k Fast-Rollout Runs

The two Minimax+CISPO comparison runs start from their pretrained/SFT models.
They use the historically matched batch accounting of 32 prompts × 8 samples =
256 trajectories per optimizer step. Each is launched as a direct Modal
`train_one` function so there is one trainer per app.

```bash
PYTHONPATH=chess-rl-miles:miles modal run --detach \
  -m chess_rl_miles.scripts.modal_train::train_one \
  --spec '6p5e18|680m|1.000|0.296' \
  --num-rollout 2000 \
  --save-interval 20 \
  --eval-interval 500 \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 8 \
  --over-sampling-batch-size 64 \
  --global-batch-size 256 \
  --run-name-suffix miles_cispo_minimax_fresh_sft_bs256_20260724 \
  --hparam-tag-suffix fresh_sft_20260724 \
  --wandb-project chess_rl_6p5e18 \
  --wandb-group multi_turn_miles_cispo_minimax_fresh_sft_bs256_20260724 \
  --wandb-team jingyanshen-new-york-university \
  --io-layout chess-rl \
  --resume-if-available
```

```bash
PYTHONPATH=chess-rl-miles:miles modal run --detach \
  -m chess_rl_miles.scripts.modal_train::train_one \
  --spec '6p5e19|680m|0.750|0.030' \
  --num-rollout 2000 \
  --save-interval 20 \
  --eval-interval 500 \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 8 \
  --over-sampling-batch-size 64 \
  --global-batch-size 256 \
  --run-name-suffix miles_cispo_minimax_fresh_sft_bs256_20260724 \
  --hparam-tag-suffix fresh_sft_20260724 \
  --wandb-project chess_rl_6p5e18 \
  --wandb-group multi_turn_miles_cispo_minimax_fresh_sft_bs256_20260724 \
  --wandb-team jingyanshen-new-york-university \
  --io-layout chess-rl \
  --resume-if-available
```

The run names are:

- `C6p5e18_680m_alpha1.000_beta0.296_miles_cispo_minimax_fresh_sft_bs256_20260724`
- `C6p5e19_680m_alpha0.750_beta0.030_miles_cispo_minimax_fresh_sft_bs256_20260724`

`--resume-if-available` starts from SFT when the new checkpoint path has no
tracker, then resumes the latest validated checkpoint on a Modal retry. In
contrast, `--resume-from-save` is strict and fails when its tracker/checkpoint
is absent.

## Output Layout

The fresh comparison runs use the default Chess-RL-style layout:

```text
/checkpoints/chess-rl-miles/<cot_type>/<hparam_tag>/<model_id>/
  checkpoints/
    iter_0000020/
    iter_0000040/
    ...
  dump_details/
  logs/
  mlflow/
  rollouts/
    training/
    validation/
```

The JSONL rollout files are written by:

- `chess_rl_miles.io.log_rollout_data`
- `chess_rl_miles.io.log_eval_rollout_data`

Custom training/eval JSONL logging is enabled by default. Miles' larger native
detail dumps are separate and require `--dump-miles-details`.

## Metrics

W&B project:

```text
https://wandb.ai/jingyanshen-new-york-university/chess_rl_6p5e18
```

Important metrics to compare with veRL:

- pass rate metrics from `--log-passrate`
- response length metrics
- rollout throughput and rollout timing
- train loss, KL, entropy, gradient norm, and learning-rate metrics
- eval metrics every 500 rollouts from `--eval-interval 500`

The old completed Miles runs stopped at `iter_0000500`. Any apparent `2500`
value in W&B for those old runs was response length, not RL step count.

## Checkpoint Upload

Convert and upload the latest Miles FSDP checkpoint to Hugging Face:

```bash
PYTHONPATH=chess-rl-miles:miles modal run --detach \
  -m chess_rl_miles.scripts.upload_checkpoints
```

Convert and upload every raw Miles checkpoint under a run:

```bash
PYTHONPATH=chess-rl-miles:miles modal run --detach \
  -m chess_rl_miles.scripts.upload_checkpoints \
  --steps all
```

The uploader converts Miles FSDP `.distcp` checkpoints with the sibling
`miles/tools/convert_fsdp_to_hf.py` script and uploads to:

```text
chess-pre-to-post/rl_<model_id>/miles_cispo_minimax/global_step_<step>
```

Each converted HF folder is deleted from Modal `/tmp` after upload, so
`--steps all` can process all saved checkpoints without accumulating the full
converted model size for every step.

The earlier converted 500-step AdamW+GRPO checkpoints were uploaded under:

- `chess-pre-to-post/rl_C6p5e18_680m_alpha1.000_beta0.296/miles_sglang_grpo_adamw_bs2048_sgl64_cvd_mrouter_ctx16fix/global_step_500`
- `chess-pre-to-post/rl_C6p5e19_680m_alpha0.750_beta0.030/miles_sglang_grpo_adamw_bs2048_sgl64_cvd_mrouter_ctx16fix/global_step_500`

## Rollout Upload

Upload rollout and detail artifacts:

```bash
PYTHONPATH=chess-rl-miles:miles modal run --detach \
  -m chess_rl_miles.scripts.upload_rollouts
```

By default, the uploader packages any available:

- `rollouts/`
- `dump_details/rollout_data/`
- `dump_details/train_data/`
- `dump_details/policy_loss_debug/`
- legacy `rollout/` state files
- `checkpoints/rollout/` state files

For a smaller rollout-only upload, use:

```bash
PYTHONPATH=chess-rl-miles:miles modal run --detach \
  -m chess_rl_miles.scripts.upload_rollouts \
  --artifact-kinds rollouts,dump_details/rollout_data \
  --compresslevel 1
```

The old 500-step Miles runs only had tiny rollout state files because
`--dump-details` and the custom rollout JSONL hooks were not enabled yet. The
new 2k resumed runs enable both, so they should produce real training and eval
rollout artifacts.

## Local Dry Run

Use a local dry run to inspect the exact Miles command without starting training:

```bash
PYTHONPATH="/Users/leonli66/Desktop/Research/RL/Chess RL/chess-rl-miles:/Users/leonli66/Desktop/Research/RL/Chess RL/miles" \
python -m chess_rl_miles.scripts.run_chess_miles \
  --miles-dir "/Users/leonli66/Desktop/Research/RL/Chess RL/miles" \
  --spec '6p5e19|680m|0.750|0.030' \
  --prepare-data \
  --prepare-sft \
  --save-dir /tmp/chess-rl-miles \
  --num-rollout 2000 \
  --eval-interval 500 \
  --run-name debug \
  --dry-run
```
