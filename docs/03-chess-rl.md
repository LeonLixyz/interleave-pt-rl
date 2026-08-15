# Chess RL (GRPO via Miles)

The policy generates a reasoning trace, may call a move-verification environment up to
6 times, and is scored by a rule-based reward on the final move sequence.

## Code

| File | Role |
|---|---|
| `chess/rl/chess_rl_miles/scripts/modal_interleave.py` | the launcher — `build_train_command` holds every hyperparameter; actions `train`, `convert` |
| `chess/rl/chess_rl_miles/scripts/run_chess_miles.py` | in-container adapter that assembles and execs the Miles command |
| `chess/rl/chess_rl_miles/batched_rollout.py` | **production rollout path** (`--batched-rollout`): token-id-only batched SGLang requests, dynamic filtering, eval rollout |
| `chess/rl/chess_rl_miles/rollout.py` | single-request rollout path (non-batched) |
| `chess/rl/chess_rl_miles/moves.py` | chess move parsing and legality |
| `chess/rl/chess_rl_miles/reward.py` | reward, matching the verl scoring semantics |
| `chess/rl/chess_rl_miles/provenance.py` | fail-closed run identity and launch records |
| `chess/rl/chess_rl_miles/data.py` | dataset constants and pinned hashes |
| `chess/rl/chess_rl_miles/io.py` | rollout JSONL and checkpoint I/O |
| `chess/rl/sitecustomize.py` | monkey-patches SGLang's detokenizer manager to flatten nested token ids |
| `miles/` | the RL framework, our fork, changes already applied |

Specialized data sources: `gate_data_source.py` (every declared prompt sampled exactly
once) and `exhaustive_data_source.py` (rollout-only pass@16 sweeps).

**`<bos>` handling lives in both `rollout.py` and `batched_rollout.py`.** Both prepend
token id 0 after `compute_prompt_ids_from_sample` and raise if the tokenizer has no
`bos_token_id`. Production uses the batched path; patching only the other one leaves
production broken.

## Config

Two sources. Environment rules in `chess/rl/config/chess_multiturn.yaml`:

```yaml
chess_multiturn: true
chess_reward_model_type: RULE_BASED
chess_difficulty_threshold: 1500.0
chess_max_env_calls: 6
chess_call_env_token: "<call_env>"
```

Everything else is emitted by `build_train_command`:

### Model and hardware

| Setting | Value |
|---|---|
| model id | `interleave_47m_qwen3` |
| profile | `small-model-h200` (refuses anything but 8×H200 / 192 GB host memory) |
| gradient checkpointing | off |
| attention | `flash_attention_3` |
| max tokens per GPU | 131,072 (only 65,536 or 131,072 accepted) — dynamic micro-batching by token budget |

### Rollout

| Setting | Value |
|---|---|
| prompts per update | 256 |
| samples per prompt | 8 |
| trajectories per update | 2,048 |
| over-sampling batch size | 256 |
| temperature / top-p | 1.0 / 1.0 |
| max prompt / response / context | 512 / 2,560 / 3,072 |
| engine | SGLang, 1 GPU per engine, server concurrency 128 (or 256) |
| eval server concurrency | 16 |
| flags | `--batched-rollout --sglang-token-id-only --use-miles-router` |
| health check interval | 30 s |
| rollout seed | 42 |
| save rollouts | on |

### Optimization

| Setting | Value |
|---|---|
| algorithm | GRPO (`--advantage-estimator grpo`, `--no-cispo`) |
| global batch | 2,048 trajectories → one optimizer step per rollout batch |
| loss aggregation | `token-mean` |
| optimizer | AdamW |
| learning rate | `--lr`, default `1e-5`; runs in this study passed `1e-4` |
| betas / eps | 0.9 / 0.999, 1e-8 |
| weight decay | 0.01 |
| KL coefficient | 0.001 |
| KL estimator | `low_var_kl` (= k3), set by `--kl-loss-type`; overridable, allowed `k1`/`k2`/`k3`/`low_var_kl` |
| LR schedule | none — constant |

### Checkpointing

| Setting | Value |
|---|---|
| save interval | 40 updates |
| run root | `/rl-checkpoints/chess-rl-miles-interleave/<run-name>/` |
| checkpoint | `iter_<step>/{model,optimizer,lr_scheduler,rng.pt,meta.json}` |
| data-sampler state | `rollout/global_dataset_state_dict_<step>.pt` (outside `iter_*`) |
| tracker | `latest_checkpointed_iteration.txt` |

### Data settings

- **all** — every drawn prompt trains (no flag).
- **dynamic filter** — `--dynamic-filter`: keep only prompt groups with both successes
  and failures, redraw the rest. On-policy.
- **solvable-only** — an offline pre-filter passed as `--train-file`: the puzzles a
  checkpoint solved 1–15 of 16 in its own evaluation, fixed for the run. Built by
  `tools/build_bands.py`. Independent of the dynamic filter.

Default file `/data/chess-rl-data/train_v4_dataset_balanced_multi_turn.parquet`
(53,225 puzzles). Any custom file needs `--train-file-sha256`; the launcher refuses on
mismatch.

## Launch

```
cd chess/rl

modal run --detach chess_rl_miles/scripts/modal_interleave.py \
  --action train \
  --hf-checkpoint /pretrain-checkpoints/interleave_50m/pretrain/<experiment>/<arm>/final \
  --run-name <run-name> \
  --num-rollout 1500 \
  --lr 1e-4 \
  --train-file /data/chess-rl-data/<dataset>.parquet \
  --train-file-sha256 <sha256>
```

Add `--dry-run` to print the exact command without allocating GPUs, `--dynamic-filter`
to enable on-policy filtering, `--kl-loss-type` to change the KL estimator.

### Convert a checkpoint for evaluation

Miles checkpoints are FSDP-sharded; the evaluator needs Hugging Face format.

```
modal run --detach chess_rl_miles/scripts/modal_interleave.py \
  --action convert \
  --run-name <run-name> \
  --hf-checkpoint <the origin HF checkpoint the run started from> \
  --output-name <run-name>-s<step> \
  --step <step>
```

Output: `/pretrain-checkpoints/interleave_50m/rl_hf/<output-name>/`.

### Extend a finished run

The provenance guard refuses to reuse a run root under different semantics, including a
different `--num-rollout`. Seed a **new** run root from the old checkpoint and let
auto-resume take it:

```
/rl-checkpoints/chess-rl-miles-interleave/<new-run>/
├── iter_0001500/                             # copied from the old run
├── rollout/global_dataset_state_dict_1499.pt # copied — the loader reads start_rollout_id-1
└── latest_checkpointed_iteration.txt         # "1500"
```

then launch with the larger `--num-rollout` and identical data and lr.
`tools/seed_resume_root2.py` does this.

### Metrics

`rollout/raw_reward` is the reward curve. `rollout/reward` and `rollout/returns` are
group-normalized by GRPO and sit near zero by construction. With no W&B key available,
rebuild curves from saved rollouts with `tools/rl_reward_curve.py` (use `--cache`).
