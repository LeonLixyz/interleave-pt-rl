# Math RL (GRPO via verl)

RL on math problems on top of an SFT'd anchor, producing the `rl` anchors.

## Code

| File | Role |
|---|---|
| `math/rl_train.py` | the launcher (Modal app `math-1b-rl`); the hydra override list **is** the config |
| `math/rl_preprocess.py` | RL data → parquet on the volume |
| `math/rl_eval.py` | merge verl FSDP shards → HF → vLLM eval on GSM8K and MATH-500 |
| `math/export_bundle.py` | FSDP → HF weights-only export for release |
| `math/rollout_stats.py`, `math/probe_rl_dataset.py` | rollout and dataset diagnostics |
| `math/external/reward_function.py` | rule-based math reward (`compute_score_batch`) |

Framework: verl 0.9 — **not included here**; original at
`~/Desktop/Research/RL/Chess RL/pretrain-rl-scaling/verl-olmo3`
(stock verl despite the name — it contains no OLMo-specific code)
Reward: `math/external/reward_function.py` (`compute_score_batch`, math_verify + boxed
extraction)

There is no YAML of its own: the hydra override list in `rl_train.py` **is** the
config, layered over verl's `verl/trainer/config/ppo_trainer.yaml`.

---

## Config

Hardware: `gpu="H200:8"`, 24h timeout, 10 retries, 8 CPU, 200 GB memory.

### Batch and sampling

| Setting | Value |
|---|---|
| train batch size | 128 prompts |
| PPO mini-batch | 128 |
| group size (samples per prompt) | 8 |
| max prompt length | 512 |
| max response length | 3,584 |
| rollout temperature | 1.0 |
| rollout engine | vLLM, tensor parallel 1 |
| GPU memory utilization | 0.85 |
| max batched tokens | 12,288 |
| PPO micro-batch per GPU | 2 |
| log-prob micro-batch | 2 |

### Algorithm

| Setting | Value |
|---|---|
| advantage estimator | `grpo` |
| KL loss | **off** (`use_kl_loss=False`, `use_kl_in_reward=False`) |
| clip ratio low / high | 0.2 / 0.26 |
| clip ratio c | 10.0 |
| loss aggregation | `seq-mean-token-mean` |
| entropy coefficient | 0.0 |
| remove padding | on |
| gradient checkpointing | on |

Note this differs from the chess RL setup, which keeps a KL term at 0.001 and
aggregates `token-mean`.

### Optimization and schedule

| Setting | Value |
|---|---|
| actor learning rate | 1e-6 |
| LR warmup | 50 steps |
| total training steps | 5,000 default; **sweeps used 3,000** |
| total epochs | 100 (not binding — steps stop it first) |
| save frequency | every 50 steps |
| test frequency | 100,000 (i.e. no mid-training eval) |
| validation samples | 8 |
| resume | `auto` |

### Reward and data

| Setting | Value |
|---|---|
| reward manager | `batch` |
| custom reward | `/root/reward_function.py::compute_score_batch` |
| reward type | `REWARD_MODEL_TYPE=RULE_BASED` |
| data | `/checkpoints/rl_data/skyeasy25k_omi2/{train,test}.parquet` |
| base model | `pre-to-post-olmo/math-1b-sft-numinamath-bs512-from-step{N}` |

Environment: `VLLM_ATTENTION_BACKEND=FLASH_ATTN`, `VLLM_USE_V1=1`, mlflow file store,
logger `['console']`.

## Launch

```
cd math
modal run --detach rl_train.py::rl_single       # one anchor
modal run --detach rl_train.py::rl_sweep_4      # anchors 10000 / 40000 / 80000 / 95368
```

Other entrypoints: `rl_sweep_remaining`, `rl_sweep_pruned_new`, `rl_finish_3k`,
`rl_custom`.

## The 7B reference

`math/external/olmo-thinking-rl-reward-run.sh` is the non-Modal bash original this was
adapted from (OLMo-3 7B, local 4-GPU): same override list, train batch 64, max prompt
1,536, response 6,656, 6 epochs, test frequency 100.
