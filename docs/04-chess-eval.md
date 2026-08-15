# Chess evaluation

There are two evaluation protocols because the older models and the native
context-2048 RL models have different context contracts. Do not compare numbers
across the two protocols without labeling the benchmark and context length.

## Code

| File | Role |
|---|---|
| `chess/eval/modal_eval_context2048_final_test.py` | **Final native context-2048 RL evaluator** on held-out B1--B5 |
| `chess/eval/context2048_eval_core.py` | Pure BOS, deterministic seed, pass@k, and metric helpers |
| `chess/eval/modal_eval_clean.py` | Older 3,072-token evaluator on the 53,225-row balanced source |
| `miles/tools/convert_fsdp_to_hf.py` | Authenticated FP32 Miles-to-Hugging-Face conversion |
| `chess/eval/tests/` | Evaluation contract tests |

The verl-based sweep evaluators remain for historical result recovery. They are
not the canonical path for the five FP32-master v13 final RL checkpoints.

## Native context-2048 final test

The complete accepted protocol and checkpoint identities are sealed in
[`decisions/CONTEXT2048_FINAL_TEST_EVALUATION.md`](decisions/CONTEXT2048_FINAL_TEST_EVALUATION.md).

### Benchmark

The held-out benchmark is the union of B1--B5. It contains 1,484 raw prompts.
The exactly-one-BOS 512-token admission rule excludes four overlength prompts,
so each checkpoint evaluates 1,480 prompts and 23,680 trajectories at 16 samples
per prompt.

The five parquet files have zero exact serialized prompt overlap with
`train_v4_dataset_balanced_multi_turn.parquet`, the source from which the RL
training cohort was constructed. Their row counts and SHA-256 hashes are pinned
in the evaluator and the decision record.

### Generation and scoring

| Setting | Value |
|---|---|
| model context | 2,048 tokens |
| prompt cap | 512 tokens, including exactly one explicit BOS |
| model-generated token budget | 1,536 tokens |
| samples per prompt | 16 |
| temperature / top-p | 1.0 / 1.0 |
| environment calls | at most 6 |
| inference precision | BF16 from an authenticated FP32 checkpoint export |
| sampling seeds | deterministic by dataset, row, sample slot, and generation round |
| scorer | `chess_rl_miles.reward._score_sample`, identical to online RL |

The same per-request seed is used for the same dataset row and sample slot in
all five checkpoints. Environment replies are taken from
`extra_info.env_replies`, as in the production rollout implementation.

The result contains unbiased pass@k for every k from 1 through 16:

```text
pass@k = mean over prompts of [1 - C(n - w, k) / C(n, k)]
```

It also records format rate, the complete win histogram, and the percentages of
prompts with zero or sixteen successful samples.

### Launch

Use the repository's Modal environment and CLI:

```bash
cd "/Users/leonli66/Desktop/Research/RL/Chess RL/interleave-pt-rl"

# Authenticate datasets, split separation, origins, and final checkpoints.
chess/pretrain-sft/.venv/bin/modal run \
  -e leon-dev chess/eval/modal_eval_context2048_final_test.py \
  --action inspect

# Required real-topology canary: 8 B1 prompts x 2 samples.
chess/pretrain-sft/.venv/bin/modal run --detach \
  -e leon-dev chess/eval/modal_eval_context2048_final_test.py \
  --action canary

# Exactly-once five-checkpoint production launch.
chess/pretrain-sft/.venv/bin/modal run --detach \
  -e leon-dev chess/eval/modal_eval_context2048_final_test.py \
  --action launch

# Read the durable ledger and per-checkpoint states.
chess/pretrain-sft/.venv/bin/modal run \
  -e leon-dev chess/eval/modal_eval_context2048_final_test.py \
  --action status
```

Production outputs are immutable under:

```text
/results/context2048-fp32-master-v13-final-b1b5-n16-v2-20260815/
  evaluation.json
  production/<checkpoint>/
    _RUNNING.json, _FAILED.json, or _SUCCESS.json
    B1..B5/summary.json
    B1..B5/generations.jsonl.gz
```

The production ledger refuses a second launch for the same version namespace.

## Older balanced-source evaluation

`modal_eval_clean.py` evaluates the 53,225-row balanced source with a 3,072-token
model context and a 2,560-token response budget. That remains valid for its
original older checkpoint family. It is not a held-out test for the current RL
runs because the RL cohort was derived from that same source, and its context
contract does not match the native 2,048-token policies.
