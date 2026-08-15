# Chess SFT (separate stage)

The alternative to mixing SFT into the pretraining stream: pretrain on chess text
only, then run a dedicated supervised stage on puzzle-solving traces.

Config file: `chess/pretrain-sft/config/configs/qwen_multiturn_sft/sft_interleave_3072.yaml`

Hyperparameters were reverse-engineered from the optimizer and scheduler state of
`chess-pre-to-post/sft_trajectory_no_labels`, which is why several are unusual values.

## Code

| File | Role |
|---|---|
| `chess/pretrain-sft/scripts/train/run_sft.py` | entry point |
| `chess/pretrain-sft/training/sft_trainer.py` | the trainer: masked loss, grad accumulation, mixed precision |
| `chess/pretrain-sft/training/sft_data_utils.py` | dataset, prompt/response masking |
| `chess/pretrain-sft/training/sft_loss.py` | masked cross-entropy that ignores prompt tokens |
| `chess/pretrain-sft/modal_scripts/launch_sft_interleave.py` | the launcher |

## Config

| Setting | Value |
|---|---|
| epochs | 3 |
| optimizer | AdamW |
| peak learning rate | **3e-4** (the yaml says 5e-5; the launcher overrides via `--lr`) |
| schedule | cosine → `eta_min` 1e-5 |
| warmup | 50 steps (absolute, not a ratio) |
| betas | 0.9 / 0.95 |
| weight decay | 0.01 |
| max grad norm | 1.0 |
| per-GPU batch | 4 |
| gradient accumulation | 32 |
| GPUs | 2 |
| **effective batch** | **256 sequences** |
| block size | 3,072 |
| precision | **fp32** — no mixed precision, unlike the pretraining path |
| packing | off |
| label smoothing | 0.0 |
| save interval | 100 steps |

Note the two deliberate differences from pretraining: fp32 instead of bf16, and a
much lower peak LR (3e-4 vs 1e-3).

### Data and masking

| Setting | Value |
|---|---|
| dataset | `200m_generated` (`sft_v1_200m_90k`), first 134 shards |
| holdout | 2 shards |
| trace field | `cot_by_method.trajectory_sep.cot_format_no_labels` |
| multi-turn | yes |
| prompt masking | on |
| environment-move masking | on |
| tokenizer | `LanTokenizerSFT`, `include_env_tokens: true` (85 tokens) |

Loss is computed only on the model's own tokens: the prompt is masked, and moves the
environment produced inside a multi-turn trace are masked too. Getting this wrong
teaches the model to predict environment output, which looks like a working run and
scores badly.

**Source dataset caveat.** Use `Pre-to-Post-2/200M_SFT_dataset` as the clean source.
The mirror `sft_v1_200m_90k` has `<verify>` tokens in the trace field; the cache
already in use was built from the clean source and its tensors are identical, but a
rebuild from the wrong mirror would silently poison the vocabulary.

For 1,024-context pretrained models the launcher adds a YaRN RoPE factor of 3.0 to
reach block size 3,072. The 47M interleaved models are natively 3,072 and need no
scaling.

## Launch

```
cd chess/pretrain-sft
modal run --detach modal_scripts/launch_sft_interleave.py \
  --pretrained-model <hf-checkpoint> --lr 3e-4
```
