# Chess pretraining (SFT mixed into the stream)

Pretraining text and supervised puzzle-solving traces are shuffled into one record
stream and trained with a single objective — not alternating batches, not a separate
stage.

## Code

Under `chess/pretrain-sft/`:

| File | Role |
|---|---|
| `scripts/train/train_interleaved_hf.py` | entry point (thin CLI shim) |
| `training/interleaved_hf_trainer.py` | the trainer — `InterleavedHFTrainer`, loss, LR arcs, resume/export |
| `training/interleaved_data.py` | the data engine — selection, SFT cache, leg manifests, stream |
| `llm_tokens/chess/lan_tokenizer_sft.py` | `LanTokenizerSFT`, the 85-token vocabulary |
| `llm_tokens/chess/tokenizer_factory.py` | `init_tokenizer(name, config)` — the single dispatch point |
| `training/hf_tokenizer_utils.py` | writes a standalone HF tokenizer into exported checkpoints |
| `modal_scripts/launch_sft_injection_ablation.py` | the launcher used for every published run |

Legacy, present but superseded: `training/mixed_trainer.py`, `training/mixed_data_utils.py`,
`scripts/train/train_mix.py` (older mixed-batch path), `scripts/train/train.py`
(pre-HF loop). Several launchers under `modal_scripts/` reference config directories
that no longer exist and cannot run.

### How the data engine is layered

```
source manifest → pretrain selection → SFT cache → leg manifest → stream
```

Each stage is hashed and checked by the next. Names that matter in
`training/interleaved_data.py`: `SourceShardManifest`, `build_pretrain_selection`,
`PretrainSelection`, `build_sft_cache`, `SFTCache`, `tokenize_masked_sft_row`,
`_mask_multi_turn_env_responses`, `_write_leg_manifest`, `LegManifest`,
`create_interleaved_dataloader`.

A leg's `order.npy` encodes the exact record order: value ≥ 0 is a local pretraining
record, negative value `-(row+1)` is a global SFT cache row, `PAD_RECORD` (int64 min)
pads to a whole number of global batches.

## Config

Base file: `chess/pretrain-sft/config/configs/interleaved_50m/base_3072.yaml`.
The launcher restates every value as an explicit override in `_run_v2r1_leg`, so
editing the yaml alone does not change a run.

### Model

| Setting | Value |
|---|---|
| parameters | 47,245,312 |
| architecture | Qwen3 |
| layers | 12 |
| hidden size | 512 |
| heads | 8 query / 4 key-value |
| head dim | 128 |
| FFN | 1,536 |
| context | 3,072 |
| dropout | 0.0 |
| attention | `sdpa` |
| parameter dtype | bf16 |
| tokenizer | `LanTokenizerSFT`, 85 tokens |

Vocabulary: 3 special (`<bos> <eos> <unk>`) + 6 pieces + 64 squares + 8 punctuation
(`x = + # O-O O-O-O . ...`) + 3 trace (`<T> </T> <sep>`) + `<call_env>` = 85. Reward
tokens exist in the tokenizer but are disabled here; enabling them gives 89 and breaks
checkpoint compatibility.

### Optimization

| Setting | Value |
|---|---|
| optimizer | AdamW |
| peak LR | 1e-3 |
| schedule | cosine to `eta_min` 1e-5 |
| warmup | 5% of steps |
| betas | 0.9 / 0.95 |
| weight decay | 0.1 |
| max grad norm | 1.0 |
| precision | bf16 |
| torch.compile | off |
| seed | 42 |
| local batch | 21 per rank |
| gradient accumulation | 1 |
| world size | 8 × H200 |
| global batch | 168 × 3,072 = 516,096 token positions |
| SFT loss weight | 1.0 |
| save interval | 200 steps |
| log interval | 10 |

Step count is derived, not chosen:
`total_steps = ceil((pretrain_records + sft_rows) / 168)`. A 5B leg with one SFT half
is 9,920 steps; a 10B run with all SFT rows is 19,840.

`sft_loss_weight: 1.0` means every optimizer step uses a globally normalized
valid-token loss — one SFT target token counts exactly as much as one pretraining
target token. The mixture is set by how many of each are in the stream.

`arc_steps` + `reset_optimizer_between_arcs` control the cosine shape: a single 10B run
is `[19840]`; the two-leg form is two runs of `[9920]` with a fresh optimizer, which
re-warms the LR to 1e-3 at the start of leg 2.

### Init modes

Mutually exclusive and fail-closed:

- `--resume <dir>` — continue a leg mid-flight with optimizer, scheduler and stream position.
- `--weights-only <dir>` — start a new leg from another checkpoint's weights only.

## Launch

```
cd chess/pretrain-sft

# 1. build selections, caches and leg manifests (CPU only)
modal run --detach modal_scripts/launch_sft_injection_ablation.py --action prep

# 2. one step on the real 8×H200 topology
modal run modal_scripts/launch_sft_injection_ablation.py --action canary-p1w1

# 3. the run
modal run --detach modal_scripts/launch_sft_injection_ablation.py --action train-p1w1
```

Actions follow the pattern `canary-<arm>` / `train-<arm>`. Trace-distillation legs use
`--action prep-trace --arm "2,4,8,16"`, then `canary-tracep2` / `train-tracep2` with the
same `--arm` list. The 10B loop legs use `prep-loop-selection`, `prep-loop`,
`canary-loop`, `train-loop` with `--arm "e2w1,e3p2"`.

Other launchers: `launch_50m_interleaved.py` (original DAG: `data-prep`, `canary`,
`train`, `p2`), `launch_50m_interleaved_20b.py` (20B fork),
`launch_interleave_pretrains.py` (20M-scale leg-2 continuations), `train.py` (generic
non-interleaved pretraining).

Checkpoints land at
`/checkpoints/interleave_50m/pretrain/sft_injection_ablation_v1_20260801/<arm>/final`.
