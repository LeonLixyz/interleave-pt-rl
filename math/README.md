# math-pretraining

Single-pass math LLM pretraining on `nvidia/Nemotron-CC-Math-v1` with per-checkpoint annealing, on Modal.

## Plan

- Base model: OLMo2-1B (`TransformerConfig.olmo2_1B_v2`) via OLMo-core, dolma2 tokenizer.
- Data: 200B-token single-pass, 70% Nemotron-CC-Math-v1 + 30% Dolmino-100B.
- Math sub-mix (70% × split): 30% `/3/` + 30% `/4plus/` + 40% `/4plus_MIND/`.
- Schedule: WSD — `ConstantWithWarmup` (2B warmup) after which LR is constant. Take checkpoints at milestones (10/25/50/65/80/100/150/200B tokens). Each fork becomes a short (~5B) linear-to-zero anneal.
- Eval: URL-level held-out split for perplexity + lm-eval-harness math benchmarks.
- Data plumbing: OLMo-core `composable` API with `MixingInstanceSource` hierarchically combining 4 token sources via weighted ratios.

## Modal volumes

| Volume | Purpose |
|---|---|
| `nemotron-cc-math-v1` | Raw Nemotron parquet shards (3, 4plus, 4plus_MIND) |
| `dolmino-mix-1124` | Raw Dolmino-100B json.zst/json.gz/jsonl.gz shards |
| `math-pretraining-tokenized` | Pre-tokenized `.npy` shards (dolma2 tokenizer); layout: `/3/`, `/4plus/`, `/4plus_MIND/`, `/dolmino/<subset>/...` |
| `olmo-core-checkpoints-v2` | Training + anneal checkpoints (shared with existing OLMo-core runs) |
| `olmo-core-cache` | HF/torchinductor/etc. caches |

## Stages

```bash
# 1. Download Math-3+ shards (~100 GiB; detached, ~10 min wall-clock)
modal run --detach download.py --subset 3

# 2. Sanity-check parquet schema (auto-detects 'text' column)
modal run tokenize_corpus.py::inspect --shard "3/part_000000.parquet"

# 3. Tokenize one shard end-to-end (smoke test, ~7 min on 32 CPU)
modal run tokenize_corpus.py::tokenize_one --shard "3/part_000000.parquet"

# 4. Fan out tokenization across all 57 shards (detached, ~10-15 min wall-clock)
modal run --detach tokenize_corpus.py --subset 3

# 5. Smoke-test training pipeline on a tiny slice (20 steps, no checkpoints)
modal run --detach train.py --gpu-type H100 --mode stable \
    --tokens 200_000_000 --warmup-tokens 1_000_000 --benchmark-steps 20 \
    --data-glob "/tokenized/3/part_000000.npy" --run-name math-1b-smoke \
    --no-compile-model --save-interval 0 --ephemeral-save-interval 0

# 6. Full stable-phase run (single node H200, ~2 days at ~3.5M tok/s):
modal run --detach train.py --gpu-type H200 --mode stable \
    --tokens 130_000_000_000 --warmup-tokens 2_000_000_000 \
    --data-glob "/tokenized/3/part_*.npy" --run-name math-1b-stable-v0

# 7. Fork-and-anneal at a checkpoint (linear LR decay to 0 over --tokens):
modal run --detach train.py --gpu-type H200 --mode anneal \
    --load-path "/checkpoints/math-1b-stable-v0/step-XXXXX" \
    --tokens 5_000_000_000 --run-name math-1b-anneal-XXXXX
```

## Observed numbers (smoke runs)

- **Raw parquet download** (Math-3+): 57 shards × ~1.8 GiB ≈ 102 GiB, ~10 min wall on 55 parallel containers (~150-200 MB/s/container).
- **Tokenization** (dolma2): 1.397B tokens per shard at 3.5M tok/s, full Math-3+ subset tokenized in ~12 min wall on 57 parallel containers.
- **Estimated total Math-3+ tokens** with dolma2: 57 × ~1.4B ≈ **~80B** (the paper's 133B figure is tokenizer-specific). Reach 100B+ by also tokenizing `4plus_MIND/` (90 shards, ~73B tokens at the paper's count).
- **Training smoke test** (20 steps, OLMo2-1B, 1 H100 node, 8 GPUs, `--no-compile-model`):
  - **MFU 27.04%**, **TPS 28,815/device**, **~230k tok/s aggregate**, **268 TFLOPs/device**.
  - 9 sec/optimizer step with global batch 2.1M tokens (gradient_accumulation=16 microbatches of 16k tokens/rank).
  - Loss starts at CE=11.87 (~log(vocab)) as expected from random init.
  - With `compile_model=True` we expect 35-40% MFU per OLMo-core's benchmarks — bumps throughput to ~300-350k tok/s.

## Cost projection for full run

- 80B stable + 7×5B anneals = 115B tokens at ~300k tok/s = **~107 GPU-hours single H100 node** = **~$3,400 on Modal** (with `compile_model=True`).
- Without compile (27% MFU): ~$4,500.

## Open items before running the full stable phase

- [ ] Decide LR schedule details (current peak LR 4e-4 from OLMo2-1B reference; Hägele et al. suggest ~half-cosine-equivalent for WSD, so 3-4e-4 is right).
- [ ] Hold-out split: `metadata.warc_filename` is in the parquet schema — use URL-level hash to split off ~1% for in-distribution PPL eval.
- [ ] Tokenize `4plus_MIND/` (only needed if pushing past 80B).
- [ ] Decide whether to include MIND in the anneal data mix (vs. just Math-4+).

## Notes

- Dataset is gated under NVIDIA Open Data License — must be accepted on HF first. The shared `huggingface-secret` token must have access.
- Phi-4 was used in dataset cleaning, which may carry license obligations on derivative model releases.
- Single-pass guarantee depends on pre-shuffled shard ordering + checkpointed consumed-token offsets — do not replay shards on resume.
