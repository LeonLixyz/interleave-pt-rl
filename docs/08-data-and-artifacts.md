# Data, artifacts, and external dependencies

None of this is stored in the repo. It is the map from a name in the code to the thing
it actually loads.

---

## Chess: Modal volumes

| Volume | Mount | Holds |
|---|---|---|
| `rl-reasoning-training-data` | `/data` | pretraining corpus, SFT cache, leg manifests, trace-transfer artifacts |
| `rl-reasoning-checkpoints` | `/checkpoints`, `/pretrain-checkpoints` | pretraining checkpoints and RL→HF exports |
| `chess-rl-miles-checkpoints` | `/rl-checkpoints` | RL run roots (Miles checkpoints + saved rollouts) |
| `chess-rl-miles-data` | `/data` | RL training parquets |
| `chess-rl-eval-results-r6` | `/results` | evaluation outputs |

### Key paths

```
/data/pretrain_v1_20b/                                   raw corpus shards (raw.NNNN.npy)
/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/
    ├── source_manifest.json                             shard inventory
    ├── pretrain_selection.json                          the frozen 10B+1 token selection
    ├── sft_cache/{input_ids.i32,labels.i32,offsets.npy,metadata.json}
    └── legs/{p1,p2}/{metadata.json,order.npy}           leg manifests
/data/sft_injection_ablation_v1_20260801/
    ├── legs/…                                           ablation leg manifests
    └── trace_transfer/                                  traces, combined caches, loop legs
/checkpoints/interleave_50m/pretrain/sft_injection_ablation_v1_20260801/<arm>/final
/pretrain-checkpoints/interleave_50m/rl_hf/<run>-s<step>  converted RL checkpoints
/rl-checkpoints/chess-rl-miles-interleave/<run>/          RL run root
/results/ablation_pass16_clean_v2_bos/<arm>/n16/<shard>/  eval results
```

### Pinned sources

| What | Identifier |
|---|---|
| pretraining corpus | HF `chess-pre-to-post/pretrain_v1_20b` @ `07dd1b7090ca5f0fb05ef624c26b20bff19483c8` |
| SFT dataset (clean) | HF `Pre-to-Post-2/200M_SFT_dataset` @ `fd343bd28f6a40fc3dab4dcfb6e74c11b7a20b90` — 77,717 rows |
| SFT cache hash | `d82378522d43d5db3e8333588c24b1f864bb9e8ecd46303e1d2cd2e31d31df98` |
| experiment version | `sft_injection_ablation_v1_20260801` |

Do **not** rebuild the SFT cache from the `sft_v1_200m_90k` mirror: its trace field
contains `<verify>` tokens. The cache in use was built from the clean source above.

### RL datasets (`chess-rl-miles-data:/chess-rl-data/`)

| File | Rows | sha256 |
|---|--:|---|
| `train_v4_dataset_balanced_multi_turn.parquet` | 53,225 | `bcf131d88cd9916f6ec12a12f8e85901a05e0dabc5d5a1fdebd8ea0b0ae86c30` |
| `train_v4_p1w1_pass16band_multi_turn.parquet` | 26,967 | `f94e9db0…` |
| `train_v4_k16band_multi_turn.parquet` | 28,774 | `e08dbc5ed1419fd03674962d935a6eb38baa0be6d6c37c41dccd7af7d18d4e93` |
| `train_v4_rollband_multi_turn.parquet` | 25,728 | `d8d2e0c621515bd6a2b58243206ede69145e2b9f2a4eda8408ec82ea9c9f9c50` |
| `train_v4_e2w1band_multi_turn.parquet` | 31,915 | `2dfaf12acfda33c1daccfea61c11ce0eedb881ec162ab7f956c7d128e5b89452` |
| `train_v4_e3p2band_multi_turn.parquet` | 31,859 | `9184d9c01f70cf03226af91e221b37dc4686f10ed38749a16015b52cc95bac29` |
| `train_v4_e2w1loopband_multi_turn.parquet` | 27,906 | `eec7155a9bd430b278ff8a6717f2d02fd849d2c76f009a66fc00cf0ca86c0bfb` |
| `train_v4_e3p2loopband_multi_turn.parquet` | 26,836 | `deec2f414455919dfc57361fcaab9e873cb92babb351b2bf82c0fae457dbfbf2` |

The `*band*` files are **solvable-only** sets: the puzzles a specific checkpoint solved
1–15 times out of 16 in its own full evaluation. Each belongs to one model; they are
not interchangeable. Built by `tools/build_bands.py`.

### Held-out evaluation shards

Held-out pretraining loss uses corpus shards the frozen 10B selection never touched,
drawn with seed `20260804`:

```
41859, 35917, 27837, 18175, 44567, 44931
```

2,048 packed 3,072-token windows = 6.3M tokens. **Any new pretraining selection must
exclude these shards** or the metric is contaminated. The fresh 5B selection built for
the 10B loop legs does exclude them; its `selection_hash` is
`fa67a458686912022c4de0f6e586e4b91f64e90b47a25ad7a071615dee3cf7d6`.

SFT loss uses 4,096 fixed probe rows (2,048 from each half of the SFT set), seed
`20260804 + 1`.

---

## Math: Modal volumes

| Volume | Holds |
|---|---|
| `nemotron-cc-math-v1` | raw math corpus |
| `dolma3-dolmino-mix-100B-1125` | general-text corpus |
| `math-pretraining-tokenized` | tokenized shards |
| `math-pretraining-untrained` | held-out / untrained extracts |
| `olmo-core-checkpoints-v2` | pretraining, anneal, SFT, RL checkpoints |
| `olmo-core-cache` | OLMo-core caches |

Names are defined in `math/common.py`.

Released weights: HF `pre-to-post-olmo/Math-Models`, one folder per anchor
(`pretrain`, `anneal`, `sft`, `rl`). RL weights are merged from verl FSDP shards
before upload (`math/export_bundle.py`).

RL data: HF `pre-to-post-olmo/rl-math-skyeasy25k-omi2` → `/checkpoints/rl_data/skyeasy25k_omi2/`.
SFT data: `numinamath_cot`, 859,490 examples.

---

## Modal secrets

| Secret | Used by | Status |
|---|---|---|
| `huggingface-secret` | all upload/download paths | working |
| `wandb-secret` | training loggers | **API key expired.** Runs from this study never uploaded metrics. Chess launchers set `logging.backend=none` / `logger=['console']`, so training is unaffected — but W&B has no data for these runs and reward curves must be rebuilt from saved rollouts with `tools/rl_reward_curve.py`. Refresh the key before relying on W&B again. |

---

## External dependencies (not included here)

| Dependency | Used by | Version / origin |
|---|---|---|
| Miles | chess RL | `radixark/miles@e20de26c94412301ba2a746e8d942220bad0d00d` (v0.2.1). **We ship a modified fork** in `miles/`; our diff is `miles/our_changes.patch` (17 modified files + `miles/miles/rollout/env_reply_ordering.py` added). |
| SGLang | chess RL inference | pinned in the Modal image; `chess/rl/sitecustomize.py` monkey-patches its detokenizer manager to flatten nested token ids |
| reward function | chess eval + RL | `pavelslab-nyu/pre2post-chess@40f04428a0a446ca319c8429bda8c0cff15b5e5a`, single file copied to `chess/eval/reward_function/` |
| OLMo-core | math pretraining | `allenai/OLMo-core`; resolved as a sibling directory of the project root by `math/train.py` |
| verl | math RL | 0.9.0.dev, original at `pretrain-rl-scaling/verl-olmo3` (stock verl) |
| LLaMA-Factory fork | math SFT | original at `pre2post-LM-SFT`; only the three `olmo_sft_1b*.yaml` files are math-specific and they are copied to `math/external/sft-configs/` |

If you restore this repo on a new machine, the four "original at …" paths are the
things that will be missing. Chess runs fully self-contained here; math pretraining,
SFT, and RL each need one external tree.

---
