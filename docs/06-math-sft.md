# Math SFT

Supervised fine-tuning of an annealed math checkpoint on NuminaMath-CoT, producing the
`sft` anchors.

## Code

`math/sft.py` (Modal app `math-1b-sft`). It is a thin Modal wrapper around a **LLaMA-Factory fork** that is not
included here. The original path is `~/Desktop/Research/RL/Chess RL/pre2post-LM-SFT`
(4.2 MB); only the three `examples/train_full/olmo_sft_1b*.yaml` files are
math-specific, and those are copied into `math/external/sft-configs/`.

Entrypoints: `smoke`, `sft_single`, `sft_numinamath`, `sft_sweep`.

---

## Config

`math/external/sft-configs/olmo_sft_1b_numinamath.yaml`

| Setting | Value |
|---|---|
| stage | `sft`, full fine-tuning |
| dataset | `numinamath_cot` (859,490 examples) |
| template | `olmo2` |
| cutoff length | 4,096 |
| packing | off |
| loss on prompt | **off** (`train_on_prompt: false`) — assistant tokens only |
| per-device batch | 32 |
| gradient accumulation | 2 |
| GPUs | 8 |
| **effective batch** | **512** |
| learning rate | 1e-5 |
| schedule | cosine, `warmup_ratio: 0.03` |
| epochs | 1 |
| precision | bf16 |
| attention | sdpa |
| Liger kernel | enabled |
| DeepSpeed | ZeRO stage 1 (`examples/deepspeed/ds_z1_config.json`) |
| saving | per epoch, weights only (`save_only_model: true`) |

### The earlier variant

`olmo_sft_1b.yaml` is the OpenThoughts recipe that preceded it: packing **on**, cutoff
8,192, per-device batch 16 × accumulation 2, 3 epochs, `warmup_ratio: 0.1`. Results in
the current line come from the NuminaMath config above; keep them distinct when
comparing checkpoints.

## Launch

```
cd math
modal run --detach sft.py::sft_numinamath
```

Output checkpoints are published as
`pre-to-post-olmo/math-1b-sft-numinamath-bs512-from-step{N}`, one per pretraining
anchor, which is exactly what `rl_train.py` consumes as its base model.
