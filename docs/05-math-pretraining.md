# Math pretraining and annealing

The second model line: a 1.5B OLMo2-architecture model pretrained on math-heavy web
text, with a separate annealed checkpoint at each milestone.

This whole tree came from `math-pretraining`, which had **no git repo** — this copy is
the only backup.

## Code

| File | Role |
|---|---|
| `math/train.py` | Modal launcher: builds the image, `torchrun`s the inner script; functions `run_h100`, `run_h200`, `run_h200_4node` |
| `math/train_inner_mix.py` | **the real training script and the authoritative config** (multi-source mix) |
| `math/train_inner.py` | legacy single-glob inner script (only with `--no-use-mix`) |
| `math/launch_anneals.py` | spawns one anneal per stable milestone |
| `math/common.py` | Modal volume and mount names shared by every math app |
| `math/download*.py`, `math/tokenize_*.py`, `math/extract_*.py` | corpus download, tokenization, held-out extraction |
| `math/eval_ppl.py`, `math/eval_downstream.py`, `math/entropy_scan.py` | evaluation apps |
| `math/convert_hf_to_dcp.py`, `math/export_bundle.py`, `math/upload_*.py` | checkpoint conversion and release |

## Config

**Python, not YAML** — the authoritative values are in `math/train_inner_mix.py` and
the `math/train.py` entrypoint defaults.

### Model

`TransformerConfig.olmo2_1B_v2` from OLMo-core.

| Setting | Value |
|---|---|
| total parameters | 1.48B (1.07B non-embedding) |
| layers | 16 |
| hidden size | 2,048 |
| FFN | 8,192 |
| attention heads | 16 (no GQA) |
| vocab | 100,278 (dolma2 tokenizer, padded) |
| RoPE θ | 1e4 |
| context length | 4,096 |
| precision | bf16 params, fp32 reduce |
| attention backend | flash-attn 2 |

### Stable pretraining

| Setting | Value |
|---|---|
| total tokens | 200B (95,368 steps) |
| sequence length | 4,096 |
| global batch | 512 × 4,096 = **2,097,152 tokens** |
| rank microbatch | 8 × 4,096 tokens (`train.py` override; the inner default is 2×) |
| optimizer | `SkipStepAdamW` |
| learning rate | 4e-4 |
| schedule | `ConstantWithWarmup`, warmup = 2B tokens (`warmup_min_lr: 0.0`) |
| betas | 0.9 / 0.95 |
| weight decay | 0.033 — **excluded for `embeddings.weight`** via an optimizer group override |
| z-loss | 1e-5 |
| max grad norm | 1.0 |
| parallelism | HSDP, wrapping by blocks |
| activation checkpointing | selected modules — `blocks.*.feed_forward` |
| float8 | disabled |
| torch.compile | on |
| seed | 1337 |
| checkpoints | every 2,500 steps; ephemeral every 250 (removed as superseded) |
| in-training evals | off (`no_evals=True`) |

### Data mixture

70% math / 30% general, composed in `DEFAULT_WEIGHTS`:

| Source | Weight |
|---|--:|
| `math_3` (Nemotron-CC-Math-v1, quality 3) | 21% |
| `math_4plus` (quality 4+) | 21% |
| `math_4plus_MIND` | 28% |
| `dolma3` | 30% |

### Annealing

One anneal per stable milestone, producing the `anneal` anchors.

| Setting | Value |
|---|---|
| tokens | 5B per anchor |
| schedule | `LinearWithWarmup`, no warmup, `alpha_f = 0.0` → LR decays 4e-4 to 0 |
| eval split seed | 20260626 |
| eval ratio | 0.15 |

`math/launch_anneals.py` resolves the deployed function
(`modal.Function.from_name("math-pretraining-train", "run_h200")`) and spawns one call
per milestone.

### Hardware

`math/train.py` exposes three functions: `run_h100` (H100:8), `run_h200` (H200:8), and
`run_h200_4node` — the last uses `@modal.experimental.clustered(size=4, rdma=True)`
for 32 GPUs over EFA/RDMA. The image is CUDA 12.4 + torch 2.8.0 + flash-attn 2.8.3 +
OLMo-core, and `torchrun` launches the inner script.

## Launch

```
cd math
modal run --detach train.py --mode stable --tokens 200000000000 --run-name math-1b-v0
modal run --detach launch_anneals.py
```

Defaults on the entrypoint: `gpu_type="H200"`, `nodes=1`, `mode="stable"`,
`tokens=200_000_000_000`, `warmup_tokens=2_000_000_000`, `lr=4e-4`,
`rank_microbatch_size_tokens=8*4096`, `compile_model=True`, `save_interval=2500`,
`ephemeral_save_interval=250`.

### External dependency

Pretraining needs **OLMo-core**, which is *not* included here. `math/train.py`
resolves it as a sibling directory of the original project root
(`LOCAL_PROJECT_DIR.parent / "OLMo-core"`). Either restore that layout or edit the
path. Version details in `08-data-and-artifacts.md`.
