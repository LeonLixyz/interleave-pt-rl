---
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
  - text-generation
  - olmo2
  - safetensors
  - math
  - reasoning
  - pretraining
  - sft
  - rl
  - grpo
---

# Math-Models

Checkpoints from a **pretraining → post-training** study on a **1B-parameter OLMo-2**
model trained on math. The repo bundles, for 15 pretraining anchors, the model at
every stage of the pipeline:

**pretrain → anneal → SFT → RL**

so you can compare how the amount of pretraining changes what post-training can add.

## Structure

Each folder is one **pretraining anchor** (a stable pretraining checkpoint at
`step{N}`, ≈ `N × 2.097M` tokens). Inside it, one subfolder per stage:

```
step{N}/
├── pretrain/        # stable pretraining snapshot @ step N
├── anneal/          # LR-annealed from the pretrain snapshot
├── sft/             # SFT (NuminaMath) on top of the anneal
└── rl/
    ├── step100/     # GRPO RL (DeepScaleR), every 100 steps ...
    ├── step200/
    └── ... up to step3000
```

Every leaf is a self-contained, ready-to-load HF model in **bf16 safetensors**
(weights + tokenizer only — no optimizer state or training logs).

| Anchor | Pretrain tokens | Anchor | Pretrain tokens |
|---|---|---|---|
| `step5000`  | 10.5B | `step45000` | 94.4B  |
| `step10000` | 21.0B | `step50000` | 104.9B |
| `step15000` | 31.5B | `step60000` | 125.8B |
| `step20000` | 41.9B | `step70000` | 146.8B |
| `step25000` | 52.4B | `step80000` | 167.8B |
| `step30000` | 62.9B | `step90000` | 188.7B |
| `step35000` | 73.4B | `step95368` | 200.0B (final) |
| `step40000` | 83.9B | | |

## Loading

The models use the standard **OLMo-2** architecture, so `transformers` loads them
directly — no `trust_remote_code`. Point `from_pretrained` at the subfolder you want.
If you don't want to clone the whole repo (it's large), snapshot just one leaf:

```python
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

path = snapshot_download(
    "pre-to-post-olmo/Math-Models",
    allow_patterns="step20000/sft/*",
    local_dir="Math-Models",
)
model = AutoModelForCausalLM.from_pretrained(f"{path}/step20000/sft")
tok = AutoTokenizer.from_pretrained(f"{path}/step20000/sft")
```

## Generating a rollout

A `generate.py` at the repo root wraps the load + chat-template + decode:

```bash
python generate.py --model step20000/sft --prompt "What is 12 * 13?"
python generate.py --model step95368/rl/step3000 --prompt "..." --max-new-tokens 1024
```

`pretrain/` and `anneal/` are base LMs (no chat template); `sft/` and `rl/` are
chat-tuned and expect the chat template (applied automatically by `generate.py`).
