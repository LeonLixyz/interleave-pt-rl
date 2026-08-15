# Checkpoint System

## Directory Structure

```
checkpoints/
  200m_C9e18_alpha0.048_beta0.04/       <- experiment_name from config
  │
  ├── config.yaml                        Training config snapshot
  │
  ├── latest/                            RESUME CHECKPOINT (overwritten each save)
  │   ├── model.safetensors              Model weights
  │   ├── optimizer.bin                  Optimizer state (AdamW momentum/variance)
  │   ├── scheduler.bin                  LR scheduler position
  │   ├── random_states_0.bin            RNG states per GPU
  │   └── training_state.json            step, epoch, total_batches, data_seed
  │
  ├── step_272/                          HF MODEL SNAPSHOT (every save_hf_interval steps)
  │   ├── config.json                    HuggingFace model config
  │   ├── model.safetensors              Model weights (no optimizer)
  │   ├── generation_config.json         Token IDs, sampling defaults
  │   ├── vocab.json                     Token vocabulary
  │   ├── tokenizer.py                   Self-contained HF tokenizer
  │   ├── tokenizer_config.json          Tokenizer config
  │   └── special_tokens_map.json        Special token mappings
  │
  ├── step_544/                          Another HF snapshot
  │   └── ...
  │
  └── final/                             FINAL HF MODEL (same format as step_*)
      ├── config.json
      ├── model.safetensors
      ├── generation_config.json
      ├── vocab.json
      ├── tokenizer.py
      ├── tokenizer_config.json
      └── special_tokens_map.json
```

## Naming

Set `training.experiment_name` in the config to control the directory name:

```yaml
training:
  experiment_name: "200m_C9e18_alpha0.048_beta0.04"
  save_dir: "checkpoints"
```

This gives: `checkpoints/200m_C9e18_alpha0.048_beta0.04/`

If `experiment_name` is not set, an auto-generated run name is used.

## Two Types of Checkpoints

### `latest/` — Resume checkpoint

Single directory, overwritten on each save. Contains everything needed to
resume training at the exact state: model weights, optimizer, LR schedule,
RNG states, and training metadata.

### `step_*/` and `final/` — HF model snapshots

Clean, self-contained HuggingFace model directories. Ready for:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("checkpoints/.../final")
```

No optimizer or scheduler state. Saved every `save_hf_interval` steps
(default: total_steps // 10) and once at end of training.

## What Auto-Resume Restores

| State                              | File                  | How                      |
|------------------------------------|-----------------------|--------------------------|
| Model weights                      | model.safetensors     | `acc.load_state()`       |
| Optimizer (AdamW momentum/variance)| optimizer.bin         | `acc.load_state()`       |
| LR scheduler position              | scheduler.bin         | `acc.load_state()`       |
| CUDA/torch/numpy RNG states        | random_states_*.bin   | `acc.load_state()`       |
| Optimizer step count                | training_state.json   | `_load_training_state()` |
| Current epoch                       | training_state.json   | `_load_training_state()` |
| Data shard seed                     | training_state.json   | `_load_training_state()` |
| Exact data position                 | Computed from step    | Batch skip on resume     |

Data ordering is deterministic: shard shuffle uses `Random(seed + epoch)`,
so the same seed always produces the same shard order per epoch.

## Config

```yaml
training:
  experiment_name: "200m_C9e18_alpha0.048_beta0.04"
  save_dir: "checkpoints"
  save_interval: 500         # Resume checkpoint (latest/)
  save_hf_interval: 250      # HF model snapshots (step_*/)
  auto_resume: true           # Find latest/ on startup
  seed: 42                    # Base seed for data ordering
```

## Upload

```bash
huggingface-cli upload your-org/model-name checkpoints/200m_C9e18_alpha0.048_beta0.04/final
```
