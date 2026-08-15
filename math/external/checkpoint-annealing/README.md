# Checkpoint annealing — OLMo-3 7B

Make 8 comparable 7B checkpoints across the pretraining-compute axis (the starting points
of the scaling study).

Intermediate pretraining checkpoints still have a non-zero learning rate, so their loss is
not comparable. Fix: resume the checkpoint (model + optimizer + data-loader state) and decay
the learning rate to 0 over a fixed 10B-token budget on the original pretraining mix
(`OLMo-mix-0625`), then convert the result to HuggingFace format for SFT/RL.

## Checkpoints

Source run `OLMo3-7B-swafix`. `STEP` = base token count = HF label. The 10B anneal adds 10B
tokens to every point, so the effective label = base + 10B.

| Base tokens | 25B | 50B | 100B | 250B | 400B | 600B | 1T | 2.5T |
|---|---|---|---|---|---|---|---|---|
| `STEP` | 6000 | 12000 | 24000 | 60000 | 95000 | 143000 | 238000 | 596000 |
| Effective | 35B | 60B | 110B | 260B | 410B | 610B | 1.01T | 2.51T |

## Files

| File | Purpose |
|---|---|
| `download_checkpoint.py` | Download one native checkpoint (`gs://ai2-llm/...` → local disk). |
| `anneal.py` | Anneal launcher (torchrun): LR → 0 over the budget on `OLMo-mix-0625`. |
| `convert_to_hf.py` | Convert an annealed checkpoint → HuggingFace format. |
| `run.sh` | Runs all three for one checkpoint: download → anneal → convert. |

## Step 0 — setup (once)

Run from the repo root (the folder containing `OLMo-core/` and `checkpoint-annealing/`):

```bash
conda create -n olmo-anneal python=3.11 -y
conda activate olmo-anneal
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e OLMo-core
pip install flash-attn==2.8.3 --no-build-isolation
pip install google-cloud-storage beaker-py beaker-gantry transformers
```

Checkpoints and data stream from the public `gs://ai2-llm` bucket (no credentials). Set
`OUT_DIR` to a directory with ~90 GB (native) + ~85 GB (annealed) free per checkpoint; on
multi-node it must be reachable from every node.

## Step 1 — test (once)

Smallest checkpoint, 2 GPUs, a few steps, then stop. Succeeds if it prints a final HF path:

```bash
conda activate olmo-anneal
cd checkpoint-annealing
OUT_DIR=/your/output bash run.sh 6000 2 --smoke
```

## Step 2 — anneal each checkpoint

One command per checkpoint (uses all GPUs on the node by default):

```bash
OUT_DIR=/your/output bash run.sh 6000      # 25B
OUT_DIR=/your/output bash run.sh 12000     # 50B
# repeat for: 24000 60000 95000 143000 238000 596000
```

Output HF checkpoint: `OUT_DIR/hf/swafix-step<STEP>/`.

## GPUs and nodes

The global batch is fixed at 4,194,304 tokens (512 sequences) and does not change with GPU
count, so:

- One anneal = one node (8 GPUs). A 10B anneal is ~2,400 steps; a single job can't use more
  than ~512 GPUs (no more data-parallel ranks than the 512-sequence batch).
- To use many nodes, run the 8 checkpoints in parallel (one per node), not one job across
  many nodes:

  ```bash
  STEPS=(6000 12000 24000 60000 95000 143000 238000 596000)
  OUT_DIR=/shared/output bash run.sh ${STEPS[$NODE_INDEX]}
  ```

## Rules for the sweep

- Identical recipe for all 8 points (same budget, data, schedule, code) — varying any is a
  confound.
- Token label = base + 10B. `anneal.py` reads the true base from `data_loader.tokens_processed`
  (not `step × batch`, which overcounts skipped steps) and logs it.
- Budget check (once): anneal `24000` (100B) at 5B / 10B / 30B; confirm loss has flattened by
  10B. If not, raise the budget and re-anneal all points at the new value.
- For a bit-faithful run, pin `OLMo-core` to the OLMo-3-era commit; the in-repo copy already
  works because `anneal.py` rebuilds the exact GQA architecture from each checkpoint's config.
