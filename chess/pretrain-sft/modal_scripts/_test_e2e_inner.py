"""Inner test script — run via accelerate launch from test_e2e.py"""
import os, sys, math, json, shutil
import numpy as np
import torch

sys.path.insert(0, "/root/chess")

# ================================================================
# TEST 0: Create synthetic data shards
# ================================================================
print("=" * 70)
print("TEST 0: Creating synthetic data")
print("=" * 70)

DATA_DIR = "/tmp/test_data"
CKPT_DIR = "/tmp/test_checkpoints"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

from llm_tokens.chess.tokenizer_factory import init_tokenizer
from omegaconf import OmegaConf as _OC
_tok = init_tokenizer(name="LanTokenizer", config=_OC.create({
    "name": "LanTokenizer",
    "special_tokens": ["<bos>", "<eos>", "<pad>"],
    "include_move_numbers": False,
}))
VOCAB_SIZE = _tok.get_vocab_size()
print(f"  LanTokenizer vocab size: {VOCAB_SIZE}")

SEQ_LEN = 1024
TOKENS_PER_SHARD = 50000
NUM_SHARDS = 40  # more data than needed — max_steps will stop us early

for i in range(NUM_SHARDS):
    tokens = np.random.randint(0, VOCAB_SIZE, size=TOKENS_PER_SHARD, dtype=np.int64)
    np.save(os.path.join(DATA_DIR, f"shard_{i:04d}.npy"), tokens)

total_tokens = NUM_SHARDS * TOKENS_PER_SHARD
print(f"  Created {NUM_SHARDS} shards, {TOKENS_PER_SHARD} tokens each")
print(f"  Total available: {total_tokens:,} tokens")

# ================================================================
# TEST 1: cache_size=0 auto-detection
# ================================================================
print("\n" + "=" * 70)
print("TEST 1: cache_size=0 auto-detection")
print("=" * 70)

from training.data_utils import ShardedPackedTextDataset

ds = ShardedPackedTextDataset(
    txt_files=[DATA_DIR], seq_len=SEQ_LEN, cache_size=0, num_shards=NUM_SHARDS,
)
print(f"  cache_size (auto): {ds.cache_size}")
print(f"  steps_per_shard:   {ds.steps_per_shard}")
print(f"  total_steps:       {ds.total_steps}")

assert ds.cache_size >= TOKENS_PER_SHARD, f"FAIL: cache_size {ds.cache_size} < shard size {TOKENS_PER_SHARD}"
print("  PASS: cache_size >= shard size")

# ================================================================
# TEST 2: Training with max_steps, LR, checkpointing
# ================================================================
print("\n" + "=" * 70)
print("TEST 2: Training with pretrain_tokens / max_steps")
print("=" * 70)

from omegaconf import OmegaConf

BATCH_SIZE = 4
GRAD_ACCUM = 2
LR = 1e-3
ETA_MIN = 1e-5
# Use only 500K tokens worth of training (out of 2M available)
PRETRAIN_TOKENS = 500_000
EXPECTED_MAX_STEPS = PRETRAIN_TOKENS // (BATCH_SIZE * SEQ_LEN * GRAD_ACCUM)
print(f"  pretrain_tokens: {PRETRAIN_TOKENS:,}")
print(f"  batch_size={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, seq_len={SEQ_LEN}")
print(f"  tokens_per_step: {BATCH_SIZE * SEQ_LEN * GRAD_ACCUM}")
print(f"  expected max_steps: {EXPECTED_MAX_STEPS}")

cfg = OmegaConf.create({
    "model": {
        "architecture": "Qwen/Qwen3-0.6B",
        "block_size": SEQ_LEN,
        "n_layer": 28,  # must match Qwen3-0.6B layer_type length
        "n_head": 4,
        "num_key_value_heads": 4,
        "n_embed": 128,
        "intermediate_size": 256,
        "dropout": 0.0,
    },
    "training": {
        "device": "cuda",
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "epochs": 1,
        "experiment_name": "test_e2e",
        "optimizer": {"name": "adamw", "lr": LR, "weight_decay": 0.1, "betas": [0.9, 0.95]},
        "scheduler": {"name": "cosine", "eta_min": ETA_MIN, "warmup_ratio": 0.05},
        "max_grad_norm": 1.0,
        "log_interval": 1,
        "num_workers": 0,
        "save_dir": CKPT_DIR,
        "cache_size": 0,
        "mixed_precision": "bf16",
        "seed": 42,
    },
    "data": {
        "txt_path": DATA_DIR,
        "pretrain_tokens": PRETRAIN_TOKENS,
        "eval_holdout": 2,
    },
    "tokenizer": {
        "name": "LanTokenizer",
        "special_tokens": ["<bos>", "<eos>", "<pad>"],
        "include_move_numbers": False,
    },
    "logging": {"backend": "wandb", "project": "test-e2e", "mode": "disabled"},
})

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

from training.trainer_hf import HFTrainer

trainer = HFTrainer(cfg)
is_main = trainer.acc.is_main_process

if is_main:
    print(f"  train_loader len (post-prepare): {len(trainer.train_loader)}")
    print(f"  num_processes: {trainer.acc.num_processes}")

# Capture LR at every step
lr_history = []
original_log = trainer._log
def capture_log(epoch, step, loss):
    lr = trainer.scheduler.get_last_lr()[0]
    lr_history.append({"step": trainer.current_step, "lr": lr, "loss": loss.item()})
    original_log(epoch, step, loss)
trainer._log = capture_log

# Run training
trainer.train()

if is_main:
    total_steps = trainer.current_step
    print(f"\n  Total optimizer steps: {total_steps}")
    print(f"  Expected max_steps:   {EXPECTED_MAX_STEPS}")

    # ================================================================
    # TEST 2a: max_steps stopping
    # ================================================================
    print("\n" + "-" * 40)
    print("TEST 2a: max_steps early stopping")
    print("-" * 40)

    # Should have stopped at max_steps, not run through all data
    data_steps = ds.total_steps // BATCH_SIZE // GRAD_ACCUM  # what full epoch would be
    stopped_early = total_steps <= EXPECTED_MAX_STEPS + 1  # +1 tolerance
    used_less_than_full = total_steps < data_steps
    print(f"  Full epoch would be: ~{data_steps} steps")
    print(f"  Actually ran: {total_steps} steps")
    print(f"  Stopped at max_steps: {'PASS' if stopped_early else 'FAIL'}")
    print(f"  Used less than full data: {'PASS' if used_less_than_full else 'FAIL'}")

    # ================================================================
    # TEST 2b: Warmup (5% of max_steps)
    # ================================================================
    print("\n" + "-" * 40)
    print("TEST 2b: Warmup verification")
    print("-" * 40)

    expected_warmup = int(EXPECTED_MAX_STEPS * 0.05)
    warmup_lrs = [(h["step"], h["lr"]) for h in lr_history if h["step"] <= expected_warmup]

    if len(warmup_lrs) >= 2:
        for s, lr in warmup_lrs[:5]:
            print(f"    step {s:4d}: lr={lr:.8f}")
        increasing = all(warmup_lrs[i][1] <= warmup_lrs[i+1][1] for i in range(len(warmup_lrs)-1))
        print(f"  Expected warmup steps: {expected_warmup}")
        print(f"  LR increasing during warmup: {'PASS' if increasing else 'FAIL'}")
    else:
        increasing = True
        print(f"  Only {len(warmup_lrs)} warmup steps (expected {expected_warmup})")

    # ================================================================
    # TEST 2c: Cosine decay + final LR
    # ================================================================
    print("\n" + "-" * 40)
    print("TEST 2c: Cosine decay + final LR")
    print("-" * 40)

    post_warmup = [(h["step"], h["lr"]) for h in lr_history if h["step"] > expected_warmup]
    if len(post_warmup) >= 2:
        decreasing = all(post_warmup[i][1] >= post_warmup[i+1][1] for i in range(len(post_warmup)-1))
        print(f"  LR decreasing after warmup: {'PASS' if decreasing else 'FAIL'}")
    else:
        decreasing = True

    all_lrs = [h["lr"] for h in lr_history]
    final_lr = lr_history[-1]["lr"] if lr_history else 0
    peak_lr = max(all_lrs) if all_lrs else 0
    no_bad = all(lr > 0 and not math.isnan(lr) for lr in all_lrs)

    reached_eta = final_lr <= ETA_MIN * 2
    peak_ok = abs(peak_lr - LR) / LR < 0.15

    print(f"  Peak LR: {peak_lr:.8f} (base={LR}) {'PASS' if peak_ok else 'FAIL'}")
    print(f"  Final LR: {final_lr:.8f} (eta_min={ETA_MIN}) {'PASS' if reached_eta else 'FAIL'}")
    print(f"  No NaN/negative: {'PASS' if no_bad else 'FAIL'}")

    # Print full LR curve sample
    print(f"\n  LR curve ({len(lr_history)} steps):")
    n = len(lr_history)
    if n > 0:
        indices = sorted(set([0, 1, 2, expected_warmup-1, expected_warmup, expected_warmup+1,
                              n//4, n//2, 3*n//4, n-2, n-1]))
        indices = [i for i in indices if 0 <= i < n]
        for i in indices:
            h = lr_history[i]
            print(f"    step {h['step']:5d}: lr={h['lr']:.8f}  loss={h['loss']:.4f}")

    # ================================================================
    # TEST 3: Checkpoint verification
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Checkpoints")
    print("=" * 70)

    from pathlib import Path
    ckpt_dir = Path(CKPT_DIR) / "test_e2e"
    latest_dir = ckpt_dir / "latest"
    final_dir = ckpt_dir / "final"

    has_latest = latest_dir.exists()
    has_final = final_dir.exists()
    print(f"  latest/ exists: {'PASS' if has_latest else 'FAIL'}")
    print(f"  final/ exists: {'PASS' if has_final else 'FAIL'}")

    has_state = False
    if has_latest:
        state_file = latest_dir / "training_state.json"
        has_state = state_file.exists()
        if has_state:
            with open(state_file) as f:
                state = json.load(f)
            step_ok = state.get("step") == total_steps
            print(f"  Saved step={state.get('step')}, actual={total_steps}: {'PASS' if step_ok else 'FAIL'}")

    if has_final:
        has_model = any(final_dir.glob("*.safetensors"))
        has_cfg = (final_dir / "config.json").exists()
        print(f"  model.safetensors: {'PASS' if has_model else 'FAIL'}")
        print(f"  config.json: {'PASS' if has_cfg else 'FAIL'}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    results = {
        "cache_size_auto": ds.cache_size >= TOKENS_PER_SHARD,
        "max_steps_stopping": stopped_early,
        "used_less_than_full_data": used_less_than_full,
        "warmup_increasing": increasing,
        "cosine_decreasing": decreasing,
        "peak_lr_ok": peak_ok,
        "final_lr_reached_eta_min": reached_eta,
        "no_bad_lrs": no_bad,
        "checkpoint_latest": has_latest,
        "checkpoint_final": has_final,
        "state_saved": has_state,
    }

    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")

    all_pass = all(results.values())
    print(f"\n  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 70)
