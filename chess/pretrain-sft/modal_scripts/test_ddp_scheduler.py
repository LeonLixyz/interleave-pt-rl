"""
Test script: verify LR scheduler behavior under DDP with Accelerate.

Creates a tiny model on 8 GPUs and checks:
1. Does acc.prepare() change dataloader length?
2. How many times does scheduler.step() get called vs expected?
3. Does the LR schedule complete its full cycle?

Usage:
  modal run modal_scripts/test_ddp_scheduler.py
"""
import os
from pathlib import Path

import modal

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .pip_install(
        "torch>=2.6.0",
        "accelerate>=1.10.0",
        "transformers>=4.50.0",
    )
)

app = modal.App("chess-test-ddp-scheduler", image=image)


@app.function(gpu="H200:8", timeout=60 * 60 * 1)
def test_scheduler():
    """Run a DDP scheduler test across 8 GPUs."""
    import subprocess, sys

    # Write the actual test script
    test_code = r'''
import torch
import math
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator

class DummyDataset(Dataset):
    """Fake dataset with known length."""
    def __init__(self, n_samples=8000, seq_len=64):
        self.n = n_samples
        self.seq_len = seq_len
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        x = torch.randint(0, 100, (self.seq_len,))
        return x, x

# Config
BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS = 1
LR = 1e-3
ETA_MIN = 1e-5
DATASET_SIZE = 8000  # known dataset size

# Setup
acc = Accelerator(
    gradient_accumulation_steps=GRAD_ACCUM,
    mixed_precision="bf16",
)

# Tiny model (input 64 -> hidden -> output 64 for MSE loss)
model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 64))
optimizer = AdamW(model.parameters(), lr=LR)

# Dataloader
ds = DummyDataset(n_samples=DATASET_SIZE, seq_len=64)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

pre_prepare_len = len(loader)

# Build scheduler BEFORE prepare (like our trainer does)
steps_per_epoch = pre_prepare_len
opt_steps = math.ceil(steps_per_epoch / GRAD_ACCUM) * EPOCHS

eta_min_ratio = ETA_MIN / LR
def warmup_cosine(step):
    warmup = 100
    if step < warmup:
        return max(1e-8, (step + 1) / warmup)
    remain = max(1, opt_steps - warmup)
    x = min(step - warmup, remain) / float(remain)
    return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * x))

scheduler = LambdaLR(optimizer, warmup_cosine)

# Prepare
model, optimizer, loader, scheduler = acc.prepare(model, optimizer, loader, scheduler)

post_prepare_len = len(loader)

if acc.is_main_process:
    print("=" * 70)
    print("DDP SCHEDULER TEST")
    print("=" * 70)
    print(f"num_processes:      {acc.num_processes}")
    print(f"gradient_accum:     {GRAD_ACCUM}")
    print(f"dataset_size:       {DATASET_SIZE}")
    print(f"batch_size:         {BATCH_SIZE}")
    print(f"pre-prepare len:    {pre_prepare_len}")
    print(f"post-prepare len:   {post_prepare_len}")
    print(f"ratio post/pre:     {post_prepare_len / pre_prepare_len:.4f}")
    print(f"scheduler opt_steps:{opt_steps}")
    print(f"expected opt steps (if DDP splits data): {math.ceil(post_prepare_len / GRAD_ACCUM)}")
    print()

# Run training loop and track LR
step_count = 0
lr_history = []

for epoch in range(EPOCHS):
    for batch_idx, (x, y) in enumerate(loader):
        with acc.accumulate(model):
            out = model(x.float())
            loss = torch.nn.functional.mse_loss(out, y.float())
            acc.backward(loss)

            if acc.sync_gradients:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step_count += 1
                lr = scheduler.get_last_lr()[0]
                lr_history.append((step_count, lr))

if acc.is_main_process:
    print(f"Total optimizer steps: {step_count}")
    print(f"Scheduler expected:    {opt_steps}")
    print(f"Ratio actual/expected: {step_count / opt_steps:.4f}")
    print()

    # Print LR at key points
    if lr_history:
        n = len(lr_history)
        indices = [0, n//4, n//2, 3*n//4, n-1]
        print("LR Schedule Sample:")
        print(f"  {'Step':>8s}  {'LR':>12s}  {'Position':>10s}")
        for i in indices:
            s, lr = lr_history[i]
            pct = s / opt_steps * 100
            print(f"  {s:>8d}  {lr:>12.6f}  {pct:>9.1f}%")

        final_lr = lr_history[-1][1]
        print(f"\n  Final LR: {final_lr:.6f}")
        print(f"  Expected final LR (eta_min): {ETA_MIN}")
        print(f"  Did schedule complete? {'YES' if final_lr < ETA_MIN * 2 else 'NO (LR still high)'}")

    print()

    # Now test what happens if we MULTIPLY steps by num_processes
    print("=" * 70)
    print("TEST 2: What if we multiply opt_steps by num_processes?")
    print("=" * 70)
    opt_steps_scaled = opt_steps * acc.num_processes
    print(f"  Scaled opt_steps: {opt_steps_scaled}")
    print(f"  Actual steps:     {step_count}")
    print(f"  Ratio actual/scaled: {step_count / opt_steps_scaled:.4f}")
    matches_scaled = abs(step_count / opt_steps_scaled - 1.0) < 0.05
    print(f"  Does scaling fix it? {'YES' if matches_scaled else 'NO'}")

    # Test dividing
    print()
    print("TEST 3: What if we divide opt_steps by num_processes?")
    opt_steps_divided = opt_steps // acc.num_processes
    print(f"  Divided opt_steps: {opt_steps_divided}")
    print(f"  Actual steps:      {step_count}")
    print(f"  Ratio actual/divided: {step_count / max(1, opt_steps_divided):.4f}")
    matches_divided = abs(step_count / max(1, opt_steps_divided) - 1.0) < 0.05
    print(f"  Does dividing fix it? {'YES' if matches_divided else 'NO'}")

    print()
    print("=" * 70)
    print("CONCLUSION:")
    if matches_scaled:
        print("  -> MULTIPLY steps_per_epoch by num_processes before building scheduler")
    elif matches_divided:
        print("  -> DIVIDE steps_per_epoch by num_processes before building scheduler")
    else:
        print(f"  -> Neither simple fix works. actual={step_count}, expected={opt_steps}, "
              f"scaled={opt_steps_scaled}, divided={opt_steps_divided}")
        print("  -> May need to investigate Accelerate's AcceleratedScheduler wrapper")
    print("=" * 70)
'''

    # Write test script
    with open("/tmp/test_ddp_sched.py", "w") as f:
        f.write(test_code)

    # Run with accelerate on 8 GPUs
    cmd = [
        "accelerate", "launch",
        "--multi_gpu",
        "--num_processes", "8",
        "--mixed_precision", "bf16",
        "/tmp/test_ddp_sched.py",
    ]
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Test failed with exit code {rc}")


@app.local_entrypoint()
def main():
    test_scheduler.remote()
