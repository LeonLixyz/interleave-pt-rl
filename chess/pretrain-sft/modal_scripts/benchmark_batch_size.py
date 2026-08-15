"""
Benchmark maximum batch size per model size on a single H200 GPU.

Tests forward+backward passes at increasing batch sizes with bf16 mixed precision.
Reports the largest batch_size that doesn't OOM for each model configuration.

Usage:
  # Benchmark all model sizes
  modal run modal_scripts/benchmark_batch_size.py

  # Benchmark a specific model size
  modal run modal_scripts/benchmark_batch_size.py --model-filter 800m
"""
import os
from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
#  Modal setup                                                                 #
# --------------------------------------------------------------------------- #

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .pip_install(
        "torch>=2.6.0",
        "transformers>=4.50.0",
        "accelerate>=1.10.0",
    )
    .run_commands(
        'python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(\'Qwen/Qwen3-0.6B\')"'
    )
)

app = modal.App("chess-benchmark-batch-size", image=image)

# --------------------------------------------------------------------------- #
#  Model configs matching C9e18_beta0.03                                       #
# --------------------------------------------------------------------------- #

# NOTE: The original configs have num_key_value_heads=4 for all models,
# but n_head must be divisible by num_key_value_heads for Qwen3 GQA.
# 100m (n_head=10) and 800m (n_head=26) are NOT divisible by 4.
# Using kv_heads=2 for those models (valid divisor).
MODEL_CONFIGS = {
    "100m": {
        "n_layer": 20, "n_head": 10, "num_key_value_heads": 2,
        "n_embed": 640, "intermediate_size": 2048,
    },
    "200m": {
        "n_layer": 24, "n_head": 16, "num_key_value_heads": 4,
        "n_embed": 896, "intermediate_size": 2304,
    },
    "400m": {
        "n_layer": 24, "n_head": 20, "num_key_value_heads": 4,
        "n_embed": 1280, "intermediate_size": 3328,
    },
    "800m": {
        "n_layer": 28, "n_head": 26, "num_key_value_heads": 2,
        "n_embed": 1664, "intermediate_size": 4352,
    },
}

BATCH_SIZES_TO_TEST = [2, 4, 8, 16, 32, 64]
SEQ_LEN = 1024
VOCAB_SIZE = 256  # LanTokenizer vocab size (small)
NUM_WARMUP_STEPS = 3
NUM_TEST_STEPS = 10


# --------------------------------------------------------------------------- #
#  Benchmark function                                                          #
# --------------------------------------------------------------------------- #

@app.function(gpu="H200:1", timeout=60 * 60 * 1)
def benchmark_model(model_name: str, model_cfg: dict):
    """Test batch sizes on a single H200, return max that works."""
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"  Config: {model_cfg}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*60}\n")

    # Build model from Qwen3 template with custom dimensions
    hf_config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    hf_config.num_hidden_layers = model_cfg["n_layer"]
    hf_config.hidden_size = model_cfg["n_embed"]
    hf_config.num_attention_heads = model_cfg["n_head"]
    hf_config.num_key_value_heads = model_cfg["num_key_value_heads"]
    hf_config.intermediate_size = model_cfg["intermediate_size"]
    hf_config.vocab_size = VOCAB_SIZE

    model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)
    model = model.cuda().bfloat16()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,} ({n_params/1e6:.1f}M)")

    # Create a simple optimizer for realistic memory usage
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    max_working_bs = 0
    results = {}

    for bs in BATCH_SIZES_TO_TEST:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            model.train()
            # Warmup
            for _ in range(NUM_WARMUP_STEPS):
                x = torch.randint(0, VOCAB_SIZE, (bs, SEQ_LEN), device="cuda")
                output = model(input_ids=x, labels=x)
                output.loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Actual benchmark
            torch.cuda.synchronize()
            import time
            start = time.time()

            for _ in range(NUM_TEST_STEPS):
                x = torch.randint(0, VOCAB_SIZE, (bs, SEQ_LEN), device="cuda")
                output = model(input_ids=x, labels=x)
                output.loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            elapsed = time.time() - start

            peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
            tokens_per_sec = (bs * SEQ_LEN * NUM_TEST_STEPS) / elapsed
            step_time_ms = (elapsed / NUM_TEST_STEPS) * 1000

            max_working_bs = bs
            results[bs] = {
                "status": "OK",
                "peak_mem_gb": peak_mem_gb,
                "tokens_per_sec": tokens_per_sec,
                "step_time_ms": step_time_ms,
            }
            print(f"  bs={bs:3d}: OK  peak_mem={peak_mem_gb:.1f}GB  "
                  f"tok/s={tokens_per_sec:.0f}  step={step_time_ms:.0f}ms")

        except torch.cuda.OutOfMemoryError:
            peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
            results[bs] = {"status": "OOM", "peak_mem_gb": peak_mem_gb}
            print(f"  bs={bs:3d}: OOM  (peak_mem={peak_mem_gb:.1f}GB)")
            # Don't try larger batch sizes
            break
        except Exception as e:
            results[bs] = {"status": f"ERROR: {e}"}
            print(f"  bs={bs:3d}: ERROR: {e}")
            break
        finally:
            optimizer.zero_grad(set_to_none=True)
            del x
            torch.cuda.empty_cache()

    print(f"\n  >>> {model_name}: max_batch_size = {max_working_bs}")
    return {
        "model": model_name,
        "n_params": n_params,
        "max_batch_size": max_working_bs,
        "details": results,
    }


# --------------------------------------------------------------------------- #
#  Entrypoint                                                                  #
# --------------------------------------------------------------------------- #

@app.local_entrypoint()
def main(model_filter: str = ""):
    models_to_test = {
        k: v for k, v in MODEL_CONFIGS.items()
        if not model_filter or k == model_filter
    }

    print(f"Benchmarking {len(models_to_test)} model sizes on H200")
    print(f"Batch sizes to test: {BATCH_SIZES_TO_TEST}")
    print(f"Sequence length: {SEQ_LEN}")
    print()

    # Launch all benchmarks in parallel
    handles = []
    for name, cfg in models_to_test.items():
        handle = benchmark_model.spawn(name, cfg)
        handles.append((name, handle))

    # Collect results
    all_results = {}
    for name, handle in handles:
        result = handle.get()
        all_results[name] = result

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':>8s}  {'Params':>8s}  {'Max BS':>8s}  {'Recommended':>30s}")
    print("-" * 60)
    for name in sorted(all_results.keys()):
        r = all_results[name]
        max_bs = r["max_batch_size"]
        n_params = r["n_params"]
        # Recommend: use max_bs with grad_accum=1
        # Effective batch with 8 GPUs = max_bs * 8
        effective = max_bs * 8
        rec = f"bs={max_bs} ga=1 eff={effective}"
        print(f"{name:>8s}  {n_params/1e6:>6.1f}M  {max_bs:>8d}  {rec:>30s}")
    print("=" * 60)
