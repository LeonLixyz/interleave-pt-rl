"""
Reconstruct the policy-entropy-vs-RL-step curve for finished verl runs by
reloading saved checkpoints and measuring mean sampled-token -logprob at
temperature 1.0 (a Monte-Carlo estimate of policy entropy over the response
distribution) on a fixed prompt set.

Why: verl logged actor/entropy only to console (wandb off, mlflow empty), and
those buffers aged out. Checkpoints exist at stride 50, so the curve is
recoverable by direct measurement.

Usage:
  modal run entropy_scan.py --anchors from-armBsmall:1500,from-step20000:3000 --stride 150
"""
import json
from pathlib import Path

import modal

CHECKPOINT_VOLUME_NAME = "olmo-core-checkpoints-v2"
CACHE_VOLUME_NAME = "olmo-core-cache"
CHECKPOINT_MOUNT = "/checkpoints"
CACHE_MOUNT = "/cache"
REMOTE_VERL_DIR = "/root/verl-olmo3"
LOCAL = Path(__file__).parent
LOCAL_VERL_DIR = str(LOCAL.parent / "pretrain-rl-scaling" / "verl-olmo3")

# Exact mirror of the proven verl merge image (interleave_prep.verl_img / rl_eval._img)
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "git", "curl", "libibverbs-dev", "libibverbs1")
    .pip_install("wheel", "packaging", "ninja", "setuptools")
    .pip_install("torch==2.8.0")
    .pip_install("huggingface_hub==0.34.4", "hf_xet==1.1.5")
    .pip_install("vllm==0.11.0", "transformers==4.57.1", "flash-attn==2.8.3",
                 extra_options="--no-build-isolation")
    .pip_install("antlr4-python3-runtime==4.13.2")
    .pip_install("ray[default]==2.43.0", "tensordict==0.10.0", "datasets==4.0.0",
                 "pyarrow==17.0.0", "pandas==2.2.3", "wandb==0.19.11", "mlflow==3.0.0")
    .pip_install("omegaconf==2.4.0.dev3", "hydra-core==1.4.0.dev1", extra_options="--no-deps")
    .pip_install("importlib-resources", "packaging")
    .pip_install("codetiming==1.4.0", "accelerate==1.2.1", "peft==0.14.0",
                 "liger-kernel==0.5.4", "pybind11", "pylatexenc", "dill==0.3.8",
                 "torchdata==0.10.0", "tensorboard", "uvicorn", "fastapi")
    .pip_install("math-verify==0.5.2")
    .add_local_dir(LOCAL_VERL_DIR, remote_path=REMOTE_VERL_DIR, copy=True,
                   ignore=[".git", ".git/**", "__pycache__/**", "docs/**", "tests/**"])
    .run_commands(f"cd {REMOTE_VERL_DIR} && pip install -e . --no-deps")
    .env({"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"})
)

checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
app = modal.App("entropy-scan", image=image)


@app.function(gpu="H200:1", timeout=60 * 40,
              volumes={CHECKPOINT_MOUNT: checkpoint_volume, CACHE_MOUNT: cache_volume})
def entropy_at(anchor: str, rl_step: int, n_prompts: int = 48, n_samples: int = 4,
               max_tokens: int = 512) -> dict:
    import os, subprocess
    os.environ["HF_HOME"] = f"{CACHE_MOUNT}/hf"
    checkpoint_volume.reload()

    run = f"math-1b-rl-deepscaler-{anchor}"
    ckpt = f"{CHECKPOINT_MOUNT}/rl/{run}/checkpoints/global_step_{rl_step}/actor"
    merged = f"/tmp/m_{anchor}_{rl_step}"
    if not Path(ckpt).exists():
        return {"anchor": anchor, "rl_step": rl_step, "error": "ckpt missing"}

    r = subprocess.run(["python", f"{REMOTE_VERL_DIR}/scripts/legacy_model_merger.py",
                        "merge", "--backend", "fsdp", "--local_dir", ckpt,
                        "--target_dir", merged], capture_output=True, text=True, timeout=1500)
    if r.returncode != 0:
        return {"anchor": anchor, "rl_step": rl_step, "error": "merge failed",
                "stderr": r.stderr[-800:]}
    import shutil
    hf_src = Path(ckpt) / "huggingface"
    for f in hf_src.glob("*"):
        if f.is_file() and not (Path(merged) / f.name).exists():
            shutil.copy(f, Path(merged) / f.name)

    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    ds = load_dataset("parquet", data_files=f"{CHECKPOINT_MOUNT}/rl_data/skyeasy25k_omi2/test.parquet")["train"]
    prompts = []
    for i in range(min(n_prompts, len(ds))):
        msgs = ds[i]["prompt"]
        prompts.append([{"role": m["role"], "content": m["content"]} for m in msgs])

    llm = LLM(model=merged, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096)
    sp = SamplingParams(temperature=1.0, top_p=1.0, n=n_samples, max_tokens=max_tokens, logprobs=0)
    outs = llm.chat(prompts, sp)

    tot_nll, tot_tok = 0.0, 0
    for o in outs:
        for c in o.outputs:
            for lp in (c.logprobs or []):
                # logprobs=0 -> dict with the sampled token's Logprob entry
                v = list(lp.values())[0].logprob
                tot_nll += -v
                tot_tok += 1
    ent = tot_nll / max(tot_tok, 1)
    print(f"[entropy] {anchor} step {rl_step}: H={ent:.4f} nats over {tot_tok} tokens", flush=True)
    return {"anchor": anchor, "rl_step": rl_step, "entropy": round(ent, 4), "tokens": tot_tok}


@app.local_entrypoint()
def main(anchors: str = "from-armBsmall:1500,from-step20000:3000", stride: int = 150):
    jobs = []
    for spec in anchors.split(","):
        anchor, last = spec.split(":")
        steps = list(range(stride, int(last) + 1, stride))
        if int(last) not in steps:
            steps.append(int(last))
        for s in steps:
            jobs.append((anchor, s, entropy_at.spawn(anchor=anchor, rl_step=s)))
    results = []
    for anchor, s, h in jobs:
        try:
            results.append(h.get())
        except Exception as e:
            results.append({"anchor": anchor, "rl_step": s, "error": str(e)[:200]})
    Path("/tmp/entropy_scan.json").write_text(json.dumps(results, indent=1))
    ok = [r for r in results if "entropy" in r]
    print(f"done: {len(ok)}/{len(results)} points -> /tmp/entropy_scan.json")
