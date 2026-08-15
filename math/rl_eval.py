"""Post-hoc eval of every RL checkpoint on GSM8K + MATH-500.

Reads verl FSDP checkpoints from the volume, merges to HF safetensors, then
runs vLLM-based eval. Writes results to /evals_rl/{run}/step{S}.json.

Usage:
    modal run --detach rl_eval.py::eval_all
    modal run rl_eval.py::eval_one --anchor step95368 --rl-step 500
"""

from __future__ import annotations

import modal

from common import (
    CACHE_MOUNT, CACHE_VOLUME_NAME,
    CHECKPOINT_MOUNT, CHECKPOINT_VOLUME_NAME,
)
from math_answer_utils import extract_last_boxed

LOCAL_VERL_DIR = "/Users/leonli66/Desktop/Research/RL/Chess RL/pretrain-rl-scaling/verl-olmo3"
REMOTE_VERL_DIR = "/root/verl-olmo3"


def _img() -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
        )
        .apt_install("build-essential", "git", "curl", "libibverbs-dev", "libibverbs1")
        .pip_install("wheel", "packaging", "ninja", "setuptools")
        .pip_install("torch==2.8.0")
        .pip_install("huggingface_hub==0.34.4", "hf_xet==1.1.5")
        .pip_install(
            "vllm==0.11.0", "transformers==4.57.1", "flash-attn==2.8.3",
            extra_options="--no-build-isolation",
        )
        .pip_install("antlr4-python3-runtime==4.13.2")
        .pip_install(
            "ray[default]==2.43.0", "tensordict==0.10.0", "datasets==4.0.0",
            "pyarrow==17.0.0", "pandas==2.2.3", "wandb==0.19.11", "mlflow==3.0.0",
        )
        .pip_install("omegaconf==2.4.0.dev3", "hydra-core==1.4.0.dev1", extra_options="--no-deps")
        .pip_install("importlib-resources", "packaging")
        .pip_install(
            "codetiming==1.4.0", "accelerate==1.2.1", "peft==0.14.0",
            "liger-kernel==0.5.4", "pybind11", "pylatexenc", "dill==0.3.8",
            "torchdata==0.10.0", "tensorboard", "uvicorn", "fastapi",
        )
        .pip_install("math-verify==0.5.2")
        .add_local_dir(
            LOCAL_VERL_DIR, remote_path=REMOTE_VERL_DIR, copy=True,
            ignore=[".git", ".git/**", "__pycache__/**", "docs/**", "tests/**"],
        )
        .run_commands(f"cd {REMOTE_VERL_DIR} && pip install -e . --no-deps")
        .env({"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"})
        .add_local_python_source("common", "math_answer_utils")
    )


app = modal.App("math-1b-rl-eval", image=_img())
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2)
hf_secret = modal.Secret.from_name("huggingface-secret")


SYSTEM_PROMPT = (
    "You are a helpful assistant. When answering math problems, first think step "
    "by step inside <think>...</think> tags, then give your final answer in "
    "\\boxed{...}."
)


@app.function(
    gpu="H200:1",
    timeout=60 * 60 * 3,
    volumes={CACHE_MOUNT: cache_volume, CHECKPOINT_MOUNT: checkpoint_volume},
    secrets=[hf_secret],
    cpu=8.0, memory=100 * 1024,
)
def eval_rl_ckpt(anchor: str, rl_step: int, n_samples: int = 4) -> dict:
    """Merge one FSDP RL ckpt to HF then eval on GSM8K + MATH-500."""
    import json, os, re, shutil, subprocess
    from pathlib import Path

    checkpoint_volume.reload()
    os.environ["HF_HOME"] = f"{CACHE_MOUNT}/hf"

    run = f"math-1b-rl-deepscaler-from-{anchor}"
    ckpt_dir = f"{CHECKPOINT_MOUNT}/rl/{run}/checkpoints/global_step_{rl_step}/actor"
    merged_dir = f"/tmp/merged_{anchor}_{rl_step}"

    if not Path(ckpt_dir).exists():
        return {"anchor": anchor, "rl_step": rl_step, "error": "ckpt not found"}

    # Merge FSDP shards to HF safetensors
    print(f"[merge] {ckpt_dir} -> {merged_dir}")
    r = subprocess.run(
        ["python", f"{REMOTE_VERL_DIR}/scripts/legacy_model_merger.py", "merge",
         "--backend", "fsdp", "--local_dir", ckpt_dir, "--target_dir", merged_dir],
        capture_output=True, text=True, timeout=1200,
    )
    if r.returncode != 0:
        return {"anchor": anchor, "rl_step": rl_step,
                "error": "merge failed", "stderr": r.stderr[-2000:]}

    # Copy tokenizer + config from the huggingface subfolder
    hf_src = f"{ckpt_dir}/huggingface"
    for f in Path(hf_src).glob("*"):
        if f.is_file() and not (Path(merged_dir) / f.name).exists():
            shutil.copy(f, Path(merged_dir) / f.name)

    # ---- eval ----
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig

    llm = LLM(model=merged_dir, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=0.85, max_model_len=4096)
    tok = llm.get_tokenizer()
    stop_ids = []
    try:
        gc = json.loads((Path(merged_dir) / "generation_config.json").read_text())
        stop_ids = gc.get("eos_token_id") or []
        if isinstance(stop_ids, int): stop_ids = [stop_ids]
    except Exception: pass

    sp = SamplingParams(temperature=0.7, max_tokens=3584, n=n_samples,
                        seed=0, stop_token_ids=stop_ids or None)

    def _mk(q):
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q}]

    def _grade(pred, gold):
        after = pred.split("</think>")[-1] if "</think>" in pred else pred
        boxed = extract_last_boxed(after)
        ans = boxed if boxed is not None else after.strip().splitlines()[-1] if after.strip() else ""
        try:
            gp = parse(f"${ans}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
            gg = parse(f"${gold}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
            return bool(verify(gg, gp))
        except Exception:
            return ans.strip() == str(gold).strip()

    def _score(samples, golds):
        total = sum(len(s) for s in samples) or 1
        correct = sum(1 for ss, g in zip(samples, golds) for s in ss if _grade(s, g))
        return {
            "n_prompts": len(samples), "n_samples": n_samples,
            "pass_at_1_avg": correct / total,
            "pass_at_n": sum(1 for ss, g in zip(samples, golds) if any(_grade(s, g) for s in ss)) / max(1, len(samples)),
        }

    result = {"anchor": anchor, "rl_step": rl_step, "per_dataset": {}}

    # GSM8K
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    prompts = [_mk(r["question"]) for r in gsm]
    golds = [r["answer"].split("####")[-1].strip() for r in gsm]
    outs = llm.chat(prompts, sp, add_generation_prompt=True)
    samples = [[o.text for o in r.outputs] for r in outs]
    result["per_dataset"]["gsm8k"] = _score(samples, golds)
    print(f"[gsm8k] {result['per_dataset']['gsm8k']}")

    # MATH-500
    m500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompts = [_mk(r["problem"]) for r in m500]
    golds = [str(r["answer"]) for r in m500]
    outs = llm.chat(prompts, sp, add_generation_prompt=True)
    samples = [[o.text for o in r.outputs] for r in outs]
    result["per_dataset"]["math500"] = _score(samples, golds)
    print(f"[math500] {result['per_dataset']['math500']}")

    # Save — merge into existing JSON so we don't clobber a sky/other block
    out_dir = Path(f"{CHECKPOINT_MOUNT}/evals_rl/{run}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"step{rl_step}.json"
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            merged = prev.get("per_dataset", {})
            merged.update(result["per_dataset"])
            result["per_dataset"] = merged
        except Exception:
            pass
    out_path.write_text(json.dumps(result, indent=2))
    checkpoint_volume.commit()
    shutil.rmtree(merged_dir, ignore_errors=True)
    return result


@app.function(
    gpu="H200:1",
    timeout=60 * 60 * 3,
    volumes={CACHE_MOUNT: cache_volume, CHECKPOINT_MOUNT: checkpoint_volume},
    secrets=[hf_secret],
    cpu=8.0, memory=100 * 1024,
)
def eval_rl_ckpt_sky(anchor: str, rl_step: int, n_samples: int = 4) -> dict:
    """Skyeasy-only post-hoc eval for an RL ckpt. Merges FSDP → HF, runs
    500 skyeasy prompts, appends 'skyeasy25k_eval' block to existing
    evals_rl/{run}/step{S}.json (or creates it if missing).
    """
    import json, os, re, shutil, subprocess
    from pathlib import Path

    checkpoint_volume.reload()
    os.environ["HF_HOME"] = f"{CACHE_MOUNT}/hf"

    run = f"math-1b-rl-deepscaler-from-{anchor}"
    ckpt_dir = f"{CHECKPOINT_MOUNT}/rl/{run}/checkpoints/global_step_{rl_step}/actor"
    merged_dir = f"/tmp/merged_{anchor}_{rl_step}_sky"
    out_path = Path(f"{CHECKPOINT_MOUNT}/evals_rl/{run}/step{rl_step}.json")

    if not Path(ckpt_dir).exists():
        return {"anchor": anchor, "rl_step": rl_step, "error": "ckpt not found"}

    # Merge FSDP → HF
    print(f"[merge] {ckpt_dir}")
    r = subprocess.run(
        ["python", f"{REMOTE_VERL_DIR}/scripts/legacy_model_merger.py", "merge",
         "--backend", "fsdp", "--local_dir", ckpt_dir, "--target_dir", merged_dir],
        capture_output=True, text=True, timeout=1200,
    )
    if r.returncode != 0:
        return {"anchor": anchor, "rl_step": rl_step,
                "error": "merge failed", "stderr": r.stderr[-2000:]}
    hf_src = f"{ckpt_dir}/huggingface"
    for f in Path(hf_src).glob("*"):
        if f.is_file() and not (Path(merged_dir) / f.name).exists():
            shutil.copy(f, Path(merged_dir) / f.name)

    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig

    llm = LLM(model=merged_dir, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=0.85, max_model_len=4096)
    try:
        gc = json.loads((Path(merged_dir) / "generation_config.json").read_text())
        stop_ids = gc.get("eos_token_id") or []
        if isinstance(stop_ids, int): stop_ids = [stop_ids]
    except Exception:
        stop_ids = []
    sp = SamplingParams(temperature=0.7, max_tokens=3584, n=n_samples,
                        seed=0, stop_token_ids=stop_ids or None)

    def _mk(q):
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q}]

    def _grade(pred, gold):
        after = pred.split("</think>")[-1] if "</think>" in pred else pred
        boxed = extract_last_boxed(after)
        ans = boxed if boxed is not None else after.strip().splitlines()[-1] if after.strip() else ""
        try:
            gp = parse(f"${ans}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
            gg = parse(f"${gold}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
            return bool(verify(gg, gp))
        except Exception:
            return ans.strip() == str(gold).strip()

    # Load skyeasy val split
    sky = load_dataset("pre-to-post-olmo/rl-math-skyeasy25k-omi2")
    split = "test" if "test" in sky else list(sky.keys())[0]
    sky = sky[split]

    def _q(row):
        p = row.get("prompt")
        if isinstance(p, list):
            p = " ".join(str(m.get("content", "")) for m in p if isinstance(m, dict))
        return str(p)

    def _gold(row):
        rm = row.get("reward_model")
        if isinstance(rm, dict):
            for k in ("ground_truth", "answer", "gold"):
                if k in rm: return str(rm[k])
        return str(row.get("answer") or row.get("solution") or "")

    prompts = [_mk(_q(r)) for r in sky]
    golds = [_gold(r) for r in sky]
    print(f"[skyeasy] {len(prompts)} prompts")
    outs = llm.chat(prompts, sp, add_generation_prompt=True)
    samples = [[o.text for o in r.outputs] for r in outs]
    total = sum(len(s) for s in samples) or 1
    correct = sum(1 for ss, g in zip(samples, golds) for s in ss if _grade(s, g))
    pass_at_1 = correct / total
    pass_at_n = sum(1 for ss, g in zip(samples, golds) if any(_grade(s, g) for s in ss)) / max(1, len(samples))
    print(f"[skyeasy] pass@1={pass_at_1:.4f}")

    # Merge into existing results.json (or create fresh)
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    else:
        existing = {"anchor": anchor, "rl_step": rl_step, "per_dataset": {}}
    existing.setdefault("per_dataset", {})["skyeasy25k_eval"] = {
        "n_prompts": len(samples), "n_samples": n_samples,
        "pass_at_1_avg": pass_at_1, "pass_at_n": pass_at_n,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(existing, indent=2))
    checkpoint_volume.commit()
    shutil.rmtree(merged_dir, ignore_errors=True)
    return existing


@app.function(
    gpu="H200:1",
    timeout=60 * 60 * 5,
    volumes={CACHE_MOUNT: cache_volume, CHECKPOINT_MOUNT: checkpoint_volume},
    secrets=[hf_secret],
    cpu=8.0, memory=100 * 1024,
)
def eval_rl_ckpt_p16(anchor: str, rl_step: int) -> dict:
    """Same as eval_rl_ckpt but with n=16 samples so we record pass@1/4/8/16."""
    import json, os, re, shutil, subprocess
    from pathlib import Path

    checkpoint_volume.reload()
    os.environ["HF_HOME"] = f"{CACHE_MOUNT}/hf"

    run = f"math-1b-rl-deepscaler-from-{anchor}"
    ckpt_dir = f"{CHECKPOINT_MOUNT}/rl/{run}/checkpoints/global_step_{rl_step}/actor"
    merged_dir = f"/tmp/merged_p16_{anchor}_{rl_step}"
    out_path = Path(f"{CHECKPOINT_MOUNT}/evals_rl_p16/{run}/step{rl_step}.json")

    if not Path(ckpt_dir).exists():
        return {"anchor": anchor, "rl_step": rl_step, "error": "ckpt not found"}

    r = subprocess.run(
        ["python", f"{REMOTE_VERL_DIR}/scripts/legacy_model_merger.py", "merge",
         "--backend", "fsdp", "--local_dir", ckpt_dir, "--target_dir", merged_dir],
        capture_output=True, text=True, timeout=1200,
    )
    if r.returncode != 0:
        return {"anchor": anchor, "rl_step": rl_step,
                "error": "merge failed", "stderr": r.stderr[-2000:]}
    hf_src = f"{ckpt_dir}/huggingface"
    for f in Path(hf_src).glob("*"):
        if f.is_file() and not (Path(merged_dir) / f.name).exists():
            shutil.copy(f, Path(merged_dir) / f.name)

    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig

    llm = LLM(model=merged_dir, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=0.85, max_model_len=4096)
    try:
        gc = json.loads((Path(merged_dir) / "generation_config.json").read_text())
        stop_ids = gc.get("eos_token_id") or []
        if isinstance(stop_ids, int): stop_ids = [stop_ids]
    except Exception:
        stop_ids = []
    sp = SamplingParams(temperature=0.7, max_tokens=3584, n=16, seed=0,
                        stop_token_ids=stop_ids or None)

    def _mk(q):
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q}]

    def _grade(pred, gold):
        after = pred.split("</think>")[-1] if "</think>" in pred else pred
        boxed = extract_last_boxed(after)
        ans = boxed if boxed is not None else after.strip().splitlines()[-1] if after.strip() else ""
        try:
            gp = parse(f"${ans}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
            gg = parse(f"${gold}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
            return bool(verify(gg, gp))
        except Exception:
            return ans.strip() == str(gold).strip()

    def _score(samples, golds):
        # samples[i] is a list of 16 strings
        import numpy as np
        N = 16
        # per-prompt boolean matrix (n_prompts x N)
        correct_mat = [[_grade(s, g) for s in ss] for ss, g in zip(samples, golds)]
        n_prompts = len(correct_mat)
        # pass@k using unbiased Chen et al. estimator:
        # pass@k = 1 - C(N-c, k) / C(N, k)  where c = # correct out of N
        from math import comb
        def pak(k):
            total = 0.0
            for row in correct_mat:
                c = sum(row); n = len(row)
                if n - c < k: total += 1.0
                else: total += 1.0 - comb(n - c, k) / comb(n, k)
            return total / max(1, n_prompts)
        return {
            "n_prompts": n_prompts, "n_samples": N,
            "pass_at_1_avg": sum(sum(r) for r in correct_mat) / max(1, n_prompts * N),
            "pass_at_4": pak(4),
            "pass_at_8": pak(8),
            "pass_at_16": pak(16),
        }

    result = {"anchor": anchor, "rl_step": rl_step, "per_dataset": {}}

    # GSM8K
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    prompts = [_mk(r["question"]) for r in gsm]
    golds = [r["answer"].split("####")[-1].strip() for r in gsm]
    outs = llm.chat(prompts, sp, add_generation_prompt=True)
    samples = [[o.text for o in r.outputs] for r in outs]
    result["per_dataset"]["gsm8k"] = _score(samples, golds)
    print(f"[gsm8k p16] {result['per_dataset']['gsm8k']}")

    # MATH-500
    m500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompts = [_mk(r["problem"]) for r in m500]
    golds = [str(r["answer"]) for r in m500]
    outs = llm.chat(prompts, sp, add_generation_prompt=True)
    samples = [[o.text for o in r.outputs] for r in outs]
    result["per_dataset"]["math500"] = _score(samples, golds)
    print(f"[math500 p16] {result['per_dataset']['math500']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    checkpoint_volume.commit()
    shutil.rmtree(merged_dir, ignore_errors=True)
    return result


@app.function(
    gpu="H200:1",
    timeout=60 * 60 * 5,
    volumes={CACHE_MOUNT: cache_volume, CHECKPOINT_MOUNT: checkpoint_volume},
    secrets=[hf_secret],
    cpu=8.0, memory=100 * 1024,
)
def eval_rl_ckpt_sky_p16(anchor: str, rl_step: int) -> dict:
    """Skyeasy-only pass@16 (n=16). Merges 'skyeasy25k_eval' block into the
    existing evals_rl_p16/{run}/step{S}.json (created by eval_rl_ckpt_p16)."""
    import json, os, re, shutil, subprocess
    from pathlib import Path
    from math import comb

    checkpoint_volume.reload()
    os.environ["HF_HOME"] = f"{CACHE_MOUNT}/hf"

    run = f"math-1b-rl-deepscaler-from-{anchor}"
    ckpt_dir = f"{CHECKPOINT_MOUNT}/rl/{run}/checkpoints/global_step_{rl_step}/actor"
    merged_dir = f"/tmp/merged_skyp16_{anchor}_{rl_step}"
    out_path = Path(f"{CHECKPOINT_MOUNT}/evals_rl_p16/{run}/step{rl_step}.json")

    if not Path(ckpt_dir).exists():
        return {"anchor": anchor, "rl_step": rl_step, "error": "ckpt not found"}

    r = subprocess.run(
        ["python", f"{REMOTE_VERL_DIR}/scripts/legacy_model_merger.py", "merge",
         "--backend", "fsdp", "--local_dir", ckpt_dir, "--target_dir", merged_dir],
        capture_output=True, text=True, timeout=1200,
    )
    if r.returncode != 0:
        return {"anchor": anchor, "rl_step": rl_step,
                "error": "merge failed", "stderr": r.stderr[-2000:]}
    hf_src = f"{ckpt_dir}/huggingface"
    for f in Path(hf_src).glob("*"):
        if f.is_file() and not (Path(merged_dir) / f.name).exists():
            shutil.copy(f, Path(merged_dir) / f.name)

    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig

    llm = LLM(model=merged_dir, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=0.85, max_model_len=4096)
    try:
        gc = json.loads((Path(merged_dir) / "generation_config.json").read_text())
        stop_ids = gc.get("eos_token_id") or []
        if isinstance(stop_ids, int): stop_ids = [stop_ids]
    except Exception:
        stop_ids = []
    sp = SamplingParams(temperature=0.7, max_tokens=3584, n=16, seed=0,
                        stop_token_ids=stop_ids or None)

    def _mk(q):
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q}]

    def _grade(pred, gold):
        after = pred.split("</think>")[-1] if "</think>" in pred else pred
        boxed = extract_last_boxed(after)
        ans = boxed if boxed is not None else after.strip().splitlines()[-1] if after.strip() else ""
        try:
            gp = parse(f"${ans}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
            gg = parse(f"${gold}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
            return bool(verify(gg, gp))
        except Exception:
            return ans.strip() == str(gold).strip()

    sky = load_dataset("pre-to-post-olmo/rl-math-skyeasy25k-omi2")
    split = "test" if "test" in sky else list(sky.keys())[0]
    sky = sky[split]

    def _q(row):
        p = row.get("prompt")
        if isinstance(p, list):
            p = " ".join(str(m.get("content", "")) for m in p if isinstance(m, dict))
        return str(p)

    def _gold(row):
        rm = row.get("reward_model")
        if isinstance(rm, dict):
            for k in ("ground_truth", "answer", "gold"):
                if k in rm: return str(rm[k])
        return str(row.get("answer") or row.get("solution") or "")

    prompts = [_mk(_q(r)) for r in sky]
    golds = [_gold(r) for r in sky]
    outs = llm.chat(prompts, sp, add_generation_prompt=True)
    samples = [[o.text for o in rr.outputs] for rr in outs]
    correct_mat = [[_grade(s, g) for s in ss] for ss, g in zip(samples, golds)]
    n_prompts = len(correct_mat)
    def pak(k):
        total = 0.0
        for row in correct_mat:
            c = sum(row); n = len(row)
            total += 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)
        return total / max(1, n_prompts)
    block = {
        "n_prompts": n_prompts, "n_samples": 16,
        "pass_at_1_avg": sum(sum(r) for r in correct_mat) / max(1, n_prompts * 16),
        "pass_at_4": pak(4), "pass_at_8": pak(8), "pass_at_16": pak(16),
    }
    print(f"[skyeasy p16] {block}")

    if out_path.exists():
        existing = json.loads(out_path.read_text())
    else:
        existing = {"anchor": anchor, "rl_step": rl_step, "per_dataset": {}}
    existing.setdefault("per_dataset", {})["skyeasy25k_eval"] = block
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(existing, indent=2))
    checkpoint_volume.commit()
    shutil.rmtree(merged_dir, ignore_errors=True)
    return existing


@app.function(
    timeout=60 * 60 * 2,
    volumes={CACHE_MOUNT: cache_volume, CHECKPOINT_MOUNT: checkpoint_volume},
    secrets=[hf_secret],
    cpu=8.0, memory=64 * 1024,
)
def rollout_lengths(anchor: str, stride: int = 50) -> list:
    """Rollout response-length stats (in tokens) for one anchor at every {stride}
    training steps, from rl/{run}/rollouts/training/*.jsonl using the model tokenizer."""
    import json, glob, os, statistics
    from pathlib import Path

    checkpoint_volume.reload()
    run = f"math-1b-rl-deepscaler-from-{anchor}"
    roll_dir = f"{CHECKPOINT_MOUNT}/rl/{run}/rollouts/training"
    if not os.path.isdir(roll_dir):
        return []

    # tokenizer from any available ckpt huggingface dir
    from transformers import AutoTokenizer
    ck_glob = sorted(glob.glob(f"{CHECKPOINT_MOUNT}/rl/{run}/checkpoints/global_step_*/actor/huggingface"))
    tok = AutoTokenizer.from_pretrained(ck_glob[-1], trust_remote_code=True)
    MAXLEN = 3584  # max_response_length used in RL

    out = []
    _all = [p for p in glob.glob(f"{roll_dir}/*.jsonl")
            if os.path.basename(p)[:-6].isdigit() and int(os.path.basename(p)[:-6]) % stride == 0]
    print(f"[{anchor}] {len(_all)} rollout files (stride={stride})", flush=True)
    files = sorted(_all,
                   key=lambda p: int(os.path.basename(p)[:-6]) if os.path.basename(p)[:-6].isdigit() else 0)
    try:
        from tqdm import tqdm
        files = tqdm(files, desc=anchor, mininterval=15, miniters=50)
    except Exception:
        pass
    for fp in files:
        step = os.path.basename(fp)[:-6]
        if not step.isdigit():
            continue
        step = int(step)
        outputs = []
        with open(fp) as f:
            for line in f:
                try:
                    outputs.append(json.loads(line).get("output", "") or "")
                except Exception:
                    pass
        if not outputs:
            continue
        # batch-tokenize (fast tokenizer), count ids per output
        enc = tok(outputs, add_special_tokens=False)["input_ids"]
        lens = [len(x) for x in enc]
        lens.sort()
        n = len(lens)
        p95 = lens[min(n - 1, int(0.95 * n))]
        trunc = sum(1 for x in lens if x >= MAXLEN) / n
        out.append({
            "rl_step": step,
            "n_rollouts": n,
            "resp_len_mean": round(sum(lens) / n, 1),
            "resp_len_median": lens[n // 2],
            "resp_len_p95": p95,
            "resp_len_max": lens[-1],
            "trunc_rate": round(trunc, 4),
        })
    return out


@app.function(
    timeout=60 * 60 * 2,
    volumes={CACHE_MOUNT: cache_volume, CHECKPOINT_MOUNT: checkpoint_volume},
    secrets=[hf_secret],
    cpu=16.0, memory=64 * 1024,
)
def sft_token_count(cutoff: int = 4096) -> dict:
    """Exact SFT training-token count: tokenize AI-MO/NuminaMath-CoT with the
    model's chat template, drop examples > cutoff, sum total + assistant tokens."""
    import os
    os.environ["HF_HOME"] = f"{CACHE_MOUNT}/hf"
    os.environ["HF_DATASETS_CACHE"] = f"{CACHE_MOUNT}/hf/datasets"
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        f"{CHECKPOINT_MOUNT}/sft/math-1b-sft-numinamath-bs512-from-step95368",
        trust_remote_code=True)
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")

    def msgs(ex):
        m = ex.get("messages")
        if m: return m
        return [{"role": "user", "content": ex.get("problem", "")},
                {"role": "assistant", "content": ex.get("solution", "")}]

    total_tok = 0; asst_tok = 0; kept = 0; dropped = 0
    B = 2000
    buf = []
    def flush(batch):
        nonlocal total_tok, asst_tok, kept, dropped
        full = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in batch]
        enc = tok(full, add_special_tokens=False)["input_ids"]
        # assistant-only: template just the user turn(s), subtract
        for m, ids in zip(batch, enc):
            L = len(ids)
            if L > cutoff:
                dropped += 1; continue
            kept += 1; total_tok += L
            try:
                uonly = tok.apply_chat_template([x for x in m if x["role"] != "assistant"],
                                                tokenize=False, add_generation_prompt=True)
                asst_tok += max(0, L - len(tok(uonly, add_special_tokens=False)["input_ids"]))
            except Exception:
                pass
    for ex in ds:
        buf.append(msgs(ex))
        if len(buf) >= B:
            flush(buf); buf = []
    if buf: flush(buf)

    return {"n_examples_total": len(ds), "kept": kept, "dropped": dropped,
            "cutoff": cutoff, "total_tokens": total_tok, "assistant_tokens": asst_tok}


@app.local_entrypoint()
def sft_tokens() -> None:
    import json
    r = sft_token_count.remote()
    print(json.dumps(r, indent=2))
    print(f"\nSFT total training tokens: {r['total_tokens']/1e9:.3f} B  "
          f"({r['total_tokens']:,})")
    print(f"SFT assistant (loss) tokens: {r['assistant_tokens']/1e9:.3f} B  "
          f"({r['assistant_tokens']:,})")
    print(f"kept {r['kept']:,} / {r['n_examples_total']:,}  (dropped {r['dropped']:,} > cutoff)")


@app.local_entrypoint()
def rollout_lengths_all() -> None:
    """Fan out rollout_lengths across all 15 anchors, write a combined CSV."""
    import csv
    anchors = ["step5000","step10000","step15000","step20000","step25000",
               "step30000","step35000","step40000","step45000","step50000",
               "step60000","step70000","step80000","step90000","step95368"]
    futures = [(a, rollout_lengths.spawn(anchor=a)) for a in anchors]
    rows = []
    for a, fut in futures:
        try:
            recs = fut.get()
            pstep = int(a.replace("step", ""))
            for r in recs:
                rows.append({"pretrain_step": pstep, **r})
            print(f"  {a}: {len(recs)} steps")
        except Exception as e:
            print(f"  ✗ {a}: {e}")
    rows.sort(key=lambda r: (r["pretrain_step"], r["rl_step"]))
    cols = ["pretrain_step","rl_step","n_rollouts","resp_len_mean",
            "resp_len_median","resp_len_p95","resp_len_max","trunc_rate"]
    out = "/tmp/rl_rollout_lengths.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


@app.local_entrypoint()
def eval_sky_p16_fill(list_path: str = "/tmp/sky_p16_missing.txt") -> None:
    """Fire skyeasy pass@16 for (anchor,step) pairs listed as 'stepN,S'."""
    pairs = []
    with open(list_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            a, s = line.split(",")
            pairs.append((a, int(s)))
    print(f"firing {len(pairs)} skyeasy pass@16 evals from {list_path}")
    futures = [(a, s, eval_rl_ckpt_sky_p16.spawn(anchor=a, rl_step=s)) for a, s in pairs]
    ok = 0
    for a, s, fut in futures:
        try:
            r = fut.get()
            if r.get("per_dataset", {}).get("skyeasy25k_eval", {}).get("pass_at_16") is not None:
                ok += 1
        except Exception as e:
            print(f"  ✗ {a} step{s}: {e}")
    print(f"filled {ok}/{len(pairs)}")


@app.local_entrypoint()
def eval_p16_all(stride: int = 250, max_step: int = 5000) -> None:
    """Fire pass@16 (n=16) sweep every {stride} RL steps across all 15 anchors."""
    anchors = ["step5000","step10000","step15000","step20000","step25000",
               "step30000","step35000","step40000","step45000","step50000",
               "step60000","step70000","step80000","step90000","step95368"]
    jobs = []
    for a in anchors:
        for s in range(stride, max_step + 1, stride):
            jobs.append((a, s))
    print(f"firing {len(jobs)} pass@16 evals")
    futures = [(a, s, eval_rl_ckpt_p16.spawn(anchor=a, rl_step=s)) for a, s in jobs]
    for a, s, f in futures:
        try:
            r = f.get()
            g = r.get("per_dataset", {}).get("gsm8k", {})
            print(f"  {a} step{s}: p1={g.get('pass_at_1_avg')} p16={g.get('pass_at_16')}")
        except Exception as e:
            print(f"  ✗ {a} step{s}: {e}")


@app.local_entrypoint()
def eval_one_sky(anchor: str = "step95368", rl_step: int = 500) -> None:
    r = eval_rl_ckpt_sky.remote(anchor=anchor, rl_step=rl_step)
    import json; print(json.dumps(r, indent=2))


@app.local_entrypoint()
def eval_main_fill(list_path: str = "/tmp/gsm_missing.txt") -> None:
    """Fire main (GSM8K+MATH-500) eval for (anchor,step) pairs listed as 'stepN,S'."""
    pairs = []
    with open(list_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            a, s = line.split(",")
            pairs.append((a, int(s)))
    print(f"firing {len(pairs)} main GSM/MATH fill evals from {list_path}")
    futures = [(a, s, eval_rl_ckpt.spawn(anchor=a, rl_step=s)) for a, s in pairs]
    ok = 0
    for a, s, fut in futures:
        try:
            r = fut.get()
            if r.get("per_dataset", {}).get("gsm8k", {}).get("pass_at_1_avg") is not None:
                ok += 1
        except Exception as e:
            print(f"  ✗ {a} step{s}: {e}")
    print(f"filled {ok}/{len(pairs)}")


@app.local_entrypoint()
def eval_p16_fill(list_path: str = "/tmp/p16_missing.txt") -> None:
    """Fire pass@16 eval only for (anchor,step) pairs listed one-per-line as 'stepN,S'."""
    pairs = []
    with open(list_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            a, s = line.split(",")
            pairs.append((a, int(s)))
    print(f"firing {len(pairs)} pass@16 fill evals from {list_path}")
    futures = [(a, s, eval_rl_ckpt_p16.spawn(anchor=a, rl_step=s)) for a, s in pairs]
    ok = 0
    for a, s, fut in futures:
        try:
            r = fut.get()
            g = r.get("per_dataset", {}).get("gsm8k", {}).get("pass_at_16")
            if g is not None:
                ok += 1
        except Exception as e:
            print(f"  ✗ {a} step{s}: {e}")
    print(f"filled {ok}/{len(pairs)}")


@app.local_entrypoint()
def eval_sky_fill(list_path: str = "/tmp/sky_missing.txt") -> None:
    """Fire skyeasy eval only for (anchor,step) pairs listed one-per-line as 'stepN,S'."""
    pairs = []
    with open(list_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            a, s = line.split(",")
            pairs.append((a, int(s)))
    print(f"firing {len(pairs)} skyeasy fill evals from {list_path}")
    futures = [(a, s, eval_rl_ckpt_sky.spawn(anchor=a, rl_step=s)) for a, s in pairs]
    ok = 0
    for a, s, fut in futures:
        try:
            r = fut.get()
            sky = r.get("per_dataset", {}).get("skyeasy25k_eval", {}).get("pass_at_1_avg")
            if sky is not None:
                ok += 1
        except Exception as e:
            print(f"  ✗ {a} step{s}: {e}")
    print(f"filled {ok}/{len(pairs)}")


@app.local_entrypoint()
def eval_sky_all(stride: int = 250, max_step: int = 5000) -> None:
    """Fire skyeasy-only eval on every {stride}th RL ckpt across all anchors.
    Sampled at 250 rather than 50 to cap cost (skyeasy rarely changes fast).
    """
    anchors = ["step5000","step10000","step15000","step20000","step25000",
               "step30000","step35000","step40000","step45000","step50000",
               "step60000","step70000","step80000","step90000","step95368"]
    jobs = []
    for a in anchors:
        for s in range(stride, max_step + 1, stride):
            jobs.append((a, s))
    print(f"firing {len(jobs)} skyeasy RL evals ({len(anchors)} anchors × every-{stride}-steps)")
    futures = [(a, s, eval_rl_ckpt_sky.spawn(anchor=a, rl_step=s)) for a, s in jobs]
    for a, s, f in futures:
        try:
            r = f.get()
            sky = r.get("per_dataset", {}).get("skyeasy25k_eval", {}).get("pass_at_1_avg")
            print(f"  {a} step{s}: sky={sky}")
        except Exception as e:
            print(f"  ✗ {a} step{s}: {e}")


@app.local_entrypoint()
def eval_one(anchor: str = "step95368", rl_step: int = 500) -> None:
    r = eval_rl_ckpt.remote(anchor=anchor, rl_step=rl_step)
    import json; print(json.dumps(r, indent=2))


@app.local_entrypoint()
def eval_all(stride: int = 200, max_step: int = 5000) -> None:
    """Fire post-hoc eval on every {stride}th RL checkpoint across 4 anchors."""
    anchors = ["step10000", "step40000", "step80000", "step95368"]
    jobs = []
    for a in anchors:
        for s in range(stride, max_step + 1, stride):
            jobs.append((a, s))
    print(f"firing {len(jobs)} RL ckpt evals ({len(anchors)} anchors × every-{stride}-steps)")
    futures = [(a, s, eval_rl_ckpt.spawn(anchor=a, rl_step=s)) for a, s in jobs]
    for a, s, f in futures:
        try:
            r = f.get()
            g = r.get("per_dataset", {}).get("gsm8k", {}).get("pass_at_1_avg")
            m = r.get("per_dataset", {}).get("math500", {}).get("pass_at_1_avg")
            print(f"  {a} step{s}: gsm={g} m500={m}")
        except Exception as e:
            print(f"  ✗ {a} step{s}: {e}")
