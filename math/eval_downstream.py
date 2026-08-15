"""Downstream eval on GSM8K + pre-to-post-olmo/rl-math-skyeasy25k-omi2 for SFT models.

Uses vLLM for batched generation, math_verify for grading.
Pass@1 with greedy decoding.

Usage:
    # Single model
    modal run eval_downstream.py::eval_single \
        --hf-repo pre-to-post-olmo/math-1b-sft-openthoughts-from-step95368

    # All landed SFT models in parallel
    modal run eval_downstream.py::eval_sweep
"""

from __future__ import annotations

import modal

from common import (
    CACHE_MOUNT,
    CACHE_VOLUME_NAME,
    CHECKPOINT_MOUNT,
    CHECKPOINT_VOLUME_NAME,
    hf_image_base,
)
from math_answer_utils import extract_last_boxed


def _img() -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
        )
        .apt_install("build-essential", "git", "curl")
        .pip_install("wheel", "packaging", "ninja", "setuptools")
        .pip_install("torch==2.6.0")
        .pip_install("torch==2.7.0")
        .pip_install(
            "vllm==0.9.2",
            "transformers==4.52.4",
            "datasets>=3.0",
            "math-verify>=0.5",
            "huggingface_hub>=0.26",
            "pandas>=2.2",
        )
        .env({"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"})
        .add_local_python_source("common", "math_answer_utils")
    )


app = modal.App("math-1b-eval-downstream", image=_img())

cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2
)
hf_secret = modal.Secret.from_name("huggingface-secret")

SYSTEM_PROMPT = (
    "You are a helpful assistant. When answering math problems, first think step "
    "by step inside <think>...</think> tags, then give your final answer in "
    "\\boxed{...}."
)


@app.function(
    gpu="H200:1",
    timeout=60 * 60 * 6,
    volumes={
        CACHE_MOUNT: cache_volume,
        CHECKPOINT_MOUNT: checkpoint_volume,
    },
    secrets=[hf_secret],
)
def eval_model(
    hf_repo: str,
    max_new_tokens: int = 8192,
    temperature: float = 0.7,
    n_samples: int = 8,
    gsm8k_limit: int | None = None,
    skyeasy_limit: int | None = None,
    repetition_penalty: float = 1.0,
) -> dict:
    """Eval one HF model on GSM8K + rl-math-skyeasy25k-omi2 eval split."""
    import json
    import os
    import re
    from pathlib import Path

    os.environ["HF_HOME"] = f"{CACHE_MOUNT}/hf"
    os.environ["HF_DATASETS_CACHE"] = f"{CACHE_MOUNT}/hf/datasets"
    os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_MOUNT}/hf/transformers"
    for d in (f"{CACHE_MOUNT}/hf", f"{CACHE_MOUNT}/hf/datasets", f"{CACHE_MOUNT}/hf/transformers"):
        Path(d).mkdir(parents=True, exist_ok=True)
    cache_volume.reload()
    checkpoint_volume.reload()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig

    tokenizer = AutoTokenizer.from_pretrained(hf_repo, use_fast=True, trust_remote_code=True)
    print(f"[tok] type={type(tokenizer).__name__}")

    # Read generation_config.eos_token_id (which can be a list) so vLLM stops on
    # <|im_end|> AND <|endoftext|> — matches HF's behavior. Config.json's single
    # eos_token_id doesn't include <|im_end|>, causing vLLM to over-generate.
    from transformers import GenerationConfig
    try:
        gcfg = GenerationConfig.from_pretrained(hf_repo)
        eos = gcfg.eos_token_id
        stop_ids = eos if isinstance(eos, list) else [eos] if eos is not None else []
    except Exception:
        stop_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    print(f"[vllm] stop_token_ids={stop_ids}")

    llm = LLM(
        model=hf_repo,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        trust_remote_code=True,
    )
    sp = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        n=n_samples,
        repetition_penalty=repetition_penalty,
        stop_token_ids=stop_ids,
        seed=0,  # deterministic — same run gives same numbers
    )

    def _generate_batched(prompts_or_msgs) -> list[list[str]]:
        """Return list of prompt-outputs; each entry is a list of n_samples decoded strings.
        Accepts either list[str] (raw prompts) or list[list[dict]] (chat messages).
        """
        if prompts_or_msgs and isinstance(prompts_or_msgs[0], list):
            outs = llm.chat(prompts_or_msgs, sp, add_generation_prompt=True)
        else:
            outs = llm.generate(prompts_or_msgs, sp)
        return [[o.text for o in r.outputs] for r in outs]

    def _make_prompt(question: str):
        # Return list-of-messages so vLLM's `llm.chat(...)` applies the template
        # (using the tokenizer's chat_template.jinja) — avoids any string re-tokenize
        # discrepancy vs HF's apply_chat_template.
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

    def _extract_pred_answer(pred: str) -> str:
        """Extract the model's answer string (whatever's inside the last \\boxed{})."""
        after_think = pred.split("</think>")[-1] if "</think>" in pred else pred
        boxed = extract_last_boxed(after_think)
        if boxed is not None:
            return boxed.strip()
        # Fallback: last line
        if after_think.strip():
            return after_think.strip().splitlines()[-1].strip()
        return ""

    def _score_multi(all_samples, golds, grade_fn, extract_fn):
        """all_samples: list of list-of-N-strings. Returns pass@1_avg, pass@N, maj@N stats."""
        from collections import Counter
        N = len(all_samples[0]) if all_samples else 0
        # Per-sample correctness matrix
        per_sample_correct = []
        for samples, gold in zip(all_samples, golds):
            per_sample_correct.append([grade_fn(s, gold) for s in samples])
        # pass@1 avg = mean of per-sample correctness across all prompts
        total_samples = sum(len(row) for row in per_sample_correct) or 1
        total_correct = sum(sum(row) for row in per_sample_correct)
        pass_at_1_avg = total_correct / total_samples
        # pass@N = fraction of prompts with at least one correct sample
        pass_at_n = sum(1 for row in per_sample_correct if any(row)) / max(1, len(per_sample_correct))
        # maj@N = majority-vote extracted answer is correct
        maj_correct = 0
        for samples, gold in zip(all_samples, golds):
            answers = [extract_fn(s) for s in samples if s]
            if not answers:
                continue
            most_common, _ = Counter(answers).most_common(1)[0]
            if grade_fn(f"\\boxed{{{most_common}}}", gold):
                maj_correct += 1
        maj_at_n = maj_correct / max(1, len(all_samples))
        return {
            "n_prompts": len(all_samples),
            "n_samples_per_prompt": N,
            "pass_at_1_avg": pass_at_1_avg,
            "pass_at_n": pass_at_n,
            "maj_at_n": maj_at_n,
        }

    def _grade(pred: str, gold: str) -> bool:
        # Extract \\boxed{...} from AFTER </think> if present; otherwise last \\boxed{}
        after_think = pred.split("</think>")[-1] if "</think>" in pred else pred
        boxed = extract_last_boxed(after_think)
        pred_answer = boxed if boxed is not None else after_think.strip().splitlines()[-1] if after_think.strip() else ""
        try:
            parsed_pred = parse(f"\\boxed{{{pred_answer}}}", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
            parsed_gold = parse(str(gold), extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
            return bool(verify(parsed_gold, parsed_pred))
        except Exception:
            return False

    results = {"hf_repo": hf_repo, "per_dataset": {}}

    # ---- GSM8K (main / test) ----
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    if gsm8k_limit:
        gsm8k = gsm8k.select(range(min(gsm8k_limit, len(gsm8k))))
    gsm_prompts = [_make_prompt(row["question"]) for row in gsm8k]
    gsm_golds = [row["answer"].split("####")[-1].strip() for row in gsm8k]

    print(f"[gsm8k] {len(gsm_prompts)} prompts, n_samples={n_samples}, temp={temperature}")
    gsm_texts = _generate_batched(gsm_prompts)  # list[list[str]]
    results["per_dataset"]["gsm8k"] = _score_multi(gsm_texts, gsm_golds, _grade, _extract_pred_answer)
    print(f"[gsm8k] {results['per_dataset']['gsm8k']}")

    # ---- MATH-500 (Hendrycks) ----
    try:
        m500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
        m500_prompts = [_make_prompt(row["problem"]) for row in m500]
        m500_golds = [str(row["answer"]) for row in m500]
        print(f"[math500] {len(m500_prompts)} prompts")
        m500_texts = _generate_batched(m500_prompts)
        results["per_dataset"]["math500"] = _score_multi(m500_texts, m500_golds, _grade, _extract_pred_answer)
        print(f"[math500] {results['per_dataset']['math500']}")
    except Exception as e:
        import traceback
        results["per_dataset"]["math500"] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
        print(f"[math500] ERROR: {e}")

    # ---- rl-math-skyeasy25k-omi2 (eval / test split) ----
    try:
        sky = load_dataset("pre-to-post-olmo/rl-math-skyeasy25k-omi2")
        # pick eval split — try common names
        for split_name in ("eval", "test", "validation", "val"):
            if split_name in sky:
                sky = sky[split_name]
                sky_used_split = split_name
                break
        else:
            # single-split fallback: 5% tail
            first_split = list(sky.keys())[0]
            ds = sky[first_split]
            sky = ds.select(range(int(len(ds) * 0.95), len(ds)))
            sky_used_split = f"{first_split}[95%:]"
        if skyeasy_limit:
            sky = sky.select(range(min(skyeasy_limit, len(sky))))

        # Detect column names — verl parquet format uses:
        #   prompt=<list of chat msgs> or <str>, reward_model={"ground_truth": ...}
        first = sky[0]

        def _extract_prompt(row):
            p = row.get("prompt") or row.get("problem") or row.get("question") or row.get("input")
            if isinstance(p, list):
                # verl format: list of {"role": "user", "content": "..."} msgs
                p = " ".join(str(m.get("content", "")) for m in p if isinstance(m, dict))
            return str(p)

        def _extract_gold(row):
            for k in ("answer", "solution", "gold", "target", "output"):
                if k in row and row[k] is not None:
                    return str(row[k])
            rm = row.get("reward_model")
            if isinstance(rm, dict):
                for k in ("ground_truth", "answer", "gold"):
                    if k in rm:
                        return str(rm[k])
            return ""

        sky_prompts = [_make_prompt(_extract_prompt(row)) for row in sky]
        sky_golds = [_extract_gold(row) for row in sky]
        prompt_col, gold_col = "prompt(auto)", "reward_model.ground_truth(auto)"

        print(f"[skyeasy/{sky_used_split}] {len(sky_prompts)} prompts, n_samples={n_samples}, temp={temperature}")
        sky_texts = _generate_batched(sky_prompts)
        sky_stats = _score_multi(sky_texts, sky_golds, _grade, _extract_pred_answer)
        results["per_dataset"]["skyeasy25k_eval"] = {
            "split": sky_used_split,
            "prompt_col": prompt_col,
            "gold_col": gold_col,
            **sky_stats,
        }
        print(f"[skyeasy] {sky_stats}")
    except Exception as e:
        import traceback
        results["per_dataset"]["skyeasy25k_eval"] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
        print(f"[skyeasy] ERROR: {e}")

    # Save
    out_dir = Path(f"{CHECKPOINT_MOUNT}/evals_downstream/{hf_repo.replace('/', '__')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    checkpoint_volume.commit()
    print(f"[out] {out_path}")

    return results


# Standard 8-shot GSM8K exemplars (Wei et al. 2022, "Chain-of-thought prompting").
_GSM8K_8SHOT = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."),
]

# Minerva-style 4-shot MATH exemplars.
_MATH_4SHOT = [
    ("Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.",
     "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x \\ge 2$, and $5-x > 0$, so $x < 5$. Combining these, the domain is $[2, 5)$. Final Answer: The final answer is $[2,5)$."),
    ("If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
     "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = 24.$ Final Answer: The final answer is $24$."),
    ("Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
     "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$: \\begin{align*} 30n&=480\\\\ \\Rightarrow\\qquad n&=480/30=16 \\end{align*} Final Answer: The final answer is $16$."),
    ("If the system of equations \\begin{align*} 6x-4y&=a,\\\\ 6y-9x &=b. \\end{align*} has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.",
     "If we multiply the first equation by $-\\frac{3}{2}$, we obtain $$6y-9x=-\\frac{3}{2}a.$$ Since we also know that $6y-9x=b$, we have $$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=-\\frac{2}{3}.$$ Final Answer: The final answer is $-\\frac{2}{3}$."),
]


def _build_gsm8k_fewshot(question: str) -> str:
    parts = []
    for q, a in _GSM8K_8SHOT:
        parts.append(f"Question: {q}\nAnswer: {a}")
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


def _build_math_fewshot(problem: str) -> str:
    parts = []
    for q, a in _MATH_4SHOT:
        parts.append(f"Problem:\n{q}\n\nSolution:\n{a}")
    parts.append(f"Problem:\n{problem}\n\nSolution:\n")
    return "\n\n".join(parts)


def _extract_gsm8k_answer(text: str) -> str:
    """Extract from 'The answer is X.' pattern; fallback to last number."""
    import re
    # Cut at next "Question:" so we don't hallucinate continued exemplars
    text = text.split("\nQuestion:")[0]
    m = list(re.finditer(r"The answer is\s*\$?([-+]?\d[\d,]*\.?\d*)", text))
    if m:
        return m[-1].group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else ""


def _extract_math_answer(text: str) -> str:
    """Extract 'Final Answer: The final answer is X.' or last \\boxed{}."""
    import re
    text = text.split("\nProblem:")[0]
    m = re.findall(r"final answer is\s*\$?(.+?)(?:\$|\.|$)", text, flags=re.IGNORECASE)
    if m:
        return m[-1].strip().rstrip("$").strip()
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


@app.function(
    gpu="H200:1",
    timeout=60 * 60 * 6,
    volumes={CACHE_MOUNT: cache_volume, CHECKPOINT_MOUNT: checkpoint_volume},
    secrets=[hf_secret],
)
def eval_fewshot(
    hf_repo: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    n_samples: int = 1,
    gsm8k_limit: int | None = None,
    math500_limit: int | None = None,
    repetition_penalty: float = 1.0,
) -> dict:
    """Few-shot eval for base + anneal (no chat template): GSM8K 8-shot, MATH 4-shot.

    Uses greedy decoding by default (deterministic, no sampling). Reports pass@1 only.
    """
    import json
    import os
    import re
    from pathlib import Path

    os.environ["HF_HOME"] = f"{CACHE_MOUNT}/hf"
    os.environ["HF_DATASETS_CACHE"] = f"{CACHE_MOUNT}/hf/datasets"
    os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_MOUNT}/hf/transformers"
    for d in (f"{CACHE_MOUNT}/hf", f"{CACHE_MOUNT}/hf/datasets", f"{CACHE_MOUNT}/hf/transformers"):
        Path(d).mkdir(parents=True, exist_ok=True)
    cache_volume.reload()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig

    llm = LLM(model=hf_repo, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=0.85, max_model_len=4096)

    sp = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        n=n_samples,
        repetition_penalty=repetition_penalty,
        stop=["\nQuestion:", "\nProblem:"],
        seed=0,
    )

    def _batched_generate(prompts):
        outs = llm.generate(prompts, sp)
        return [[o.text for o in r.outputs] for r in outs]

    results = {"hf_repo": hf_repo, "mode": "fewshot", "per_dataset": {}}

    # ---- GSM8K 8-shot ----
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    if gsm8k_limit:
        gsm = gsm.select(range(min(gsm8k_limit, len(gsm))))
    gsm_prompts = [_build_gsm8k_fewshot(row["question"]) for row in gsm]
    gsm_golds = [row["answer"].split("####")[-1].strip().replace(",", "") for row in gsm]
    print(f"[gsm8k 8-shot] {len(gsm_prompts)} prompts")
    gsm_outs = _batched_generate(gsm_prompts)
    correct = 0
    for outs, gold in zip(gsm_outs, gsm_golds):
        pred = _extract_gsm8k_answer(outs[0])
        try:
            if float(pred) == float(gold):
                correct += 1
        except ValueError:
            if pred.strip() == gold.strip():
                correct += 1
    acc = correct / max(1, len(gsm_prompts))
    results["per_dataset"]["gsm8k"] = {
        "n_prompts": len(gsm_prompts), "n_shot": 8, "pass_at_1": acc,
    }
    print(f"[gsm8k 8-shot] pass@1={acc:.4f}")

    # ---- MATH-500 4-shot ----
    try:
        m500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
        if math500_limit:
            m500 = m500.select(range(min(math500_limit, len(m500))))
        m500_prompts = [_build_math_fewshot(row["problem"]) for row in m500]
        m500_golds = [str(row["answer"]) for row in m500]
        print(f"[math500 4-shot] {len(m500_prompts)} prompts")
        m500_outs = _batched_generate(m500_prompts)
        correct = 0
        for outs, gold in zip(m500_outs, m500_golds):
            pred = _extract_math_answer(outs[0])
            try:
                gp = parse(f"${pred}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
                gg = parse(f"${gold}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
                if verify(gg, gp):
                    correct += 1
                    continue
            except Exception:
                pass
            if pred.strip() == gold.strip():
                correct += 1
        acc = correct / max(1, len(m500_prompts))
        results["per_dataset"]["math500"] = {
            "n_prompts": len(m500_prompts), "n_shot": 4, "pass_at_1": acc,
        }
        print(f"[math500 4-shot] pass@1={acc:.4f}")
    except Exception as e:
        import traceback
        results["per_dataset"]["math500"] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
        print(f"[math500 4-shot] ERROR: {e}")

    # ---- save ----
    out_dir = Path(f"{CHECKPOINT_MOUNT}/evals_fewshot/{hf_repo.replace('/', '__')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    checkpoint_volume.commit()
    print(f"[out] {out_dir}/results.json")

    return results


@app.local_entrypoint()
def eval_fewshot_single(hf_repo: str, gsm8k_limit: int = 0, math500_limit: int = 0) -> None:
    import json
    r = eval_fewshot.remote(
        hf_repo=hf_repo,
        gsm8k_limit=gsm8k_limit or None,
        math500_limit=math500_limit or None,
    )
    print(json.dumps(r, indent=2))


@app.local_entrypoint()
def eval_fewshot_sweep(kind: str = "both") -> None:
    """kind: 'stable' | 'anneal' | 'both'"""
    STABLE = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 95368]
    ANNEAL = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000,
              55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 95368]
    repos = []
    if kind in ("stable", "both"):
        repos += [f"pre-to-post-olmo/math-1b-stable-step{s}" for s in STABLE]
    if kind in ("anneal", "both"):
        repos += [f"pre-to-post-olmo/math-1b-anneal-from-step{s}" for s in ANNEAL]
    print(f"firing {len(repos)} few-shot evals in parallel")
    futures = [(r, eval_fewshot.spawn(hf_repo=r)) for r in repos]
    for r, f in futures:
        try:
            res = f.get()
            g = res["per_dataset"].get("gsm8k", {}).get("pass_at_1")
            m = res["per_dataset"].get("math500", {}).get("pass_at_1")
            print(f"  {r.rsplit('/',1)[-1]}: gsm8k={g} math500={m}")
        except Exception as e:
            print(f"  ✗ {r}: {e}")


SFT_REPOS = [
    "pre-to-post-olmo/math-1b-sft-openthoughts-from-step10000",
    "pre-to-post-olmo/math-1b-sft-openthoughts-from-step20000",
    "pre-to-post-olmo/math-1b-sft-openthoughts-from-step30000",
    "pre-to-post-olmo/math-1b-sft-openthoughts-from-step40000",
    "pre-to-post-olmo/math-1b-sft-openthoughts-from-step50000",
    "pre-to-post-olmo/math-1b-sft-openthoughts-from-step60000",
    "pre-to-post-olmo/math-1b-sft-openthoughts-from-step70000",
    "pre-to-post-olmo/math-1b-sft-openthoughts-from-step80000",
    "pre-to-post-olmo/math-1b-sft-openthoughts-from-step90000",
    "pre-to-post-olmo/math-1b-sft-openthoughts-from-step95368",
]


@app.local_entrypoint()
def eval_single(
    hf_repo: str,
    gsm8k_limit: int = 0,
    skyeasy_limit: int = 0,
    n_samples: int = 8,
    temperature: float = 0.7,
    max_new_tokens: int = 8192,
) -> None:
    r = eval_model.remote(
        hf_repo=hf_repo,
        max_new_tokens=max_new_tokens,
        n_samples=n_samples,
        temperature=temperature,
        gsm8k_limit=gsm8k_limit or None,
        skyeasy_limit=skyeasy_limit or None,
    )
    import json
    print(json.dumps(r, indent=2))


SFT_5K_REPOS = [
    f"pre-to-post-olmo/math-1b-sft-numinamath-bs512-from-step{a}"
    for a in (5000, 15000, 25000, 35000, 45000, 55000, 65000, 75000, 85000, 95000)
]


@app.local_entrypoint()
def eval_sweep_5k(
    gsm8k_limit: int = 0, skyeasy_limit: int = 0, math500_limit: int = 0,
    n_samples: int = 8, temperature: float = 0.7,
) -> None:
    """Fire eval on the 10 every-5k NuminaMath SFT repos in parallel."""
    import json
    print(f"firing {len(SFT_5K_REPOS)} downstream evals (5k stride) in parallel")
    futures = []
    for repo in SFT_5K_REPOS:
        f = eval_model.spawn(
            hf_repo=repo,
            n_samples=n_samples,
            temperature=temperature,
            gsm8k_limit=gsm8k_limit or None,
            skyeasy_limit=skyeasy_limit or None,
        )
        futures.append((repo, f))
    all_results = []
    for repo, f in futures:
        try:
            r = f.get()
        except Exception as e:
            r = {"hf_repo": repo, "error": str(e)}
        all_results.append(r)
        gsm = r.get("per_dataset", {}).get("gsm8k", {}).get("accuracy")
        m500 = r.get("per_dataset", {}).get("math500", {}).get("accuracy")
        sky = r.get("per_dataset", {}).get("skyeasy25k_eval", {}).get("accuracy")
        print(f"  {repo.rsplit('/',1)[-1]}: gsm8k={gsm} math500={m500} skyeasy={sky}")
    print()
    print(json.dumps(all_results, indent=2))


@app.local_entrypoint()
def eval_sweep(
    gsm8k_limit: int = 0, skyeasy_limit: int = 0,
    n_samples: int = 8, temperature: float = 0.7,
) -> None:
    """Fire eval on all landed SFT repos in parallel."""
    import json
    print(f"firing {len(SFT_REPOS)} downstream evals in parallel")
    futures = []
    for repo in SFT_REPOS:
        f = eval_model.spawn(
            hf_repo=repo,
            n_samples=n_samples,
            temperature=temperature,
            gsm8k_limit=gsm8k_limit or None,
            skyeasy_limit=skyeasy_limit or None,
        )
        futures.append((repo, f))
    all_results = []
    for repo, f in futures:
        try:
            r = f.get()
        except Exception as e:
            r = {"hf_repo": repo, "error": str(e)}
        all_results.append(r)
        gsm = r.get("per_dataset", {}).get("gsm8k", {}).get("accuracy")
        sky = r.get("per_dataset", {}).get("skyeasy25k_eval", {}).get("accuracy")
        print(f"  {repo.rsplit('/',1)[-1]}: gsm8k={gsm} skyeasy={sky}")

    print()
    print(json.dumps(all_results, indent=2))
