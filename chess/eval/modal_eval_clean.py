"""Clean, faithful pass@k evaluator — bare vLLM, no verl trainer, no giant tensors.

Faithfully replicates the official evaluator's semantics:
  * multi-turn loop from fsdp_workers.generate_multi_turn_sequences:
      - generate with stop on <call_env>, global model-token budget 2560
      - on <call_env>: append next pre-scripted env reply (extra_info.env_replies),
        env replies do NOT count against the model-token budget
      - finish on EOS, empty output, exhausted replies, 6 env calls, or budget
  * prompt = dataset "prompt" string (already ends in <T>), tokenizer from the
    checkpoint (trust_remote_code), add_special_tokens=False, 512-token cap
  * sampling: temperature 1.0, top-p 1.0 (official val_kwargs)
  * scoring: verl/reward_function_multiturn.compute_score_batch imported
    VERBATIM; solution text decoded with skip_special_tokens=True (official
    batch reward manager behaviour)
  * results stream to CPU/disk per row — memory does not scale with dataset

  modal run Eval/modal_eval_clean.py --action canary                 # A1, 256 prompts, n=16
  modal run --detach Eval/modal_eval_clean.py --action launch        # 8 arms x 4 shards, n=16
  modal run Eval/modal_eval_clean.py --action merge
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

HERE = Path(__file__).resolve().parent
# Parquet eval shards are not stored in this repo (104 MB). Place or symlink
# them at chess/eval/test_data/ — see docs/08-data-and-artifacts.md.
EVAL_DATA_ROOT = Path(os.environ.get("CHESS_EVAL_DATA_ROOT", HERE / "test_data"))
_REWARD_CANDIDATES = (
    HERE / "reward_function" / "reward_function_multiturn.py",
    HERE / "pre2post-chess" / "rl" / "verl" / "reward_function_multiturn.py",
)
REWARD_FN_LOCAL = next(
    (p for p in _REWARD_CANDIDATES if p.is_file()), _REWARD_CANDIDATES[0]
)
REWARD_FN_REMOTE = "/opt/clean_eval/reward_function_multiturn.py"
REMOTE_EVAL_DATA = "/eval-data"

EXPERIMENT_VERSION = "sft_injection_ablation_v1_20260801"
CKPT_MOUNT = "/pretrain-checkpoints"
CKPT_ROOT = f"{CKPT_MOUNT}/interleave_50m/pretrain/{EXPERIMENT_VERSION}"
ARMS = ["A1", "A2", "A3", "A4", "A2R", "B1", "B2", "B3", "B4", "B1H", "B2H", "B3H", "B4H", "E2W1", "P1W1", "E3P2", "E1UP2", "E1DP2", "LR4P2", "TRACEP2", "TRACEP2K2", "TRACEP2K4", "TRACEP2K8", "TRACEP2K16", "TRACEP2ROLL", "E2W1LOOP", "E3P2LOOP"]

SHARDS = [f"eval_train_v4_balanced_shard{i}" for i in range(4)]
FULL_DATASET_SHA256 = (
    "bcf131d88cd9916f6ec12a12f8e85901a05e0dabc5d5a1fdebd8ea0b0ae86c30"
)

RESULTS_VOLUME_NAME = "chess-rl-eval-results-r6"
RESULTS_ROOT = "/results"
NAMESPACE = "ablation_pass16_clean_v2_bos"

# official evaluator semantics
RESPONSE_BUDGET = 2560     # model-generated tokens per sample (env replies free)
MAX_ENV_CALLS = 6
PROMPT_CAP = 512
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_MODEL_LEN = 3072

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("curl", "git")
    .pip_install(
        "vllm==0.8.5",           # same engine+version as the official evaluator
        "transformers==4.51.3",
        "tokenizers==0.21.4",
        "huggingface_hub==0.36.2",
        "pandas==2.3.3",
        "pyarrow==23.0.1",
        "numpy==2.2.6",
        "chess==1.11.2",         # reward fn legality checks
    )
    .add_local_file(str(REWARD_FN_LOCAL), remote_path=REWARD_FN_REMOTE, copy=True)
    .add_local_dir(str(EVAL_DATA_ROOT), remote_path=REMOTE_EVAL_DATA)
)

results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(
    "rl-reasoning-checkpoints", create_if_missing=False
)

app = modal.App(
    "chess-ablation-clean-eval",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)


class _State:
    __slots__ = (
        "row_idx", "sample_idx", "prompt_len", "ctx", "model_tokens",
        "env_calls", "env_replies", "finished",
    )

    def __init__(self, row_idx: int, sample_idx: int, prompt_ids: list[int],
                 env_replies: list[str]):
        self.row_idx = row_idx
        self.sample_idx = sample_idx
        self.prompt_len = len(prompt_ids)
        self.ctx = list(prompt_ids)
        self.model_tokens = 0
        self.env_calls = 0
        self.env_replies = list(env_replies)
        self.finished = False

    @property
    def response_ids(self) -> list[int]:
        return self.ctx[self.prompt_len:]


@app.function(
    gpu="H200",
    cpu=16.0,
    memory=64 * 1024,
    timeout=60 * 60 * 12,
    retries=0,
    volumes={RESULTS_ROOT: results_volume, CKPT_MOUNT: checkpoint_volume},
)
def eval_arm(
    arm: str,
    dataset: str,
    n_samples: int = 16,
    max_prompts: int = 0,
    hf_repo: str = "",
    hf_subfolder: str = "",
    stride: int = 1,
    model_subpath: str = "",
) -> str:
    import importlib.util
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    checkpoint_volume.reload()
    results_volume.reload()

    tag = f"{dataset}" + (f"_first{max_prompts}" if max_prompts else "") + (
        f"_stride{stride}" if stride > 1 else "")
    result_root = Path(RESULTS_ROOT) / NAMESPACE / arm / f"n{n_samples}" / tag
    marker = result_root / "success.json"
    if marker.is_file():
        return marker.read_text()

    if hf_repo:
        from huggingface_hub import snapshot_download
        pattern = f"{hf_subfolder}/*" if hf_subfolder else "*"
        local = snapshot_download(hf_repo, allow_patterns=[pattern])
        model_path = Path(local) / hf_subfolder if hf_subfolder else Path(local)
    elif model_subpath:
        # arbitrary checkpoint on the volume, e.g. RL HF exports under
        # interleave_50m/rl_hf/<name>
        model_path = Path(CKPT_MOUNT) / model_subpath
    else:
        model_path = Path(CKPT_ROOT) / arm.lower() / "final"
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path)
    # clamp to the model's own context if smaller than ours
    cfg = json.loads((model_path / "config.json").read_text())
    model_ctx = int(cfg.get("max_position_embeddings", MAX_MODEL_LEN))
    max_len = min(MAX_MODEL_LEN, model_ctx)

    # --- reward function, imported verbatim ---------------------------------
    spec = importlib.util.spec_from_file_location("reward_mt", REWARD_FN_REMOTE)
    reward_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reward_mod)
    compute_score_batch = reward_mod.compute_score_batch

    # --- data ----------------------------------------------------------------
    frame = pd.read_parquet(Path(REMOTE_EVAL_DATA) / f"{dataset}.parquet")
    if stride > 1:
        frame = frame.iloc[::stride]
    if max_prompts:
        frame = frame.iloc[:max_prompts]

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    call_env_id = tokenizer.convert_tokens_to_ids("<call_env>")
    eos_id = tokenizer.eos_token_id
    if call_env_id is None or call_env_id == tokenizer.unk_token_id:
        raise RuntimeError("<call_env> not in vocabulary")

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        raise RuntimeError("tokenizer has no bos token")
    states: list[_State] = []
    skipped_overlong = 0
    for row_idx, row in enumerate(frame.itertuples(index=False)):
        # training sequences always start with <bos>; replicate that here
        prompt_ids = [bos_id] + tokenizer(
            row.prompt, add_special_tokens=False
        ).input_ids
        if len(prompt_ids) - 1 > PROMPT_CAP:  # official filter_overlong_prompts (bos excluded)
            skipped_overlong += 1
            continue
        raw_replies = row.extra_info.get("env_replies", [])
        replies = [str(r) for r in (raw_replies if raw_replies is not None else [])]
        for sample_idx in range(n_samples):
            states.append(_State(row_idx, sample_idx, prompt_ids, replies))
    print(f"[clean-eval] {arm}/{dataset}: {len(frame)} prompts "
          f"({skipped_overlong} overlong skipped) x {n_samples} = {len(states)} samples")

    llm = LLM(
        model=str(model_path),
        trust_remote_code=True,
        gpu_memory_utilization=0.60,
        max_model_len=max_len,
        seed=0,
        enforce_eager=False,
        dtype="bfloat16",
    )

    reply_tok_cache: dict[str, list[int]] = {}
    round_no = 0
    while True:
        active = [s for s in states if not s.finished]
        if not active:
            break
        round_no += 1
        prompts, params = [], []
        for s in active:
            budget = min(RESPONSE_BUDGET - s.model_tokens,
                         max_len - len(s.ctx))
            prompts.append(TokensPrompt(prompt_token_ids=s.ctx))
            params.append(SamplingParams(
                temperature=TEMPERATURE, top_p=TOP_P,
                max_tokens=max(budget, 1),
                stop_token_ids=[call_env_id],
            ))
        outs = llm.generate(prompts, params, use_tqdm=False)
        n_done = 0
        for s, out in zip(active, outs):
            comp = out.outputs[0]
            tokens = list(comp.token_ids)
            # vLLM excludes the matched stop token — re-append it (official
            # loop keeps <call_env> in the sequence).
            if comp.finish_reason == "stop" and comp.stop_reason == call_env_id:
                tokens.append(call_env_id)
            if not tokens:
                s.finished = True
                n_done += 1
                continue
            remaining = RESPONSE_BUDGET - s.model_tokens
            tokens = tokens[:remaining]
            if not tokens:
                s.finished = True
                n_done += 1
                continue
            if call_env_id in tokens:
                cut = tokens.index(call_env_id) + 1
                kept = tokens[:cut]
                s.ctx.extend(kept)
                s.model_tokens += len(kept)
                s.env_calls += 1
                reply = s.env_replies.pop(0) if s.env_replies else None
                if not reply:
                    s.finished = True
                else:
                    if reply not in reply_tok_cache:
                        reply_tok_cache[reply] = tokenizer(
                            reply, add_special_tokens=False
                        ).input_ids
                    s.ctx.extend(reply_tok_cache[reply])   # env tokens: budget-free
                if s.env_calls >= MAX_ENV_CALLS:
                    s.finished = True
                if len(s.ctx) >= max_len:
                    s.finished = True
            else:
                s.ctx.extend(tokens)
                s.model_tokens += len(tokens)
                s.finished = True
            if s.finished:
                n_done += 1
        print(f"[clean-eval] round {round_no}: {len(active)} active, {n_done} finished")

    # --- score (CPU) -----------------------------------------------------------
    texts = [
        tokenizer.decode(s.response_ids, skip_special_tokens=True) for s in states
    ]
    rows_meta = frame.reset_index(drop=True)
    data_sources = [rows_meta.iloc[s.row_idx]["data_source"] for s in states]
    ground_truths = [
        rows_meta.iloc[s.row_idx]["reward_model"]["ground_truth"] for s in states
    ]
    extra_infos = [dict(rows_meta.iloc[s.row_idx]["extra_info"]) for s in states]
    scored = compute_score_batch(data_sources, texts, ground_truths, extra_infos)

    # --- persist + summarize -----------------------------------------------
    result_root.mkdir(parents=True, exist_ok=True)
    per_prompt: dict[int, dict[str, int]] = {}
    with gzip.open(result_root / "generations.jsonl.gz", "wt") as handle:
        for s, text, res in zip(states, texts, scored):
            score = float(res["score"]) if isinstance(res, dict) else float(res)
            fmt = 0 <= text.find("</T>") < text.find("<call_env>") \
                if "<call_env>" in text and "</T>" in text else False
            entry = per_prompt.setdefault(
                s.row_idx, {"n": 0, "wins": 0, "fmt": 0}
            )
            entry["n"] += 1
            entry["wins"] += int(score > 0)
            entry["fmt"] += int(fmt)
            handle.write(json.dumps({
                "row": int(s.row_idx), "sample": int(s.sample_idx),
                "score": score, "format_ok": bool(fmt),
                "model_tokens": s.model_tokens, "env_calls": s.env_calls,
                "response": text,
            }) + "\n")

    hist: dict[int, int] = {}
    fmt_hits = total_rows = 0
    for entry in per_prompt.values():
        hist[entry["wins"]] = hist.get(entry["wins"], 0) + 1
        fmt_hits += entry["fmt"]
        total_rows += entry["n"]
    prompts_n = len(per_prompt)

    def pass_at_k(k: int) -> float:
        total = 0.0
        for wins, count in hist.items():
            if wins <= 0:
                v = 0.0
            elif n_samples - wins < k:
                v = 1.0
            else:
                v = 1.0 - math.comb(n_samples - wins, k) / math.comb(n_samples, k)
            total += v * count
        return total / max(1, prompts_n)

    summary = {
        "schema": "ablation-clean-eval-summary-v1",
        "arm": arm, "dataset": dataset, "n_samples": n_samples,
        "max_prompts": max_prompts,
        "prompts": prompts_n, "rows": total_rows,
        "skipped_overlong": skipped_overlong,
        "pass_at_1": pass_at_k(1),
        "pass_at_8": pass_at_k(8) if n_samples >= 8 else None,
        "pass_at_16": pass_at_k(16) if n_samples >= 16 else None,
        "format_rate": fmt_hits / max(1, total_rows),
        "variance_rate": sum(
            c for w, c in hist.items() if 0 < w < n_samples
        ) / max(1, prompts_n),
        "solved_prompts": sum(c for w, c in hist.items() if w > 0),
        "wins_histogram": {str(k): v for k, v in sorted(hist.items())},
        "model_path": str(model_path),
        "engine": "vllm-0.8.5-bare",
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    marker.write_text(json.dumps(summary, indent=2, sort_keys=True))
    results_volume.commit()
    print(json.dumps({k: summary[k] for k in (
        "arm", "dataset", "prompts", "pass_at_1", "pass_at_8", "pass_at_16",
        "format_rate", "variance_rate")}, indent=2))
    return json.dumps(summary)


@app.local_entrypoint()
def main(action: str = "canary", arm: str = "", n_samples: int = 16) -> None:
    if action == "canary":
        print(eval_arm.remote(arm or "A1", SHARDS[0], n_samples, 256))
    elif action == "prev-sft":
        # previous staged-SFT 50M reference: full 4-shard eval, n=16
        sub = arm or "C6p5e18_50m_alpha0.400_beta0.023"
        calls = {
            shard: eval_arm.spawn(
                f"PREV-{sub}", shard, n_samples, 0,
                "chess-pre-to-post/sft_trajectory_no_labels", sub,
            ) for shard in SHARDS
        }
        print(json.dumps({k: c.object_id for k, c in calls.items()}, indent=2))
    elif action == "launch":
        calls = {}
        for a in ARMS:
            for shard in SHARDS:
                calls[f"{a}/{shard[-6:]}"] = eval_arm.spawn(a, shard, n_samples, 0)
        print(json.dumps({k: c.object_id for k, c in calls.items()}, indent=2))
    elif action == "launch-one":
        if not arm:
            raise ValueError("launch-one requires --arm")
        calls = {
            shard: eval_arm.spawn(arm, shard, n_samples, 0) for shard in SHARDS
        }
        print(json.dumps({k: c.object_id for k, c in calls.items()}, indent=2))
    elif action == "rl-point":
        # --arm "<run>@<step>" evaluates the converted RL HF export at
        # interleave_50m/rl_hf/<run>-s<step>, stride-8 subset across all shards.
        run, step_s = arm.split("@")
        name = f"{run}-s{int(step_s):04d}"
        calls = {
            shard: eval_arm.spawn(
                f"RL-{name}", shard, n_samples, 0, "", "", 8,
                f"interleave_50m/rl_hf/{name}",
            ) for shard in SHARDS
        }
        print(json.dumps({k: c.object_id for k, c in calls.items()}, indent=2))
    elif action == "rl-full":
        # full 4-shard endpoint eval for a converted RL export ("<run>@<step>")
        run, step_s = arm.split("@")
        name = f"{run}-s{int(step_s):04d}"
        calls = {
            shard: eval_arm.spawn(
                f"RL-{name}", shard, n_samples, 0, "", "", 1,
                f"interleave_50m/rl_hf/{name}",
            ) for shard in SHARDS
        }
        print(json.dumps({k: c.object_id for k, c in calls.items()}, indent=2))
    elif action == "rl-base":
        # stride-8 baseline for a pretrain endpoint arm (step-0 curve point)
        calls = {
            shard: eval_arm.spawn(arm, shard, n_samples, 0, "", "", 8)
            for shard in SHARDS
        }
        print(json.dumps({k: c.object_id for k, c in calls.items()}, indent=2))
    elif action == "launch-hist":
        calls = {}
        for a in ("B1H", "B2H", "B3H", "B4H"):
            for shard in SHARDS:
                calls[f"{a}/{shard[-6:]}"] = eval_arm.spawn(a, shard, n_samples, 0)
        print(json.dumps({k: c.object_id for k, c in calls.items()}, indent=2))
    elif action == "merge":
        import subprocess as sp
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            sp.run(["modal", "volume", "get", RESULTS_VOLUME_NAME, NAMESPACE,
                    tmp, "--force"], capture_output=True, text=True)
            root = Path(tmp) / NAMESPACE
            matrix: dict[str, Any] = {}
            for a in ARMS:
                hist: dict[int, int] = {}
                fmt = rows = 0
                done = 0
                for marker in sorted(root.glob(f"{a}/n{n_samples}/*shard*/success.json")):
                    d = json.loads(marker.read_text())
                    if d.get("max_prompts"):
                        continue
                    done += 1
                    for w, c in d["wins_histogram"].items():
                        hist[int(w)] = hist.get(int(w), 0) + int(c)
                    fmt += round(d["format_rate"] * d["rows"])
                    rows += d["rows"]
                if not hist:
                    matrix[a] = {"shards_done": done, "status": "pending"}
                    continue
                prompts = sum(hist.values())

                def pak(k: int) -> float:
                    total = 0.0
                    for w, c in hist.items():
                        if w <= 0:
                            v = 0.0
                        elif n_samples - w < k:
                            v = 1.0
                        else:
                            v = 1.0 - math.comb(n_samples - w, k) / math.comb(n_samples, k)
                        total += v * c
                    return total / prompts

                matrix[a] = {
                    "shards_done": done,
                    "prompts": prompts,
                    "pass@1": round(pak(1), 4),
                    "pass@8": round(pak(8), 4),
                    "pass@16": round(pak(16), 4),
                    "format_rate": round(fmt / max(1, rows), 4),
                    "variance_rate": round(sum(
                        c for w, c in hist.items() if 0 < w < n_samples
                    ) / prompts, 4),
                }
            print(json.dumps(matrix, indent=2))
    else:
        raise ValueError(f"Unknown action: {action}")
