"""Select one verified-correct RL trace per prompt from the lr1e-4 endpoint's
full-set eval generations and stage them as an SFT-style parquet.

Quality gates per trace (Exp4 criteria): score == 1, format_ok, exactly one
</T>, at least one <call_env>, model_tokens <= 2560. One trace per prompt,
chosen deterministically. Prompt/response are reshaped to the SFT cache frame:
pgn = prompt minus trailing "<T>", resp = "<T> " + trace.
"""
import gzip
import hashlib
import json
from pathlib import Path

import modal

app = modal.App("trace-extract")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pandas", "pyarrow", "numpy"
).add_local_dir(
    "/Users/leonli66/Desktop/Research/RL/Chess RL/Eval/test_data",
    remote_path="/eval-data",
)
results_vol = modal.Volume.from_name("chess-rl-eval-results-r6", create_if_missing=False)
data_vol = modal.Volume.from_name("rl-reasoning-training-data", create_if_missing=False)

GEN_ROOT = "/results/ablation_pass16_clean_v2_bos/RL-p1w1-band-lr1e4-rl1500-s1500/n16"
OUT_DIR = "/data/sft_injection_ablation_v1_20260801/trace_transfer"
SELECT_SEED = 20260806


@app.function(image=image, cpu=8.0, memory=48 * 1024, timeout=3600,
              volumes={"/results": results_vol, "/data": data_vol})
def extract(k: int = 1) -> str:
    import numpy as np
    import pandas as pd

    results_vol.reload()
    records = []
    stats = {"prompts": 0, "candidates": 0, "rejected_quality": 0,
             "prompts_with_pick": 0}
    for sh in range(4):
        df = pd.read_parquet(f"/eval-data/eval_train_v4_balanced_shard{sh}.parquet")
        df = df.reset_index(drop=True)
        by_row: dict[int, list[dict]] = {}
        with gzip.open(f"{GEN_ROOT}/eval_train_v4_balanced_shard{sh}/generations.jsonl.gz", "rt") as fh:
            for line in fh:
                d = json.loads(line)
                if d["score"] != 1 or not d["format_ok"]:
                    continue
                stats["candidates"] += 1
                text = d["response"]
                if text.count("</T>") != 1 or "<call_env>" not in text \
                        or d["model_tokens"] > 2560:
                    stats["rejected_quality"] += 1
                    continue
                by_row.setdefault(d["row"], []).append(d)
        stats["prompts"] += len(df)
        for row_idx, cands in sorted(by_row.items()):
            cands.sort(key=lambda c: c["sample"])
            # within-prompt exact-text dedup before selection
            seen = set()
            uniq = []
            for c in cands:
                if c["response"] not in seen:
                    seen.add(c["response"])
                    uniq.append(c)
            rng = np.random.default_rng(SELECT_SEED + sh * 1_000_000 + row_idx)
            picks = [uniq[i] for i in rng.permutation(len(uniq))[: int(k)]]
            prompt = str(df.iloc[row_idx]["prompt"])
            if not prompt.rstrip().endswith("<T>"):
                raise RuntimeError(f"prompt does not end with <T>: shard {sh} row {row_idx}")
            pgn = prompt.rstrip()[: -len("<T>")].strip()
            extra = df.iloc[row_idx].get("extra_info") or {}
            for pick in picks:
                records.append({
                    "pgn": pgn,
                    "resp": "<T> " + pick["response"].strip(),
                    "shard": sh,
                    "row": int(row_idx),
                    "sample": int(pick["sample"]),
                    "model_tokens": int(pick["model_tokens"]),
                    "env_calls": int(pick["env_calls"]),
                    "puzzle_id": str(extra.get("PuzzleId", "")) if isinstance(extra, dict) else "",
                })
            stats["prompts_with_pick"] += 1

    out = pd.DataFrame(records)
    # exact-duplicate guard (plan Exp4: dedup exact (prompt, response))
    before = len(out)
    out = out.drop_duplicates(subset=["pgn", "resp"]).reset_index(drop=True)
    stats["dedup_dropped"] = before - len(out)
    stats["final_rows"] = len(out)

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    suffix = f"_k{k}" if int(k) != 1 else ""
    path = Path(OUT_DIR) / f"traces{suffix}.parquet"
    out.to_parquet(path, index=False)
    stats["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    stats["generator_checkpoint"] = "rl_hf/p1w1-band-lr1e4-rl1500-s1500"
    stats["select_seed"] = SELECT_SEED
    stats["k"] = int(k)
    (Path(OUT_DIR) / f"traces_summary{suffix}.json").write_text(json.dumps(stats, indent=2))
    data_vol.commit()
    return json.dumps(stats)


@app.local_entrypoint()
def main(ks: str = "1") -> None:
    calls = {kk: extract.spawn(int(kk)) for kk in ks.split(",")}
    for kk, c in calls.items():
        print(f"RESULT k={kk}::" + c.get())
