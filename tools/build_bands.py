"""Build solvable-only band datasets (1-15 wins of 16) from a checkpoint's own
full-eval generations, for RL training."""
import gzip
import hashlib
import json
from pathlib import Path

import modal

app = modal.App("build-bands")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pandas", "pyarrow", "numpy"
).add_local_dir(
    "/Users/leonli66/Desktop/Research/RL/Chess RL/Eval/test_data",
    remote_path="/eval-data",
)
results_vol = modal.Volume.from_name("chess-rl-eval-results-r6", create_if_missing=False)
miles_data_vol = modal.Volume.from_name("chess-rl-miles-data", create_if_missing=False)

NS = "/results/ablation_pass16_clean_v2_bos"


@app.function(image=image, cpu=8.0, memory=32 * 1024, timeout=1800,
              volumes={"/results": results_vol, "/data": miles_data_vol})
def build(arm: str, out_name: str) -> str:
    import pandas as pd

    results_vol.reload()
    frames = []
    hist = {}
    for sh in range(4):
        wins = {}
        with gzip.open(f"{NS}/{arm}/n16/eval_train_v4_balanced_shard{sh}/generations.jsonl.gz", "rt") as fh:
            for line in fh:
                d = json.loads(line)
                wins[d["row"]] = wins.get(d["row"], 0) + int(d["score"] > 0)
        df = pd.read_parquet(f"/eval-data/eval_train_v4_balanced_shard{sh}.parquet").reset_index(drop=True)
        for w in wins.values():
            hist[w] = hist.get(w, 0) + 1
        sel = sorted(r for r, w in wins.items() if 1 <= w <= 15)
        frames.append(df.iloc[sel])
    out = pd.concat(frames, ignore_index=True)
    path = Path(f"/data/chess-rl-data/{out_name}")
    out.to_parquet(path, index=False)
    miles_data_vol.commit()
    return json.dumps({
        "arm": arm, "rows": len(out),
        "never_solved": hist.get(0, 0), "always_solved": hist.get(16, 0),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "path": str(path),
    })


@app.local_entrypoint()
def main() -> None:
    jobs = {
        "E2W1LOOP": "train_v4_e2w1loopband_multi_turn.parquet",
        "E3P2LOOP": "train_v4_e3p2loopband_multi_turn.parquet",
    }
    calls = {a: build.spawn(a, n) for a, n in jobs.items()}
    for a, c in calls.items():
        print("RESULT::" + c.get())
