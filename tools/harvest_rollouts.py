"""Harvest ALL positive traces from the lr1e-4 RL run's own training rollouts
(zero extra generation FLOPs) and stage them as a trace parquet."""
import hashlib
import json
from pathlib import Path

import modal

app = modal.App("harvest-rollouts")
image = modal.Image.debian_slim(python_version="3.11").pip_install("pandas", "pyarrow")
rl_vol = modal.Volume.from_name("chess-rl-miles-checkpoints", create_if_missing=False)
data_vol = modal.Volume.from_name("rl-reasoning-training-data", create_if_missing=False)

RUN = "/rl/chess-rl-miles-interleave/p1w1-band-lr1e4-rl1500"
OUT_DIR = "/data/sft_injection_ablation_v1_20260801/trace_transfer"


@app.function(image=image, cpu=8.0, memory=64 * 1024, timeout=3600 * 2,
              volumes={"/rl": rl_vol, "/data": data_vol})
def harvest() -> str:
    import pandas as pd

    rl_vol.reload()
    seen: set[tuple[str, str]] = set()
    records = []
    stats = {"rollout_files": 0, "rows": 0, "positives": 0,
             "rejected_quality": 0, "dup_dropped": 0}
    files = sorted(Path(f"{RUN}/rollouts/training").glob("rollout_*.jsonl"),
                   key=lambda p: int(p.stem.split("_")[1]))
    stats["rollout_files"] = len(files)
    for f in files:
        with open(f) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stats["rows"] += 1
                rw = d.get("reward")
                if isinstance(rw, str):
                    try:
                        rw = eval(rw)
                    except Exception:
                        rw = {}
                score = float(rw.get("score", 0.0)) if isinstance(rw, dict) else float(rw or 0)
                if score != 1.0:
                    continue
                stats["positives"] += 1
                out = d.get("output") or ""
                if out.count("</T>") != 1 or "<call_env>" not in out \
                        or int(d.get("model_token_count", 0)) > 2560:
                    stats["rejected_quality"] += 1
                    continue
                prompt = str(d.get("input") or "")
                key = (prompt, out)
                if key in seen:
                    stats["dup_dropped"] += 1
                    continue
                seen.add(key)
                if not prompt.rstrip().endswith("<T>"):
                    continue
                records.append({
                    "pgn": prompt.rstrip()[: -len("<T>")].strip(),
                    "resp": "<T> " + out.strip(),
                    "rollout_step": int(d.get("rollout_id", d.get("step", -1))),
                })
    out_df = pd.DataFrame(records)
    stats["final_rows"] = len(out_df)
    path = Path(OUT_DIR) / "traces_roll.parquet"
    out_df.to_parquet(path, index=False)
    stats["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    stats["source"] = "training rollouts of p1w1-band-lr1e4-rl1500 (no extra generation)"
    (Path(OUT_DIR) / "traces_summary_roll.json").write_text(json.dumps(stats, indent=2))
    data_vol.commit()
    return json.dumps(stats)


@app.local_entrypoint()
def main() -> None:
    print("RESULT::" + harvest.remote())
