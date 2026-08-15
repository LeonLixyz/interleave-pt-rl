"""
Aggregate per-step training-rollout stats for RL runs (reward + exploration),
reconstructed exactly from the rollout dumps at
  /checkpoints/rl/<run>/rollouts/training/<step>.jsonl  (1024 rows = 128 prompts x 8)

Per step we compute:
  - reward_mean: mean score (== critic/score/mean verl logged)
  - zero_adv_frac: fraction of prompt-groups where all 8 samples share one score
      (GRPO advantage == 0 for the whole group -> zero learning signal)
  - uniq_frac: mean over groups of (# distinct outputs / group size)  [diversity]
  - len_mean / len_std: output length (chars) mean / within-group std mean

Usage:
  modal run rollout_stats.py --runs from-armBsmall,from-step20000 --stride 5
"""
import json
from collections import defaultdict
from pathlib import Path

import modal

CHECKPOINT_VOLUME_NAME = "olmo-core-checkpoints-v2"
CHECKPOINT_MOUNT = "/checkpoints"

image = modal.Image.debian_slim(python_version="3.11")
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2)
app = modal.App("rollout-stats", image=image)


@app.function(timeout=60 * 60, volumes={CHECKPOINT_MOUNT: checkpoint_volume}, cpu=8)
def stats_for_run(run: str, stride: int = 5) -> dict:
    import statistics
    checkpoint_volume.reload()
    d = Path(f"{CHECKPOINT_MOUNT}/rl/math-1b-rl-deepscaler-{run}/rollouts/training")
    steps = sorted(int(p.stem) for p in d.glob("*.jsonl") if p.stem.isdigit())
    picked = [s for s in steps if s % stride == 0 or s == steps[-1] or s <= 3]
    out = []
    for s in picked:
        try:
            rows = [json.loads(l) for l in open(d / f"{s}.jsonl")]
        except Exception:
            continue
        if not rows:
            continue
        groups = defaultdict(list)
        for r in rows:
            groups[hash(r["input"])].append(r)
        scores = [r["score"] for r in rows]
        all0 = all1 = 0
        lstds = []
        for g in groups.values():
            gs = [x["score"] for x in g]
            if max(gs) == 0:
                all0 += 1          # unsolved: too hard, zero gradient
            elif min(gs) >= 1:
                all1 += 1          # saturated: fully solved, zero gradient
            if len(g) > 1:
                lens = [len(x["output"]) for x in g]
                lstds.append(statistics.pstdev(lens))
        n = len(groups)
        out.append({
            "step": s,
            "reward_mean": round(sum(scores) / len(scores), 4),
            "all0_frac": round(all0 / n, 4),
            "all1_frac": round(all1 / n, 4),
            "zero_adv_frac": round((all0 + all1) / n, 4),
            "len_std_mean": round(sum(lstds) / len(lstds), 1) if lstds else None,
            "n_groups": n,
        })
        if len(out) % 50 == 0:
            print(f"[{run}] {len(out)} steps done (latest {s})", flush=True)
    print(f"[{run}] DONE: {len(out)} steps, range {out[0]['step']}..{out[-1]['step']}" if out else f"[{run}] EMPTY", flush=True)
    return {"run": run, "stats": out}


@app.local_entrypoint()
def main(runs: str = "from-armBsmall,from-step20000", stride: int = 5):
    handles = [(r, stats_for_run.spawn(run=r, stride=stride)) for r in runs.split(",")]
    results = {}
    for r, h in handles:
        results[r] = h.get()
        print(f"{r}: {len(results[r]['stats'])} points")
    outp = Path("/tmp/rollout_stats.json")
    outp.write_text(json.dumps(results))
    print(f"wrote {outp}")
