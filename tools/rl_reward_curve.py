"""Compute per-rollout mean reward curves for the E1 RL runs, server-side."""
import json

import modal

app = modal.App("rl-reward-curve")
ckpt_vol = modal.Volume.from_name("chess-rl-miles-checkpoints", create_if_missing=False)
ROOT = "/rl-checkpoints/chess-rl-miles-interleave"


@app.function(volumes={"/rl-checkpoints": ckpt_vol}, timeout=3600, cpu=8.0)
def curve(run_name: str, start_after: int = -1) -> str:
    from pathlib import Path

    ckpt_vol.reload()
    out = []
    files = sorted(
        (p for p in Path(f"{ROOT}/{run_name}/rollouts/training").glob("rollout_*.jsonl")
         if int(p.stem.split("_")[1]) > start_after),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    for f in files:
        scores = []
        fmt = 0
        n = 0
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                rw = r.get("reward")
                if isinstance(rw, str):
                    try:
                        rw = eval(rw)  # dict repr written by the reward fn
                    except Exception:
                        rw = {}
                s = float(rw.get("score", 0.0)) if isinstance(rw, dict) else float(rw or 0.0)
                scores.append(s)
                if isinstance(rw, dict) and rw.get("extracted_moves"):
                    fmt += 1
        if n:
            out.append({
                "rollout": int(f.stem.split("_")[1]),
                "mean_reward": sum(scores) / n,
                "format_rate": fmt / n,
                "rows": n,
            })
    return json.dumps({"run": run_name, "points": out})


@app.local_entrypoint()
def main(
    runs: str = ("e1-u-rl1500,e1-d-rl1500,e2-u-rl3000,e2-d-rl3000,"
                 "e3-u-rl3000,e3-d-rl3000,e1-u-rl1500-leg2,b2h-u-rl1500,p1w1-band-rl1500,p1w1-band-lr1e4-rl1500,e1-d-rl1500-leg2"),
    cache: str = "",
) -> None:
    import pathlib
    cached: dict[str, list] = {}
    if cache and pathlib.Path(cache).exists():
        for line in open(cache):
            if not line.strip().startswith("{"):
                continue
            d = json.loads(line)
            cached[d["run"]] = d["points"]
    calls = {}
    for run in runs.split(","):
        run = run.strip()
        start_after = max((p["rollout"] for p in cached.get(run, [])), default=-1)
        calls[run] = curve.spawn(run, start_after)
    for run, c in calls.items():
        new = json.loads(c.get())["points"]
        merged = cached.get(run, []) + new
        print(json.dumps({"run": run, "points": merged}))
