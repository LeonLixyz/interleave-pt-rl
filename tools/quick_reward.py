import json
import modal
from pathlib import Path

app = modal.App("quick-reward")
ckpt_vol = modal.Volume.from_name("chess-rl-miles-checkpoints", create_if_missing=False)
ROOT = "/rl-checkpoints/chess-rl-miles-interleave"


@app.function(volumes={"/rl-checkpoints": ckpt_vol}, timeout=900, cpu=4.0)
def tail_reward(run: str, n_last: int = 15) -> str:
    ckpt_vol.reload()
    files = sorted(Path(f"{ROOT}/{run}/rollouts/training").glob("rollout_*.jsonl"),
                   key=lambda p: int(p.stem.split("_")[1]))[-n_last:]
    vals = []
    for f in files:
        s = n = 0
        for line in open(f):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            rw = d.get("reward")
            if isinstance(rw, str):
                try: rw = eval(rw)
                except Exception: rw = {}
            s += float(rw.get("score", 0)) if isinstance(rw, dict) else float(rw or 0)
            n += 1
        if n: vals.append((int(f.stem.split("_")[1]), s / n))
    return json.dumps({"run": run, "last": vals[-1] if vals else None,
                       "mean_last": round(sum(v for _, v in vals) / len(vals), 4) if vals else None})


@app.local_entrypoint()
def main(runs: str = "") -> None:
    runs = [r.strip() for r in runs.split(",") if r.strip()] or [
        "k16band-lr1e4-rl1500", "e2w1band-lr1e4-rl1500",
        "e3p2band-lr1e4-rl1500", "rollband-lr1e4-rl1500"]
    calls = [tail_reward.spawn(r) for r in runs]
    for c in calls:
        print("RESULT::" + c.get())
