"""Merge all completed stride-8 RL eval points into rl_pass_curve_full.json."""
import json
import subprocess
from math import comb
from pathlib import Path

S = Path("/private/tmp/claude-501/-Users-leonli66-Desktop-Research-RL-Chess-RL/cf172269-6c26-448b-8a10-904c1e927526/scratchpad")


def run_key(run: str) -> str:
    if run.endswith("-leg2"):
        return run.replace("-rl1500-leg2", "-leg2")
    return run.replace("-rl1500", "").replace("-rl3000", "")


def main() -> None:
    ledger = [l.split() for l in (S / "rl_eval_ledger.txt").read_text().splitlines() if l.strip()]
    # skip full-eval (non-stride) endpoint entries
    skip = {("e1-u-rl1500", "1500"), ("e2-u-rl3000", "3000"), ("e3-u-rl3000", "3000")}
    prev = {}
    cache_file = S / "rl_pass_curve_full.json"
    if cache_file.exists():
        for r in json.loads(cache_file.read_text()):
            prev[(r["run"], r["step"])] = r

    out = []
    missing = 0
    for run, step in ledger:
        if (run, step) in skip:
            continue
        key = (run_key(run), int(step))
        if key in prev:
            out.append(prev[key])
            continue
        arm = f"RL-{run}-s{int(step):04d}"
        hist = {}
        fmt = rows = ok = 0
        for sh in range(4):
            loc = S / "tmp_pt.json"
            r = subprocess.run(
                ["modal", "volume", "get", "chess-rl-eval-results-r6",
                 f"ablation_pass16_clean_v2_bos/{arm}/n16/eval_train_v4_balanced_shard{sh}_stride8/success.json",
                 str(loc), "--force"], capture_output=True)
            if r.returncode != 0:
                break
            d = json.loads(loc.read_text())
            for w, c in d["wins_histogram"].items():
                hist[int(w)] = hist.get(int(w), 0) + int(c)
            fmt += round(d["format_rate"] * d["rows"])
            rows += d["rows"]
            ok += 1
        if ok < 4:
            missing += 1
            continue
        n = 16
        prompts = sum(hist.values())
        pak = lambda k: sum(c * (1 - comb(n - w, k) / comb(n, k)) for w, c in hist.items()) / prompts
        out.append({"run": key[0], "step": key[1], "pass1": round(pak(1), 4),
                    "pass8": round(pak(8), 4), "pass16": round(pak(16), 4),
                    "format": round(fmt / rows, 4)})
    print(f"pass-curve points: {len(out)} complete, {missing} pending")
    cache_file.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
