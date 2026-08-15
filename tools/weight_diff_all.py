"""Weight change of every RL run: latest/endpoint checkpoint vs its RL-start checkpoint."""
import json

import modal

app = modal.App("chess-weight-diff-all")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy", "safetensors", "torch==2.9.0")
ckpt_vol = modal.Volume.from_name("rl-reasoning-checkpoints", create_if_missing=False)

BASE = "/checkpoints/interleave_50m"
PT = f"{BASE}/pretrain/sft_injection_ablation_v1_20260801"
RL = f"{BASE}/rl_hf"

PAIRS = [
    ("RL(all,1e-5,1500) from 5B", f"{RL}/e1-u-rl1500-s1500", f"{PT}/p1w1/final", 1500),
    ("RL(filter,1e-5,1500) from 5B", f"{RL}/e1-d-rl1500-s1500", f"{PT}/p1w1/final", 1500),
    ("RL(solv,1e-5,1500) from 5B", f"{RL}/p1w1-band-rl1500-s1500", f"{PT}/p1w1/final", 1500),
    ("RL(solv,1e-4,1500) from 5B", f"{RL}/p1w1-band-lr1e4-rl1500-s1500", f"{PT}/p1w1/final", 1500),
    ("RL(all,1e-5,3000) from 10B", f"{RL}/e2-u-rl3000-s3000", f"{PT}/e2w1/final", 3000),
    ("RL(filter,1e-5,3000) from 10B", f"{RL}/e2-d-rl3000-s3000", f"{PT}/e2w1/final", 3000),
    ("RL(all,1e-5,3000) from 10B2r", f"{RL}/e3-u-rl3000-s3000", f"{PT}/e3p2/final", 3000),
    ("RL(filter,1e-5,3000) from 10B2r", f"{RL}/e3-d-rl3000-s3000", f"{PT}/e3p2/final", 3000),
    ("RL(all,1e-5,1500) from SFT-stage", f"{RL}/b2h-u-rl1500-s1500", f"{PT}/b2h/final", 1500),
]


@app.function(image=image, cpu=8.0, memory=48 * 1024, timeout=2400,
              volumes={"/checkpoints": ckpt_vol})
def diff() -> str:
    import torch
    from safetensors.torch import load_file

    ckpt_vol.reload()
    cache: dict[str, dict] = {}

    def load(path):
        if path not in cache:
            cache[path] = {k: v.to(torch.float64)
                           for k, v in load_file(f"{path}/model.safetensors").items()}
        return cache[path]

    out = []
    for label, rl_path, start_path, steps in PAIRS:
        try:
            a, b = load(rl_path), load(start_path)
        except Exception as e:
            out.append({"run": label, "error": str(e)[:100]})
            continue
        num = den = dot = na = nb = 0.0
        changed = total = 0
        for k in b:
            x, y = a[k], b[k]
            d = x - y
            num += float((d * d).sum())
            den += float((y * y).sum())
            dot += float((x * y).sum())
            na += float((x * x).sum())
            nb += float((y * y).sum())
            changed += int((d != 0).sum())
            total += d.numel()
        out.append({
            "run": label, "steps": steps,
            "relative_l2_pct": round(100 * (num ** 0.5) / (den ** 0.5), 4),
            "cosine": round(dot / ((na ** 0.5) * (nb ** 0.5)), 7),
            "params_changed": changed,
            "params_total": total,
            "changed_pct": round(100 * changed / total, 2),
        })
        # free the RL checkpoint, keep start checkpoints (reused)
        cache.pop(rl_path, None)
    return json.dumps(out)


@app.local_entrypoint()
def main() -> None:
    print("RESULT::" + diff.remote())
