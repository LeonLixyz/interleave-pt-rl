"""Eval-loss evaluator: held-out pretraining CE + SFT-half CE + uniform mix.

PT held-out: deterministic shards from pretrain_v1_20b that the frozen 10B
selection never touched, packed into 3,073-token windows exactly like training.
SFT: fixed probe rows from the v2r1 SFT cache, split into the P1 half (negative
codes in legs/p1/order.npy) and the P2 half (complement). For P1-stage models
the P2 half is genuinely held-out; for full-exposure models both halves were
seen exactly once during training.
"""
import json

import modal

app = modal.App("chess-eval-loss")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.9.0", "transformers==4.57.0", "numpy", "safetensors", "accelerate",
    "chess",
)
data_vol = modal.Volume.from_name("rl-reasoning-training-data", create_if_missing=False)
ckpt_vol = modal.Volume.from_name("rl-reasoning-checkpoints", create_if_missing=False)

V2R1 = "/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate"
SOURCE = "/data/pretrain_v1_20b"
CKPT_ROOT = "/checkpoints/interleave_50m/pretrain/sft_injection_ablation_v1_20260801"
SEQ = 3072
IGNORE = -100
HELDOUT_SEED = 20260804
N_PT_WINDOWS = 2048
N_SFT_ROWS_PER_HALF = 2048
# training mixture share of SFT supervised targets (52,482,753 / (1e10 + 52,482,753))
SFT_SHARE = 52_482_753 / (10_000_000_000 + 52_482_753)


@app.function(
    gpu="H200:1", cpu=16.0, memory=64 * 1024, timeout=3600, image=image,
    volumes={"/data": data_vol, "/checkpoints": ckpt_vol},
)
def eval_loss(ckpt_subpath: str, tag: str) -> str:
    from pathlib import Path

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data_vol.reload()
    ckpt_vol.reload()

    ckpt = Path(CKPT_ROOT) / ckpt_subpath
    tok = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt), torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()

    # ---- held-out pretraining windows -------------------------------------
    selection = json.load(open(f"{V2R1}/pretrain_selection.json"))

    def selected_shard_ids(obj) -> set[int]:
        ids: set[int] = set()

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k in ("shard", "shard_id", "shard_index") and isinstance(v, int):
                        ids.add(v)
                    elif k in ("file", "path", "shard_name") and isinstance(v, str) and "raw." in v:
                        ids.add(int(v.split("raw.")[1].split(".npy")[0]))
                    else:
                        walk(v)
            elif isinstance(o, list):
                for x in o:
                    walk(x)

        walk(obj)
        return ids

    used = selected_shard_ids(selection)
    all_ids = sorted(
        int(p.name.split(".")[1]) for p in Path(SOURCE).glob("raw.*.npy")
    )
    unused = [i for i in all_ids if i not in used]
    if not unused:
        raise RuntimeError("no held-out shards found; selection parse failed")
    rng = np.random.default_rng(HELDOUT_SEED)
    order = rng.permutation(len(unused))
    need = N_PT_WINDOWS * SEQ + 1
    stream: list[np.ndarray] = []
    total = 0
    picked = []
    for j in order:
        sid = unused[int(j)]
        arr = np.load(f"{SOURCE}/raw.{sid:04d}.npy") if Path(
            f"{SOURCE}/raw.{sid:04d}.npy").exists() else np.load(f"{SOURCE}/raw.{sid}.npy")
        stream.append(arr.astype(np.int64))
        picked.append(sid)
        total += len(arr)
        if total >= need:
            break
    tokens = np.concatenate(stream)[:need]
    windows = tokens[: N_PT_WINDOWS * SEQ + 1]

    pt_sum = 0.0
    pt_cnt = 0
    bs = 16
    with torch.no_grad():
        for i in range(0, N_PT_WINDOWS, bs):
            n = min(bs, N_PT_WINDOWS - i)
            ids = np.stack([
                windows[(i + k) * SEQ: (i + k) * SEQ + SEQ + 1] for k in range(n)
            ])
            x = torch.from_numpy(ids[:, :SEQ]).cuda()
            y = torch.from_numpy(ids[:, 1: SEQ + 1]).cuda()
            logits = model(input_ids=x).logits.float()
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            )
            pt_sum += ce.item()
            pt_cnt += y.numel()
    pt_loss = pt_sum / pt_cnt

    # ---- SFT probe rows -----------------------------------------------------
    offsets = np.load(f"{V2R1}/sft_cache/offsets.npy")
    input_ids = np.memmap(f"{V2R1}/sft_cache/input_ids.i32", dtype="<i4", mode="r")
    labels = np.memmap(f"{V2R1}/sft_cache/labels.i32", dtype="<i4", mode="r")
    order_p1 = np.load(f"{V2R1}/legs/p1/order.npy")
    neg = order_p1[order_p1 < 0]
    p1_rows = set((-neg - 1).tolist())
    all_rows = set(range(len(offsets) - 1))
    p2_rows = sorted(all_rows - p1_rows)
    p1_rows = sorted(p1_rows)

    def sft_ce(rows: list[int]) -> tuple[float, int]:
        rng2 = np.random.default_rng(HELDOUT_SEED + 1)
        probe = sorted(rng2.choice(rows, size=min(N_SFT_ROWS_PER_HALF, len(rows)),
                                   replace=False).tolist())
        s = 0.0
        c = 0
        with torch.no_grad():
            for i in range(0, len(probe), 16):
                batch = probe[i: i + 16]
                seqs = []
                labs = []
                for r in batch:
                    a, b = int(offsets[r]), int(offsets[r + 1])
                    seqs.append(np.asarray(input_ids[a:b], dtype=np.int64))
                    labs.append(np.asarray(labels[a:b], dtype=np.int64))
                L = max(len(q) for q in seqs)
                x = np.zeros((len(batch), L), dtype=np.int64)
                y = np.full((len(batch), L), IGNORE, dtype=np.int64)
                for k, (q, l) in enumerate(zip(seqs, labs)):
                    x[k, : len(q)] = q
                    y[k, : len(l)] = l
                xt = torch.from_numpy(x).cuda()
                yt = torch.from_numpy(y).cuda()
                logits = model(input_ids=xt).logits.float()
                ce = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), yt.reshape(-1),
                    reduction="sum", ignore_index=IGNORE,
                )
                s += ce.item()
                c += int((yt != IGNORE).sum().item())
        return s / c, c

    sft_p1_loss, n1 = sft_ce(p1_rows)
    sft_p2_loss, n2 = sft_ce(p2_rows)
    sft_loss = (sft_p1_loss * n1 + sft_p2_loss * n2) / (n1 + n2)
    uniform = (1 - SFT_SHARE) * pt_loss + SFT_SHARE * sft_loss

    out = {
        "tag": tag,
        "checkpoint": str(ckpt),
        "pt_heldout_loss": round(pt_loss, 5),
        "pt_windows": N_PT_WINDOWS,
        "pt_heldout_shards": picked,
        "sft_p1half_loss": round(sft_p1_loss, 5),
        "sft_p2half_loss": round(sft_p2_loss, 5),
        "sft_loss": round(sft_loss, 5),
        "sft_tokens": n1 + n2,
        "uniform_loss": round(uniform, 5),
        "sft_share": SFT_SHARE,
        "p1_rows": len(p1_rows),
        "p2_rows": len(p2_rows),
    }
    print(json.dumps(out))
    return json.dumps(out)


@app.local_entrypoint()
def main(targets: str = "p1w1/final:P1W1,e2w1/final:E2W1") -> None:
    calls = {}
    for t in targets.split(","):
        sub, tag = t.strip().split(":")
        calls[tag] = eval_loss.spawn(sub, tag)
    for tag, c in calls.items():
        print(c.get())
