"""Download training loss curve via Modal (uses wandb-secret on Modal, no local login needed).

Outputs files in CWD locally via Function.local() returning bytes.

Usage:
    modal run download_wandb_curve_modal.py
"""

from __future__ import annotations

import modal

from common import hf_image_base


def _img() -> modal.Image:
    return (
        hf_image_base()
        .pip_install("wandb>=0.18", "matplotlib>=3.8", "pandas>=2.2")
        .add_local_python_source("common")
    )


app = modal.App("dl-wandb-curve", image=_img())
wandb_secret = modal.Secret.from_name("wandb-secret")


@app.function(secrets=[wandb_secret], timeout=600, cpu=2.0)
def fetch(
    run_path: str = "leon-modal-modal/math-pretraining/nkhcxisa",
    keys: str = "train/CE loss,optim/LR,throughput/device/MFU,train/Z loss",
) -> dict:
    import io
    import wandb
    import pandas as pd
    import matplotlib.pyplot as plt

    api = wandb.Api()
    # If run_path matches multiple resumes by run-name, fetch ALL of them
    # (resumes after timeouts create separate wandb runs sharing the same
    # `name` but different IDs).
    if run_path.count("/") == 2 and "*" not in run_path:
        # Specific run id: fetch all sibling runs with the same `name`.
        primary = api.run(run_path)
        entity, project = primary.entity, primary.project
        runs = api.runs(
            path=f"{entity}/{project}",
            filters={"display_name": primary.name},
            order="+created_at",
        )
        runs = list(runs)
        print(f"[wandb] primary {run_path} → name={primary.name!r}, found {len(runs)} siblings")
    else:
        runs = [api.run(run_path)]
    # Enumerate available keys via the FIRST run.
    run0 = runs[0]
    summary_keys = sorted(list(run0.summary.keys()))
    available_keys = set()
    for i, row in enumerate(run0.scan_history(page_size=10)):
        available_keys.update(row.keys())
        if i >= 10:
            break
    available_keys = sorted(available_keys)

    # Pick a loss key, LR key, MFU key
    def _find(candidates):
        for c in candidates:
            for k in available_keys:
                if c.lower() == k.lower() or c.lower() in k.lower():
                    return k
        return None

    loss_key = _find(["train/CE loss", "train/ce_loss", "CE loss", "train_loss", "loss"])
    lr_key = _find(["optim/LR", "lr", "learning_rate"])
    mfu_key = _find(["throughput/device/MFU", "MFU", "mfu"])
    zloss_key = _find(["train/Z loss", "z_loss"])

    chosen = [k for k in (loss_key, lr_key, mfu_key, zloss_key) if k]
    if not chosen:
        return {
            "error": "no useful keys found",
            "summary_keys": summary_keys,
            "available_keys": available_keys,
        }

    # Concatenate history across ALL resume-sibling runs. Each run uses its
    # OWN _step (relative to that run's start). Each Modal restart resumes
    # from a checkpoint, so its `_step` 0 maps to the global step we
    # resumed at — but we can't recover that from wandb alone reliably.
    # Workaround: dedup by (loss_key value range) — instead, just take ALL
    # rows from ALL runs and emit them with a `run_id` column. We'll renumber
    # later via OLMo-core's checkpoint sequence (which records true global step).
    all_rows = []
    wanted = set(chosen) | {"_step"}
    for r in runs:
        for row in r.scan_history():
            d = {k: row.get(k) for k in wanted}
            d["wandb_run_id"] = r.id
            d["wandb_run_created"] = str(r.created_at)
            all_rows.append(d)
    hist = pd.DataFrame(all_rows)
    print(f"[wandb] concatenated {len(hist)} rows across {len(runs)} runs")

    csv_bytes = hist.to_csv(index=False).encode()

    if not loss_key or loss_key not in hist.columns:
        return {
            "error": f"loss_key={loss_key!r} not in returned columns",
            "columns": list(hist.columns),
            "available_keys": available_keys,
        }
    loss_col = loss_key

    fig, ax = plt.subplots(figsize=(10, 5))
    df = hist.dropna(subset=[loss_col])
    ax.plot(df["_step"], df[loss_col], lw=0.7, color="steelblue", label="CE loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("train/CE loss")
    ax.set_title(f"{run0.name} — training loss")
    ax.grid(alpha=0.3)
    gbs = 512 * 4096
    sec = ax.secondary_xaxis(
        "top",
        functions=(lambda x: x * gbs / 1e9, lambda x: x * 1e9 / gbs),
    )
    sec.set_xlabel("training tokens (B)")
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    png_bytes = buf.getvalue()

    return {
        "csv": csv_bytes,
        "png": png_bytes,
        "n_rows": len(hist),
        "columns": list(hist.columns),
        "loss_key": loss_key,
        "lr_key": lr_key,
        "mfu_key": mfu_key,
        "available_keys": available_keys,
    }


@app.local_entrypoint()
def main(
    run_path: str = "leon-modal-modal/math-pretraining/nkhcxisa",
    out_csv: str = "training_curve.csv",
    out_png: str = "training_curve.png",
) -> None:
    from pathlib import Path
    result = fetch.remote(run_path=run_path)
    if "error" in result:
        print(f"[err] {result['error']}")
        if "available_keys" in result:
            print(f"[available_keys] ({len(result['available_keys'])}):")
            for k in result["available_keys"]:
                print(f"  {k}")
        return
    Path(out_csv).write_bytes(result["csv"])
    Path(out_png).write_bytes(result["png"])
    print(f"[csv] {Path(out_csv).resolve()} ({len(result['csv'])} bytes)")
    print(f"[png] {Path(out_png).resolve()} ({len(result['png'])} bytes)")
    print(f"[meta] rows={result['n_rows']} cols={result['columns']}")
    print(f"[chosen] loss={result['loss_key']} lr={result['lr_key']} mfu={result['mfu_key']}")
