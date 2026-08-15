"""Peek at the extracted untrained corpus — verify it's real text, not garbage.

For each source, opens the first .npy file, decodes a few random 4096-token
chunks back to text via the dolma2 tokenizer, and prints them. Also reports
shape + sizes so you can sanity-check the extraction.

Usage:
    modal run peek_untrained.py
"""

from __future__ import annotations

import modal

from common import (
    UNTRAINED_MOUNT,
    hf_image_base,
    untrained_volume,
)


def _img() -> modal.Image:
    return (
        hf_image_base()
        .pip_install("torch==2.6.0", "transformers>=4.46")
        .add_local_dir(
            "/Users/leonli66/Desktop/Research/RL/Chess RL/OLMo-core",
            remote_path="/root/OLMo-core",
            copy=True,
            ignore=[".git", ".git/**", ".venv/**", "__pycache__/**", "build/**"],
        )
        .run_commands("cd /root/OLMo-core && pip install -e .")
        .add_local_python_source("common")
    )


app = modal.App("peek-untrained", image=_img())


@app.function(
    volumes={UNTRAINED_MOUNT: untrained_volume},
    timeout=600,
    cpu=2.0,
    memory=16 * 1024,
)
def peek(sequence_length: int = 4096, n_samples: int = 2, max_chars: int = 600) -> None:
    import json
    from pathlib import Path
    import numpy as np
    from olmo_core.data import TokenizerConfig

    untrained_volume.reload()
    tk_cfg = TokenizerConfig.dolma2()
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(tk_cfg.identifier)

    manifest_path = Path(UNTRAINED_MOUNT) / "untrained_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "files"}
                          for k, v in manifest["per_source"].items()}, indent=2))
        print()

    for label in ("math_3", "math_4plus_MIND", "dolma3"):
        src_dir = Path(UNTRAINED_MOUNT) / label
        files = sorted(src_dir.glob("part_*.npy"))
        if not files:
            print(f"=== {label}: no files ===\n")
            continue
        f0 = files[0]
        size_bytes = f0.stat().st_size
        n_inst = size_bytes // (sequence_length * 4)
        mm = np.memmap(f0, dtype=np.uint32, mode="r", shape=(n_inst, sequence_length))
        total_files = len(files)
        total_bytes = sum(p.stat().st_size for p in files)
        print(f"=== {label} ===")
        print(f"  files: {total_files}, total: {total_bytes/1e9:.2f} GB "
              f"({total_bytes//(sequence_length*4):,} instances)")
        print(f"  first file: {f0.name}, {n_inst:,} instances x {sequence_length} tokens")

        rng = np.random.default_rng(seed=42)
        idxs = rng.choice(n_inst, size=min(n_samples, n_inst), replace=False)
        for k, i in enumerate(idxs):
            ids = mm[int(i)].tolist()
            text = tk.decode(ids, skip_special_tokens=False)
            snippet = text[:max_chars].replace("\n", " ⏎ ")
            print(f"\n  --- sample {k+1} (instance {int(i):,}) ---")
            print(f"  first 8 ids: {ids[:8]}")
            print(f"  text[:{max_chars}]: {snippet}{'…' if len(text) > max_chars else ''}")
        print()


@app.local_entrypoint()
def main() -> None:
    peek.remote()
