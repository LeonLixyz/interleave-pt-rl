"""Check param counts for all model sizes with kv_heads=4."""
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.6.0", "transformers>=4.50.0"
).run_commands(
    'python -c "from transformers import AutoConfig; AutoConfig.from_pretrained(\'Qwen/Qwen3-0.6B\')"'
)
app = modal.App("check-params", image=image)

MODELS = {
    "50M":   {"n_layer": 12, "n_head": 8,  "kv": 4, "hidden": 512,  "inter": 1536, "head_dim": 128},
    "110M":  {"n_layer": 12, "n_head": 12, "kv": 4, "hidden": 768,  "inter": 2304, "head_dim": 128},
    "200M":  {"n_layer": 24, "n_head": 12, "kv": 4, "hidden": 768,  "inter": 2304, "head_dim": 128},
    "410M":  {"n_layer": 28, "n_head": 16, "kv": 4, "hidden": 1024, "inter": 3072, "head_dim": 128},
    "670M":  {"n_layer": 30, "n_head": 20, "kv": 4, "hidden": 1280, "inter": 3840, "head_dim": 128},
    "1B":    {"n_layer": 32, "n_head": 24, "kv": 4, "hidden": 1536, "inter": 4608, "head_dim": 128},
}

@app.function()
def check():
    from transformers import AutoModelForCausalLM, AutoConfig

    print(f"{'':>12s} {'50M':>8s} {'110M':>8s} {'200M':>8s} {'410M':>8s} {'670M':>8s} {'1B':>8s}")
    print("-" * 68)

    rows = {}
    for name, c in MODELS.items():
        hf = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        hf.num_hidden_layers = c["n_layer"]
        hf.hidden_size = c["hidden"]
        hf.num_attention_heads = c["n_head"]
        hf.num_key_value_heads = c["kv"]
        hf.intermediate_size = c["inter"]
        hf.vocab_size = 81
        if "head_dim" in c:
            hf.head_dim = c["head_dim"]
        for attr in ['layer_type', 'layer_types']:
            if hasattr(hf, attr):
                setattr(hf, attr, ["dense"] * c["n_layer"])

        model = AutoModelForCausalLM.from_config(hf, trust_remote_code=True)
        n = sum(p.numel() for p in model.parameters())
        rows[name] = {"params": n, **c}
        del model

    # Print table matching screenshot format
    keys = ["50M", "110M", "200M", "410M", "670M", "1B"]

    def row(label, fn):
        vals = [fn(rows[k]) for k in keys]
        print(f"{label:>12s} " + " ".join(f"{v:>8s}" for v in vals))

    row("hidden_size", lambda r: str(r["hidden"]))
    row("n_head",      lambda r: str(r["n_head"]))
    row("kv_heads",    lambda r: str(r["kv"]))
    row("head_dim",    lambda r: str(r["hidden"] // r["n_head"]))
    row("n_layer",     lambda r: str(r["n_layer"]))
    row("intermediate",lambda r: str(r["inter"]))
    row("params",      lambda r: f"{r['params']/1e6:.1f}M")

@app.local_entrypoint()
def main():
    check.remote()
