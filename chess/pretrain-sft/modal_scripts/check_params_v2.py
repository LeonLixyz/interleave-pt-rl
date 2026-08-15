"""Check param counts with and without explicit head_dim=128."""
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.6.0", "transformers>=4.50.0"
).run_commands(
    'python -c "from transformers import AutoConfig; AutoConfig.from_pretrained(\'Qwen/Qwen3-0.6B\')"'
)
app = modal.App("check-params-v2", image=image)

MODELS = [
    ("50M",   12, 8,  512,  1536),
    ("100M",  12, 12, 768,  2304),
    ("200M",  24, 12, 768,  2304),
    ("410M",  28, 16, 1024, 3072),
    ("680M",  30, 20, 1280, 3840),
    ("1B",    32, 24, 1536, 4608),
]

@app.function()
def check():
    from transformers import AutoModelForCausalLM, AutoConfig

    print(f"{'Model':<6s} {'head_dim':>8s} {'Q_proj':>10s} {'K_proj':>10s} {'params':>12s}")
    print("-" * 52)

    for name, nl, nh, hidden, inter in MODELS:
        for hd_override in [None, 128]:
            hf = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
            hf.num_hidden_layers = nl
            hf.hidden_size = hidden
            hf.num_attention_heads = nh
            hf.num_key_value_heads = 4
            hf.intermediate_size = inter
            hf.vocab_size = 81
            if hd_override:
                hf.head_dim = hd_override
            for attr in ['layer_type', 'layer_types']:
                if hasattr(hf, attr):
                    setattr(hf, attr, ["dense"] * nl)

            model = AutoModelForCausalLM.from_config(hf, trust_remote_code=True)
            n = sum(p.numel() for p in model.parameters())

            # Check Q proj size
            q_size = "?"
            k_size = "?"
            try:
                q = model.model.layers[0].self_attn.q_proj
                k = model.model.layers[0].self_attn.k_proj
                q_size = f"{q.in_features}x{q.out_features}"
                k_size = f"{k.in_features}x{k.out_features}"
            except:
                pass

            actual_hd = hf.head_dim if hasattr(hf, 'head_dim') and hf.head_dim else hidden // nh
            label = f"hd={actual_hd}"
            print(f"{name:<6s} {label:>8s} {q_size:>10s} {k_size:>10s} {n/1e6:>10.1f}M")
            del model
        print()

@app.local_entrypoint()
def main():
    check.remote()
