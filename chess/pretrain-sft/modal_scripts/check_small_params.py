"""Param counts for candidate 5M / 10M small-model configs.

n_head must be in {2, 4}. head_dim=128 fixed. kv_heads <= n_head and divides n_head.
"""
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.6.0", "transformers>=4.50.0"
).run_commands(
    'python -c "from transformers import AutoConfig; AutoConfig.from_pretrained(\'Qwen/Qwen3-0.6B\')"'
)
app = modal.App("check-small-params", image=image)

CANDIDATES = [
    # name,                      n_layer, n_head, kv, hidden, inter, head_dim
    ("5M_A_L6_h256_nh2",          6,  2, 2,  256,  768, 128),
    ("5M_B_L6_h256_nh4",          6,  4, 4,  256,  768, 128),
    ("5M_C_L4_h256_nh4",          4,  4, 4,  256,  768, 128),
    ("5M_D_L8_h256_nh2",          8,  2, 2,  256,  768, 128),
    # 10M candidates
    ("10M_A_L6_h384_nh4_i1024",   6,  4, 4,  384, 1024, 128),
    ("10M_B_L6_h384_nh4_i1152",   6,  4, 4,  384, 1152, 128),
    ("10M_C_L6_h384_nh2_i1152",   6,  2, 2,  384, 1152, 128),
    ("10M_D_L12_h256_nh2",       12,  2, 2,  256,  768, 128),
    ("10M_E_L8_h384_nh4_i1024",   8,  4, 4,  384, 1024, 128),
    ("10M_F_L6_h448_nh4_i1280",   6,  4, 4,  448, 1280, 128),
    ("10M_G_L6_h512_nh4_i1280",   6,  4, 4,  512, 1280, 128),
]

@app.function()
def check():
    from transformers import AutoModelForCausalLM, AutoConfig

    print(f"{'name':<28s} {'L':>3s} {'nh':>3s} {'kv':>3s} {'hidden':>6s} {'inter':>6s} {'hd':>4s} {'params':>10s}")
    print("-" * 75)

    for name, nl, nh, kv, hidden, inter, hd in CANDIDATES:
        hf = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        hf.num_hidden_layers = nl
        hf.hidden_size = hidden
        hf.num_attention_heads = nh
        hf.num_key_value_heads = kv
        hf.intermediate_size = inter
        hf.head_dim = hd
        hf.vocab_size = 82
        for attr in ["layer_type", "layer_types"]:
            if hasattr(hf, attr):
                setattr(hf, attr, ["dense"] * nl)

        model = AutoModelForCausalLM.from_config(hf, trust_remote_code=True)
        n = sum(p.numel() for p in model.parameters())
        print(f"{name:<28s} {nl:>3d} {nh:>3d} {kv:>3d} {hidden:>6d} {inter:>6d} {hd:>4d} {n/1e6:>8.2f}M")
        del model


@app.local_entrypoint()
def main():
    check.remote()
