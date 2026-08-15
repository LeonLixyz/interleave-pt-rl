"""Quick param count check for different n_head / kv_head combos."""
import modal

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .pip_install("torch>=2.6.0", "transformers>=4.50.0")
    .run_commands('python -c "from transformers import AutoConfig; AutoConfig.from_pretrained(\'Qwen/Qwen3-0.6B\')"')
)

app = modal.App("check-params", image=image)

@app.function(gpu="H200:1", timeout=60*60)
def check():
    from transformers import AutoModelForCausalLM, AutoConfig

    VOCAB = 256

    configs = [
        # Targeting ~800M with kv=4, n_head%4=0, hidden%n_head=0
        ("h=1512,nh=28,kv=4,i=4096,L=28",  {"n_layer": 28, "n_head": 28, "kv": 4, "hidden": 1512, "inter": 4096}),
        ("h=1512,nh=28,kv=4,i=3840,L=28",  {"n_layer": 28, "n_head": 28, "kv": 4, "hidden": 1512, "inter": 3840}),
        ("h=1536,nh=24,kv=4,i=4096,L=28",  {"n_layer": 28, "n_head": 24, "kv": 4, "hidden": 1536, "inter": 4096}),
        ("h=1536,nh=24,kv=4,i=3840,L=28",  {"n_layer": 28, "n_head": 24, "kv": 4, "hidden": 1536, "inter": 3840}),
        ("h=1400,nh=28,kv=4,i=3584,L=28",  {"n_layer": 28, "n_head": 28, "kv": 4, "hidden": 1400, "inter": 3584}),
        ("h=1456,nh=28,kv=4,i=3840,L=28",  {"n_layer": 28, "n_head": 28, "kv": 4, "hidden": 1456, "inter": 3840}),
        ("h=1344,nh=28,kv=4,i=3584,L=28",  {"n_layer": 28, "n_head": 28, "kv": 4, "hidden": 1344, "inter": 3584}),
    ]

    print(f"{'Config':<35s}  {'Params':>12s}  {'head_dim':>8s}")
    print("-" * 65)

    for name, c in configs:
        hf_config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        hf_config.num_hidden_layers = c["n_layer"]
        hf_config.hidden_size = c["hidden"]
        hf_config.num_attention_heads = c["n_head"]
        hf_config.num_key_value_heads = c["kv"]
        hf_config.intermediate_size = c["inter"]
        hf_config.vocab_size = VOCAB

        model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)
        n = sum(p.numel() for p in model.parameters())
        head_dim = c["hidden"] // c["n_head"]
        print(f"{name:<35s}  {n/1e6:>10.1f}M  {head_dim:>8d}")
        del model

@app.local_entrypoint()
def main():
    check.remote()
