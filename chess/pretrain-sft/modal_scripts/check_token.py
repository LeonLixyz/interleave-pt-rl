import modal, os

image = modal.Image.debian_slim().pip_install("huggingface-hub>=0.28.0")
app = modal.App("test-hf-token", image=image, secrets=[modal.Secret.from_name("huggingface-secret")])

@app.function()
def check():
    token = os.environ.get("HF_TOKEN", "NOT SET")
    print(f"HF_TOKEN: {token[:10]}...{token[-5:]}")
    print(f"Length: {len(token)}")

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    info = api.whoami()
    print(f"Logged in as: {info.get('name', 'unknown')}")

    # Try listing models in chess-pre-to-post org
    try:
        models = list(api.list_models(author="chess-pre-to-post"))
        print(f"Models in chess-pre-to-post: {len(models)}")
        for m in models:
            print(f"  - {m.id}")
    except Exception as e:
        print(f"Error listing models: {e}")

    # Also check datasets
    try:
        datasets = list(api.list_datasets(author="chess-pre-to-post"))
        print(f"\nDatasets in chess-pre-to-post: {len(datasets)}")
        for d in datasets:
            print(f"  - {d.id}")
    except Exception as e:
        print(f"Error listing datasets: {e}")

    # Try direct repo info
    for rtype in ["model", "dataset"]:
        try:
            info = api.repo_info("chess-pre-to-post/pretrain_v1_20b", repo_type=rtype)
            print(f"\nFound as {rtype}: {info.id}")
        except Exception as e:
            print(f"pretrain_v1_20b as {rtype}: {e}")

@app.local_entrypoint()
def main():
    check.remote()
