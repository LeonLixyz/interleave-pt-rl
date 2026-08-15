"""verl 0.9 GRPO RL for our 1B SFT models on rl-math-skyeasy25k-omi2.

Adapts pretrain-rl-scaling/olmo-thinking-rl-reward-run.sh for Modal. Runs verl
main_ppo with our 1B-adapted config. Overnight-safe: auto-resume via Modal
retries + verl's `trainer.resume_mode=auto`. No mid-training eval (test_freq
set very high) — pure training focus.

Usage:
    # Fire single RL run:
    modal run --detach rl_train.py::rl_train_fn \
        --base-repo pre-to-post-olmo/math-1b-sft-numinamath-bs512-from-step95368 \
        --anchor 95368

    # Fire the 4-way sweep (step10k, 40k, 80k, 95368):
    modal run --detach rl_train.py::rl_sweep_4
"""

from __future__ import annotations

import modal
from common import (
    CACHE_MOUNT, CACHE_VOLUME_NAME,
    CHECKPOINT_MOUNT, CHECKPOINT_VOLUME_NAME,
    hf_image_base,
)

LOCAL_VERL_DIR = "/Users/leonli66/Desktop/Research/RL/Chess RL/pretrain-rl-scaling/verl-olmo3"
LOCAL_REWARD_FN = "/Users/leonli66/Desktop/Research/RL/Chess RL/pretrain-rl-scaling/reward_function.py"
REMOTE_VERL_DIR = "/root/verl-olmo3"
REMOTE_REWARD_FN = "/root/reward_function.py"


def _img() -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
        )
        .apt_install("build-essential", "git", "curl", "libibverbs-dev", "libibverbs1")
        .pip_install("wheel", "packaging", "ninja", "setuptools")
        .pip_install("torch==2.8.0")
        # Pin hub FIRST — otherwise pip backtracks through hundreds of versions
        # looking for the hf_xet extra when a later dep requests it.
        .pip_install("huggingface_hub==0.34.4", "hf_xet==1.1.5")
        .pip_install(
            "vllm==0.11.0",
            "transformers==4.57.1",
            # Rebuild flash-attn against torch 2.8 to avoid ABI mismatch.
            "flash-attn==2.8.3",
            extra_options="--no-build-isolation",
        )
        # antlr4 dep conflict resolution:
        #   omegaconf 2.3 needs antlr 4.9 (ATN v3); math-verify's
        #   latex2sympy2_extended needs antlr 4.13 (ATN v4). Fix: use
        #   omegaconf 2.4.0.dev3 which is regenerated against antlr 4.13.
        .pip_install("antlr4-python3-runtime==4.13.2")
        .pip_install(
            "ray[default]==2.43.0",
            "tensordict==0.10.0",
            "datasets==4.0.0",
            "mlflow==3.0.0",
            "wandb==0.19.11",
            "pyarrow==17.0.0",
            "pandas==2.2.3",
        )
        # omegaconf 2.4 dev supports antlr 4.13; matching hydra-core dev.
        .pip_install("omegaconf==2.4.0.dev3", "hydra-core==1.4.0.dev1",
                     extra_options="--no-deps")
        .pip_install("importlib-resources", "packaging")
        # verl runtime deps (installed --no-deps below), plus math-verify.
        .pip_install(
            "codetiming==1.4.0",
            "accelerate==1.2.1",
            "peft==0.14.0",
            "liger-kernel==0.5.4",
            "pybind11",
            "pylatexenc",
            "dill==0.3.8",
            "torchdata==0.10.0",
            "tensorboard",
            "uvicorn",
            "fastapi",
        )
        .pip_install("math-verify==0.5.2")
        .add_local_dir(
            LOCAL_VERL_DIR,
            remote_path=REMOTE_VERL_DIR,
            copy=True,
            ignore=[".git", ".git/**", "__pycache__/**", "docs/**", "tests/**"],
        )
        # Verl deps already installed above. Install verl editable, no-deps
        # so it doesn't try to pull hydra-core==1.3.2 back in.
        .run_commands(f"cd {REMOTE_VERL_DIR} && pip install -e . --no-deps")
        .add_local_file(LOCAL_REWARD_FN, remote_path=REMOTE_REWARD_FN, copy=True)
        .add_local_python_source("common")
    )


app = modal.App("math-1b-rl", image=_img())
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2)
hf_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb-secret")


@app.function(
    gpu="H200:8",
    timeout=60 * 60 * 24,
    retries=modal.Retries(max_retries=10, initial_delay=60, backoff_coefficient=1.0),
    volumes={CACHE_MOUNT: cache_volume, CHECKPOINT_MOUNT: checkpoint_volume},
    secrets=[hf_secret, wandb_secret],
    cpu=8.0,
    memory=200 * 1024,
)
def rl_train_fn(
    base_repo: str,
    anchor: int,
    train_batch_size: int = 128,
    ppo_mini_batch: int = 128,
    group_size: int = 8,
    max_prompt_length: int = 512,
    max_response_length: int = 3584,
    actor_lr: float = 1e-6,
    lr_warmup_steps: int = 50,
    total_training_steps: int = 5000,
    total_epochs: int = 100,     # unbounded — capped by step_cap
    save_freq: int = 50,
    test_freq: int = 100000,     # effectively no mid-training eval
    val_n_samples: int = 8,
    rollout_temperature: float = 1.0,
    ppo_micro_batch_per_gpu: int = 2,
    log_prob_micro_batch: int = 2,
    gpu_memory_utilization: float = 0.85,
    max_num_batched_tokens: int = 12288,
    run_name: str | None = None,
) -> dict:
    import os
    import subprocess
    from pathlib import Path

    cache_volume.reload()
    checkpoint_volume.reload()

    if run_name is None:
        run_name = f"math-1b-rl-deepscaler-from-step{anchor}"

    save_dir = f"{CHECKPOINT_MOUNT}/rl/{run_name}"
    checkpoint_dir = f"{save_dir}/checkpoints"
    log_dir = f"{save_dir}/logs"
    mlflow_dir = f"{save_dir}/mlflow"
    rollout_dir = f"{save_dir}/rollouts/training"
    validation_dir = f"{save_dir}/rollouts/validation"
    for d in (checkpoint_dir, log_dir, mlflow_dir, rollout_dir, validation_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    train_parquet = f"{CHECKPOINT_MOUNT}/rl_data/skyeasy25k_omi2/train.parquet"
    val_parquet = f"{CHECKPOINT_MOUNT}/rl_data/skyeasy25k_omi2/test.parquet"

    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "REWARD_MODEL_TYPE": "RULE_BASED",
        "HF_HOME": f"{CACHE_MOUNT}/hf",
        "TRANSFORMERS_CACHE": f"{CACHE_MOUNT}/hf/transformers",
        "HF_DATASETS_CACHE": f"{CACHE_MOUNT}/hf/datasets",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "VLLM_USE_V1": "1",
        "HYDRA_FULL_ERROR": "1",
        "MLFLOW_TRACKING_URI": f"file://{mlflow_dir}",
        "MLFLOW_ALLOW_FILE_STORE": "true",
        "MLFLOW_EXPERIMENT_NAME": run_name,
        "WANDB_PROJECT": "math-1b-rl",
        "WANDB_RUN_ID": run_name,
    })
    for d in (env["HF_HOME"], env["TRANSFORMERS_CACHE"], env["HF_DATASETS_CACHE"]):
        Path(d).mkdir(parents=True, exist_ok=True)

    cwd = save_dir  # cd to neutral dir so `python -m verl...` uses installed verl

    kl_args = [
        "actor_rollout_ref.actor.use_kl_loss=False",
    ]
    clip_args = [
        "actor_rollout_ref.actor.clip_ratio_low=0.2",
        "actor_rollout_ref.actor.clip_ratio_high=0.26",
        "actor_rollout_ref.actor.clip_ratio_c=10.0",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
    ]

    cmd = [
        "python3", "-m", "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={train_parquet}",
        f"data.val_files=['{val_parquet}']",
        f"data.train_batch_size={train_batch_size}",
        f"data.max_prompt_length={max_prompt_length}",
        f"data.max_response_length={max_response_length}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={base_repo}",
        f"actor_rollout_ref.actor.optim.lr={actor_lr}",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps={lr_warmup_steps}",
        "actor_rollout_ref.model.use_remove_padding=True",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={ppo_micro_batch_per_gpu}",
        *kl_args,
        *clip_args,
        "actor_rollout_ref.actor.entropy_coeff=0.0",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={log_prob_micro_batch}",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={max_num_batched_tokens}",
        f"actor_rollout_ref.rollout.temperature={rollout_temperature}",
        f"actor_rollout_ref.rollout.n={group_size}",
        f"actor_rollout_ref.rollout.val_kwargs.n={val_n_samples}",
        "actor_rollout_ref.rollout.val_kwargs.temperature=1",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={log_prob_micro_batch}",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "algorithm.use_kl_in_reward=False",
        "reward_model.reward_manager=batch",
        f"custom_reward_function.path={REMOTE_REWARD_FN}",
        "custom_reward_function.name=compute_score_batch",
        "trainer.critic_warmup=0",
        "trainer.default_hdfs_dir=null",
        f"trainer.default_local_dir={checkpoint_dir}",
        "trainer.resume_mode=auto",
        "trainer.logger=['console']",  # wandb disabled — proto mismatch in this image; mlflow file store used
        f"trainer.project_name=math-1b-rl",
        f"trainer.experiment_name={run_name}",
        "trainer.n_gpus_per_node=8",
        "trainer.nnodes=1",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={test_freq}",
        f"trainer.total_epochs={total_epochs}",
        f"trainer.total_training_steps={total_training_steps}",
        f"trainer.rollout_data_dir={rollout_dir}",
        f"trainer.validation_data_dir={validation_dir}",
    ]

    log_file = f"{log_dir}/rl_train.log"
    print(f"[rl] cwd={cwd}")
    print(f"[rl] log={log_file}")
    print(f"[rl] base_repo={base_repo}")
    print(f"[rl] anchor={anchor}")
    print(f"[rl] cmd:", " ".join(cmd[:6]), "...")

    # Tee output so we can see progress via Modal logs AND persist to volume
    with open(log_file, "a") as f:
        f.write(f"\n=== new run {run_name} ===\n")
        f.write(" ".join(cmd) + "\n\n")
    result = subprocess.run(cmd, cwd=cwd, env=env, check=False)
    checkpoint_volume.commit()

    return {
        "run_name": run_name,
        "base_repo": base_repo,
        "anchor": anchor,
        "return_code": result.returncode,
        "save_dir": save_dir,
    }


ANCHORS_4WAY = [10000, 40000, 80000, 95368]

# All other pretrain anchors with an SFT model on HF (55k SFT not landed)
ANCHORS_REMAINING = [
    5000, 15000, 20000, 25000, 30000, 35000, 45000, 50000, 60000,
    65000, 70000, 75000, 85000, 90000, 95000,
]  # 15 anchors

# Pruned sweep: every 5k below 50k, every 10k above 50k (~15 total including
# the 4 rl_sweep_4 anchors already covered).
ANCHORS_PRUNED_NEW = [
    5000, 15000, 20000, 25000, 30000, 35000, 45000, 50000,   # every-5k ≤ 50k
    60000, 70000, 90000,                                      # every-10k > 50k
]  # 11 new (10k/40k/80k/95368 already running via rl_sweep_4)


@app.local_entrypoint()
def rl_custom(base_repo: str, run_name: str, total_training_steps: int = 1500,
              anchor: int = 0) -> None:
    """Interleave experiment: RL from an arbitrary base model (e.g. a volume path)
    with a custom run name and step budget."""
    print(f"[rl_custom] base={base_repo} run={run_name} steps={total_training_steps}")
    r = rl_train_fn.remote(base_repo=base_repo, anchor=anchor, run_name=run_name,
                           total_training_steps=total_training_steps)
    print(r)


@app.local_entrypoint()
def rl_sweep_pruned_new() -> None:
    """Fire RL for the pruned set: every-5k ≤ 50k, every-10k > 50k (11 new)."""
    print(f"firing {len(ANCHORS_PRUNED_NEW)} RL runs (pruned)")
    futures = []
    for a in ANCHORS_PRUNED_NEW:
        base = f"pre-to-post-olmo/math-1b-sft-numinamath-bs512-from-step{a}"
        run_name = f"math-1b-rl-deepscaler-from-step{a}"
        f = rl_train_fn.spawn(base_repo=base, anchor=a, run_name=run_name)
        futures.append((a, f))
    for a, f in futures:
        try:
            r = f.get()
            print(f"  ✓ from step{a}: {r}")
        except Exception as e:
            print(f"  ✗ from step{a}: {e}")


@app.local_entrypoint()
def rl_sweep_remaining() -> None:
    """Fire RL for every pretrain-scale SFT anchor not covered by rl_sweep_4."""
    print(f"firing {len(ANCHORS_REMAINING)} RL runs")
    futures = []
    for a in ANCHORS_REMAINING:
        base = f"pre-to-post-olmo/math-1b-sft-numinamath-bs512-from-step{a}"
        run_name = f"math-1b-rl-deepscaler-from-step{a}"
        f = rl_train_fn.spawn(base_repo=base, anchor=a, run_name=run_name)
        futures.append((a, f))
    for a, f in futures:
        try:
            r = f.get()
            print(f"  ✓ from step{a}: {r}")
        except Exception as e:
            print(f"  ✗ from step{a}: {e}")


@app.local_entrypoint()
def rl_sweep_4() -> None:
    """Fire 4 parallel RL runs from every-40B pretrain anchor SFT models."""
    print(f"firing {len(ANCHORS_4WAY)} RL runs in parallel")
    futures = []
    for a in ANCHORS_4WAY:
        base = f"pre-to-post-olmo/math-1b-sft-numinamath-bs512-from-step{a}"
        run_name = f"math-1b-rl-deepscaler-from-step{a}"
        f = rl_train_fn.spawn(base_repo=base, anchor=a, run_name=run_name)
        futures.append((a, f))
    for a, f in futures:
        try:
            r = f.get()
            print(f"  ✓ from step{a}: {r}")
        except Exception as e:
            print(f"  ✗ from step{a}: {e}")


ANCHORS_FINISH_3K = [5000, 15000, 20000, 50000, 70000]


@app.local_entrypoint()
def rl_finish_3k() -> None:
    """Resume the 5 under-3k anchors and cap them at 3000 steps."""
    print(f"firing {len(ANCHORS_FINISH_3K)} RL runs to reach 3000 steps")
    futures = []
    for a in ANCHORS_FINISH_3K:
        base = f"pre-to-post-olmo/math-1b-sft-numinamath-bs512-from-step{a}"
        run_name = f"math-1b-rl-deepscaler-from-step{a}"
        f = rl_train_fn.spawn(
            base_repo=base, anchor=a, run_name=run_name,
            total_training_steps=3000,
        )
        futures.append((a, f))
    for a, f in futures:
        try:
            r = f.get()
            print(f"  ✓ from step{a}: {r}")
        except Exception as e:
            print(f"  ✗ from step{a}: {e}")


@app.local_entrypoint()
def rl_single(anchor: int = 95368, total_training_steps: int = 5000) -> None:
    base = f"pre-to-post-olmo/math-1b-sft-numinamath-bs512-from-step{anchor}"
    r = rl_train_fn.remote(
        base_repo=base, anchor=anchor,
        total_training_steps=total_training_steps,
        run_name=f"math-1b-rl-deepscaler-from-step{anchor}",
    )
    print(r)
