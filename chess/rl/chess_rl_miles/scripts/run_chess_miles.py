from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from chess_rl_miles.data import (
    COT_TYPE,
    DEFAULT_EVAL_FILE,
    DEFAULT_TRAIN_FILE,
    DEFAULT_TRAIN_FILE_SHA256,
    ensure_chess_data,
    ensure_sft_model,
    model_id_from_spec,
)

SPECS = [
    "6p5e18|20m|1.000|0.008",
    # interleave experiment (alpha tokens are free-form labels, not fractions):
    # 0.600i = SFT on alpha1.000@step_60480; 1.000iB / 1.000iC = SFT on leg-2 finals
    "6p5e18|20m|0.600i|0.008",
    "6p5e18|20m|1.000iB|0.008",
    "6p5e18|20m|1.000iC|0.008",
    "6p5e18|32m|0.200|0.013",
    "6p5e18|32m|0.400|0.013",
    "6p5e18|50m|0.100|0.023",
    "6p5e18|50m|0.750|0.023",
    "6p5e18|50m|1.000|0.023",
    "6p5e19|50m|0.180|0.002",
    "6p5e18|410m|0.750|0.148",
    "6p5e18|410m|1.000|0.148",
    "6p5e18|680m|0.750|0.296",
    "6p5e18|680m|1.000|0.296",
    "6p5e19|200m|0.200|0.007",
    "6p5e19|200m|0.400|0.007",
    "6p5e19|200m|0.750|0.007",
    "6p5e19|200m|1.000|0.007",
    "6p5e19|680m|0.400|0.030",
    "6p5e19|680m|0.750|0.030",
    "6p5e19|680m|0.200|0.030",
    "6p5e19|680m|1.000|0.030",
    "6p5e19|680m|1.500|0.030",
    "6p5e19|680m|2.000|0.030",
    "6p5e19|680m|3.000|0.030",
]

STANDARD_POLICY_UPDATE_PROFILE = "standard"
SMALL_MODEL_H200_POLICY_UPDATE_PROFILE = "small-model-h200"
SMALL_MODEL_H200_MAX_TOKENS = frozenset({65_536, 131_072})


def _default_paths():
    project_dir = Path(__file__).resolve().parents[2]
    workspace = project_dir.parent
    return project_dir, workspace / "miles"


def _split_extra_args(extra_args: list[str] | None) -> list[str]:
    if not extra_args:
        return []
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    if len(extra_args) == 1 and extra_args[0].strip() == "":
        return []
    return extra_args


def _tag_float(value: float | str) -> str:
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return text.replace("e-0", "e-").replace("e+0", "e+")


def _default_hparam_tag(args: argparse.Namespace) -> str:
    suffixes = [args.optim_tag, "cispo" if args.cispo else args.advantage_estimator, "miles"]
    if args.hparam_tag_suffix:
        suffixes.append(args.hparam_tag_suffix.strip("_"))
    return (
        f"multi_turn_lr{_tag_float(args.lr)}"
        f"_bs{args.global_batch_size}"
        f"_kl{_tag_float(args.kl_loss_coef)}"
        f"_res{args.rollout_max_response_len}"
        f"_{'_'.join(s for s in suffixes if s)}"
    )


def _ensure_miles_suffix(name: str) -> str:
    return name if "miles" in name.lower() else f"{name}_miles"


def _verify_train_file(path: Path, expected_sha256: str | None) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"RL training data does not exist: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()

    expected = (expected_sha256 or "").strip().lower()
    if expected.startswith("sha256:"):
        expected = expected.removeprefix("sha256:")
    if expected and (len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected)):
        raise ValueError("--train-file-sha256 must be a 64-character hexadecimal SHA256")
    if expected and actual_sha256 != expected:
        raise ValueError(
            "RL training data SHA256 mismatch: "
            f"path={path} expected={expected} actual={actual_sha256}"
        )

    status = "verified" if expected else "logged-only"
    print(
        f"[data] train_file={path} bytes={path.stat().st_size} "
        f"sha256={actual_sha256} status={status}",
        flush=True,
    )
    return actual_sha256


def _validate_policy_update_profile(args: argparse.Namespace) -> None:
    profile = args.small_model_profile
    if profile == STANDARD_POLICY_UPDATE_PROFILE:
        return
    if profile != SMALL_MODEL_H200_POLICY_UPDATE_PROFILE:
        raise ValueError(f"Unknown policy-update profile: {profile}")

    violations: list[str] = []
    if args.train_backend != "fsdp":
        violations.append("train_backend must be fsdp")
    if args.actor_num_nodes != 1 or args.actor_num_gpus_per_node != 8:
        violations.append("actor topology must be exactly 1 node x 8 GPUs")
    if args.gradient_checkpointing:
        violations.append("gradient checkpointing must be disabled")
    if args.fsdp_cpu_offload:
        violations.append("FSDP CPU offload must be disabled")
    if args.rollout_max_context_len > 3_072:
        violations.append("rollout context must be <= 3072 tokens")
    if args.max_tokens_per_gpu not in SMALL_MODEL_H200_MAX_TOKENS:
        allowed = ", ".join(
            f"{value:,}" for value in sorted(SMALL_MODEL_H200_MAX_TOKENS)
        )
        violations.append(f"max_tokens_per_gpu must be one of: {allowed}")
    if violations:
        raise ValueError(
            f"{SMALL_MODEL_H200_POLICY_UPDATE_PROFILE} profile rejected: "
            + "; ".join(violations)
        )


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    project_dir, default_miles_dir = _default_paths()
    miles_dir = Path(args.miles_dir or default_miles_dir).resolve()
    project_dir = Path(args.project_dir or project_dir).resolve()
    _validate_policy_update_profile(args)

    if args.prepare_data:
        ensure_chess_data(args.data_dir)

    model_path = Path(args.hf_checkpoint).resolve() if args.hf_checkpoint else None
    model_id = args.model_id
    if model_path is None:
        if not args.spec:
            raise ValueError("Pass --hf-checkpoint or --spec.")
        model_id = model_id or model_id_from_spec(args.spec)
        if args.prepare_sft:
            model_path = ensure_sft_model(model_id, args.sft_root, cot_type=args.cot_type)
        else:
            model_path = Path(args.sft_root) / args.cot_type / model_id

    default_train_file = (Path(args.data_dir) / DEFAULT_TRAIN_FILE).resolve()
    train_file = Path(args.train_file or default_train_file).resolve()
    expected_train_sha256 = args.train_file_sha256
    if expected_train_sha256 is None and train_file == default_train_file:
        expected_train_sha256 = DEFAULT_TRAIN_FILE_SHA256
    _verify_train_file(train_file, expected_train_sha256)
    eval_file = Path(args.eval_file or Path(args.data_dir) / DEFAULT_EVAL_FILE).resolve()
    run_name = args.run_name or (model_id or model_path.name)
    wandb_run_name = args.wandb_run_name or _ensure_miles_suffix(run_name)
    config_path = (
        Path(args.custom_config).resolve()
        if args.custom_config
        else project_dir / "config/chess_multiturn.yaml"
    )
    save_dir = Path(args.save_dir).resolve()
    if args.io_layout == "chess-rl":
        if not model_id:
            model_id = model_path.name
        hparam_tag = args.hparam_tag or _default_hparam_tag(args)
        artifact_root = save_dir / args.cot_type / hparam_tag / model_id
        checkpoint_dir = artifact_root / "checkpoints"
    else:
        artifact_root = save_dir / run_name
        checkpoint_dir = artifact_root

    log_dir = artifact_root / "logs"
    mlflow_dir = artifact_root / "mlflow"
    rollout_dir = artifact_root / "rollouts" / "training"
    validation_dir = artifact_root / "rollouts" / "validation"
    for path in (checkpoint_dir, log_dir, mlflow_dir, rollout_dir, validation_dir):
        path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(miles_dir / "train.py"),
        "--train-backend",
        args.train_backend,
        "--hf-checkpoint",
        str(model_path),
        "--load",
        str(args.load or model_path),
        "--ref-load",
        str(args.ref_load or model_path),
        "--save",
        str(checkpoint_dir),
        "--prompt-data",
        str(train_file),
        "--data-source-path",
        args.data_source_path,
        "--input-key",
        "prompt",
        "--label-key",
        "reward_model",
        "--metadata-key",
        "extra_info",
        "--custom-generate-function-path",
        "chess_rl_miles.rollout.generate",
        "--custom-rm-path",
        "chess_rl_miles.reward.reward_func",
        "--custom-config-path",
        str(config_path),
        "--reward-key",
        "score",
        "--rollout-shuffle",
        "--rollout-seed",
        str(args.rollout_seed),
        "--num-rollout",
        str(args.num_rollout),
        "--rollout-batch-size",
        str(args.rollout_batch_size),
        "--n-samples-per-prompt",
        str(args.n_samples_per_prompt),
        "--over-sampling-batch-size",
        str(args.over_sampling_batch_size),
        "--rollout-max-prompt-len",
        str(args.rollout_max_prompt_len),
        "--rollout-prompt-reserved-prefix-tokens",
        str(args.rollout_prompt_reserved_prefix_tokens),
        "--rollout-max-response-len",
        str(args.rollout_max_response_len),
        "--rollout-max-context-len",
        str(args.rollout_max_context_len),
        "--chess-context-margin-tokens",
        str(args.chess_context_margin_tokens),
        "--rollout-temperature",
        str(args.rollout_temperature),
        "--rollout-top-p",
        str(args.rollout_top_p),
        "--global-batch-size",
        str(args.global_batch_size),
        "--num-steps-per-rollout",
        str(args.num_steps_per_rollout),
        "--policy-loss-agg-mode",
        args.policy_loss_agg_mode,
        "--advantage-estimator",
        "cispo" if args.cispo else args.advantage_estimator,
        "--cispo-clip-min",
        str(args.cispo_clip_min),
        "--cispo-clip-max",
        str(args.cispo_clip_max),
        "--use-kl-loss",
        "--kl-loss-coef",
        str(args.kl_loss_coef),
        "--kl-loss-type",
        args.kl_loss_type,
        "--entropy-coef",
        "0.0",
        "--optimizer",
        "adam",
        "--lr",
        str(args.lr),
        "--lr-decay-style",
        "constant",
        "--weight-decay",
        str(args.weight_decay),
        "--adam-beta1",
        str(args.adam_beta1),
        "--adam-beta2",
        str(args.adam_beta2),
        "--adam-eps",
        str(args.adam_eps),
        "--clip-grad",
        str(args.clip_grad),
        "--rollout-num-gpus-per-engine",
        str(args.rollout_num_gpus_per_engine),
        "--sglang-server-concurrency",
        str(args.sglang_server_concurrency),
        "--sglang-dtype",
        args.sglang_dtype,
        "--eval-sglang-server-concurrency",
        str(args.eval_sglang_server_concurrency),
        "--eval-generate-max-retries",
        str(args.eval_generate_max_retries),
        "--sglang-mem-fraction-static",
        str(args.sglang_mem_fraction_static),
        "--sglang-chunked-prefill-size",
        str(args.sglang_chunked_prefill_size),
        "--sglang-cuda-graph-backend-prefill",
        "disabled",
        "--sglang-decode-log-interval",
        "1000",
        "--rollout-health-check-interval",
        str(args.rollout_health_check_interval),
        "--actor-num-nodes",
        str(args.actor_num_nodes),
        "--actor-num-gpus-per-node",
        str(args.actor_num_gpus_per_node),
        "--num-gpus-per-node",
        str(args.actor_num_gpus_per_node),
        "--colocate",
        "--attn-implementation",
        args.attn_implementation,
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu",
        str(args.max_tokens_per_gpu),
        "--log-multi-turn",
        "--custom-rollout-log-function-path",
        "chess_rl_miles.io.log_rollout_data",
        "--custom-eval-rollout-log-function-path",
        "chess_rl_miles.io.log_eval_rollout_data",
    ]

    initial_adam_values = (
        args.initial_adam_checkpoint,
        args.initial_adam_completion_sha256,
        args.initial_adam_source_tree_sha256,
        args.initial_adam_step,
    )
    if any(value not in (None, "", 0) for value in initial_adam_values):
        if not all(value not in (None, "", 0) for value in initial_adam_values):
            raise ValueError(
                "initial Adam import requires checkpoint, completion SHA-256, "
                "source-tree SHA-256, and step together"
            )
        cmd.extend(
            [
                "--initial-adam-checkpoint",
                str(args.initial_adam_checkpoint),
                "--initial-adam-completion-sha256",
                str(args.initial_adam_completion_sha256),
                "--initial-adam-source-tree-sha256",
                str(args.initial_adam_source_tree_sha256),
                "--initial-adam-step",
                str(args.initial_adam_step),
            ]
        )

    if args.sglang_context_length is not None:
        if args.sglang_context_length != args.rollout_max_context_len:
            raise ValueError(
                "--sglang-context-length must equal --rollout-max-context-len so generation and policy "
                "training use the same position limit"
            )
        cmd.extend(["--sglang-context-length", str(args.sglang_context_length)])

    if args.use_fault_tolerance:
        cmd.append("--use-fault-tolerance")

    if args.log_passrate:
        cmd.append("--log-passrate")

    if args.gradient_checkpointing:
        cmd.append("--gradient-checkpointing")

    # The chess rollout always sends input IDs and consumes output IDs. The
    # model's remote-code tokenizer is intentionally slow and otherwise runs
    # incremental decode inside SGLang on every generation step.
    if args.sglang_token_id_only:
        cmd.append("--sglang-skip-tokenizer-init")
    if args.sglang_enable_deterministic_inference:
        cmd.append("--sglang-enable-deterministic-inference")
    if args.debug_rollout_only:
        cmd.append("--debug-rollout-only")

    # A zero interval is useful for short throughput canaries: Miles otherwise
    # forces a checkpoint on the final rollout even when the interval exceeds
    # the run length. Omitting the option disables periodic/final saves while
    # retaining --save as the output location for normal runs and diagnostics.
    if args.save_interval > 0:
        cmd.extend(["--save-interval", str(args.save_interval)])

    if args.use_miles_router:
        cmd.append("--use-miles-router")

    if args.batched_rollout:
        cmd.extend([
            "--rollout-function-path",
            "chess_rl_miles.batched_rollout.ChessBatchedRolloutFn",
        ])

    if args.dump_miles_details:
        cmd.extend(["--dump-details", str(artifact_root / "dump_details")])

    if args.balance_data:
        cmd.append("--balance-data")
    if args.use_rollout_logprobs:
        cmd.append("--use-rollout-logprobs")
    if args.fsdp_cpu_offload:
        cmd.append("--fsdp-cpu-offload")
    if args.dynamic_filter:
        cmd.extend([
            "--dynamic-sampling-filter-path",
            "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
        ])
    if args.eval_interval > 0:
        cmd.extend([
            "--eval-interval",
            str(args.eval_interval),
            "--eval-prompt-data",
            "chess_eval",
            str(eval_file),
            "--n-samples-per-eval-prompt",
            str(args.n_samples_per_eval_prompt),
            "--eval-max-response-len",
            str(args.eval_max_response_len),
            "--eval-top-p",
            "1",
            "--skip-eval-before-train",
        ])
    if args.wandb_project and os.environ.get("WANDB_API_KEY"):
        cmd.extend([
            "--use-wandb",
            "--wandb-project",
            args.wandb_project,
            "--wandb-group",
            args.wandb_group,
        ])
        if args.wandb_team:
            cmd.extend(["--wandb-team", args.wandb_team])
        cmd.extend([
            "--wandb-key",
            os.environ["WANDB_API_KEY"],
            "--wandb-run-name",
            wandb_run_name,
            "--disable-wandb-random-suffix",
        ])
        if args.wandb_run_id:
            cmd.extend(["--wandb-run-id", args.wandb_run_id])

    cmd.extend(_split_extra_args(args.extra_args))

    env = os.environ.copy()
    cpu_count = os.cpu_count() or 32
    env["PYTHONPATH"] = f"{project_dir}:{miles_dir}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "true"
    env["OMP_NUM_THREADS"] = str(args.cpu_threads or cpu_count)
    env["RAYON_NUM_THREADS"] = str(args.cpu_threads or cpu_count)
    env["SGLANG_CPU_THREAD_POOL_SIZE"] = str(args.cpu_threads or cpu_count)
    env["MILES_DISABLE_TQDM"] = "1"
    env["TQDM_DISABLE"] = "1"
    env["REWARD_MODEL_TYPE"] = args.reward_model_type
    env["CHESS_RL_MILES_SMALL_MODEL_PROFILE"] = args.small_model_profile
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["MLFLOW_TRACKING_URI"] = f"file://{mlflow_dir}"
    if args.batched_rollout:
        env["MILES_EXPERIMENTAL_ROLLOUT_REFACTOR"] = "1"
    if args.chess_deterministic_seed_by_sample_index:
        env["CHESS_RL_MILES_DETERMINISTIC_SEED_MODE"] = "sample-index"
    if args.save_rollouts:
        env["CHESS_RL_MILES_ARTIFACT_ROOT"] = str(artifact_root)
    return cmd, env


def parse_args() -> argparse.Namespace:
    project_dir, default_miles_dir = _default_paths()
    parser = argparse.ArgumentParser(description="Run Chess-RL on Miles/SGLang with CISPO.")
    parser.add_argument("--miles-dir", default=str(default_miles_dir))
    parser.add_argument("--project-dir", default=str(project_dir))
    parser.add_argument("--spec", default=SPECS[0], choices=SPECS)
    parser.add_argument("--all-specs", action="store_true", help="Run every known Chess-RL SFT spec sequentially.")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--hf-checkpoint", default=None)
    parser.add_argument("--sft-root", default="/sft")
    parser.add_argument("--cot-type", default=COT_TYPE)
    parser.add_argument("--data-dir", default="/root/chess-rl-data")
    parser.add_argument("--train-file", default=None)
    parser.add_argument(
        "--data-source-path",
        default="miles.rollout.data_source.RolloutDataSourceWithBuffer",
        help=(
            "Fully qualified Miles rollout data-source class. The default "
            "preserves production behavior; fixed statistical gates use the "
            "fail-closed chess exact-once source."
        ),
    )
    parser.add_argument(
        "--train-file-sha256",
        default=None,
        help=(
            "Optional expected SHA256 for --train-file. The known balanced-data "
            "checksum is enforced automatically when the default file is used; "
            "pass an empty value to log without enforcing it."
        ),
    )
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--save-dir", default="/checkpoints/chess-rl-miles")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--io-layout", choices=["chess-rl", "flat"], default="chess-rl")
    parser.add_argument("--hparam-tag", default="")
    parser.add_argument("--hparam-tag-suffix", default="")
    parser.add_argument("--save-rollouts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dump-miles-details", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--custom-config", default=None)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--prepare-sft", action="store_true")
    parser.add_argument("--train-backend", choices=["fsdp", "megatron"], default="fsdp")
    parser.add_argument(
        "--small-model-profile",
        choices=[
            STANDARD_POLICY_UPDATE_PROFILE,
            SMALL_MODEL_H200_POLICY_UPDATE_PROFILE,
        ],
        default=STANDARD_POLICY_UPDATE_PROFILE,
        help=(
            "Fail-closed policy-update configuration profile. The standard "
            "profile preserves historical behavior; small-model-h200 enforces "
            "the validated 8-GPU FSDP topology and token-budget allowlist."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable activation checkpointing in the Miles FSDP actor. Enabled "
            "by default for historical large-model runs."
        ),
    )
    parser.add_argument("--load", default=None)
    parser.add_argument("--ref-load", default=None)
    parser.add_argument("--num-rollout", type=int, default=500)
    parser.add_argument(
        "--rollout-seed",
        type=int,
        default=42,
        help="Seed used by Miles for prompt shuffling and per-sample rollout seeds.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=20,
        help="Checkpoint interval; use 0 to disable checkpoints for short benchmarks.",
    )
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--n-samples-per-prompt", type=int, default=8)
    parser.add_argument("--over-sampling-batch-size", type=int, default=256)
    parser.add_argument("--global-batch-size", type=int, default=2048)
    parser.add_argument("--num-steps-per-rollout", type=int, default=1)
    parser.add_argument("--policy-loss-agg-mode", choices=["seq-mean-token-mean", "token-mean"], default="token-mean")
    parser.add_argument(
        "--balance-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Balance samples across data-parallel ranks by sequence length. "
            "Enabled by default to match Chess-RL/veRL trainer.balance_batch=True."
        ),
    )
    parser.add_argument("--optim-tag", default="minimax")
    parser.add_argument("--rollout-max-prompt-len", type=int, default=512)
    parser.add_argument(
        "--rollout-prompt-reserved-prefix-tokens",
        type=int,
        default=1,
        help="Reserve one prefilter token for the BOS inserted by chess rollout.",
    )
    parser.add_argument("--rollout-max-response-len", type=int, default=2560)
    parser.add_argument("--rollout-max-context-len", type=int, default=3072)
    parser.add_argument(
        "--chess-context-margin-tokens",
        type=int,
        default=0,
        help="Reserved positions below rollout_max_context_len; exact-context runs use zero.",
    )
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-top-p", type=float, default=1.0)
    parser.add_argument("--advantage-estimator", default="grpo")
    parser.add_argument("--cispo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cispo-clip-min", type=float, default=0.0)
    parser.add_argument("--cispo-clip-max", type=float, default=5.0)
    parser.add_argument("--kl-loss-coef", type=float, default=0.001)
    parser.add_argument(
        "--kl-loss-type",
        choices=["k1", "k2", "k3", "low_var_kl"],
        default="low_var_kl",
        help="KL estimator forwarded to the Miles policy objective.",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-15)
    parser.add_argument("--initial-adam-checkpoint", default=None)
    parser.add_argument("--initial-adam-completion-sha256", default=None)
    parser.add_argument("--initial-adam-source-tree-sha256", default=None)
    parser.add_argument("--initial-adam-step", type=int, default=None)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--actor-num-nodes", type=int, default=1)
    parser.add_argument("--actor-num-gpus-per-node", type=int, default=8)
    parser.add_argument("--rollout-num-gpus-per-engine", type=int, default=1)
    parser.add_argument(
        "--sglang-server-concurrency",
        type=int,
        default=128,
        help="Per-engine rollout concurrency; 128 is validated on the 8-GPU chess workload.",
    )
    parser.add_argument(
        "--sglang-dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help=(
            "Inference dtype loaded in memory by SGLang. Pin this explicitly because SGLang's auto mode "
            "loads an FP32 HF checkpoint as FP16, while the default Miles training compute dtype is BF16."
        ),
    )
    parser.add_argument(
        "--sglang-context-length",
        type=int,
        default=None,
        help="Explicit SGLang max model length; when set it must match --rollout-max-context-len.",
    )
    parser.add_argument("--eval-sglang-server-concurrency", type=int, default=16)
    parser.add_argument("--eval-generate-max-retries", type=int, default=300)
    parser.add_argument(
        "--use-fault-tolerance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable Miles' destructive rollout health watchdog. Fixed "
            "exact-once rollout gates disable it so a busy SGLang health "
            "probe cannot kill an otherwise recoverable engine."
        ),
    )
    parser.add_argument(
        "--rollout-health-check-interval",
        type=float,
        default=30.0,
        help=(
            "Seconds between Miles health probes. Fixed blinded gates use "
            "1e18 to suppress both destructive and router background probes."
        ),
    )
    parser.add_argument(
        "--log-passrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable Miles pass-rate metric logging. Fixed blinded gates "
            "disable it so no outcome aggregate can be emitted before the "
            "six-cell barrier."
        ),
    )
    parser.add_argument("--use-miles-router", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--batched-rollout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Batch each prompt's sibling trajectories into SGLang requests (enabled by default).",
    )
    parser.add_argument(
        "--sglang-token-id-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable SGLang's server tokenizer and exchange token IDs only (enabled by default).",
    )
    parser.add_argument(
        "--sglang-enable-deterministic-inference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Assign deterministic per-sample SGLang seeds from "
            "rollout_seed + sample index. Disabled by default to preserve "
            "existing production-run identity."
        ),
    )
    parser.add_argument(
        "--chess-deterministic-seed-by-sample-index",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use rollout_seed + the global Miles sample index, rather than "
            "rollout_seed + sibling index, for deterministic chess sampling. "
            "This gives fixed statistical gates one unique seed per "
            "prompt/sibling trajectory."
        ),
    )
    parser.add_argument(
        "--debug-rollout-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Generate and persist rollouts without constructing an actor "
            "optimizer or applying a policy update."
        ),
    )
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.82)
    parser.add_argument("--sglang-chunked-prefill-size", type=int, default=131072)
    parser.add_argument("--max-tokens-per-gpu", type=int, default=9216)
    parser.add_argument("--attn-implementation", default="flash_attention_3")
    parser.add_argument("--fsdp-cpu-offload", action="store_true")
    parser.add_argument(
        "--use-rollout-logprobs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use SGLang rollout logprobs as old policy logprobs. Disabled by "
            "default so Chess-Miles matches Chess-RL/veRL, which recomputes "
            "old_log_probs with the actor after rollout."
        ),
    )
    parser.add_argument("--dynamic-filter", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--n-samples-per-eval-prompt", type=int, default=8)
    parser.add_argument("--eval-max-response-len", type=int, default=2560)
    parser.add_argument("--reward-model-type", default="RULE_BASED")
    parser.add_argument("--wandb-project", default="chess_rl_6p5e18")
    parser.add_argument("--wandb-group", default="multi_turn_miles")
    parser.add_argument("--wandb-team", default=os.environ.get("WANDB_ENTITY", "jingyanshen-new-york-university"))
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = SPECS if args.all_specs else [args.spec]
    if args.all_specs and args.hf_checkpoint:
        raise ValueError("--all-specs expects --spec/--sft-root layout, not one --hf-checkpoint.")

    rc = 0
    for spec in specs:
        args.spec = spec
        if args.all_specs:
            args.model_id = model_id_from_spec(spec)
            args.run_name = args.model_id
        cmd, env = build_command(args)
        redacted = list(cmd)
        for idx, item in enumerate(redacted[:-1]):
            if item == "--wandb-key":
                redacted[idx + 1] = "<redacted>"
        print(" ".join(redacted))
        if not args.dry_run:
            rc = subprocess.call(cmd, env=env)
            if rc != 0:
                return rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
