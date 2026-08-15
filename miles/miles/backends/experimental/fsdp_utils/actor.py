import hashlib
import json
import logging
import os
from pathlib import Path
from argparse import Namespace

import ray
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoConfig

from miles.ray.train_actor import TrainRayActor
from miles.utils import train_dump_utils, train_metric_utils
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.ray_utils import Box
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils import init_tracking

from ....utils.profile_utils import TrainProfiler
from ...training_utils.ci_utils import check_grad_norm
from ...training_utils.data import (
    DataIterator,
    get_batch,
    get_data_iterator,
    get_rollout_data,
    peek_supervised_token_count,
)
from ...training_utils.log_utils import (
    aggregate_forward_results,
    aggregate_train_losses,
    log_rollout_data,
    log_train_step,
)
from ...training_utils.loss import compute_advantages_and_returns, get_log_probs_and_entropy, loss_function
from ...training_utils.parallel import get_parallel_state, set_parallel_state
from . import checkpoint
from .lr_scheduler import get_lr_scheduler
from .initial_adam import (
    assert_initial_adam_step_progression,
    initial_adam_spec_from_args,
    install_initial_adam_state,
    prepare_initial_adam_state,
    validate_initial_adam_resume_evidence,
)
from .parallel import create_fsdp_parallel_state
from .precision import (
    GRADIENT_REDUCTION_DTYPE,
    MASTER_PARAMETER_DTYPE,
    assert_finite_training_value,
    assert_fp32_gradients,
    assert_fp32_master_parameters,
    assert_fp32_training_state,
    compute_dtype,
    precision_contract,
    upcast_model_to_fp32_,
    validate_policy_logging_wrapper,
)
from .update_weight_utils import UpdateWeightFromDistributed, UpdateWeightFromTensor

logger = logging.getLogger(__name__)


def _record_precision_gate_weight_versions(
    *,
    actor_global_step: int,
    updater_weight_version: int,
    engine_weight_versions: list[str],
) -> None:
    leg = os.environ.get("CHESS_RL_MILES_PRECISION_GATE_LEG")
    artifact_root = os.environ.get("CHESS_RL_MILES_ARTIFACT_ROOT")
    if leg is None:
        return
    if leg not in {"1", "2"} or not artifact_root:
        raise RuntimeError(
            "Precision-gate weight evidence requires leg 1/2 and an artifact root"
        )
    core = {
        "schema": "miles-precision-gate-weight-version-v1",
        "leg": int(leg),
        "actor_global_step": int(actor_global_step),
        "expected_weight_version": int(updater_weight_version),
        "engine_weight_versions": list(engine_weight_versions),
    }
    digest = hashlib.sha256(
        json.dumps(
            core,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    output = (
        Path(artifact_root)
        / "precision_gate"
        / f"leg_{leg}_weight_versions.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {**core, "evidence_sha256": digest}
    descriptor = os.open(
        output,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    try:
        os.write(
            descriptor,
            (json.dumps(payload, sort_keys=True) + "\n").encode(),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class FSDPTrainRayActor(TrainRayActor):
    """Simplified TrainRayActor for pure HF+FSDP training.

    Responsibilities:
      * Initialize model/tokenizer on rank0 sequentially to avoid race on cache
      * Wrap model with FSDP
      * Provide minimal train / save / update_weights hooks compatible with existing RayTrainGroup

    Weight update strategy:
      * Rank0 gathers state_dict (full) and broadcasts tensor-by-tensor.
      * For small models this is fine; for larger models consider sharded state_dict type.
    """

    @with_defer(lambda: Timer().start("train_wait"))
    def init(self, args: Namespace, role: str, with_ref: bool = False) -> int:  # type: ignore[override]
        super().init(args, role, with_ref)

        if args.dumper_enable:
            from sglang.srt.debug_utils.dumper import dumper

            dumper.apply_source_patches()

        # Setup ParallelState for both CP and non-CP cases
        set_parallel_state(create_fsdp_parallel_state(args))

        torch.manual_seed(args.seed)

        self.train_parallel_config = {
            "dp_size": get_parallel_state().intra_dp.size,
        }

        if self.args.debug_rollout_only:
            return 0

        self.fsdp_cpu_offload = getattr(self.args, "fsdp_cpu_offload", False)
        # Offload train and fsdp cpu offload cannot be used together, fsdp_cpu_offload is more aggressive
        if self.args.offload_train and self.fsdp_cpu_offload:
            self.args.offload_train = False

        self._enable_true_on_policy_optimizations(args)
        if dist.get_rank() == 0:
            init_tracking(args, primary=False)

        if getattr(self.args, "start_rollout_id", None) is None:
            self.args.start_rollout_id = 0

        self.prof = TrainProfiler(args)

        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = load_tokenizer(
                    self.args.hf_checkpoint, chat_template_path=self.args.chat_template_path, trust_remote_code=True
                )
                # Vision models have `vision_config` in the config
                if hasattr(self.hf_config, "vision_config"):
                    self.processor = load_processor(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        init_context = self._get_init_weight_context_manager()

        with init_context():
            model = self.get_model_cls().from_pretrained(
                self.args.hf_checkpoint,
                trust_remote_code=True,
                attn_implementation=self.args.attn_implementation,
                dtype=MASTER_PARAMETER_DTYPE,
            )

        # FSDP2 performs the optimizer step in each sharded parameter's
        # original dtype.  An HF checkpoint whose config says bfloat16 would
        # otherwise silently create BF16 master parameters and BF16 Adam
        # moments, causing small updates (notably at lr=1e-5) to round away.
        upcast_model_to_fp32_(model)
        assert_fp32_master_parameters(model, where="after Hugging Face load and FP32 upcast")
        model.train()

        full_state = model.state_dict()

        initial_adam_spec = initial_adam_spec_from_args(args)
        initial_adam_prepared: dict[str, object] | None = None
        initial_adam_prepare_result: list[str | None] = [None]
        if initial_adam_spec is not None and dist.get_rank() == 0:
            try:
                initial_adam_prepared = prepare_initial_adam_state(
                    model,
                    args=args,
                )
            except Exception as exc:
                initial_adam_prepare_result[0] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(initial_adam_prepare_result, src=0)
        if initial_adam_prepare_result[0] is not None:
            raise RuntimeError(
                "initial Adam source authentication failed: "
                f"{initial_adam_prepare_result[0]}"
            )

        model = apply_fsdp2(
            model, mesh=get_parallel_state().dp_mesh, cpu_offload=self.fsdp_cpu_offload, args=self.args
        )

        model = self._fsdp2_load_full_state_dict(
            model, full_state, get_parallel_state().dp_mesh, cpu_offload=True if self.fsdp_cpu_offload else None
        )
        assert_fp32_master_parameters(model, where="after FSDP wrapping and full-state load")

        self.model = model

        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if args.optimizer == "adam":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}. Supported options: 'adam'")

        assert_fp32_training_state(
            self.model,
            self.optimizer,
            where="after AdamW construction",
            require_optimizer_state=False,
        )
        self._optimizer_precision_verified = False
        self._gradient_precision_verified = False
        self._forward_precision_verified: set[str] = set()
        self._forward_precision_dtypes: dict[str, str] = {}
        if dist.get_rank() == 0:
            logger.info("FSDP training precision contract: %s", precision_contract(args))

        # Initialize LR scheduler
        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)

        self.global_step = 0
        self.micro_step = 0

        checkpoint_payload = checkpoint.load(self)
        if checkpoint_payload is None:
            self._initial_adam_import_evidence = (
                install_initial_adam_state(
                    self.model,
                    self.optimizer,
                    initial_adam_prepared,
                )
                if initial_adam_spec is not None
                else None
            )
        else:
            self._initial_adam_import_evidence = (
                validate_initial_adam_resume_evidence(
                    self.args,
                    (checkpoint_payload.get("metadata") or {}).get(
                        "initial_adam_import"
                    ),
                )
            )
        self._initial_adam_step_progression = (
            assert_initial_adam_step_progression(
                self.optimizer,
                self._initial_adam_import_evidence,
                rl_global_step=(
                    int((checkpoint_payload.get("metadata") or {}).get("global_step", 0))
                    if checkpoint_payload is not None
                    else 0
                ),
            )
        )
        assert_fp32_training_state(
            self.model,
            self.optimizer,
            where="after distributed-checkpoint load",
            require_optimizer_state=bool(self.optimizer.state),
        )
        self._optimizer_precision_verified = bool(self.optimizer.state)

        # Create separate ref model if needed (kept in CPU until needed)
        self.ref_model = None
        if with_ref:
            self.ref_model = self._create_ref_model(args.ref_load)

        self.weight_updater = (
            UpdateWeightFromTensor(self.args, self.model)
            if self.args.colocate
            else UpdateWeightFromDistributed(self.args, self.model)
        )

        checkpoint.finalize_load(self, checkpoint_payload)
        # Keep SGLang's policy version monotonic across a new process. The
        # unconditional initial sync in train.py advances this baseline to
        # global_step + 1, exactly as an uninterrupted run would.
        self.weight_updater.weight_version = int(self.global_step)
        if self.weight_updater.weight_version < 0:
            raise RuntimeError(
                "Restored optimizer step cannot initialize a negative rollout weight version"
            )

        # Initialize data packing parameters
        self.max_tokens_per_gpu = args.max_tokens_per_gpu  # From main arguments

        if self.args.offload_train:
            self.sleep()

        self.prof.on_init_end()

        return int(getattr(self.args, "start_rollout_id", 0))

    def get_model_cls(self):
        # Vision models have `vision_config` in the config
        if hasattr(self.hf_config, "vision_config"):
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        else:
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM

    def _enable_true_on_policy_optimizations(self, args):
        if args.true_on_policy_mode:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            from .models.qwen3_moe import apply_true_on_policy_patch_for_qwen3_moe

            logger.info("FSDPTrainRayActor call enable_batch_invariant_mode for true-on-policy")
            enable_batch_invariant_mode(
                # In Qwen3, rope `inv_freq_expanded.float() @ position_ids_expanded.float()` uses bmm
                # and disabling it will make it aligned
                enable_bmm=False,
            )

            apply_true_on_policy_patch_for_qwen3_moe()
        else:
            from .models.qwen3_moe_hf import apply_fsdp_moe_patch

            apply_fsdp_moe_patch()

    def _get_init_weight_context_manager(self):
        """Get context manager for model initialization.

        Returns a callable that creates a context manager.
        Uses meta device (no memory allocation) for non-rank-0 processes,
        UNLESS tie_word_embeddings=True (which causes hangs with meta tensors).

        Ref: verl/utils/fsdp_utils.py::get_init_weight_context_manager
        NOTE: tie_word_embedding causes meta_tensor init to hang
        """
        from accelerate import init_empty_weights

        # Check if model uses tied word embeddings (which doesn't work with meta tensors)
        use_meta_tensor = not self.hf_config.tie_word_embeddings

        def cpu_init_weights():
            return torch.device("cpu")

        if use_meta_tensor:
            # Rank 0: CPU, others: meta device (memory efficient for large models)
            return init_empty_weights if dist.get_rank() != 0 else cpu_init_weights
        else:
            logger.info(f"[Rank {dist.get_rank()}] tie_word_embeddings=True, loading full model to CPU on all ranks")
            return cpu_init_weights

    def _fsdp2_load_full_state_dict(self, model, full_state, device_mesh, cpu_offload):
        """Load full state dict into FSDP2 model with efficient broadcast from rank 0.

        This function loads weights from rank 0 and broadcasts to all other ranks,
        avoiding the need for each rank to load the full model from disk.

        Args:
            model: FSDP2-wrapped model
            full_state: State dict (only rank 0 has real weights, others have empty dict)
            device_mesh: Device mesh for FSDP
            cpu_offload: If not None, enables StateDictOptions cpu_offload

        Ref:verl/utils/fsdp_utils.py::fsdp2_load_full_state_dict
        """
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        # Rank 0: move with weights, others: allocate empty tensors on device
        if dist.get_rank() == 0:
            model = model.to(device=torch.cuda.current_device(), non_blocking=True)
        else:
            # to_empty creates tensors on device without initializing memory
            model = model.to_empty(device=torch.cuda.current_device())

        is_cpu_offload = cpu_offload is not None
        options = StateDictOptions(full_state_dict=True, cpu_offload=is_cpu_offload, broadcast_from_rank0=True)

        set_model_state_dict(model, full_state, options=options)

        # set_model_state_dict will not broadcast buffers, so we need to broadcast them manually.
        for _name, buf in model.named_buffers():
            dist.broadcast(buf, src=0)

        if is_cpu_offload:
            model.to("cpu", non_blocking=True)
            for buf in model.buffers():
                buf.data = buf.data.to(torch.cuda.current_device())

        return model

    @timer
    def sleep(self) -> None:
        """Pause CUDA memory for all tracked tensors."""
        if not self.args.offload_train:
            return

        print_memory("before offload model")

        self.model.cpu()
        move_torch_optimizer(self.optimizer, "cpu")
        clear_memory()
        dist.barrier(group=get_gloo_group())
        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        """Resume CUDA memory for all tracked tensors."""
        if not self.args.offload_train:
            return

        self.model.cuda()
        move_torch_optimizer(self.optimizer, "cuda")
        dist.barrier(group=get_gloo_group())
        print_memory("after wake_up model")

    def save_model(
        self,
        rollout_id: int,
        force_sync: bool = False,
        rollout_state: dict | None = None,
    ) -> None:
        """Delegate checkpoint saving to the shared checkpoint utilities."""
        if self.args.debug_rollout_only or self.args.save is None:
            return

        assert not self.args.async_save, "FSDPTrainRayActor does not support async_save yet."
        assert_fp32_training_state(
            self.model,
            self.optimizer,
            where=f"before checkpoint save at rollout {rollout_id}",
            require_optimizer_state=bool(self.optimizer.state),
        )
        checkpoint.save(self, rollout_id, rollout_state=rollout_state)

    def _compute_log_prob(
        self,
        model_tag: str,
        data_iterator: DataIterator,
        num_microbatches: list[int],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:
        """Compute token log-probabilities using data iterator.

        Parameters:
            model_tag: Which parameters to use, e.g. "actor" or "ref".
            data_iterator: DataIterator providing micro-batches.
            num_microbatches: List of number of microbatches per step.
            store_prefix: Prefix to use for keys in outputs (e.g., "ref_").

        Returns:
            A lightweight dictionary keyed by f"{store_prefix}log_probs".

        Note:
            Uses separate ref model when model_tag == "ref". The ref model is
            loaded from CPU to GPU on-demand and offloaded back after use.
        """
        # Select which model to use
        if model_tag == "ref" and self.ref_model is not None:
            if not self.fsdp_cpu_offload:
                self.model.cpu()
                torch.cuda.empty_cache()
                dist.barrier(group=get_gloo_group())

            active_model = self.ref_model
            active_model.eval()
        else:
            active_model = self.model

        try:
            forward_data_store = []
            data_iterator.reset()

            with timer(f"{store_prefix}log_probs"), torch.no_grad():
                num_steps_per_rollout = len(num_microbatches)
                for step_id in range(num_steps_per_rollout):
                    for _ in self.prof.iterate_train_log_probs(
                        tqdm(
                            range(num_microbatches[step_id]),
                            desc=f"{store_prefix}log_probs",
                            disable=dist.get_rank() != 0,
                        )
                    ):
                        forward_only_keys = [
                            "tokens",
                            "loss_masks",
                            "multimodal_train_inputs",
                            "total_lengths",
                            "response_lengths",
                            "max_seq_lens",
                        ]
                        batch = get_batch(
                            data_iterator,
                            forward_only_keys,
                            self.args.data_pad_size_multiplier,
                            self.args.qkv_format,
                            get_position_ids=True,
                        )

                        model_args = self._get_model_inputs_args(batch)
                        raw_logits = active_model(**model_args).logits
                        self._verify_forward_dtype(raw_logits, model_tag=model_tag)
                        logits = raw_logits.float()

                        result = get_log_probs_and_entropy(
                            logits=logits,
                            args=self.args,
                            unconcat_tokens=batch["unconcat_tokens"],
                            total_lengths=batch["total_lengths"],
                            response_lengths=batch["response_lengths"],
                            with_entropy=(store_prefix == ""),
                            max_seq_lens=batch.get("max_seq_lens", None),
                        )

                        batch_result = {
                            f"{store_prefix}log_probs": result["log_probs"],
                        }
                        if store_prefix == "" and "entropy" in result:
                            batch_result["entropy"] = result["entropy"]
                        forward_data_store.append(batch_result)

            rollout_data = aggregate_forward_results(forward_data_store, data_iterator, self.args, store_prefix)

            return rollout_data

        finally:
            # Restore actor model if it was offloaded
            if model_tag == "ref" and self.ref_model is not None:
                torch.cuda.empty_cache()
                dist.barrier(group=get_gloo_group())

                if not self.fsdp_cpu_offload:
                    self.model.cuda()
                    dist.barrier(group=get_gloo_group())

    def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
        """Run one training update over a rollout batch.

        Parameters:
            rollout_id: Monotonic id for logging.
            rollout_data_ref: A Box handle wrapping a Ray object reference to a
                dictionary with rollout tensors and metadata (e.g., `tokens`,
                `loss_masks`, `rewards`, `response_lengths`, optional
                `rollout_log_probs`, etc.). It will be fetched and partitioned
                by `process_rollout_data` based on data-parallel rank/size.
        """
        if self.args.offload_train:
            self.wake_up()

        with inverse_timer("train_wait"), timer("train"):
            rollout_data = get_rollout_data(self.args, rollout_data_ref)
            if self.args.debug_rollout_only:
                return
            # Reset after rollout transfer so the measurement covers only the
            # policy-update path (reference log probs, actor log probs, and
            # actor backward/update). The global MAX below makes the canary
            # fail-safe against one data-parallel rank carrying longer samples.
            torch.cuda.reset_peak_memory_stats()
            self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)
            torch.cuda.synchronize()
            peak_memory = torch.tensor(
                [
                    torch.cuda.max_memory_allocated(),
                    torch.cuda.max_memory_reserved(),
                ],
                dtype=torch.float64,
                device=torch.cuda.current_device(),
            )
            dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
            if dist.get_rank() == 0:
                allocated, reserved = (
                    peak_memory / float(1024**3)
                ).tolist()
                logger.info(
                    "policy_update_memory %s: "
                    "{'peak_allocated_gb_global_max': %.3f, "
                    "'peak_reserved_gb_global_max': %.3f}",
                    rollout_id,
                    allocated,
                    reserved,
                )

        train_metric_utils.log_perf_data_raw(
            rollout_id=rollout_id,
            args=self.args,
            is_primary_rank=dist.get_rank() == 0,
            compute_total_fwd_flops=None,
        )

    def _train_core(self, rollout_id: int, rollout_data) -> None:
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        data_iterator = data_iterator[0]

        assert (
            len(num_microbatches) > 0
        ), f"Invalid num_microbatches {num_microbatches} for micro_batch_size {self.args.micro_batch_size} and global_batch_size {self.args.global_batch_size}"

        if self.ref_model is not None:
            ref_results = self._compute_log_prob("ref", data_iterator, num_microbatches, store_prefix="ref_")
            rollout_data.update(ref_results)

        actor_results = self._compute_log_prob("actor", data_iterator, num_microbatches)
        if "entropy" in actor_results:
            assert_finite_training_value(
                actor_results["entropy"],
                name="rollout_entropy",
                where=f"actor log-probability pass at rollout {rollout_id}",
            )
        rollout_data.update(actor_results)

        compute_advantages_and_returns(self.args, rollout_data)

        log_rollout_data(rollout_id, self.args, rollout_data)

        with timer("actor_train"):
            data_iterator.reset()
            num_steps_per_rollout = len(num_microbatches)

            for step_id in range(num_steps_per_rollout):
                self.optimizer.zero_grad(set_to_none=True)

                global_token_count = None
                use_token_mean_loss = self.args.calculate_per_token_loss or getattr(
                    self.args, "policy_loss_agg_mode", None
                ) == "token-mean"
                if use_token_mean_loss:
                    global_token_count = self._global_supervised_token_count(
                        data_iterator,
                        num_microbatches=num_microbatches[step_id],
                    )

                losses_reduced = []
                for _ in self.prof.iterate_train_actor(
                    tqdm(range(num_microbatches[step_id]), desc="actor_train", disable=dist.get_rank() != 0)
                ):
                    batch = get_batch(
                        data_iterator,
                        [
                            "tokens",
                            "loss_masks",
                            "multimodal_train_inputs",
                            "total_lengths",
                            "response_lengths",
                            "max_seq_lens",
                            "log_probs",
                            "advantages",
                            "returns",
                            "ref_log_probs",
                            "rollout_log_probs",
                        ],
                        self.args.data_pad_size_multiplier,
                        self.args.qkv_format,
                        get_position_ids=True,
                    )

                    log_dict = self._train_step(
                        batch=batch,
                        step_id=step_id,
                        num_microbatches=num_microbatches[step_id],
                        global_token_count=global_token_count,
                    )
                    self.micro_step += 1
                    losses_reduced.append(log_dict)

                if not self._gradient_precision_verified:
                    assert_fp32_gradients(
                        self.model,
                        where=f"after first complete backward accumulation at rollout {rollout_id}",
                    )
                    if dist.get_rank() == 0:
                        logger.info(
                            "Verified FP32 accumulated/reduced gradients before first AdamW step at rollout %s",
                            rollout_id,
                        )
                    self._gradient_precision_verified = True
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                grad_norm = grad_norm.full_tensor().item()
                assert_finite_training_value(
                    grad_norm,
                    name="grad_norm",
                    where=f"before AdamW step at rollout {rollout_id}",
                )

                self.optimizer.step()
                self.global_step += 1
                self._initial_adam_step_progression = (
                    assert_initial_adam_step_progression(
                        self.optimizer,
                        self._initial_adam_import_evidence,
                        rl_global_step=self.global_step,
                    )
                )
                if not self._optimizer_precision_verified:
                    assert_fp32_training_state(
                        self.model,
                        self.optimizer,
                        where=f"after first AdamW step at rollout {rollout_id}",
                        require_optimizer_state=True,
                    )
                    self._optimizer_precision_verified = True
                self.lr_scheduler.step()

                if self.args.ci_test:
                    check_grad_norm(
                        args=self.args,
                        grad_norm=grad_norm,
                        rollout_id=rollout_id,
                        step_id=step_id,
                        role="actor",
                        rank=get_parallel_state().intra_dp_cp.rank,
                    )

                loss_dict = aggregate_train_losses(losses_reduced)

                extra_metrics = {}
                for param_group_id, param_group in enumerate(self.optimizer.param_groups):
                    extra_metrics[f"lr-pg_{param_group_id}"] = param_group["lr"]

                log_train_step(
                    args=self.args,
                    loss_dict=loss_dict,
                    grad_norm=grad_norm,
                    rollout_id=rollout_id,
                    step_id=step_id,
                    num_steps_per_rollout=num_steps_per_rollout,
                    role="actor",
                    extra_metrics=extra_metrics,
                )

        self.prof.step(rollout_id=rollout_id)

        if self.args.save_debug_train_data is not None:
            train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        # Update ref model if needed (copy actor weights to ref)
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and self.ref_model is not None
        ):
            if dist.get_rank() == 0:
                logger.info(f"Updating ref model at rollout_id {rollout_id}")
            # Copy actor model state to ref model
            actor_state = self.model.state_dict()
            self.ref_model.load_state_dict(actor_state)
            self.ref_model.cpu()

    def _global_supervised_token_count(self, data_iterator: DataIterator, *, num_microbatches: int) -> torch.Tensor:
        """Count supervised tokens once for the complete optimizer update."""
        local_token_count = peek_supervised_token_count(data_iterator, num_microbatches).to(
            device=torch.cuda.current_device(),
            dtype=torch.float64,
        )

        parallel_state = get_parallel_state()
        dist.all_reduce(local_token_count, op=dist.ReduceOp.SUM, group=parallel_state.intra_dp.group)
        token_count_value = local_token_count.item()
        if not torch.isfinite(local_token_count).item() or token_count_value <= 0:
            raise RuntimeError(
                f"Global token-mean loss requires a positive finite supervised-token count; got {token_count_value}"
            )
        return local_token_count

    def _train_step(self, batch, step_id, num_microbatches, global_token_count=None):
        # Prepare model inputs
        model_args = self._get_model_inputs_args(batch)
        raw_logits = self.model(**model_args).logits
        self._verify_forward_dtype(raw_logits, model_tag="actor_train")
        logits = raw_logits.float()

        loss, normalizer, log_dict = loss_function(
            args=self.args,
            batch=batch,
            num_microbatches=num_microbatches,
            logits=logits,
            apply_megatron_loss_scaling=False,
            global_token_count=global_token_count,
        )
        assert_finite_training_value(
            loss,
            name="loss",
            where=f"actor train microbatch for optimizer step {step_id}",
        )
        validate_policy_logging_wrapper(
            log_dict,
            where=f"actor train microbatch for optimizer step {step_id}",
        )

        loss.backward()

        return log_dict

    def _verify_forward_dtype(self, logits: torch.Tensor, *, model_tag: str) -> None:
        if model_tag in self._forward_precision_verified:
            return
        expected_dtype = compute_dtype(self.args)
        if logits.dtype != expected_dtype:
            raise RuntimeError(
                f"FSDP forward precision violation for {model_tag}: expected logits {expected_dtype}, "
                f"got {logits.dtype}"
            )
        self._forward_precision_verified.add(model_tag)
        self._forward_precision_dtypes[model_tag] = str(logits.dtype).removeprefix(
            "torch."
        )
        if dist.get_rank() == 0:
            logger.info("Verified %s forward logits dtype: %s", model_tag, logits.dtype)

    @timer
    def update_weights(self) -> None:  # type: ignore[override]
        """Synchronize actor weights to rollout engines.

        Handles both colocated and distributed update modes. In offload mode,
        wakes up parameters as needed to perform the update.
        """
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        rollout_engines, rollout_engine_lock, num_new_engines, engine_gpu_counts, engine_gpu_offsets = ray.get(
            self.rollout_manager.get_updatable_engines_and_lock.remote()
        )
        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_num_new_engines.remote())

        self.weight_updater.update_weights()

        version_error: str | None = None
        observed_versions: list[str] = []
        if dist.get_rank() == 0:
            try:
                observed_versions = [
                    str(value)
                    for value in ray.get(
                        [
                            engine.get_weight_version.remote()
                            for engine in rollout_engines
                            if engine is not None
                        ]
                    )
                ]
                expected_version = str(self.weight_updater.weight_version)
                if not observed_versions:
                    raise RuntimeError(
                        "No live rollout engine reported a weight version after synchronization"
                    )
                mismatches = [
                    (index, value)
                    for index, value in enumerate(observed_versions)
                    if value != expected_version
                ]
                if mismatches:
                    raise RuntimeError(
                        "Rollout weight version mismatch after synchronization: "
                        f"expected={expected_version} actual={observed_versions}"
                    )
                if self.weight_updater.weight_version != self.global_step + 1:
                    raise RuntimeError(
                        "Rollout weight version is not continuous with the optimizer step: "
                        f"global_step={self.global_step} "
                        f"weight_version={self.weight_updater.weight_version}"
                    )
                _record_precision_gate_weight_versions(
                    actor_global_step=self.global_step,
                    updater_weight_version=self.weight_updater.weight_version,
                    engine_weight_versions=observed_versions,
                )
            except Exception as exc:
                version_error = f"{type(exc).__name__}: {exc}"

        version_errors = [version_error]
        dist.broadcast_object_list(
            version_errors,
            src=0,
            group=get_gloo_group(),
        )
        if version_errors[0] is not None:
            raise RuntimeError(version_errors[0])

        clear_memory()

    def _create_ref_model(self, ref_load_path: str | None):
        """Create and initialize a separate reference model with FSDP2 CPUOffloadPolicy.

        Parameters:
            ref_load_path: Path to a directory containing a HF checkpoint. If
                None, a ValueError is raised.

        Returns:
            FSDP2-wrapped ref model with CPU offload enabled

        Note:
            Creates a separate FSDP2 model instance for the reference model.
            ALWAYS uses CPUOffloadPolicy for the reference model to save memory,
            regardless of the actor model's CPU offload setting.
        """
        if ref_load_path is None:
            raise ValueError("ref_load_path must be provided when loading reference model")

        if os.path.isdir(ref_load_path):
            logger.info(f"[Rank {dist.get_rank()}] Creating separate ref model from {ref_load_path}")

            init_context = self._get_init_weight_context_manager()

            with init_context():
                ref_model = self.get_model_cls().from_pretrained(
                    ref_load_path,
                    trust_remote_code=True,
                    attn_implementation=self.args.attn_implementation,
                    dtype=MASTER_PARAMETER_DTYPE,
                )

            upcast_model_to_fp32_(ref_model)
            assert_fp32_master_parameters(ref_model, where="after reference HF load and FP32 upcast")
            full_state = ref_model.state_dict()

            # Always use CPUOffloadPolicy for reference, let FSDP2 handle the offload. It is faster than model.cpu().
            ref_model = apply_fsdp2(ref_model, mesh=get_parallel_state().dp_mesh, cpu_offload=True, args=self.args)
            ref_model = self._fsdp2_load_full_state_dict(
                ref_model, full_state, get_parallel_state().dp_mesh, cpu_offload=True
            )
            assert_fp32_master_parameters(ref_model, where="after reference FSDP full-state load")

            logger.info(f"[Rank {dist.get_rank()}] Reference model created with FSDP2 CPUOffloadPolicy")
            return ref_model
        else:
            raise NotImplementedError(f"Loading from checkpoint file {ref_load_path} not yet implemented")

    def _get_model_inputs_args(self, batch: dict) -> dict:
        input_ids = batch["tokens"]
        position_ids = batch["position_ids"]

        if get_parallel_state().cp.size > 1:
            # TODO: Pin ring_flash_attn for torch 2.11+ compatibility; keep this local import to unblock non-FSDP+CP paths.
            from ring_flash_attn import update_ring_flash_attn_params

            if "cu_seqlens" in batch:
                cu_seqlens = batch["cu_seqlens"]
                if not cu_seqlens.is_cuda:
                    cu_seqlens = cu_seqlens.cuda()
                update_ring_flash_attn_params(cu_seqlens, self.cp_group)

            input_ids = torch.chunk(input_ids, get_parallel_state().cp.size, dim=1)[get_parallel_state().cp.rank]
            position_ids = torch.chunk(position_ids, get_parallel_state().cp.size, dim=1)[get_parallel_state().cp.rank]

        model_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": None,
        }

        if batch.get("multimodal_train_inputs"):
            model_args.update(batch["multimodal_train_inputs"])

        return model_args


@torch.no_grad()
def move_torch_optimizer(optimizer, device):
    """ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py"""
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device, non_blocking=True)

    torch.cuda.synchronize()


def apply_fsdp2(model, mesh=None, cpu_offload=False, args=None):
    """Apply FSDP v2 to the model.

    Args:
        model: The model to wrap with FSDP
        mesh: Optional DeviceMesh for FSDP. If None, uses all ranks.
        cpu_offload: If True, offload parameters, gradients, and optimizer states
            to CPU. The optimizer step will run on CPU. (Default: False)
        args: Arguments containing precision settings (fp16/bf16)

    Ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    layer_cls_to_wrap = model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and next(iter(layer_cls_to_wrap)) is not None

    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
        or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
    ]

    # MixedPrecisionPolicy casts only the unsharded parameters used for
    # forward/backward. The sharded optimizer-facing parameters remain FP32.
    param_dtype = compute_dtype(args)
    reduce_dtype = GRADIENT_REDUCTION_DTYPE

    logger.info(f"FSDP MixedPrecision Policy: param_dtype={param_dtype}, reduce_dtype={reduce_dtype}")

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
    }

    # Apply FSDP to each module (offload_policy=None is equivalent to not passing it)
    for module in modules:
        fully_shard(module, **fsdp_kwargs)

    # Apply FSDP to the top-level model
    fully_shard(model, **fsdp_kwargs)

    return model
