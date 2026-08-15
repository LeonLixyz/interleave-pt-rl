from sglang.srt.server_args import ServerArgs
from miles.utils.http_utils import _wrap_ipv6


_RENAMED_PARALLEL_SIZE_ARGS = (
    ("sglang_dp_size", "sglang_data_parallel_size", 1),
    ("sglang_pp_size", "sglang_pipeline_parallel_size", 1),
    ("sglang_ep_size", "sglang_expert_parallel_size", 1),
    ("sglang_attn_cp_size", "sglang_attention_context_parallel_size", 1),
)


def normalize_renamed_args(args):
    """Normalize SGLang Namespace fields that changed names across releases.

    Newer SGLang releases use the short dataclass field names as argparse
    destinations (for example, ``dp_size``), while retaining the long names
    only as CLI aliases. Older releases exposed the long names on the parsed
    Namespace. Canonical destinations take precedence when both shapes are
    present.
    """
    for canonical_name, legacy_name, default in _RENAMED_PARALLEL_SIZE_ARGS:
        if not hasattr(args, canonical_name):
            setattr(args, canonical_name, getattr(args, legacy_name, default))

    # SGLang replaced the piecewise-CUDA-graph booleans with one explicit
    # prefill-backend destination. Keep both Namespace shapes coherent without
    # overriding an explicitly parsed canonical backend.
    canonical_backend_name = "sglang_cuda_graph_backend_prefill"
    backend_is_available = hasattr(args, canonical_backend_name)
    backend = getattr(args, canonical_backend_name, None)
    legacy_disable = getattr(args, "sglang_disable_piecewise_cuda_graph", False)
    legacy_enforce = getattr(args, "sglang_enforce_piecewise_cuda_graph", False)

    if backend is not None:
        args.sglang_disable_piecewise_cuda_graph = backend == "disabled"
        args.sglang_enforce_piecewise_cuda_graph = backend == "tc_piecewise"
    elif backend_is_available:
        if legacy_enforce:
            setattr(args, canonical_backend_name, "tc_piecewise")
        elif legacy_disable:
            setattr(args, canonical_backend_name, "disabled")


def set_colocate_cuda_graph_default(args):
    """Disable only prefill CUDA graphs when colocate mode has no explicit policy.

    Returns ``True`` when a default was applied. Decode CUDA graphs are left
    enabled; this is the canonical equivalent of the legacy
    ``--disable-piecewise-cuda-graph`` behavior.
    """
    normalize_renamed_args(args)
    prefill_backend = getattr(args, "sglang_cuda_graph_backend_prefill", None)
    legacy_disable = getattr(args, "sglang_disable_piecewise_cuda_graph", False)
    legacy_enforce = getattr(args, "sglang_enforce_piecewise_cuda_graph", False)
    if prefill_backend is not None or legacy_disable or legacy_enforce:
        return False

    args.sglang_disable_piecewise_cuda_graph = True
    if hasattr(args, "sglang_cuda_graph_backend_prefill"):
        args.sglang_cuda_graph_backend_prefill = "disabled"
    return True


# TODO: use all sglang router arguments with `--sglang-router` prefix
def add_sglang_router_arguments(parser):
    """
    Add arguments to the parser for the SGLang router.
    """
    parser.add_argument(
        "--sglang-router-ip",
        type=str,
        default=None,
        help="IP address of the SGLang router",
    )
    parser.add_argument(
        "--sglang-router-port",
        type=int,
        default=None,
        help="Port of the SGLang router",
    )
    parser.add_argument(
        "--sglang-router-policy",
        type=str,
        default=None,
        help="Routing policy for the SGLang router (e.g., 'consistent_hashing', 'round_robin')",
    )
    parser.add_argument(
        "--sglang-router-request-timeout-secs",
        type=int,
        default=14400,
        help="Timeout for requests to the SGLang router in seconds",
    )
    return parser


def add_sglang_arguments(parser):
    """
    Add arguments to the parser for the SGLang server.
    """
    parser = add_sglang_router_arguments(parser)
    parser.add_argument("--sglang-server-concurrency", type=int, default=512)
    parser.add_argument(
        "--sglang-dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help=(
            "Explicit in-memory SGLang model dtype. FP32 training checkpoints are cast at load and during live "
            "weight synchronization; auto dtype is deliberately not permitted."
        ),
    )

    old_add_argument = parser.add_argument

    skipped_args = [
        "model_path",
        "config",
        "trust_remote_code",
        "random_seed",
        # memory
        "enable_memory_saver",
        # distributed
        "tp_size",
        "port",
        "nnodes",
        "node_rank",
        "dist_init_addr",
        "gpu_id_step",
        "base_gpu_id",
        "nccl_port",
        "skip_server_warmup",
        "enable_return_routed_experts",
        # Miles owns this explicit, fail-closed option above.
        "dtype",
    ]

    def new_add_argument_wrapper(*name_or_flags, **kwargs):
        """
        Add arguments to the parser, ensuring that the server arguments are prefixed and skippable.
        """
        # Determine the canonical name for skip check (e.g., "model_path")
        canonical_name_for_skip_check = None
        if "dest" in kwargs:
            canonical_name_for_skip_check = kwargs["dest"]
        else:
            for flag_name_candidate in name_or_flags:
                if isinstance(flag_name_candidate, str) and flag_name_candidate.startswith("--"):
                    # Derive from first long flag: --foo-bar -> foo_bar
                    stem = flag_name_candidate[2:]
                    canonical_name_for_skip_check = stem.replace("-", "_")
                    break
            # If no long flag and no dest, skip logic might not catch it unless short flags imply a dest.

        if canonical_name_for_skip_check and canonical_name_for_skip_check in skipped_args:
            return  # Skip this entire argument definition

        # If not skipped, proceed to prefix flags and dest
        new_name_or_flags_list = []
        for item_flag in name_or_flags:
            if isinstance(item_flag, str) and item_flag.startswith("-"):
                original_flag_stem = item_flag.lstrip("-")  # "foo-bar" from "--foo-bar", or "f" from "-f"
                prefixed_item = f"--sglang-{original_flag_stem}"
                new_name_or_flags_list.append(prefixed_item)
            else:
                # Positional arguments or non-string items
                new_name_or_flags_list.append(item_flag)

        # Prepare kwargs for the actual add_argument call.
        # Make a copy to avoid modifying the original kwargs dict.
        final_kwargs = kwargs.copy()

        # If 'dest' is explicitly provided and is a string, prefix it.
        # This ensures the attribute on the args namespace becomes, e.g., args.sglang_dest_name.
        if "dest" in final_kwargs and isinstance(final_kwargs["dest"], str):
            original_dest = final_kwargs["dest"]
            # Avoid double prefixing if dest somehow already starts with sglang_
            if not original_dest.startswith("sglang_"):
                final_kwargs["dest"] = f"sglang_{original_dest}"
        # If 'dest' is not explicitly provided (or is None/not a string),
        # argparse will derive 'dest' from the (now prefixed) flag names.
        # E.g., if the first flag is "--sglang-foo-bar", argparse sets dest to "sglang_foo_bar".

        old_add_argument(*new_name_or_flags_list, **final_kwargs)

    parser.add_argument = new_add_argument_wrapper
    ServerArgs.add_cli_args(parser)
    parser.add_argument = old_add_argument

    parser.add_argument(
        "--sglang-config",
        type=str,
        default=None,
        help=(
            "Path to a YAML config for SGLang engine deployment. "
            "Defines server_groups with worker_type (regular/prefill/decode/placeholder), "
            "num_gpus per group, and optional per-group 'overrides' dict of "
            "ServerArgs field names that override the base --sglang-* CLI args. "
            "Placeholder groups reserve GPU slots without creating engines. "
            "Mutually exclusive with --prefill-num-servers."
        ),
    )

    return parser


def validate_args(args):
    normalize_renamed_args(args)

    args.sglang_tp_size = args.rollout_num_gpus_per_engine

    if args.true_on_policy_mode:
        args.sglang_enable_deterministic_inference = True

    if getattr(args, "recompute_logprobs_via_prefill", False):
        args.sglang_enable_prefill_only_deterministic_inference = True
        args.sglang_enable_deterministic_inference = True

    if args.sglang_dp_size > 1:
        assert getattr(args, "sglang_enable_dp_attention", False)

    if args.sglang_router_policy:
        from miles.utils.environ import enable_experimental_rollout_refactor

        assert (
            not enable_experimental_rollout_refactor()
        ), "--sglang-router-policy is not supported with MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1"

    if getattr(args, "sglang_router_ip", None):
        args.sglang_router_ip = _wrap_ipv6(args.sglang_router_ip)
