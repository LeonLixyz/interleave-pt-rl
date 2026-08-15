import argparse
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


MILES_ARGUMENTS_PATH = (
    Path(__file__).resolve().parents[3]
    / "miles"
    / "miles"
    / "backends"
    / "sglang_utils"
    / "arguments.py"
)
MILES_ENGINE_PATH = MILES_ARGUMENTS_PATH.with_name("sglang_engine.py")


@dataclass
class _EngineServerArgs:
    model_path: str = ""
    trust_remote_code: bool = False
    random_seed: int = 0
    enable_memory_saver: bool = False
    host: str = ""
    port: int = 0
    nccl_port: int = 0
    nnodes: int = 1
    node_rank: int = 0
    dist_init_addr: str = ""
    gpu_id_step: int = 1
    base_gpu_id: int = 0
    tp_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1
    skip_server_warmup: bool = False
    enable_draft_weights_cpu_backup: bool = False
    dtype: str = "auto"
    context_length: int | None = None


class _CurrentServerArgs:
    @staticmethod
    def add_cli_args(parser):
        parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="auto")
        parser.add_argument("--tp-size", dest="tp_size", type=int, default=1)
        parser.add_argument(
            "--dp-size",
            "--data-parallel-size",
            dest="dp_size",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--pp-size",
            "--pipeline-parallel-size",
            dest="pp_size",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--ep-size",
            "--expert-parallel-size",
            dest="ep_size",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--attn-cp-size",
            "--attention-context-parallel-size",
            dest="attn_cp_size",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--cuda-graph-backend-prefill",
            choices=("disabled", "breakable", "tc_piecewise"),
            default=None,
        )
        parser.add_argument(
            "--disable-piecewise-cuda-graph",
            action="store_const",
            dest="cuda_graph_backend_prefill",
            const="disabled",
        )


@pytest.fixture
def sglang_arguments(monkeypatch):
    fake_server_args = types.ModuleType("sglang.srt.server_args")
    fake_server_args.ServerArgs = _CurrentServerArgs
    fake_http_utils = types.ModuleType("miles.utils.http_utils")
    fake_http_utils._wrap_ipv6 = lambda value: value
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", fake_server_args)
    monkeypatch.setitem(sys.modules, "miles.utils.http_utils", fake_http_utils)

    spec = importlib.util.spec_from_file_location("_miles_sglang_arguments_under_test", MILES_ARGUMENTS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sglang_engine(monkeypatch):
    fake_router = types.ModuleType("sglang_router")
    fake_router.__version__ = "0.3.0"
    fake_server_args = types.ModuleType("sglang.srt.server_args")
    fake_server_args.ServerArgs = _EngineServerArgs
    fake_srt_utils = types.ModuleType("sglang.srt.utils")
    fake_srt_utils.kill_process_tree = lambda *args, **kwargs: None
    fake_lora = types.ModuleType("miles.backends.megatron_utils.lora_utils")
    fake_lora.LORA_ADAPTER_NAME = "default"
    fake_lora.convert_target_modules_to_hf = lambda value: value
    fake_lora.is_lora_enabled = lambda args: False
    fake_ray_actor = types.ModuleType("miles.ray.ray_actor")
    fake_ray_actor.RayActor = object
    fake_env_report = types.ModuleType("miles.utils.env_report")
    fake_env_report.collect_and_print_node_env_report = lambda **kwargs: None
    fake_http_utils = types.ModuleType("miles.utils.http_utils")
    fake_http_utils.get_host_info = lambda: ("host", "127.0.0.1")

    monkeypatch.setitem(sys.modules, "sglang_router", fake_router)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", fake_server_args)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", fake_srt_utils)
    monkeypatch.setitem(
        sys.modules,
        "miles.backends.megatron_utils.lora_utils",
        fake_lora,
    )
    monkeypatch.setitem(sys.modules, "miles.ray.ray_actor", fake_ray_actor)
    monkeypatch.setitem(sys.modules, "miles.utils.env_report", fake_env_report)
    monkeypatch.setitem(sys.modules, "miles.utils.http_utils", fake_http_utils)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    spec = importlib.util.spec_from_file_location(
        "_miles_sglang_engine_under_test",
        MILES_ENGINE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_namespace(**overrides):
    values = {
        "rollout_num_gpus_per_engine": 2,
        "true_on_policy_mode": False,
        "recompute_logprobs_via_prefill": False,
        "sglang_enable_dp_attention": True,
        "sglang_router_policy": None,
        "sglang_router_ip": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_current_parser_destinations_are_canonical(sglang_arguments):
    parser = argparse.ArgumentParser()
    sglang_arguments.add_sglang_arguments(parser)

    args = parser.parse_args(
        [
            "--sglang-data-parallel-size",
            "4",
            "--sglang-pipeline-parallel-size",
            "2",
            "--sglang-expert-parallel-size",
            "8",
            "--sglang-attention-context-parallel-size",
            "3",
            "--sglang-disable-piecewise-cuda-graph",
            "--sglang-dtype",
            "bfloat16",
        ]
    )

    assert args.sglang_dp_size == 4
    assert args.sglang_pp_size == 2
    assert args.sglang_ep_size == 8
    assert args.sglang_attn_cp_size == 3
    assert args.sglang_cuda_graph_backend_prefill == "disabled"
    assert args.sglang_dtype == "bfloat16"
    assert not hasattr(args, "sglang_data_parallel_size")
    assert not hasattr(args, "sglang_disable_piecewise_cuda_graph")


def test_sglang_dtype_is_explicit_and_rejects_auto(sglang_arguments):
    parser = argparse.ArgumentParser()
    sglang_arguments.add_sglang_arguments(parser)
    assert parser.parse_args([]).sglang_dtype == "bfloat16"
    with pytest.raises(SystemExit):
        parser.parse_args(["--sglang-dtype", "auto"])


def test_actual_compute_server_args_pins_bf16_and_native_2048_context(
    sglang_engine,
):
    args = SimpleNamespace(
        rollout_num_gpus_per_engine=1,
        num_gpus_per_node=8,
        hf_checkpoint="/models/context2048",
        seed=42,
        offload_rollout=False,
        sglang_dp_size=1,
        sglang_pp_size=1,
        sglang_ep_size=1,
        sglang_dtype="bfloat16",
        sglang_context_length=2048,
        use_rollout_routing_replay=False,
    )

    server_args, _ = sglang_engine._compute_server_args(
        args,
        rank=0,
        dist_init_addr="127.0.0.1:1234",
        nccl_port=1235,
        host="127.0.0.1",
        port=30000,
        base_gpu_id=0,
    )

    assert server_args["dtype"] == "bfloat16"
    assert server_args["context_length"] == 2048
    assert sglang_engine._validate_actual_server_runtime(
        {"dtype": "torch.bfloat16", "context_length": 2048},
        server_args,
    ) == {
        "dtype": "bfloat16",
        "context_length": 2048,
        "server_version": None,
    }


def test_validate_preserves_current_destinations(sglang_arguments):
    args = _validate_namespace(
        sglang_dp_size=4,
        sglang_pp_size=2,
        sglang_ep_size=8,
        sglang_attn_cp_size=3,
        sglang_cuda_graph_backend_prefill="breakable",
        sglang_data_parallel_size=99,
    )

    sglang_arguments.validate_args(args)

    assert args.sglang_tp_size == 2
    assert (
        args.sglang_dp_size,
        args.sglang_pp_size,
        args.sglang_ep_size,
        args.sglang_attn_cp_size,
    ) == (4, 2, 8, 3)
    assert args.sglang_cuda_graph_backend_prefill == "breakable"
    assert args.sglang_disable_piecewise_cuda_graph is False


def test_validate_accepts_legacy_destinations(sglang_arguments):
    args = _validate_namespace(
        sglang_data_parallel_size=4,
        sglang_pipeline_parallel_size=2,
        sglang_expert_parallel_size=8,
        sglang_attention_context_parallel_size=3,
        sglang_disable_piecewise_cuda_graph=True,
    )

    sglang_arguments.validate_args(args)

    assert (
        args.sglang_dp_size,
        args.sglang_pp_size,
        args.sglang_ep_size,
        args.sglang_attn_cp_size,
    ) == (4, 2, 8, 3)
    assert args.sglang_disable_piecewise_cuda_graph is True


def test_validate_defaults_missing_parallel_destinations_to_one(sglang_arguments):
    args = _validate_namespace()

    sglang_arguments.validate_args(args)

    assert (
        args.sglang_dp_size,
        args.sglang_pp_size,
        args.sglang_ep_size,
        args.sglang_attn_cp_size,
    ) == (1, 1, 1, 1)


def test_colocate_default_disables_only_prefill_cuda_graphs(sglang_arguments):
    args = argparse.Namespace(sglang_cuda_graph_backend_prefill=None)

    assert sglang_arguments.set_colocate_cuda_graph_default(args) is True
    assert args.sglang_cuda_graph_backend_prefill == "disabled"
    assert args.sglang_disable_piecewise_cuda_graph is True
    assert not hasattr(args, "sglang_disable_cuda_graph")


@pytest.mark.parametrize("backend", ["disabled", "breakable", "tc_piecewise"])
def test_colocate_preserves_explicit_current_backend(sglang_arguments, backend):
    args = argparse.Namespace(sglang_cuda_graph_backend_prefill=backend)

    assert sglang_arguments.set_colocate_cuda_graph_default(args) is False
    assert args.sglang_cuda_graph_backend_prefill == backend


def test_colocate_accepts_legacy_piecewise_flags(sglang_arguments):
    disabled = argparse.Namespace(sglang_disable_piecewise_cuda_graph=True)
    enforced = argparse.Namespace(sglang_enforce_piecewise_cuda_graph=True)

    assert sglang_arguments.set_colocate_cuda_graph_default(disabled) is False
    assert sglang_arguments.set_colocate_cuda_graph_default(enforced) is False
    assert not hasattr(disabled, "sglang_cuda_graph_backend_prefill")
    assert not hasattr(enforced, "sglang_cuda_graph_backend_prefill")
