import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest
import requests
import torch


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
SGLANG_ENGINE_PATH = WORKSPACE_ROOT / "miles" / "miles" / "backends" / "sglang_utils" / "sglang_engine.py"
FSDP_UPDATE_PATH = (
    WORKSPACE_ROOT
    / "miles"
    / "miles"
    / "backends"
    / "experimental"
    / "fsdp_utils"
    / "update_weight_utils.py"
)


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_source(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sglang_engine_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "sglang_router", _module("sglang_router", __version__="0.3.0"))
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", _module("sglang.srt.server_args", ServerArgs=object))
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.utils",
        _module("sglang.srt.utils", kill_process_tree=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "miles.backends.megatron_utils.lora_utils",
        _module(
            "miles.backends.megatron_utils.lora_utils",
            LORA_ADAPTER_NAME="default",
            convert_target_modules_to_hf=lambda value: value,
            is_lora_enabled=lambda _args: False,
        ),
    )
    monkeypatch.setitem(sys.modules, "miles.ray.ray_actor", _module("miles.ray.ray_actor", RayActor=object))
    monkeypatch.setitem(
        sys.modules,
        "miles.utils.env_report",
        _module("miles.utils.env_report", collect_and_print_node_env_report=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "miles.utils.http_utils",
        _module("miles.utils.http_utils", get_host_info=lambda: ("host", "127.0.0.1")),
    )
    return _load_source("_miles_sglang_engine_under_test", SGLANG_ENGINE_PATH)


def test_current_server_uses_begin_update_end_payloads(sglang_engine_module):
    engine = object.__new__(sglang_engine_module.SGLangEngine)
    calls = []

    def make_request(endpoint, payload):
        calls.append((endpoint, payload))
        return {"success": True, "message": "Success"}

    engine._make_request = make_request

    assert engine.begin_weight_update() == {"success": True, "message": "Success"}
    assert engine.end_weight_update() == {"success": True, "message": "Success"}
    assert calls == [
        ("begin_weight_update", {"selector": "all"}),
        ("end_weight_update", {}),
    ]
    assert engine._weight_update_sessions_supported is True


@pytest.mark.parametrize("status_code", [404, 405])
def test_legacy_server_caches_transaction_endpoint_as_noop(sglang_engine_module, status_code):
    engine = object.__new__(sglang_engine_module.SGLangEngine)
    calls = []

    def make_request(endpoint, payload):
        calls.append((endpoint, payload))
        response = requests.Response()
        response.status_code = status_code
        raise requests.exceptions.HTTPError(response=response)

    engine._make_request = make_request

    begin_result = engine.begin_weight_update()
    end_result = engine.end_weight_update()

    assert begin_result["success"] is True
    assert begin_result["session_supported"] is False
    assert end_result["success"] is True
    assert end_result["session_supported"] is False
    assert calls == [("begin_weight_update", {"selector": "all"})]
    assert engine._weight_update_sessions_supported is False


def test_non_capability_http_errors_are_not_suppressed(sglang_engine_module):
    engine = object.__new__(sglang_engine_module.SGLangEngine)

    def make_request(_endpoint, _payload):
        response = requests.Response()
        response.status_code = 500
        raise requests.exceptions.HTTPError(response=response)

    engine._make_request = make_request

    with pytest.raises(requests.exceptions.HTTPError):
        engine.begin_weight_update()

    assert not hasattr(engine, "_weight_update_sessions_supported")


def test_missing_end_endpoint_is_not_hidden_after_begin_succeeds(sglang_engine_module):
    engine = object.__new__(sglang_engine_module.SGLangEngine)

    def make_request(endpoint, _payload):
        if endpoint == "begin_weight_update":
            return {"success": True, "message": "Success"}
        response = requests.Response()
        response.status_code = 404
        raise requests.exceptions.HTTPError(response=response)

    engine._make_request = make_request

    engine.begin_weight_update()
    with pytest.raises(requests.exceptions.HTTPError):
        engine.end_weight_update()

    assert engine._weight_update_sessions_supported is True


@pytest.fixture
def fsdp_update_module(monkeypatch):
    fake_ray = _module("ray")
    fake_ray.get = lambda refs: refs
    fake_ray_actor = _module("ray.actor", ActorHandle=object)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.actor", fake_ray_actor)

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.utils.patch_torch",
        _module("sglang.srt.utils.patch_torch", monkey_patch_torch_reductions=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.utils",
        _module("sglang.srt.utils", MultiprocessingSerializer=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.weight_sync.tensor_bucket",
        _module("sglang.srt.weight_sync.tensor_bucket", FlattenedTensorBucket=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "miles.utils.distributed_utils",
        _module(
            "miles.utils.distributed_utils",
            get_gloo_group=lambda: "gloo",
            init_process_group=lambda **_kwargs: None,
        ),
    )
    return _load_source(
        "miles.backends.experimental.fsdp_utils.update_weight_utils",
        FSDP_UPDATE_PATH,
    )


class _RemoteMethod:
    def __init__(self, name, events, result=None):
        self.name = name
        self.events = events
        self.result = {"success": True, "message": "Success"} if result is None else result

    def remote(self, *_args, **_kwargs):
        self.events.append(self.name)
        return self.result


class _Engine:
    def __init__(self, events, begin_result=None, end_result=None):
        self.pause_generation = _RemoteMethod("pause", events)
        self.flush_cache = _RemoteMethod("flush", events)
        self.begin_weight_update = _RemoteMethod("begin", events, begin_result)
        self.end_weight_update = _RemoteMethod("end", events, end_result)
        self.continue_generation = _RemoteMethod("continue", events)


class _Parameter:
    def __init__(self):
        self.dtype = torch.float32

    def is_floating_point(self):
        return True

    def to(self, *, dtype):
        result = _Parameter()
        result.dtype = dtype
        return result

    def numel(self):
        return 1

    def element_size(self):
        return 2

    def cuda(self):
        return self


class _Model:
    def state_dict(self):
        return {"weight": _Parameter()}


class _Distributed:
    def __init__(self, events):
        self.events = events

    def get_rank(self):
        return 0

    def barrier(self, group=None):
        self.events.append(f"barrier:{group}")


def test_fsdp_weight_sync_wraps_all_buckets_in_one_transaction(fsdp_update_module, monkeypatch):
    events = []
    monkeypatch.setattr(fsdp_update_module, "dist", _Distributed(events))

    class Updater(fsdp_update_module.UpdateWeight):
        def connect_rollout_engines(self, *_args, **_kwargs):
            pass

        def update_bucket_weights(self, named_tensors, weight_version=None):
            events.append(f"update:{weight_version}:{len(named_tensors)}")

    updater = Updater(
        Namespace(
            update_weight_buffer_size=1024,
            sglang_dtype="bfloat16",
        ),
        _Model(),
    )
    updater.rollout_engines = [_Engine(events)]

    updater.update_weights()

    assert events == [
        "pause",
        "flush",
        "begin",
        "barrier:gloo",
        "update:1:1",
        "barrier:gloo",
        "end",
        "continue",
        "barrier:gloo",
    ]


def test_session_rpc_rejects_unsuccessful_result(fsdp_update_module):
    engines = [_Engine([], begin_result={"success": False, "message": "session rejected"})]

    with pytest.raises(RuntimeError, match="begin_weight_update.*session rejected"):
        fsdp_update_module._run_weight_update_session_rpc(engines, "begin_weight_update")


def test_session_rpc_accepts_legacy_none_result(fsdp_update_module):
    engines = [_Engine([], begin_result=None)]
    engines[0].begin_weight_update.result = None

    fsdp_update_module._run_weight_update_session_rpc(engines, "begin_weight_update")
