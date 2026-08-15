import asyncio
import importlib
import inspect
import sys
import types
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest


def _install_module(monkeypatch, name, *, package=False, **attributes):
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)

    parent_name, _, child_name = name.rpartition(".")
    if parent_name and parent_name in sys.modules:
        monkeypatch.setattr(sys.modules[parent_name], child_name, module, raising=False)
    return module


@pytest.fixture
def batched_rollout(monkeypatch):
    """Import the dispatch module without requiring the GPU Miles runtime locally."""

    class Placeholder:
        pass

    class GenerateFnInput:
        def __init__(self, *, state, sample, sampling_params, evaluation):
            self.state = state
            self.sample = sample
            self.sampling_params = sampling_params
            self.evaluation = evaluation

    async def async_unused(*args, **kwargs):
        del args, kwargs

    def unused(*args, **kwargs):
        del args, kwargs

    _install_module(monkeypatch, "miles", package=True)
    _install_module(monkeypatch, "miles.rollout", package=True)
    _install_module(monkeypatch, "miles.rollout.filter_hub", package=True)
    _install_module(monkeypatch, "miles.rollout.generate_utils", package=True)
    _install_module(monkeypatch, "miles.rollout.inference_rollout", package=True)
    _install_module(monkeypatch, "miles.utils", package=True)

    _install_module(
        monkeypatch,
        "miles.rollout.base_types",
        GenerateFnInput=GenerateFnInput,
        RolloutFnConstructorInput=Placeholder,
        RolloutFnEvalInput=Placeholder,
        RolloutFnEvalOutput=Placeholder,
        RolloutFnInput=Placeholder,
        RolloutFnOutput=Placeholder,
        RolloutFnTrainInput=Placeholder,
        RolloutFnTrainOutput=Placeholder,
    )
    _install_module(
        monkeypatch,
        "miles.rollout.filter_hub.base_types",
        MetricGatherer=Placeholder,
        call_dynamic_filter=unused,
    )
    _install_module(
        monkeypatch,
        "miles.rollout.generate_utils.prefill_logprobs",
        recompute_samples_rollout_logprobs_via_prefill=async_unused,
    )
    _install_module(
        monkeypatch,
        "miles.rollout.inference_rollout.inference_rollout_common",
        GenerateState=Placeholder,
    )
    _install_module(
        monkeypatch,
        "miles.rollout.inference_rollout.inference_rollout_eval",
        eval_rollout_single_dataset=async_unused,
    )
    _install_module(
        monkeypatch,
        "miles.rollout.inference_rollout.inference_rollout_train",
        abort=async_unused,
    )
    _install_module(
        monkeypatch,
        "miles.rollout.rm_hub",
        async_rm=async_unused,
        batched_async_rm=async_unused,
    )
    _install_module(monkeypatch, "miles.utils.dumper_utils")
    _install_module(monkeypatch, "miles.utils.http_utils", post=async_unused)
    _install_module(monkeypatch, "miles.utils.misc", load_function=unused)
    _install_module(monkeypatch, "miles.utils.types", Sample=Placeholder)

    import chess_rl_miles

    rollout_stub = _install_module(
        monkeypatch,
        "chess_rl_miles.rollout",
        _append_tokens=unused,
        _call_env_token_id=unused,
        _context_budget=unused,
        _context_limit=unused,
        _cut_at_call_token=unused,
        _next_env_reply=async_unused,
        _record_meta=unused,
        _response_from_output=unused,
        _routing_headers=unused,
        _sampling_params_with_stop=unused,
        _tokenize_env_reply=async_unused,
    )
    monkeypatch.setattr(chess_rl_miles, "rollout", rollout_stub, raising=False)
    monkeypatch.delitem(sys.modules, "chess_rl_miles.batched_rollout", raising=False)

    module = importlib.import_module("chess_rl_miles.batched_rollout")
    monkeypatch.setattr(module, "_BATCH_GENERATE_UNSUPPORTED", False)
    monkeypatch.setattr(module, "_BATCH_GENERATE_ATTEMPTS", 0)
    monkeypatch.setattr(module, "_BATCH_GENERATE_SUCCESSES", 0)
    monkeypatch.setattr(module, "_BATCH_GENERATE_FALLBACKS", 0)
    return module


def _payload(token, *, temperature=1.0):
    return {
        "input_ids": [token],
        "sampling_params": {"temperature": temperature, "max_new_tokens": 4},
        "return_logprob": False,
        "return_routed_experts": False,
    }


def test_batch_dispatch_preserves_order_and_headers(batched_rollout, monkeypatch):
    calls = []
    headers = {"X-SMG-Routing-Key": "chess-group-7"}

    async def fake_post(url, payload, max_retries, headers=None):
        calls.append((url, deepcopy(payload), max_retries, headers))
        assert isinstance(payload["input_ids"][0], list)
        return [{"token": input_ids[0]} for input_ids in payload["input_ids"]]

    monkeypatch.setattr(batched_rollout, "post", fake_post)
    payloads = [
        _payload(0, temperature=0.7),
        _payload(1, temperature=0.8),
        _payload(2, temperature=0.7),
        _payload(3, temperature=0.8),
    ]

    outputs = asyncio.run(
        batched_rollout._post_many_generate(
            "http://router/generate",
            payloads,
            max_retries=9,
            headers=headers,
        )
    )

    assert [output["token"] for output in outputs] == [0, 1, 2, 3]
    assert len(calls) == 1
    assert calls[0][1]["input_ids"] == [[0], [1], [2], [3]]
    assert calls[0][1]["sampling_params"] == [payload["sampling_params"] for payload in payloads]
    assert calls[0][2] == 1
    assert calls[0][3] is headers
    assert batched_rollout._BATCH_GENERATE_ATTEMPTS == 1
    assert batched_rollout._BATCH_GENERATE_SUCCESSES == 1
    assert batched_rollout._BATCH_GENERATE_FALLBACKS == 0


def test_single_dispatch_forwards_headers(batched_rollout, monkeypatch):
    calls = []
    headers = {"X-SMG-Routing-Key": "chess-group-8"}

    async def fake_post(url, payload, max_retries, headers=None):
        calls.append((url, deepcopy(payload), max_retries, headers))
        return {"token": payload["input_ids"][0]}

    monkeypatch.setattr(batched_rollout, "post", fake_post)

    outputs = asyncio.run(
        batched_rollout._post_many_generate(
            "http://router/generate",
            [_payload(8)],
            max_retries=11,
            headers=headers,
        )
    )

    assert outputs == [{"token": 8}]
    assert calls == [("http://router/generate", _payload(8), 11, headers)]


def test_invalid_batch_falls_back_in_order_and_stays_on_fallback(batched_rollout, monkeypatch):
    calls = []
    headers = {"X-SMG-Routing-Key": "chess-group-9"}

    async def fake_post(url, payload, max_retries, headers=None):
        calls.append((url, deepcopy(payload), max_retries, headers))
        if isinstance(payload["input_ids"][0], list):
            return {"unexpected": "not-a-list"}
        return {"token": payload["input_ids"][0]}

    monkeypatch.setattr(batched_rollout, "post", fake_post)
    payloads = [_payload(4), _payload(5)]

    first_outputs = asyncio.run(
        batched_rollout._post_many_generate(
            "http://router/generate",
            payloads,
            max_retries=13,
            headers=headers,
        )
    )
    second_outputs = asyncio.run(
        batched_rollout._post_many_generate(
            "http://router/generate",
            payloads,
            max_retries=13,
            headers=headers,
        )
    )

    assert [output["token"] for output in first_outputs] == [4, 5]
    assert [output["token"] for output in second_outputs] == [4, 5]
    assert batched_rollout._BATCH_GENERATE_UNSUPPORTED is True
    assert len(calls) == 5
    assert calls[0][2] == 1
    assert all(call[2] == 13 for call in calls[1:])
    assert all(call[3] is headers for call in calls)
    assert sum(isinstance(call[1]["input_ids"][0], list) for call in calls) == 1
    assert batched_rollout._BATCH_GENERATE_ATTEMPTS == 1
    assert batched_rollout._BATCH_GENERATE_SUCCESSES == 0
    assert batched_rollout._BATCH_GENERATE_FALLBACKS == 1


@pytest.mark.parametrize("failure_kind", ["read_error", "http_500"])
def test_transient_batch_failure_retries_batching_on_next_call(
    batched_rollout,
    monkeypatch,
    failure_kind,
):
    calls = []
    batch_attempts = 0

    async def fake_post(url, payload, max_retries, headers=None):
        nonlocal batch_attempts
        del headers
        calls.append((deepcopy(payload), max_retries))
        if isinstance(payload["input_ids"][0], list):
            batch_attempts += 1
            if batch_attempts == 1:
                request = httpx.Request("POST", url)
                if failure_kind == "read_error":
                    raise httpx.ReadError("transient router disconnect", request=request)
                response = httpx.Response(500, request=request)
                raise httpx.HTTPStatusError(
                    "transient router error",
                    request=request,
                    response=response,
                )
            return [{"token": input_ids[0]} for input_ids in payload["input_ids"]]
        return {"token": payload["input_ids"][0]}

    monkeypatch.setattr(batched_rollout, "post", fake_post)
    payloads = [_payload(10), _payload(11)]

    first_outputs = asyncio.run(
        batched_rollout._post_many_generate(
            "http://router/generate",
            payloads,
            max_retries=7,
        )
    )
    second_outputs = asyncio.run(
        batched_rollout._post_many_generate(
            "http://router/generate",
            payloads,
            max_retries=7,
        )
    )

    assert [output["token"] for output in first_outputs] == [10, 11]
    assert [output["token"] for output in second_outputs] == [10, 11]
    assert batched_rollout._BATCH_GENERATE_UNSUPPORTED is False
    assert batch_attempts == 2
    batch_max_retries = [
        max_retries
        for payload, max_retries in calls
        if isinstance(payload["input_ids"][0], list)
    ]
    assert batch_max_retries == [
        1,
        1,
    ]
    assert batched_rollout._BATCH_GENERATE_ATTEMPTS == 2
    assert batched_rollout._BATCH_GENERATE_SUCCESSES == 1
    assert batched_rollout._BATCH_GENERATE_FALLBACKS == 1


@pytest.mark.parametrize("status_code", [400, 404, 405, 415, 422])
def test_definitive_http_batch_rejection_stays_disabled(
    batched_rollout,
    monkeypatch,
    status_code,
):
    calls = []

    async def fake_post(url, payload, max_retries, headers=None):
        del headers
        calls.append((deepcopy(payload), max_retries))
        if isinstance(payload["input_ids"][0], list):
            request = httpx.Request("POST", url)
            response = httpx.Response(status_code, request=request)
            raise httpx.HTTPStatusError(
                "batch request is unsupported",
                request=request,
                response=response,
            )
        return {"token": payload["input_ids"][0]}

    monkeypatch.setattr(batched_rollout, "post", fake_post)
    payloads = [_payload(12), _payload(13)]

    for _ in range(2):
        outputs = asyncio.run(
            batched_rollout._post_many_generate(
                "http://router/generate",
                payloads,
                max_retries=5,
            )
        )
        assert [output["token"] for output in outputs] == [12, 13]

    assert batched_rollout._BATCH_GENERATE_UNSUPPORTED is True
    assert sum(isinstance(payload["input_ids"][0], list) for payload, _ in calls) == 1
    assert batched_rollout._BATCH_GENERATE_ATTEMPTS == 1
    assert batched_rollout._BATCH_GENERATE_SUCCESSES == 0
    assert batched_rollout._BATCH_GENERATE_FALLBACKS == 1


def test_group_rollout_forwards_one_stable_header(batched_rollout, monkeypatch):
    class Status:
        PENDING = "pending"
        ABORTED = "aborted"
        COMPLETED = "completed"
        TRUNCATED = "truncated"

    class Sample:
        def __init__(self):
            self.group_index = 17
            self.status = Sample.Status.PENDING
            self.tokens = []
            self.response = ""
            self.response_length = 0
            self.loss_mask = []
            self.rollout_log_probs = None
            self.reward = 1.0
            self.metadata = {}

    Sample.Status = Status

    class Tokenizer:
        def decode(self, tokens, skip_special_tokens=False):
            del tokens, skip_special_tokens
            return "done"

    args = SimpleNamespace(
        chess_call_env_token="<call_env>",
        use_rollout_logprobs=False,
        chess_max_env_calls=1,
        rollout_max_response_len=4,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        generate_max_retries=3,
        sglang_enable_deterministic_inference=False,
        use_rollout_routing_replay=False,
        group_rm=False,
    )
    state = SimpleNamespace(
        args=args,
        tokenizer=Tokenizer(),
        generate_fn_semaphore=asyncio.Semaphore(2),
        aborted=False,
    )
    samples = [Sample(), Sample()]
    expected_headers = {"X-SMG-Routing-Key": "chess-group-17"}
    routed = []
    dispatched = []

    def fake_routing_headers(received_args, received_sample):
        routed.append((received_args, received_sample))
        received_sample.session_id = "chess-group-17"
        return expected_headers

    def fake_initialize(input, call_env_token):
        del call_env_token
        input.sample.tokens = [1]
        return batched_rollout._SampleState(input.sample, [])

    async def fake_post_many(url, payloads, max_retries, *, headers=None):
        dispatched.append((url, deepcopy(payloads), max_retries, headers))
        return [{"meta_info": {"finish_reason": {"type": "stop"}}} for _ in payloads]

    def fake_append(sample, response_tokens, tokens, logprobs, loss_mask_val):
        del logprobs
        sample.tokens.extend(tokens)
        response_tokens.extend(tokens)
        sample.loss_mask.extend([loss_mask_val] * len(tokens))

    monkeypatch.setattr(batched_rollout, "Sample", Sample)
    monkeypatch.setattr(batched_rollout, "_routing_headers", fake_routing_headers)
    monkeypatch.setattr(batched_rollout, "_call_env_token_id", lambda tokenizer, token: None)
    monkeypatch.setattr(batched_rollout, "_initialize_sample", fake_initialize)
    monkeypatch.setattr(batched_rollout, "_context_budget", lambda args, total, remaining: remaining)
    monkeypatch.setattr(
        batched_rollout,
        "_sampling_params_with_stop",
        lambda args, params, max_new_tokens, call_env_id: {"max_new_tokens": max_new_tokens},
    )
    monkeypatch.setattr(batched_rollout, "_post_many_generate", fake_post_many)
    monkeypatch.setattr(batched_rollout, "_record_meta", lambda sample, args, meta: None)
    monkeypatch.setattr(
        batched_rollout,
        "_response_from_output",
        lambda output, tokenizer, require_logprobs: ("done", [7], [0.0]),
    )
    monkeypatch.setattr(
        batched_rollout,
        "_cut_at_call_token",
        lambda text, tokens, logprobs, tokenizer, call_env_id, call_env_token: (
            False,
            tokens,
            logprobs,
            text,
        ),
    )
    monkeypatch.setattr(batched_rollout, "_append_tokens", fake_append)

    result = asyncio.run(
        batched_rollout.generate_group_batched(
            state,
            samples,
            sampling_params={"max_new_tokens": 4},
        )
    )

    assert result == samples
    assert routed == [(args, samples[0])]
    assert [sample.session_id for sample in samples] == ["chess-group-17", "chess-group-17"]
    assert len(dispatched) == 1
    assert dispatched[0][3] is expected_headers


def test_deterministic_sibling_seeds_are_42_to_49_across_continuations(
    batched_rollout,
    monkeypatch,
):
    class Status:
        PENDING = "pending"
        ABORTED = "aborted"
        COMPLETED = "completed"
        TRUNCATED = "truncated"

    class Sample:
        def __init__(self, index):
            self.index = index
            self.group_index = 23
            self.status = Sample.Status.PENDING
            self.tokens = []
            self.response = ""
            self.response_length = 0
            self.loss_mask = []
            self.rollout_log_probs = None
            self.reward = 0.0
            self.metadata = {}

    Sample.Status = Status

    class Tokenizer:
        def decode(self, tokens, skip_special_tokens=False):
            del tokens, skip_special_tokens
            return "done"

    args = SimpleNamespace(
        chess_call_env_token="<call_env>",
        use_rollout_logprobs=False,
        chess_max_env_calls=2,
        rollout_max_response_len=8,
        rollout_seed=42,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        generate_max_retries=3,
        sglang_enable_deterministic_inference=True,
        use_rollout_routing_replay=False,
        group_rm=False,
    )
    state = SimpleNamespace(
        args=args,
        tokenizer=Tokenizer(),
        generate_fn_semaphore=asyncio.Semaphore(8),
        aborted=False,
    )
    samples = [Sample(index) for index in range(8)]
    dispatched_seed_vectors = []
    dispatch_number = 0

    def fake_initialize(input, call_env_token):
        del call_env_token
        input.sample.tokens = [1]
        return batched_rollout._SampleState(input.sample, [])

    async def fake_post_many(
        url, payloads, max_retries, *, headers=None
    ):
        nonlocal dispatch_number
        del url, max_retries, headers
        dispatch_number += 1
        dispatched_seed_vectors.append(
            [
                payload["sampling_params"]["sampling_seed"]
                for payload in payloads
            ]
        )
        return [
            {
                "turn": dispatch_number,
                "meta_info": {"finish_reason": {"type": "stop"}},
            }
            for _ in payloads
        ]

    def fake_append(
        sample, response_tokens, tokens, logprobs, loss_mask_val
    ):
        del logprobs
        sample.tokens.extend(tokens)
        response_tokens.extend(tokens)
        sample.loss_mask.extend([loss_mask_val] * len(tokens))

    async def fake_env_reply(env_replies, index):
        del env_replies, index
        return "environment reply"

    monkeypatch.setattr(batched_rollout, "Sample", Sample)
    monkeypatch.setattr(
        batched_rollout, "_routing_headers", lambda args, sample: None
    )
    monkeypatch.setattr(
        batched_rollout,
        "_call_env_token_id",
        lambda tokenizer, token: 99,
    )
    monkeypatch.setattr(
        batched_rollout, "_initialize_sample", fake_initialize
    )
    monkeypatch.setattr(
        batched_rollout,
        "_context_budget",
        lambda args, total, remaining: remaining,
    )
    monkeypatch.setattr(
        batched_rollout, "_context_limit", lambda args: None
    )
    monkeypatch.setattr(
        batched_rollout,
        "_sampling_params_with_stop",
        lambda args, params, max_new_tokens, call_env_id: {
            **params,
            "max_new_tokens": max_new_tokens,
        },
    )
    monkeypatch.setattr(
        batched_rollout, "_post_many_generate", fake_post_many
    )
    monkeypatch.setattr(
        batched_rollout, "_record_meta", lambda sample, args, meta: None
    )
    monkeypatch.setattr(
        batched_rollout,
        "_response_from_output",
        lambda output, tokenizer, require_logprobs: (
            "call" if output["turn"] == 1 else "done",
            [7],
            [0.0],
        ),
    )
    monkeypatch.setattr(
        batched_rollout,
        "_cut_at_call_token",
        lambda text, tokens, logprobs, tokenizer, call_env_id, call_env_token: (
            text == "call",
            tokens,
            logprobs,
            text,
        ),
    )
    monkeypatch.setattr(batched_rollout, "_append_tokens", fake_append)
    monkeypatch.setattr(
        batched_rollout, "_next_env_reply", fake_env_reply
    )
    monkeypatch.setattr(
        batched_rollout,
        "_tokenize_env_reply",
        lambda tokenizer, text: _async_value([8, 9]),
    )

    result = asyncio.run(
        batched_rollout.generate_group_batched(
            state,
            samples,
            sampling_params={"temperature": 1.0, "max_new_tokens": 8},
        )
    )

    assert result == samples
    assert dispatched_seed_vectors == [
        list(range(42, 50)),
        list(range(42, 50)),
    ]
    assert [
        sample.metadata["sampling_seed"] for sample in samples
    ] == list(range(42, 50))
    assert [
        sample.metadata["sampling_seed_sibling_index"]
        for sample in samples
    ] == list(range(8))


def test_groups_are_sorted_by_known_env_depth(batched_rollout):
    groups = [
        [SimpleNamespace(metadata={"env_replies": ["a"]})],
        [SimpleNamespace(metadata={"env_replies": ["a", "b", "c"]})],
        [SimpleNamespace(metadata={"env_replies": []})],
    ]

    sorted_groups = batched_rollout._sort_groups_by_env_reply_count(groups)

    assert [len(group[0].metadata["env_replies"]) for group in sorted_groups] == [3, 1, 0]


def test_gate_sampling_seed_uses_global_sample_index(
    batched_rollout, monkeypatch
):
    monkeypatch.setenv(
        "CHESS_RL_MILES_DETERMINISTIC_SEED_MODE",
        "sample-index",
    )
    seed, mode = batched_rollout._deterministic_sampling_seed(
        SimpleNamespace(rollout_seed=1_567_877_051),
        SimpleNamespace(index=2_049),
        sibling_index=1,
    )

    assert seed == 1_567_879_100
    assert mode == "sample-index"


def test_group_semaphore_capacity_counts_complete_groups(batched_rollout):
    args = SimpleNamespace(
        n_samples_per_prompt=8,
        sglang_server_concurrency=64,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=1,
    )

    assert batched_rollout._group_semaphore_capacity(args) == 64


def test_group_semaphore_rejects_group_larger_than_total_concurrency(batched_rollout):
    args = SimpleNamespace(
        n_samples_per_prompt=16,
        sglang_server_concurrency=1,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=1,
    )

    with pytest.raises(ValueError, match="16 > 8"):
        batched_rollout._group_semaphore_capacity(args)


def test_eval_uses_its_configured_concurrency_without_mutating_training_state(
    batched_rollout,
    monkeypatch,
):
    class FakeGenerateState:
        def __init__(self, args):
            self.args = args
            self.generate_fn_semaphore = asyncio.Semaphore(
                args.sglang_server_concurrency
                * args.rollout_num_gpus
                // args.rollout_num_gpus_per_engine
            )
            self.aborted = False

    args = SimpleNamespace(
        n_samples_per_prompt=8,
        sglang_server_concurrency=128,
        eval_sglang_server_concurrency=16,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=1,
        group_rm=False,
        eval_datasets=["chess-eval"],
    )
    constructor_input = SimpleNamespace(args=args, data_source=object())
    seen_states = []

    async def fake_eval(state, dataset_cfg, prompt_dataset_cache):
        del prompt_dataset_cache
        seen_states.append(state)
        return {dataset_cfg: {"rewards": []}}

    monkeypatch.setattr(batched_rollout, "GenerateState", FakeGenerateState)
    monkeypatch.setattr(batched_rollout, "eval_rollout_single_dataset", fake_eval)
    monkeypatch.setattr(
        batched_rollout,
        "RolloutFnEvalOutput",
        lambda *, data: SimpleNamespace(data=data),
    )

    rollout_fn = batched_rollout.ChessBatchedRolloutFn(constructor_input)
    training_semaphore = rollout_fn.state.generate_fn_semaphore

    result = asyncio.run(rollout_fn._call_eval(SimpleNamespace()))

    assert training_semaphore._value == 128 * 8
    assert rollout_fn.state.generate_fn_semaphore is training_semaphore
    assert rollout_fn.eval_state.generate_fn_semaphore._value == 16 * 8
    assert seen_states == [rollout_fn.eval_state]
    assert result.data == {"chess-eval": {"rewards": []}}


def test_eval_concurrency_defaults_to_training_concurrency(batched_rollout):
    args = SimpleNamespace(
        sglang_server_concurrency=64,
        eval_sglang_server_concurrency=None,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=2,
    )

    assert batched_rollout._eval_semaphore_capacity(args) == 64 * 4


def test_generate_logs_positive_attempts_before_all_one_dynamic_drop(
    batched_rollout,
    monkeypatch,
):
    class Gatherer:
        def on_dynamic_filter_drop(self, reason):
            assert reason == "all_one"

        def collect(self):
            return {}

    class State:
        def __init__(self):
            self.args = SimpleNamespace(
                rollout_global_dataset=True,
                dynamic_sampling_filter_path="dynamic-filter",
                rollout_batch_size=1,
                n_samples_per_prompt=2,
                over_sampling_batch_size=1,
                rollout_sample_filter_path=None,
                rollout_all_samples_process_path=None,
                sglang_router_ip="127.0.0.1",
                sglang_router_port=30000,
            )
            self.sampling_params = {}
            self.aborted = False
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

    def sample(group_index, sample_index, score):
        return SimpleNamespace(
            group_index=group_index,
            index=sample_index,
            prompt="prompt",
            response="response",
            label="label",
            reward={"score": score},
            metadata={},
        )

    dropped_all_one = [
        sample(1, 0, 1.0),
        sample(1, 1, 1.0),
    ]
    accepted_mixed = [
        sample(2, 2, 1.0),
        sample(2, 3, 0.0),
    ]
    queued = iter([[dropped_all_one], [accepted_mixed]])
    captured = {}

    async def return_group(group):
        return group

    def submit(state, groups):
        del state
        return [asyncio.create_task(return_group(group)) for group in groups]

    def dynamic_filter(filter_fn, args, group):
        del filter_fn, args
        if group[0].group_index == 1:
            return SimpleNamespace(keep=False, reason="all_one")
        return SimpleNamespace(keep=True, reason=None)

    def log_attempts(rollout_id, args, all_samples):
        del args
        captured["rollout_id"] = rollout_id
        captured["groups"] = all_samples
        return sum(
            sample.reward["score"] == 1.0
            for group in all_samples
            for sample in group
        )

    _install_module(
        monkeypatch,
        "chess_rl_miles.io",
        log_all_attempts_positive=log_attempts,
    )
    monkeypatch.setattr(batched_rollout, "MetricGatherer", Gatherer)
    monkeypatch.setattr(batched_rollout, "load_function", lambda path: None)
    monkeypatch.setattr(batched_rollout, "call_dynamic_filter", dynamic_filter)
    monkeypatch.setattr(batched_rollout, "_submit_generate_tasks", submit)
    monkeypatch.setattr(
        batched_rollout.dumper_utils,
        "configure_sglang",
        lambda args: _async_value(None),
        raising=False,
    )
    monkeypatch.setattr(
        batched_rollout,
        "abort",
        lambda state, pendings, rollout_id: _async_value([]),
    )
    monkeypatch.setattr(
        batched_rollout,
        "recompute_samples_rollout_logprobs_via_prefill",
        lambda *args, **kwargs: _async_value(None),
    )
    monkeypatch.setattr(
        batched_rollout,
        "RolloutFnTrainOutput",
        lambda *, samples, metrics: SimpleNamespace(
            samples=samples,
            metrics=metrics,
        ),
    )

    state = State()
    result, aborted = asyncio.run(
        batched_rollout._generate_rollout_async(
            state,
            rollout_id=7,
            data_source=lambda count: next(queued),
        )
    )

    assert aborted == []
    assert result.samples == [accepted_mixed]
    assert captured["rollout_id"] == 7
    assert [
        group[0].group_index for group in captured["groups"]
    ] == [1, 2]
    assert result.metrics[
        "rollout/chess_positive_attempts/all_before_dynamic_filter"
    ] == 3


def test_strict_exact_once_source_fails_on_generation_task_exception(
    batched_rollout, monkeypatch
):
    class Gatherer:
        def collect(self):
            return {}

    class State:
        def __init__(self):
            self.args = SimpleNamespace(
                rollout_global_dataset=True,
                rollout_batch_size=1,
                n_samples_per_prompt=1,
                over_sampling_batch_size=1,
                dynamic_sampling_filter_path=None,
            )

    class StrictSource:
        strict_exact_once = True

        def get_samples(self, count):
            assert count == 1
            return [[SimpleNamespace(metadata={})]]

    async def fail_group(group):
        del group
        raise RuntimeError("synthetic generation failure")

    def submit(state, groups):
        del state
        return [
            asyncio.create_task(fail_group(group)) for group in groups
        ]

    monkeypatch.setattr(batched_rollout, "MetricGatherer", Gatherer)
    monkeypatch.setattr(batched_rollout, "load_function", lambda path: None)
    monkeypatch.setattr(batched_rollout, "_submit_generate_tasks", submit)
    monkeypatch.setattr(
        batched_rollout.dumper_utils,
        "configure_sglang",
        lambda args: _async_value(None),
        raising=False,
    )

    source = StrictSource()
    with pytest.raises(
        RuntimeError,
        match="refuses prompt replacement",
    ):
        asyncio.run(
            batched_rollout._generate_rollout_async(
                State(),
                rollout_id=0,
                data_source=source.get_samples,
            )
        )


def test_strict_gate_source_disables_positive_attempt_aggregation(
    batched_rollout,
):
    source = inspect.getsource(
        batched_rollout._generate_rollout_async
    )

    assert "if strict_exact_once:" in source
    strict_branch = source.split("if strict_exact_once:", 1)[1]
    assert "positive_attempts = 0" in strict_branch
    assert (
        strict_branch.index("positive_attempts = 0")
        < strict_branch.index("log_all_attempts_positive")
    )
    assert "metrics = {} if strict_exact_once" in source
    assert (
        '"rollout/chess_positive_attempts/'
        'all_before_dynamic_filter"'
    ) in source


def test_strict_gate_sample_logs_redact_outcome_fields(
    batched_rollout, caplog
):
    sample = SimpleNamespace(
        index=17,
        status="completed",
        prompt="SECRET_PROMPT",
        response="SECRET_RESPONSE",
        label="SECRET_LABEL",
        reward={"score": 1.0},
    )

    with caplog.at_level("INFO"):
        batched_rollout._log_rollout_sample(
            "First",
            sample,
            strict_exact_once=True,
        )

    rendered = caplog.text
    assert "outcome redacted" in rendered
    assert "index=17" in rendered
    for secret in (
        "SECRET_PROMPT",
        "SECRET_RESPONSE",
        "SECRET_LABEL",
        "score",
    ):
        assert secret not in rendered


async def _async_value(value):
    return value
