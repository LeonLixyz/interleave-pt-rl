import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from miles.rollout.base_types import GenerateFnInput
from miles.utils.types import Sample

from chess_rl_miles import rollout


class _Tokenizer:
    bos_token_id = 0
    env_token = "<call_env>"
    unk_token_id = -1

    def _convert_token_to_id(self, token):
        return 99 if token == self.env_token else self.unk_token_id

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "<call_env>":
            return [99]
        if text == "opponent move":
            return [55]
        return [ord(char) for char in text]

    def decode(self, tokens, skip_special_tokens=False):
        del skip_special_tokens
        return "".join("<call_env>" if token == 99 else chr(token) for token in tokens)


def _args(**overrides):
    values = {
        "partial_rollout": False,
        "use_rollout_logprobs": False,
        "use_rollout_routing_replay": False,
        "use_miles_router": True,
        "sglang_router_policy": None,
        "sglang_router_ip": "127.0.0.1",
        "sglang_router_port": 30000,
        "sglang_speculative_algorithm": None,
        "rollout_max_response_len": 32,
        "rollout_max_prompt_len": 32,
        "rollout_max_context_len": 64,
        "chess_context_margin_tokens": 0,
        "chess_max_env_calls": 6,
        "generate_max_retries": 2,
        "eval_generate_max_retries": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_generate_reuses_one_routing_key_across_turns(monkeypatch):
    seen = []
    outputs = iter(
        [
            {
                "output_ids": [99],
                "meta_info": {"finish_reason": {"type": "stop"}},
            },
            {
                "output_ids": [100, 111, 110, 101],
                "meta_info": {"finish_reason": {"type": "stop"}},
            },
        ]
    )

    async def fake_post(url, payload, max_retries, headers):
        seen.append((url, deepcopy(payload), max_retries, headers))
        return next(outputs)

    monkeypatch.setattr(rollout, "post", fake_post)
    monkeypatch.setattr(rollout, "compute_prompt_ids_from_sample", lambda state, sample: [1, 2])

    args = _args()
    sample = Sample(metadata={"env_replies": ["opponent move"]})
    state = SimpleNamespace(args=args, tokenizer=_Tokenizer())
    result = asyncio.run(
        rollout.generate(
            GenerateFnInput(
                state=state,
                sample=sample,
                sampling_params={"max_new_tokens": 32},
                evaluation=False,
            )
        )
    )

    assert result.samples is sample
    assert len(seen) == 2
    assert sample.session_id is not None
    assert seen[0][3] == {"X-SMG-Routing-Key": sample.session_id}
    assert seen[1][3] == seen[0][3]
    assert seen[1][1]["input_ids"] == [0, 1, 2, 99, 55]


def test_routing_headers_do_not_opt_in_unkeyed_routers():
    sample = Sample()

    assert rollout._routing_headers(_args(use_miles_router=False), sample) is None
    assert sample.session_id is None


def test_routing_headers_preserve_existing_session_id():
    sample = Sample(session_id="existing-session")

    assert rollout._routing_headers(_args(), sample) == {"X-SMG-Routing-Key": "existing-session"}


def test_routing_headers_co_locate_prompt_group_siblings():
    first = Sample(group_index=17)
    second = Sample(group_index=17)

    first_headers = rollout._routing_headers(_args(), first)
    second_headers = rollout._routing_headers(_args(), second)

    assert first_headers == {"X-SMG-Routing-Key": "chess-group-17"}
    assert second_headers == first_headers
    assert first.session_id == second.session_id == "chess-group-17"


def test_tokenizer_free_server_requires_token_id_stop():
    with pytest.raises(ValueError, match="requires the chess <call_env> stop token"):
        rollout._sampling_params_with_stop(
            _args(sglang_skip_tokenizer_init=True),
            {"max_new_tokens": 32},
            32,
            None,
        )
