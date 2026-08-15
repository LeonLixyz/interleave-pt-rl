from __future__ import annotations

import argparse
import asyncio
import uuid
from collections import OrderedDict
from copy import deepcopy
from threading import Lock
from typing import Any

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import compute_prompt_ids_from_sample
from miles.utils.http_utils import post
from miles.utils.types import Sample

from chess_rl_miles.moves import CALL_ENV_TOKEN, parse_env_replies
from chess_rl_miles.prompt_tokens import (
    ensure_exactly_one_leading_bos,
    leading_bos_evidence,
)

_TOKEN_CACHE_LIMIT = 200_000
_TOKEN_CACHE: OrderedDict[tuple[int, str], list[int]] = OrderedDict()
_TOKEN_CACHE_LOCK = Lock()
_ROUTING_KEY_HEADER = "X-SMG-Routing-Key"


def _tokenize_cached(tokenizer, text: str) -> list[int]:
    key = (id(tokenizer), text)
    with _TOKEN_CACHE_LOCK:
        cached = _TOKEN_CACHE.get(key)
        if cached is not None:
            _TOKEN_CACHE.move_to_end(key)
            return list(cached)

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    with _TOKEN_CACHE_LOCK:
        _TOKEN_CACHE[key] = list(token_ids)
        _TOKEN_CACHE.move_to_end(key)
        while len(_TOKEN_CACHE) > _TOKEN_CACHE_LIMIT:
            _TOKEN_CACHE.popitem(last=False)
    return list(token_ids)


async def _tokenize_env_reply(tokenizer, text: str) -> list[int]:
    return await asyncio.to_thread(_tokenize_cached, tokenizer, text)


def _call_env_token_id(tokenizer, call_env_token: str) -> int | None:
    env_token = getattr(tokenizer, "env_token", call_env_token)
    token_id = None
    if hasattr(tokenizer, "_convert_token_to_id"):
        token_id = tokenizer._convert_token_to_id(env_token)
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        token_ids = tokenizer.encode(call_env_token, add_special_tokens=False)
        if len(token_ids) == 1:
            token_id = token_ids[0]
    return token_id


def _sampling_params_with_stop(args: Any, sampling_params: dict[str, Any], max_new_tokens: int, call_env_id: int | None):
    params = deepcopy(sampling_params)
    params["max_new_tokens"] = max(max_new_tokens, 1)
    params["no_stop_trim"] = True
    params["spaces_between_special_tokens"] = False

    if getattr(args, "sglang_skip_tokenizer_init", False) and call_env_id is None:
        raise ValueError(
            "--sglang-skip-tokenizer-init requires the chess <call_env> stop token "
            "to resolve to a token ID; string stop sequences are unavailable when "
            "the SGLang tokenizer is disabled."
        )

    if call_env_id is not None:
        stop_ids = list(params.get("stop_token_ids") or [])
        if call_env_id not in stop_ids:
            stop_ids.append(call_env_id)
        params["stop_token_ids"] = stop_ids
    else:
        stops = params.get("stop")
        if stops is None:
            stops = []
        elif isinstance(stops, str):
            stops = [stops]
        else:
            stops = list(stops)
        if CALL_ENV_TOKEN not in stops:
            stops.append(CALL_ENV_TOKEN)
        params["stop"] = stops
    return params


def _response_from_output(output: dict[str, Any], tokenizer, *, require_logprobs: bool):
    meta_info = output.get("meta_info") or {}
    token_logprobs = meta_info.get("output_token_logprobs")
    if token_logprobs is not None:
        tokens = [item[1] for item in token_logprobs]
        logprobs = [float(item[0]) for item in token_logprobs]
        return output.get("text") or tokenizer.decode(tokens, skip_special_tokens=False), tokens, logprobs

    if require_logprobs:
        raise RuntimeError("SGLang response did not include output_token_logprobs while --use-rollout-logprobs is set")

    output_ids = output.get("output_ids")
    if output_ids is not None:
        tokens = [int(token_id) for token_id in output_ids]
        return output.get("text") or tokenizer.decode(tokens, skip_special_tokens=False), tokens, [0.0] * len(tokens)

    text = output.get("text") or ""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return text, tokens, [0.0] * len(tokens)


def _cut_at_call_token(
    text: str,
    tokens: list[int],
    logprobs: list[float],
    tokenizer,
    call_env_id: int | None,
    call_env_token: str,
):
    if call_env_id is not None and call_env_id in tokens:
        cut = tokens.index(call_env_id) + 1
        return True, tokens[:cut], logprobs[:cut], tokenizer.decode(tokens[:cut], skip_special_tokens=False)

    idx = text.find(call_env_token)
    if idx < 0:
        return False, tokens, logprobs, text

    cut_text = text[: idx + len(call_env_token)]
    cut_tokens = tokenizer.encode(cut_text, add_special_tokens=False)
    if len(cut_tokens) <= len(tokens):
        return True, cut_tokens, logprobs[: len(cut_tokens)], cut_text
    return True, tokens, logprobs, text


def _append_tokens(
    sample: Sample,
    response_tokens: list[int],
    tokens: list[int],
    logprobs: list[float],
    loss_mask_val: int,
):
    sample.tokens.extend(tokens)
    response_tokens.extend(tokens)
    sample.loss_mask.extend([loss_mask_val] * len(tokens))
    if sample.rollout_log_probs is not None:
        sample.rollout_log_probs.extend(logprobs)
    sample.response_length = len(response_tokens)


def _record_meta(sample: Sample, args: Any, meta_info: dict[str, Any]):
    if getattr(args, "sglang_speculative_algorithm", None):
        sample.spec_info.add(meta_info)
    sample.prefix_cache_info.add(meta_info)
    if "weight_version" in meta_info:
        sample.weight_versions.append(meta_info["weight_version"])


async def _next_env_reply(env_replies: list[str], turn_idx: int) -> str | None:
    return await asyncio.to_thread(lambda: env_replies[turn_idx] if turn_idx < len(env_replies) else None)


def _context_limit(args: Any) -> int | None:
    max_context_len = getattr(args, "rollout_max_context_len", None)
    if max_context_len is None:
        return None
    # The configured SGLang/policy context is the actual hard budget.  A
    # hidden fallback margin made the 2048-token profile silently behave like
    # 2032 tokens in the batched rollout path, whose class has no custom CLI
    # argument registration.
    margin = max(0, int(getattr(args, "chess_context_margin_tokens", 0)))
    return max_context_len - margin


def _context_budget(args: Any, total_tokens: int, remaining_model_tokens: int) -> int:
    context_limit = _context_limit(args)
    if context_limit is None:
        return remaining_model_tokens
    return min(remaining_model_tokens, context_limit - total_tokens)


def _routing_headers(args: Any, sample: Sample) -> dict[str, str] | None:
    """Return a stable prompt-group routing key when the active router supports it."""
    uses_consistent_hashing = getattr(args, "sglang_router_policy", None) == "consistent_hashing"
    if not getattr(args, "use_miles_router", False) and not uses_consistent_hashing:
        return None

    if sample.group_index is not None:
        sample.session_id = f"chess-group-{sample.group_index}"
    elif sample.session_id is None:
        sample.session_id = uuid.uuid4().hex
    return {_ROUTING_KEY_HEADER: sample.session_id}


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = input.sample
    tokenizer = input.state.tokenizer
    call_env_token = getattr(args, "chess_call_env_token", CALL_ENV_TOKEN)
    call_env_id = _call_env_token_id(tokenizer, call_env_token)

    assert not args.partial_rollout, "Chess multi-turn rollout does not support partial rollout yet."
    assert sample.status in {Sample.Status.PENDING, Sample.Status.ABORTED}, f"{sample.status=}"

    prompt_ids = list(compute_prompt_ids_from_sample(input.state, sample))
    # Training sequences always start with <bos>; the prompt must replicate that
    # frame. compute_prompt_ids_from_sample uses add_special_tokens=False, so
    # prepend bos explicitly (missing bos was the train/eval mismatch bug).
    prompt_ids = ensure_exactly_one_leading_bos(prompt_ids, tokenizer)
    prompt_cap = int(getattr(args, "rollout_max_prompt_len", 0))
    if prompt_cap <= 0 or len(prompt_ids) > prompt_cap:
        raise RuntimeError(
            "Post-BOS chess rollout prompt exceeds the configured prompt cap: "
            f"tokens={len(prompt_ids)} cap={prompt_cap}"
        )
    sample.metadata = sample.metadata or {}
    sample.metadata.update(leading_bos_evidence(prompt_ids, tokenizer))
    sample.tokens = list(prompt_ids)
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    require_logprobs = bool(getattr(args, "use_rollout_logprobs", False))
    sample.rollout_log_probs = [] if require_logprobs else None
    sample.status = Sample.Status.PENDING

    response_tokens: list[int] = []
    env_replies = parse_env_replies(sample.metadata)
    env_replies_used: list[str] = []
    max_env_calls = int(getattr(args, "chess_max_env_calls", 6))
    max_model_tokens = int(getattr(args, "rollout_max_response_len", input.sampling_params.get("max_new_tokens", 1024)))
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    headers = _routing_headers(args, sample)

    model_tokens = 0
    finish_type = None

    for turn_idx in range(max_env_calls + 1):
        remaining_model_tokens = max_model_tokens - model_tokens
        max_new_tokens = _context_budget(args, len(sample.tokens), remaining_model_tokens)
        if max_new_tokens <= 0:
            sample.status = Sample.Status.TRUNCATED
            break

        payload = {
            "input_ids": sample.tokens,
            "sampling_params": _sampling_params_with_stop(args, input.sampling_params, max_new_tokens, call_env_id),
            "return_logprob": require_logprobs,
            "return_routed_experts": getattr(args, "use_rollout_routing_replay", False),
        }
        max_retries = (
            getattr(args, "eval_generate_max_retries", 300)
            if input.evaluation
            else getattr(args, "generate_max_retries", 60)
        )
        output = await post(url, payload, max_retries=max_retries, headers=headers)
        meta_info = output.get("meta_info") or {}
        finish_type = (meta_info.get("finish_reason") or {}).get("type")
        _record_meta(sample, args, meta_info)

        if finish_type == "abort":
            sample.status = Sample.Status.ABORTED
            break

        response_text, new_tokens, new_logprobs = _response_from_output(
            output,
            tokenizer,
            require_logprobs=require_logprobs,
        )
        hit_call, new_tokens, new_logprobs, response_text = _cut_at_call_token(
            response_text,
            new_tokens[:max_new_tokens],
            new_logprobs[:max_new_tokens],
            tokenizer,
            call_env_id,
            call_env_token,
        )

        if not new_tokens:
            sample.status = Sample.Status.COMPLETED
            break

        _append_tokens(sample, response_tokens, new_tokens, new_logprobs, loss_mask_val=1)
        model_tokens += len(new_tokens)

        if finish_type == "length" and not hit_call:
            sample.status = Sample.Status.TRUNCATED
            break
        if not hit_call:
            sample.status = Sample.Status.COMPLETED
            break

        obs_text = await _next_env_reply(env_replies, len(env_replies_used))
        if not obs_text:
            sample.status = Sample.Status.COMPLETED
            break

        env_replies_used.append(obs_text)
        obs_tokens = await _tokenize_env_reply(tokenizer, obs_text)
        context_remaining = None
        context_limit = _context_limit(args)
        if context_limit is not None:
            context_remaining = context_limit - len(sample.tokens)
            obs_tokens = obs_tokens[: max(context_remaining, 0)]
        if not obs_tokens:
            sample.status = Sample.Status.TRUNCATED if context_remaining == 0 else Sample.Status.COMPLETED
            break

        _append_tokens(sample, response_tokens, obs_tokens, [0.0] * len(obs_tokens), loss_mask_val=0)
        if len(env_replies_used) >= max_env_calls:
            sample.status = Sample.Status.COMPLETED
            break
    else:
        sample.status = Sample.Status.COMPLETED

    if sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.TRUNCATED if finish_type == "length" else Sample.Status.COMPLETED

    sample.response = tokenizer.decode(response_tokens, skip_special_tokens=False)
    sample.response_length = len(response_tokens)
    sample.metadata = sample.metadata or {}
    sample.metadata["n_env_calls"] = len(env_replies_used)
    sample.metadata["max_env_calls_used"] = max_env_calls
    sample.metadata["env_replies_used"] = env_replies_used
    sample.metadata["model_token_count"] = model_tokens
    sample.metadata["env_token_count"] = sample.response_length - model_tokens
    return GenerateFnOutput(samples=sample)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--chess-max-env-calls", type=int, default=6)
    parser.add_argument("--chess-call-env-token", type=str, default=CALL_ENV_TOKEN)
    parser.add_argument(
        "--chess-context-margin-tokens",
        type=int,
        default=0,
        help="Reserved positions below rollout_max_context_len; exact-context runs use zero.",
    )
    return parser


generate.add_arguments = _add_arguments
