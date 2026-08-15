from __future__ import annotations

import asyncio
import logging
import os
import sys
from copy import copy, deepcopy
from typing import Any

from tqdm import tqdm

from miles.rollout.base_types import (
    GenerateFnInput,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.rollout.generate_utils.prefill_logprobs import recompute_samples_rollout_logprobs_via_prefill
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.rollout.inference_rollout.inference_rollout_eval import eval_rollout_single_dataset
from miles.rollout.inference_rollout.inference_rollout_train import abort
from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.utils import dumper_utils
from miles.utils.http_utils import post
from miles.utils.misc import load_function
from miles.utils.types import Sample

from chess_rl_miles.moves import CALL_ENV_TOKEN, parse_env_replies
from chess_rl_miles.prompt_tokens import (
    ensure_exactly_one_leading_bos,
    leading_bos_evidence,
)
from chess_rl_miles.rollout import (
    _append_tokens,
    _call_env_token_id,
    _context_budget,
    _context_limit,
    _cut_at_call_token,
    _next_env_reply,
    _record_meta,
    _response_from_output,
    _routing_headers,
    _sampling_params_with_stop,
    _tokenize_env_reply,
)

logger = logging.getLogger(__name__)

_BATCH_GENERATE_UNSUPPORTED = False
_DETERMINISTIC_SEED_MODE_ENV = (
    "CHESS_RL_MILES_DETERMINISTIC_SEED_MODE"
)
_BATCH_GENERATE_ATTEMPTS = 0
_BATCH_GENERATE_SUCCESSES = 0
_BATCH_GENERATE_FALLBACKS = 0

_DEFINITIVE_BATCH_REJECTION_STATUS_CODES = frozenset({400, 404, 405, 415, 422})


class _BatchGenerateResponseShapeError(RuntimeError):
    """The server accepted a batch request but did not return one result per input."""


def _is_definitive_batch_incompatibility(exc: Exception) -> bool:
    if isinstance(exc, _BatchGenerateResponseShapeError):
        return True
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is None:
        status_code = getattr(exc, "status_code", None)
    return status_code in _DEFINITIVE_BATCH_REJECTION_STATUS_CODES


def _progress_enabled() -> bool:
    if os.environ.get("MILES_DISABLE_TQDM", "").lower() in {"1", "true", "yes"}:
        return False
    stream = getattr(sys, "stderr", None)
    if stream is None or getattr(stream, "closed", False):
        return False
    try:
        return stream.isatty()
    except Exception:
        return False


def _log_rollout_sample(
    event: str,
    sample: Sample,
    *,
    strict_exact_once: bool,
) -> None:
    if strict_exact_once:
        logger.info(
            "%s rollout sample: outcome redacted for strict exact-once gate "
            "(index=%s, status=%s)",
            event,
            getattr(sample, "index", None),
            getattr(sample, "status", None),
        )
        return
    logger.info(
        "%s rollout sample: %s, label: %s, reward: %s",
        event,
        [str(sample.prompt) + sample.response],
        sample.label,
        sample.reward,
    )


def _sort_groups_by_env_reply_count(groups: list[list[Sample]]) -> list[list[Sample]]:
    """Schedule longer known-depth chess trajectories first to reduce tail idle time."""
    return sorted(groups, key=lambda group: len(parse_env_replies(group[0].metadata)), reverse=True)


def _group_semaphore_capacity(args: Any) -> int:
    group_size = int(args.n_samples_per_prompt)
    total_concurrency = (
        int(args.sglang_server_concurrency)
        * int(args.rollout_num_gpus)
        // int(args.rollout_num_gpus_per_engine)
    )
    if group_size > total_concurrency:
        raise ValueError(
            "Chess batched rollout requires n_samples_per_prompt <= total SGLang concurrency "
            f"({group_size} > {total_concurrency})"
        )
    return total_concurrency // group_size


def _eval_semaphore_capacity(args: Any) -> int:
    per_engine_concurrency = getattr(args, "eval_sglang_server_concurrency", None)
    if per_engine_concurrency is None:
        per_engine_concurrency = args.sglang_server_concurrency
    num_engines = int(args.rollout_num_gpus) // int(args.rollout_num_gpus_per_engine)
    return int(per_engine_concurrency) * num_engines


async def _post_many_generate(
    url: str,
    payloads: list[dict[str, Any]],
    max_retries: int,
    *,
    headers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Post generate requests, using SGLang's batch payload when safe.

    The batch endpoint is an optimization only. If the active SGLang build or
    router definitively rejects batch-shaped /generate requests, we permanently
    fall back to concurrent single-sequence requests for this process. Transient
    transport and server failures fall back only for the current request.
    """
    global _BATCH_GENERATE_ATTEMPTS
    global _BATCH_GENERATE_FALLBACKS
    global _BATCH_GENERATE_SUCCESSES
    global _BATCH_GENERATE_UNSUPPORTED

    if len(payloads) == 1 or _BATCH_GENERATE_UNSUPPORTED:
        return await asyncio.gather(
            *(post(url, payload, max_retries=max_retries, headers=headers) for payload in payloads)
        )

    routed_expert_values = {payload.get("return_routed_experts", False) for payload in payloads}
    if len(routed_expert_values) != 1:
        return await asyncio.gather(
            *(post(url, payload, max_retries=max_retries, headers=headers) for payload in payloads)
        )

    # SGLang accepts one sampling-parameter dict per input. Chess siblings have
    # different remaining token budgets after their first turn, so preserving
    # the heterogeneous list is essential: partitioning by sampling params
    # would turn later turns into serial singleton requests.
    batch_payload = {
        "input_ids": [payload["input_ids"] for payload in payloads],
        "sampling_params": [payload["sampling_params"] for payload in payloads],
        "return_logprob": [payload.get("return_logprob", False) for payload in payloads],
        "return_routed_experts": routed_expert_values.pop(),
    }
    _BATCH_GENERATE_ATTEMPTS += 1
    try:
        batch_output = await post(url, batch_payload, max_retries=1, headers=headers)
        if not isinstance(batch_output, list) or len(batch_output) != len(payloads):
            raise _BatchGenerateResponseShapeError(
                f"Unexpected batch /generate response: {type(batch_output).__name__}, "
                f"expected list[{len(payloads)}]"
            )
        _BATCH_GENERATE_SUCCESSES += 1
        return batch_output
    except Exception as exc:
        _BATCH_GENERATE_FALLBACKS += 1
        definitive_incompatibility = _is_definitive_batch_incompatibility(exc)
        if definitive_incompatibility:
            _BATCH_GENERATE_UNSUPPORTED = True
        logger.warning(
            "Batch /generate failed; falling back to single requests%s: %r",
            " and disabling batching" if definitive_incompatibility else " for this request",
            exc,
        )
        return await asyncio.gather(
            *(post(url, payload, max_retries=max_retries, headers=headers) for payload in payloads)
        )


async def _acquire_sample_slots(state: GenerateState, count: int) -> None:
    for _ in range(count):
        await state.generate_fn_semaphore.acquire()


def _release_sample_slots(state: GenerateState, count: int) -> None:
    for _ in range(count):
        state.generate_fn_semaphore.release()


class _SampleState:
    def __init__(self, sample: Sample, env_replies: list[str]) -> None:
        self.sample = sample
        self.response_tokens: list[int] = []
        self.env_replies = env_replies
        self.env_replies_used: list[str] = []
        self.model_tokens = 0
        self.finish_type: str | None = None


def _deterministic_sampling_seed(
    args,
    sample: Sample,
    sibling_index: int,
) -> tuple[int, str]:
    mode = os.environ.get(_DETERMINISTIC_SEED_MODE_ENV, "sibling-index")
    if mode == "sibling-index":
        return int(args.rollout_seed) + sibling_index, mode
    if mode == "sample-index":
        sample_index = getattr(sample, "index", None)
        if (
            isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or sample_index < 0
        ):
            raise ValueError(
                "sample-index deterministic seeding requires a non-negative "
                "integer Miles sample index"
            )
        return int(args.rollout_seed) + sample_index, mode
    raise ValueError(
        f"unsupported deterministic sampling seed mode: {mode!r}"
    )


def _initialize_sample(input: GenerateFnInput, call_env_token: str) -> _SampleState:
    args = input.args
    sample = input.sample
    tokenizer = input.state.tokenizer

    assert not args.partial_rollout, "Chess batched multi-turn rollout does not support partial rollout yet."
    assert sample.status in {Sample.Status.PENDING, Sample.Status.ABORTED}, f"{sample.status=}"

    from miles.rollout.generate_utils.generate_endpoint_utils import compute_prompt_ids_from_sample

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
    sample.rollout_log_probs = [] if bool(getattr(args, "use_rollout_logprobs", False)) else None
    sample.status = Sample.Status.PENDING

    return _SampleState(sample, parse_env_replies(sample.metadata))


async def generate_group_batched(
    state: GenerateState,
    group: list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> list[Sample]:
    args = state.args
    tokenizer = state.tokenizer
    call_env_token = getattr(args, "chess_call_env_token", CALL_ENV_TOKEN)
    call_env_id = _call_env_token_id(tokenizer, call_env_token)
    require_logprobs = bool(getattr(args, "use_rollout_logprobs", False))
    max_env_calls = int(getattr(args, "chess_max_env_calls", 6))
    max_model_tokens = int(getattr(args, "rollout_max_response_len", sampling_params.get("max_new_tokens", 1024)))
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    headers = _routing_headers(args, group[0])
    if headers is not None:
        for sample in group[1:]:
            sample.session_id = group[0].session_id
    max_retries = (
        getattr(args, "eval_generate_max_retries", 300)
        if evaluation
        else getattr(args, "generate_max_retries", 60)
    )

    group_semaphore = getattr(state, "chess_group_semaphore", None)
    if group_semaphore is not None:
        await group_semaphore.acquire()
    else:
        await _acquire_sample_slots(state, len(group))
    try:
        if state.aborted:
            for sample in group:
                sample.status = Sample.Status.ABORTED
            return group

        sample_states: list[_SampleState] = []
        for idx, sample in enumerate(group):
            current_sampling_params = deepcopy(sampling_params)
            sampling_seed = None
            sampling_seed_mode = None
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_seed, sampling_seed_mode = (
                    _deterministic_sampling_seed(args, sample, idx)
                )
                current_sampling_params["sampling_seed"] = sampling_seed
            sample_state = _initialize_sample(
                GenerateFnInput(
                    state=state,
                    sample=sample,
                    sampling_params=current_sampling_params,
                    evaluation=evaluation,
                ),
                call_env_token,
            )
            if sampling_seed is not None:
                sample_state.sample.metadata["sampling_seed"] = sampling_seed
                sample_state.sample.metadata[
                    "sampling_seed_sibling_index"
                ] = idx
                sample_state.sample.metadata["sampling_seed_mode"] = (
                    sampling_seed_mode
                )
            sample_states.append(sample_state)

        active = list(sample_states)
        sampling_params_by_sample = {id(sample): deepcopy(sampling_params) for sample in group}
        if getattr(args, "sglang_enable_deterministic_inference", False):
            for idx, sample in enumerate(group):
                sampling_seed, _ = _deterministic_sampling_seed(
                    args, sample, idx
                )
                sampling_params_by_sample[id(sample)][
                    "sampling_seed"
                ] = sampling_seed

        for _turn_idx in range(max_env_calls + 1):
            if state.aborted:
                for sample_state in active:
                    sample_state.sample.status = Sample.Status.ABORTED
                break

            request_states: list[_SampleState] = []
            payloads: list[dict[str, Any]] = []
            for sample_state in active:
                sample = sample_state.sample
                remaining_model_tokens = max_model_tokens - sample_state.model_tokens
                max_new_tokens = _context_budget(args, len(sample.tokens), remaining_model_tokens)
                if max_new_tokens <= 0:
                    sample.status = Sample.Status.TRUNCATED
                    continue

                payloads.append(
                    {
                        "input_ids": sample.tokens,
                        "sampling_params": _sampling_params_with_stop(
                            args,
                            sampling_params_by_sample[id(sample)],
                            max_new_tokens,
                            call_env_id,
                        ),
                        "return_logprob": require_logprobs,
                        "return_routed_experts": getattr(args, "use_rollout_routing_replay", False),
                    }
                )
                request_states.append(sample_state)

            if not request_states:
                break

            outputs = await _post_many_generate(
                url,
                payloads,
                max_retries=max_retries,
                headers=headers,
            )
            next_active: list[_SampleState] = []

            for sample_state, output, payload in zip(request_states, outputs, payloads, strict=True):
                sample = sample_state.sample
                meta_info = output.get("meta_info") or {}
                finish_type = (meta_info.get("finish_reason") or {}).get("type")
                sample_state.finish_type = finish_type
                _record_meta(sample, args, meta_info)

                if finish_type == "abort":
                    sample.status = Sample.Status.ABORTED
                    continue

                response_text, new_tokens, new_logprobs = _response_from_output(
                    output,
                    tokenizer,
                    require_logprobs=require_logprobs,
                )
                max_new_tokens = payload["sampling_params"]["max_new_tokens"]
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
                    continue

                _append_tokens(sample, sample_state.response_tokens, new_tokens, new_logprobs, loss_mask_val=1)
                sample_state.model_tokens += len(new_tokens)

                if finish_type == "length" and not hit_call:
                    sample.status = Sample.Status.TRUNCATED
                    continue
                if not hit_call:
                    sample.status = Sample.Status.COMPLETED
                    continue

                obs_text = await _next_env_reply(sample_state.env_replies, len(sample_state.env_replies_used))
                if not obs_text:
                    sample.status = Sample.Status.COMPLETED
                    continue

                sample_state.env_replies_used.append(obs_text)
                obs_tokens = await _tokenize_env_reply(tokenizer, obs_text)
                context_remaining = None
                context_limit = _context_limit(args)
                if context_limit is not None:
                    context_remaining = context_limit - len(sample.tokens)
                    obs_tokens = obs_tokens[: max(context_remaining, 0)]
                if not obs_tokens:
                    sample.status = Sample.Status.TRUNCATED if context_remaining == 0 else Sample.Status.COMPLETED
                    continue

                _append_tokens(sample, sample_state.response_tokens, obs_tokens, [0.0] * len(obs_tokens), loss_mask_val=0)
                if len(sample_state.env_replies_used) >= max_env_calls:
                    sample.status = Sample.Status.COMPLETED
                    continue
                next_active.append(sample_state)

            active = next_active
            if not active:
                break

        for sample_state in sample_states:
            sample = sample_state.sample
            if sample.status == Sample.Status.PENDING:
                sample.status = (
                    Sample.Status.TRUNCATED if sample_state.finish_type == "length" else Sample.Status.COMPLETED
                )
            sample.response = tokenizer.decode(sample_state.response_tokens, skip_special_tokens=False)
            sample.response_length = len(sample_state.response_tokens)
            sample.metadata = sample.metadata or {}
            sample.metadata["n_env_calls"] = len(sample_state.env_replies_used)
            sample.metadata["max_env_calls_used"] = max_env_calls
            sample.metadata["env_replies_used"] = sample_state.env_replies_used
            sample.metadata["model_token_count"] = sample_state.model_tokens
            sample.metadata["env_token_count"] = sample.response_length - sample_state.model_tokens

        if state.aborted:
            return group

        if args.group_rm:
            await batched_async_rm(args, group, inplace_set_reward_field=True)
        else:
            samples_need_reward = [
                sample for sample in group if sample.status != Sample.Status.ABORTED and sample.reward is None
            ]
            rewards = await asyncio.gather(*(async_rm(args, sample) for sample in samples_need_reward))
            for sample, reward in zip(samples_need_reward, rewards, strict=True):
                sample.reward = reward

        return group
    finally:
        if group_semaphore is not None:
            group_semaphore.release()
        else:
            _release_sample_slots(state, len(group))


def _submit_generate_tasks(state: GenerateState, samples: list[list[Sample]]) -> list[asyncio.Task]:
    return [
        asyncio.create_task(
            generate_group_batched(
                state,
                group,
                sampling_params=state.sampling_params.copy(),
                evaluation=False,
            )
        )
        for group in samples
    ]


async def _generate_rollout_async(state: GenerateState, rollout_id: int, data_source) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    args = state.args
    assert args.rollout_global_dataset
    owner = getattr(data_source, "__self__", None)
    strict_exact_once = bool(
        getattr(owner, "strict_exact_once", False)
    )

    await dumper_utils.configure_sglang(args)
    dynamic_filter = load_function(args.dynamic_sampling_filter_path)
    metric_gatherer = MetricGatherer()
    batch_counters_at_start = (
        _BATCH_GENERATE_ATTEMPTS,
        _BATCH_GENERATE_SUCCESSES,
        _BATCH_GENERATE_FALLBACKS,
    )
    target_data_size = args.rollout_batch_size

    pendings: set[asyncio.Task] = set()
    data: list[list[Sample]] = []
    all_data: list[list[Sample]] = []
    do_print = True
    pbar = tqdm(
        total=target_data_size * args.n_samples_per_prompt,
        desc="Chess batched rollout generation",
        disable=not _progress_enabled(),
    )

    while len(data) < target_data_size:
        while len(data) + len(pendings) < target_data_size:
            samples = data_source(args.over_sampling_batch_size)
            samples = _sort_groups_by_env_reply_count(samples)
            pendings.update(_submit_generate_tasks(state, samples))

        done, pendings = await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                group: list[Sample] = task.result()
            except Exception as exc:
                logger.error("[chess-batched-rollout] Task raised exception: %r", exc, exc_info=True)
                owner = getattr(data_source, "__self__", None)
                if bool(getattr(owner, "strict_exact_once", False)):
                    for pending in pendings:
                        pending.cancel()
                    if pendings:
                        await asyncio.gather(
                            *pendings, return_exceptions=True
                        )
                    pbar.close()
                    raise RuntimeError(
                        "strict exact-once rollout gate refuses prompt "
                        "replacement after a generation-task exception"
                    ) from exc
                continue

            if do_print:
                sample = group[0]
                _log_rollout_sample(
                    "First",
                    sample,
                    strict_exact_once=strict_exact_once,
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                continue

            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    sample = data[-1][0]
    _log_rollout_sample(
        "Finish",
        sample,
        strict_exact_once=strict_exact_once,
    )

    if strict_exact_once:
        # With no filtering and an exact 256-prompt pull, reaching the target
        # means every submitted group is complete.  An unconditional abort RPC
        # is both unnecessary and dangerous here: Miles' health watchdog can
        # remove a busy engine immediately before this cleanup call, causing a
        # completed fixed gate to fail while contacting the now-dead worker.
        if pendings:
            for pending in pendings:
                pending.cancel()
            await asyncio.gather(*pendings, return_exceptions=True)
            raise RuntimeError(
                "strict exact-once rollout reached its target with pending "
                "generation tasks"
            )
        aborted_samples: list[list[Sample]] = []
    else:
        aborted_samples = await abort(state, pendings, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0].index)
    all_samples = sorted(all_data, key=lambda group: group[0].index)

    # The normal Miles logger only receives ``data`` after dynamic filtering.
    # Preserve every completed positive attempt (including all-one groups that
    # the dynamic filter drops) for the Exp 4 behavioral replay corpus.  A
    # blinded strict gate is the deliberate exception: writing its positive
    # counts or histogram before the cross-cell barrier would leak aggregate
    # outcomes and is unrelated to any later replay stage.
    if strict_exact_once:
        positive_attempts = 0
    else:
        from chess_rl_miles.io import log_all_attempts_positive

        positive_attempts = log_all_attempts_positive(
            rollout_id,
            args,
            all_samples,
        )
    metrics_positive_attempts = int(positive_attempts)

    state.reset()

    if f := load_function(args.rollout_sample_filter_path):
        f(args, data)
    if f := load_function(args.rollout_all_samples_process_path):
        f(args, all_samples, data_source)

    await recompute_samples_rollout_logprobs_via_prefill(
        args,
        [sample for group in data for sample in group],
        url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
        sampling_params=state.sampling_params,
    )

    metrics = {} if strict_exact_once else metric_gatherer.collect()
    if not strict_exact_once:
        metrics[
            "rollout/chess_positive_attempts/all_before_dynamic_filter"
        ] = metrics_positive_attempts
    metrics.update(
        {
            "rollout/chess_batch_generate/attempts": (
                _BATCH_GENERATE_ATTEMPTS - batch_counters_at_start[0]
            ),
            "rollout/chess_batch_generate/successes": (
                _BATCH_GENERATE_SUCCESSES - batch_counters_at_start[1]
            ),
            "rollout/chess_batch_generate/fallbacks": (
                _BATCH_GENERATE_FALLBACKS - batch_counters_at_start[2]
            ),
            "rollout/chess_batch_generate/unsupported": int(
                _BATCH_GENERATE_UNSUPPORTED
            ),
        }
    )
    return RolloutFnTrainOutput(samples=data, metrics=metrics), aborted_samples


class ChessBatchedRolloutFn:
    def __init__(self, input: RolloutFnConstructorInput):
        self.data_source = input.data_source
        self.state = GenerateState(input.args)
        # Reserve capacity atomically by complete prompt groups. Acquiring the
        # shared per-sample semaphore one slot at a time can deadlock when
        # multiple groups each hold a partial reservation.
        self.state.chess_group_semaphore = asyncio.Semaphore(_group_semaphore_capacity(input.args))
        # The shared Miles eval helper reads generate_fn_semaphore from its
        # GenerateState. Keep a shallow state copy so eval can use its lower
        # concurrency without mutating or racing the training state.
        self.eval_state = copy(self.state)
        self.eval_state.generate_fn_semaphore = asyncio.Semaphore(_eval_semaphore_capacity(input.args))
        self.eval_prompt_dataset_cache: dict[str, Any] = {}

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return await self._call_eval(input)
        return await self._call_train(input)

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        output, aborted_samples = await _generate_rollout_async(
            self.state,
            input.rollout_id,
            self.data_source.get_samples,
        )
        self.data_source.add_samples(aborted_samples)
        return output

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
        assert not self.state.args.group_rm, "Group RM is not supported for eval rollout"
        self.eval_state.aborted = self.state.aborted
        coros = []
        for dataset_cfg in getattr(self.eval_state.args, "eval_datasets", []) or []:
            coros.append(eval_rollout_single_dataset(self.eval_state, dataset_cfg, self.eval_prompt_dataset_cache))
        results_list = await asyncio.gather(*coros)
        results = {k: v for r in results_list for k, v in r.items()}
        return RolloutFnEvalOutput(data=results)
