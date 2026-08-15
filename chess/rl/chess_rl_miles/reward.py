from __future__ import annotations

import asyncio
import os
from typing import Any

from miles.utils.types import Sample

from chess_rl_miles.moves import (
    CALL_ENV_TOKEN,
    check_move_legality,
    extract_all_my_moves,
    extract_first_move,
    extract_move_after_thinking,
    is_complete_move,
    is_move_number,
    majority_vote,
    parse_ground_truth,
    safe_move_to_uci,
    setup_move_uci,
    side_to_move_after_setup,
)

MAJORITY_REWARD_TYPES = {
    "MAJORITY_VOTE",
    "MAJORITY_50_DIFFICULT_RULE_50_EASY",
    "REVERSE_MAJORITY_50_EASY_RULE_50_DIFFICULT",
}


def _reward_model_type(args: Any, sample: Sample | None = None) -> str:
    metadata = getattr(sample, "metadata", None) if sample is not None else None
    metadata = metadata if isinstance(metadata, dict) else {}
    return (
        metadata.get("reward_model_type")
        or getattr(args, "chess_reward_model_type", None)
        or os.environ.get("REWARD_MODEL_TYPE")
        or "RULE_BASED"
    ).upper()


def _is_multiturn(args: Any, sample: Sample) -> bool:
    if getattr(args, "chess_multiturn", None) is not None:
        return bool(args.chess_multiturn)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return bool(metadata.get("env_replies")) or CALL_ENV_TOKEN in sample.response


def _fen(sample: Sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return str(metadata.get("FEN") or metadata.get("fen") or "")


def _data_source(sample: Sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return str(metadata.get("data_source") or metadata.get("source") or "puzzle")


def _setup_move(sample: Sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return setup_move_uci(metadata)


def _model_side_to_move(sample: Sample) -> str | None:
    setup = _setup_move(sample)
    if not setup:
        return None
    return side_to_move_after_setup(_fen(sample), setup)


def _sample_move_to_uci(sample: Sample, move: str | None) -> str:
    side = _model_side_to_move(sample)
    stripped = str(move or "").rstrip("+#").strip()
    if stripped in {"O-O", "O-O-O"} and side is None:
        return ""
    return safe_move_to_uci(move, side_to_move=side or "white")


def _first_move_legality(sample: Sample, uci_move: str | None) -> float:
    setup = _setup_move(sample)
    if not setup:
        return 0.0
    return check_move_legality(
        _fen(sample),
        uci_move,
        setup_uci=setup,
    )


def _score_multiturn(sample: Sample, target_moves: list[str], reward_type: str, majority_target: list[str] | None = None):
    extracted_ucis = [
        _sample_move_to_uci(sample, move)
        for move in extract_all_my_moves(
            sample.response,
            call_env_token=CALL_ENV_TOKEN,
        )
    ]
    active_target = majority_target if majority_target is not None else target_moves

    if not extracted_ucis:
        score = 0.0
    else:
        score = float(len(extracted_ucis) == len(active_target) and all(a == b for a, b in zip(extracted_ucis, active_target, strict=False)))

    first_pred = extracted_ucis[0] if extracted_ucis else ""
    first_gt = target_moves[0] if target_moves else ""
    result = {
        "score": score,
        "ground_truth": str(target_moves),
        "reward_method": "CHESS_MULTITURN_PARSING" if reward_type == "RULE_BASED" else reward_type,
        "extracted_moves": ",".join(extracted_ucis),
        "target_moves": ",".join(active_target),
        "original_target_moves": ",".join(target_moves),
        "data_source": _data_source(sample),
        "first_move_score": float(bool(first_pred) and first_pred == first_gt),
        "first_move_legality_score": _first_move_legality(sample, first_pred),
    }
    if majority_target is not None:
        result["majority_gt"] = ",".join(majority_target)
        result["ground_truth_score"] = float(
            len(extracted_ucis) == len(target_moves) and all(a == b for a, b in zip(extracted_ucis, target_moves, strict=False))
        )
    return result


def _score_eval_only_nonthink(sample: Sample, target_moves: list[str], reward_type: str):
    extracted_moves = []
    for token in sample.response.strip().split():
        if is_move_number(token):
            continue
        if is_complete_move(token):
            extracted_moves.append(token.strip())

    correct = 0
    total_player = 0
    for idx in range(0, len(target_moves), 2):
        total_player += 1
        if idx < len(extracted_moves):
            pred_uci = _sample_move_to_uci(sample, extracted_moves[idx])
            if pred_uci == target_moves[idx]:
                correct += 1

    extracted_move = extracted_moves[0] if extracted_moves else ""
    first_pred = _sample_move_to_uci(sample, extracted_move)
    first_gt = target_moves[0] if target_moves else ""
    return {
        "score": float(correct / total_player) if total_player > 0 else 0.0,
        "ground_truth": str(target_moves),
        "reward_method": "CHESS_MOVE_PARSING",
        "extracted_move": extracted_move,
        "extracted_move_uci": first_pred,
        "format_score": 0.0,
        "first_move_score": float(bool(first_pred) and first_pred == first_gt),
        "first_move_legality_score": _first_move_legality(sample, first_pred),
        "data_source": _data_source(sample),
    }


def _score_single_turn(sample: Sample, target_moves: list[str], reward_type: str, majority_target: list[str] | None = None):
    if reward_type == "EVAL_ONLY_NONTHINK_BASED":
        return _score_eval_only_nonthink(sample, target_moves, reward_type)

    extracted_move, follows_format = extract_move_after_thinking(sample.response, strict_single_close=True)
    if extracted_move is None and not follows_format:
        extracted_move = extract_first_move(sample.response)

    extracted_uci = _sample_move_to_uci(sample, extracted_move)
    active_target = majority_target if majority_target is not None else target_moves
    processed_gt = active_target[0] if active_target else ""

    score = float(bool(extracted_uci) and extracted_uci == processed_gt)
    if reward_type == "RULE_FORMAT_BASED" and not follows_format:
        score = 0.0

    result = {
        "score": score,
        "ground_truth": str(target_moves),
        "reward_method": "CHESS_MOVE_PARSING" if reward_type == "RULE_BASED" else reward_type,
        "extracted_move": extracted_move or "",
        "extracted_move_uci": extracted_uci,
        "format_score": float(follows_format),
        "first_move_score": float(bool(extracted_uci) and extracted_uci == (target_moves[0] if target_moves else "")),
        "first_move_legality_score": _first_move_legality(sample, extracted_uci),
        "data_source": _data_source(sample),
    }
    if majority_target is not None:
        result["majority_gt"] = ",".join(majority_target)
        result["ground_truth_score"] = float(bool(extracted_uci) and extracted_uci == (target_moves[0] if target_moves else ""))
    return result


def _should_use_majority(args: Any, sample: Sample, reward_type: str) -> bool:
    if reward_type == "MAJORITY_VOTE":
        return True

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    difficulty = metadata.get("difficulty", metadata.get("Rating"))
    try:
        difficulty_val = float(difficulty)
    except (TypeError, ValueError):
        difficulty_val = None

    threshold = float(getattr(args, "chess_difficulty_threshold", 1500))
    is_difficult = difficulty_val is not None and difficulty_val >= threshold
    if reward_type == "MAJORITY_50_DIFFICULT_RULE_50_EASY":
        return is_difficult
    if reward_type == "REVERSE_MAJORITY_50_EASY_RULE_50_DIFFICULT":
        return not is_difficult
    return False


def _score_sample(
    args: Any,
    sample: Sample,
    *,
    reward_type: str | None = None,
    majority_target: list[str] | None = None,
    majority_ratio: float | None = None,
):
    reward_type = reward_type or _reward_model_type(args, sample)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    target_moves = parse_ground_truth(sample.label, metadata)

    if _is_multiturn(args, sample):
        result = _score_multiturn(sample, target_moves, reward_type, majority_target=majority_target)
    else:
        result = _score_single_turn(sample, target_moves, reward_type, majority_target=majority_target)

    if majority_ratio is not None:
        result["majority_ratio"] = float(majority_ratio)
    return result


def _candidate_moves(args: Any, sample: Sample) -> list[str]:
    if _is_multiturn(args, sample):
        return [
            _sample_move_to_uci(sample, move)
            for move in extract_all_my_moves(
                sample.response,
                call_env_token=CALL_ENV_TOKEN,
            )
        ]
    move, follows_format = extract_move_after_thinking(sample.response, strict_single_close=True)
    if move is None and not follows_format:
        move = extract_first_move(sample.response)
    uci = _sample_move_to_uci(sample, move)
    return [uci] if uci else []


def _score_group(args: Any, samples: list[Sample]):
    reward_type = _reward_model_type(args, samples[0] if samples else None)
    use_majority = bool(samples) and reward_type in MAJORITY_REWARD_TYPES and _should_use_majority(args, samples[0], reward_type)

    majority_target = None
    majority_ratio = None
    if use_majority:
        majority_target, majority_ratio = majority_vote([_candidate_moves(args, sample) for sample in samples])

    return [
        _score_sample(
            args,
            sample,
            reward_type=reward_type,
            majority_target=majority_target if use_majority else None,
            majority_ratio=majority_ratio if use_majority else None,
        )
        for sample in samples
    ]


async def reward_func(args: Any, samples: Sample | list[Sample], **kwargs):
    if isinstance(samples, list):
        return await asyncio.to_thread(_score_group, args, samples)
    if not isinstance(samples, Sample):
        raise TypeError(f"Expected Sample or list[Sample], got {type(samples).__name__}")
    return await asyncio.to_thread(_score_sample, args, samples)


def add_reward_arguments(parser):
    parser.add_argument("--chess-reward-model-type", type=str, default=None)
    parser.add_argument("--chess-multiturn", action="store_true", default=None)
    parser.add_argument("--chess-singleturn", action="store_false", dest="chess_multiturn")
    parser.add_argument("--chess-difficulty-threshold", type=float, default=1500.0)
    return parser


reward_func.add_arguments = add_reward_arguments
