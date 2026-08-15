from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from miles.utils.types import Sample

from chess_rl_miles.moves import (
    check_move_legality,
    safe_move_to_uci,
    setup_move_uci,
    side_to_move_after_setup,
)
from chess_rl_miles.reward import _score_sample


BLACK_CASTLE_FEN = "4k2r/8/8/8/8/8/P7/4K3 w k - 0 1"
BLACK_SETUP = "a2a3"
WHITE_CASTLE_FEN = "4k3/7p/8/8/8/8/8/R3K3 b Q - 0 1"
WHITE_SETUP = "h7h6"


def _offline_reward_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "eval/reward_function/reward_function_multiturn.py"
    )
    spec = importlib.util.spec_from_file_location(
        "context2048_offline_reward_for_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "move,side,expected",
    [
        ("O-O", "white", "e1g1"),
        ("O-O", "black", "e8g8"),
        ("O-O-O", "white", "e1c1"),
        ("O-O-O", "black", "e8c8"),
    ],
)
def test_castling_conversion_uses_actual_side_to_move(move, side, expected):
    assert safe_move_to_uci(move, side_to_move=side) == expected


def test_puzzle_position_applies_setup_move_before_legality_check():
    metadata = {
        "FEN": BLACK_CASTLE_FEN,
        "Moves": f"{BLACK_SETUP} e8g8",
        "first_move_uci": BLACK_SETUP,
    }

    assert setup_move_uci(metadata) == BLACK_SETUP
    assert side_to_move_after_setup(BLACK_CASTLE_FEN, BLACK_SETUP) == "black"
    assert check_move_legality(BLACK_CASTLE_FEN, "e8g8") == 0.0
    assert (
        check_move_legality(
            BLACK_CASTLE_FEN,
            "e8g8",
            setup_uci=BLACK_SETUP,
        )
        == 1.0
    )


@pytest.mark.parametrize(
    "fen,setup,response,target",
    [
        (BLACK_CASTLE_FEN, BLACK_SETUP, "</T> O-O <call_env>", "e8g8"),
        (WHITE_CASTLE_FEN, WHITE_SETUP, "</T> O-O-O <call_env>", "e1c1"),
    ],
)
def test_online_and_offline_rewards_match_for_castling(
    fen,
    setup,
    response,
    target,
):
    metadata = {
        "FEN": fen,
        "Moves": f"{setup} {target}",
        "first_move_uci": setup,
        "env_replies": [],
        "data_source": "puzzle",
    }
    sample = Sample(
        response=response,
        label=str([target]),
        metadata=metadata,
    )
    online = _score_sample(
        SimpleNamespace(
            chess_multiturn=True,
            chess_reward_model_type="RULE_BASED",
        ),
        sample,
    )

    offline_module = _offline_reward_module()
    offline = offline_module.compute_score_batch(
        ["puzzle"],
        [response],
        [str([target])],
        [metadata],
    )[0]

    assert online["score"] == offline["score"] == 1.0
    assert online["first_move_score"] == offline["first_move_score"] == 1.0
    assert (
        online["first_move_legality_score"]
        == offline["first_move_legality_score"]
        == 1.0
    )
    assert online["extracted_moves"] == offline["extracted_moves"] == target


def test_invalid_setup_move_fails_closed_for_castling_and_legality():
    metadata = {
        "FEN": BLACK_CASTLE_FEN,
        "first_move_uci": "a2a5",
        "env_replies": [],
        "data_source": "puzzle",
    }
    sample = Sample(
        response="</T> O-O <call_env>",
        label="['e8g8']",
        metadata=metadata,
    )

    result = _score_sample(
        SimpleNamespace(
            chess_multiturn=True,
            chess_reward_model_type="RULE_BASED",
        ),
        sample,
    )

    assert result["score"] == 0.0
    assert result["extracted_moves"] == ""
    assert result["first_move_legality_score"] == 0.0


@pytest.mark.parametrize(
    "metadata",
    [
        {"FEN": BLACK_CASTLE_FEN, "env_replies": []},
        {
            "FEN": BLACK_CASTLE_FEN,
            "first_move_uci": BLACK_SETUP,
            "Moves": "a2a4 e8g8",
            "env_replies": [],
        },
    ],
)
def test_missing_or_conflicting_setup_metadata_fails_closed(metadata):
    sample = Sample(
        response="</T> O-O <call_env>",
        label="['e8g8']",
        metadata=metadata,
    )

    online = _score_sample(
        SimpleNamespace(
            chess_multiturn=True,
            chess_reward_model_type="RULE_BASED",
        ),
        sample,
    )
    offline = _offline_reward_module().compute_score_batch(
        ["puzzle"],
        [sample.response],
        [sample.label],
        [metadata],
    )[0]

    assert online["score"] == offline["score"] == 0.0
    assert online["extracted_moves"] == offline["extracted_moves"] == ""
    assert (
        online["first_move_legality_score"]
        == offline["first_move_legality_score"]
        == 0.0
    )
