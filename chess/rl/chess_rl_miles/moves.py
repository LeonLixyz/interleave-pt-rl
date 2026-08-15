from __future__ import annotations

import ast
import json
import re
from collections import Counter
from typing import Any

CALL_ENV_TOKEN = "<call_env>"
THINK_END_TOKEN = "</T>"

_LAN_MOVE_RE = re.compile(r"^[PNBRQK][a-h][1-8](x)?[a-h][1-8](=[QRBN])?[+#]?$")
_MOVE_NUMBER_RE = re.compile(r"^\d+\.{1,3}$")


def parse_literal(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            pass
    return value


def parse_ground_truth(label: Any, metadata: dict[str, Any] | None = None) -> list[str]:
    metadata = metadata or {}
    value = parse_literal(label)

    if isinstance(value, dict):
        for key in ("ground_truth", "original_gt", "majority_gt", "answer", "label"):
            if key in value:
                value = parse_literal(value[key])
                break

    if value in (None, ""):
        for key in ("ground_truth", "original_gt", "majority_gt", "answer", "label"):
            if key in metadata:
                value = parse_literal(metadata[key])
                break

    if isinstance(value, dict):
        value = value.get("ground_truth", value)
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item for item in str(value).strip().split() if item]


def parse_env_replies(metadata: dict[str, Any] | None) -> list[str]:
    metadata = metadata or {}
    value = parse_literal(metadata.get("env_replies", []))
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def lan_to_uci(lan: str, side_to_move: str = "white") -> str:
    lan = lan.rstrip("+#").strip()

    if lan == "O-O":
        if side_to_move == "white":
            return "e1g1"
        if side_to_move == "black":
            return "e8g8"
        raise ValueError("Invalid side_to_move for castling")
    if lan == "O-O-O":
        if side_to_move == "white":
            return "e1c1"
        if side_to_move == "black":
            return "e8c8"
        raise ValueError("Invalid side_to_move for castling")

    match = re.match(r"^([PNBRQK])([a-h][1-8])(x)?([a-h][1-8])(=([QRBN]))?$", lan)
    if not match:
        raise ValueError(f"Invalid LAN format: {lan}")

    _, from_sq, _, to_sq, _, promo = match.groups()
    uci = from_sq + to_sq
    if promo:
        uci += promo.lower()
    return uci


def safe_move_to_uci(
    move: str | None,
    *,
    side_to_move: str = "white",
) -> str:
    if not move:
        return ""
    move = move.strip()
    try:
        return lan_to_uci(move, side_to_move=side_to_move)
    except ValueError:
        return ""


def setup_move_uci(metadata: dict[str, Any] | None) -> str:
    """Return the move already shown at the end of a puzzle prompt.

    Lichess puzzle FENs describe the position immediately before the setup
    move.  The model is asked to move only after that move has been played.
    """

    metadata = metadata or {}
    moves = str(metadata.get("Moves") or metadata.get("moves") or "").split()
    moves_setup = moves[0] if moves else ""
    explicit = str(metadata.get("first_move_uci") or "").strip()
    if explicit and moves_setup and explicit != moves_setup:
        return ""
    return explicit or moves_setup


def board_after_setup_move(
    fen: str | None,
    setup_uci: str | None = None,
):
    """Build the position in which the model makes its first move.

    Returns ``None`` for malformed metadata.  This keeps diagnostics and
    castling conversion fail-closed instead of silently evaluating the move
    on the pre-puzzle position.
    """

    if not fen:
        return None
    try:
        import chess

        board = chess.Board(fen)
        if setup_uci:
            move = chess.Move.from_uci(str(setup_uci))
            if move not in board.legal_moves:
                return None
            board.push(move)
        return board
    except Exception:
        return None


def side_to_move_after_setup(
    fen: str | None,
    setup_uci: str | None = None,
) -> str | None:
    board = board_after_setup_move(fen, setup_uci)
    if board is None:
        return None
    return "white" if board.turn else "black"


def is_complete_move(text: str | None) -> bool:
    if not text:
        return False
    move = text.strip().rstrip("+#")
    return move in {"O-O", "O-O-O"} or bool(_LAN_MOVE_RE.match(text.strip()))


def is_move_number(text: str | None) -> bool:
    return bool(text and _MOVE_NUMBER_RE.match(text))


def extract_first_move(text: str) -> str | None:
    for move in text.strip().split():
        if is_move_number(move):
            continue
        if is_complete_move(move):
            return move
    return None


def extract_last_move(text: str) -> str | None:
    for move in reversed(text.strip().split()):
        if is_move_number(move):
            continue
        if is_complete_move(move):
            return move
    return None


def extract_move_after_thinking(text: str, *, strict_single_close: bool = False) -> tuple[str | None, bool]:
    text = text.strip()
    has_closing = THINK_END_TOKEN in text
    follows_format = has_closing and (not strict_single_close or text.count(THINK_END_TOKEN) == 1)
    if not follows_format:
        return None, False

    text_after_thinking = text[text.find(THINK_END_TOKEN) + len(THINK_END_TOKEN) :].strip()
    if not text_after_thinking:
        return None, True
    return extract_first_move(text_after_thinking), True


def extract_all_my_moves(text: str, call_env_token: str = CALL_ENV_TOKEN) -> list[str | None]:
    moves: list[str | None] = []
    for segment in text.split(call_env_token)[:-1]:
        move, _ = extract_move_after_thinking(segment)
        moves.append(move or extract_last_move(segment))
    return moves


def extract_all_my_moves_uci(
    text: str,
    call_env_token: str = CALL_ENV_TOKEN,
    *,
    side_to_move: str = "white",
) -> list[str]:
    return [
        safe_move_to_uci(move, side_to_move=side_to_move)
        for move in extract_all_my_moves(text, call_env_token=call_env_token)
    ]


def majority_vote(values: list[list[str]]) -> tuple[list[str], float]:
    normalized = [tuple(value) for value in values if any(value)]
    if not normalized:
        return [], 0.0
    winner, count = Counter(normalized).most_common(1)[0]
    return list(winner), count / len(values)


def check_move_legality(
    fen: str | None,
    uci_move: str | None,
    *,
    setup_uci: str | None = None,
) -> float:
    if not fen or not uci_move:
        return 0.0
    try:
        import chess

        board = board_after_setup_move(fen, setup_uci)
        if board is None:
            return 0.0
        move = chess.Move.from_uci(uci_move)
        return 1.0 if move in board.legal_moves else 0.0
    except Exception:
        return 0.0
