from __future__ import annotations

import ast
import json
from typing import Any


def _parse_env_replies(value: Any) -> Any:
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


def env_reply_count(sample: Any) -> int:
    metadata = getattr(sample, "metadata", None) or {}
    if not isinstance(metadata, dict):
        return 0

    value = _parse_env_replies(metadata.get("env_replies"))
    if value in (None, ""):
        return 0
    if isinstance(value, dict):
        value = _parse_env_replies(value.get("env_replies"))
    if isinstance(value, str):
        return 1
    if isinstance(value, (list, tuple)):
        return len(value)
    return 1


def first_sample(group: Any) -> Any:
    item = group[0]
    while isinstance(item, list):
        item = item[0]
    return item


def sort_groups_by_env_reply_count(groups: list[Any]) -> list[Any]:
    """Order known-depth multi-turn samples like Chess-RL's rollout path."""
    return sorted(groups, key=lambda group: env_reply_count(first_sample(group)), reverse=True)
