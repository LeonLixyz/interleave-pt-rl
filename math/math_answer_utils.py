"""Small, dependency-free helpers shared by the math evaluators."""

from __future__ import annotations


_BOXED_PREFIX = r"\boxed{"


def _is_escaped(text: str, index: int) -> bool:
    """Return whether the character at ``index`` has an odd slash prefix."""
    slash_count = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        slash_count += 1
        index -= 1
    return slash_count % 2 == 1


def _find_balanced_close(text: str, open_index: int) -> int | None:
    depth = 1
    for index in range(open_index + 1, len(text)):
        char = text[index]
        if char not in "{}" or _is_escaped(text, index):
            continue
        if char == "{":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return index
    return None


def extract_last_boxed(text: str) -> str | None:
    """Return the contents of the last complete ``\\boxed{...}`` expression.

    Nested, balanced braces are retained in the returned content. Escaped braces
    (``\\{`` and ``\\}``) are treated as literal characters. ``None`` means no
    complete box was found; an empty string represents a complete ``\\boxed{}``.
    """
    last: str | None = None
    search_from = 0

    while True:
        marker_index = text.find(_BOXED_PREFIX, search_from)
        if marker_index < 0:
            return last

        open_index = marker_index + len(_BOXED_PREFIX) - 1
        close_index = _find_balanced_close(text, open_index)
        if close_index is not None:
            last = text[open_index + 1 : close_index]

        # Advance by one so a nested or later box is considered independently.
        search_from = marker_index + 1
