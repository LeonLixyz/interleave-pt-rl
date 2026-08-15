"""Exact tokenizer identity shared by RL checkpoint validation.

The PT/SFT package is not mounted in the RL Modal image, so importing its
contract by a sibling filesystem path would make the production deployment
fragile.  This module intentionally carries the same small canonical mapping
and pins its independently computed digest.  A special-token-only check is
insufficient: swapping any two ordinary chess tokens changes the meaning of
the corresponding embedding rows.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


_BASE_TOKENS = (
    ["<bos>", "<eos>", "<unk>"]
    + list("KQRBNP")
    + [f"{file}{rank}" for file in "abcdefgh" for rank in "12345678"]
    + ["x", "=", "+", "#", "O-O", "O-O-O", ".", "..."]
)
EXPECTED_RL_VOCAB_85 = {
    token: token_id
    for token_id, token in enumerate(
        [*dict.fromkeys(_BASE_TOKENS), "<T>", "</T>", "<sep>", "<call_env>"]
    )
}
EXPECTED_RL_VOCAB_MAPPING_SHA256 = (
    "f0366c5dc44ada849282959e67b172da79264c0b9336707c03648c430ccf0651"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def vocab_mapping_sha256(vocab: Mapping[str, int]) -> str:
    return hashlib.sha256(_canonical_json(dict(vocab))).hexdigest()


def validate_exact_rl_vocab(vocab: Mapping[str, Any]) -> dict[str, int]:
    """Require the complete canonical 85-token token-to-ID mapping."""

    if not isinstance(vocab, Mapping):
        raise RuntimeError("tokenizer vocabulary must be a mapping")
    normalized: dict[str, int] = {}
    for token, token_id in vocab.items():
        if not isinstance(token, str) or not token:
            raise RuntimeError("tokenizer vocabulary keys must be nonempty strings")
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise RuntimeError(
                f"tokenizer ID for {token!r} must be an integer, got {token_id!r}"
            )
        normalized[token] = token_id
    if normalized != EXPECTED_RL_VOCAB_85:
        mismatches = [
            f"{token!r}: observed={normalized.get(token)!r}, expected={token_id}"
            for token, token_id in EXPECTED_RL_VOCAB_85.items()
            if normalized.get(token) != token_id
        ]
        unexpected = sorted(set(normalized) - set(EXPECTED_RL_VOCAB_85))
        raise RuntimeError(
            "production RL requires the complete exact 85-token tokenizer "
            "mapping; "
            + "; ".join((mismatches + [f"unexpected={unexpected}"])[:12])
        )
    observed_sha256 = vocab_mapping_sha256(normalized)
    if observed_sha256 != EXPECTED_RL_VOCAB_MAPPING_SHA256:
        raise RuntimeError("canonical RL tokenizer mapping digest drifted")
    return normalized


if vocab_mapping_sha256(EXPECTED_RL_VOCAB_85) != EXPECTED_RL_VOCAB_MAPPING_SHA256:
    raise RuntimeError("embedded RL tokenizer contract digest is invalid")


__all__ = [
    "EXPECTED_RL_VOCAB_85",
    "EXPECTED_RL_VOCAB_MAPPING_SHA256",
    "validate_exact_rl_vocab",
    "vocab_mapping_sha256",
]
