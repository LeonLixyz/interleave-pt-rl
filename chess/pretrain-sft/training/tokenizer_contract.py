"""Canonical tokenizer identities for the controlled 81/85-token runs.

Special-token spot checks are not enough for a weights-only handoff: swapping
two ordinary chess tokens preserves the vocabulary size and all special token
IDs while changing the meaning of two embedding rows.  This module therefore
defines and hashes the complete token-to-ID mapping used by production.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


_BASE_TOKENS = (
    ["<bos>", "<eos>", "<unk>"]
    + list("KQRBNP")
    + [f"{file}{rank}" for file in "abcdefgh" for rank in "12345678"]
    + ["x", "=", "+", "#", "O-O", "O-O-O", ".", "..."]
)
EXPECTED_VOCAB_81 = {
    token: token_id for token_id, token in enumerate(dict.fromkeys(_BASE_TOKENS))
}
EXPECTED_VOCAB_85 = {
    **EXPECTED_VOCAB_81,
    "<T>": 81,
    "</T>": 82,
    "<sep>": 83,
    "<call_env>": 84,
}
EXPECTED_VOCABS = {81: EXPECTED_VOCAB_81, 85: EXPECTED_VOCAB_85}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_vocab_mapping(
    vocab: Mapping[str, Any],
    *,
    expected_size: int | None = None,
) -> dict[str, int]:
    """Return a strict token-to-ID mapping and reject lossy coercions."""

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
        normalized[token] = int(token_id)
    size = len(normalized)
    if expected_size is not None and size != int(expected_size):
        raise RuntimeError(
            f"tokenizer vocabulary size drifted: {size} != {int(expected_size)}"
        )
    if set(normalized.values()) != set(range(size)):
        raise RuntimeError("tokenizer IDs must be a bijection over the vocabulary")
    return normalized


def vocab_mapping_sha256(vocab: Mapping[str, Any]) -> str:
    normalized = normalize_vocab_mapping(vocab)
    return hashlib.sha256(_canonical_json(normalized)).hexdigest()


def expected_vocab_mapping(expected_size: int) -> dict[str, int]:
    try:
        return dict(EXPECTED_VOCABS[int(expected_size)])
    except KeyError as exc:
        raise ValueError(
            f"unsupported tokenizer contract size {expected_size}"
        ) from exc


def validate_expected_vocab_mapping(
    vocab: Mapping[str, Any],
    *,
    expected_size: int,
) -> dict[str, int]:
    """Require equality with the complete production mapping."""

    observed = normalize_vocab_mapping(vocab, expected_size=expected_size)
    expected = expected_vocab_mapping(expected_size)
    if observed != expected:
        mismatches = [
            f"{token!r}: observed={observed.get(token)!r}, expected={token_id}"
            for token, token_id in expected.items()
            if observed.get(token) != token_id
        ]
        unexpected = sorted(set(observed) - set(expected))
        raise RuntimeError(
            "complete tokenizer mapping drifted; "
            + "; ".join((mismatches + [f"unexpected={unexpected}"])[:12])
        )
    return observed


def validate_hf_tokenizer_contract(
    path: Path,
    *,
    expected_vocab_size: int,
    expected_context_length: int,
) -> dict[str, Any]:
    """Validate and fingerprint every tokenizer/model handoff field."""

    path = Path(path)
    expected_vocab_size = int(expected_vocab_size)
    vocab_path = path / "vocab.json"
    tokenizer_config_path = path / "tokenizer_config.json"
    model_config_path = path / "config.json"
    for required in (vocab_path, tokenizer_config_path, model_config_path):
        if not required.is_file() or required.is_symlink():
            raise RuntimeError(
                f"HF tokenizer contract lacks regular file {required}"
            )
    try:
        vocab_value = json.loads(vocab_path.read_text(encoding="utf-8"))
        tokenizer_config = json.loads(
            tokenizer_config_path.read_text(encoding="utf-8")
        )
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid tokenizer JSON under {path}") from exc
    if not isinstance(tokenizer_config, Mapping) or not isinstance(
        model_config, Mapping
    ):
        raise RuntimeError("tokenizer and model configs must be JSON mappings")
    vocab = validate_expected_vocab_mapping(
        vocab_value,
        expected_size=expected_vocab_size,
    )
    expected_special_tokens = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<bos>",
        "model_max_length": int(expected_context_length),
    }
    for key, expected in expected_special_tokens.items():
        if tokenizer_config.get(key) != expected:
            raise RuntimeError(
                f"HF tokenizer config {key} drifted: "
                f"{tokenizer_config.get(key)!r} != {expected!r}"
            )
    expected_model_fields = {
        "vocab_size": expected_vocab_size,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 0,
        "max_position_embeddings": int(expected_context_length),
    }
    for key, expected in expected_model_fields.items():
        if model_config.get(key) != expected:
            raise RuntimeError(
                f"HF model config {key} drifted: "
                f"{model_config.get(key)!r} != {expected!r}"
            )
    return {
        "schema": "chess-tokenizer-contract-v1",
        "vocab_size": expected_vocab_size,
        "token_ids": {
            token: token_id
            for token, token_id in vocab.items()
            if token in {"<bos>", "<eos>", "<unk>", "<T>", "</T>", "<sep>", "<call_env>"}
        },
        "vocab_mapping": vocab,
        "vocab_mapping_sha256": vocab_mapping_sha256(vocab),
        "model_max_length": int(expected_context_length),
        "vocab_file_sha256": sha256_file(vocab_path),
        "vocab_sha256": sha256_file(vocab_path),
        "tokenizer_config_sha256": sha256_file(tokenizer_config_path),
        "model_config_sha256": sha256_file(model_config_path),
    }


def validate_vocab_transition(
    source_vocab: Mapping[str, Any],
    destination_vocab: Mapping[str, Any],
    *,
    allow_vocab_expansion: bool,
) -> dict[str, Any]:
    """Authenticate an equal-vocab or canonical 81-to-85 row transition."""

    source = normalize_vocab_mapping(source_vocab)
    destination = normalize_vocab_mapping(destination_vocab)
    validate_expected_vocab_mapping(source, expected_size=len(source))
    validate_expected_vocab_mapping(destination, expected_size=len(destination))
    if len(source) == len(destination):
        if source != destination:
            raise RuntimeError("same-size source and destination vocabularies differ")
        transition = "identity"
    elif len(source) == 81 and len(destination) == 85 and allow_vocab_expansion:
        destination_prefix = {
            token: token_id
            for token, token_id in destination.items()
            if token_id < 81
        }
        if destination_prefix != source:
            raise RuntimeError(
                "85-token vocabulary IDs 0:81 do not exactly match the "
                "81-token source vocabulary"
            )
        transition = "81-to-85"
    else:
        raise RuntimeError(
            "unsupported tokenizer transition: "
            f"{len(source)} -> {len(destination)}, "
            f"allow_vocab_expansion={allow_vocab_expansion}"
        )
    return {
        "schema": "chess-tokenizer-transition-v1",
        "transition": transition,
        "source_vocab_size": len(source),
        "destination_vocab_size": len(destination),
        "source_vocab_mapping": source,
        "destination_vocab_mapping": destination,
        "source_vocab_mapping_sha256": vocab_mapping_sha256(source),
        "destination_vocab_mapping_sha256": vocab_mapping_sha256(destination),
        "shared_prefix_mapping_sha256": vocab_mapping_sha256(
            {
                token: token_id
                for token, token_id in destination.items()
                if token_id < len(source)
            }
        ),
    }


__all__ = [
    "EXPECTED_VOCAB_81",
    "EXPECTED_VOCAB_85",
    "expected_vocab_mapping",
    "normalize_vocab_mapping",
    "sha256_file",
    "validate_expected_vocab_mapping",
    "validate_hf_tokenizer_contract",
    "validate_vocab_transition",
    "vocab_mapping_sha256",
]
