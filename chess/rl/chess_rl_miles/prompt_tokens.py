from __future__ import annotations


def ensure_exactly_one_leading_bos(prompt_ids: list[int], tokenizer) -> list[int]:
    """Return prompt IDs with exactly one BOS, at position zero.

    The chess training format has one leading BOS. Tokenization uses
    ``add_special_tokens=False``, so rollout prepends it when absent and fails
    closed if the source prompt already contains a duplicate BOS anywhere.
    """
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None:
        raise RuntimeError(
            "tokenizer has no bos_token_id; chess rollout prompts require the training-time <bos> prefix"
        )
    bos_id = int(bos_id)
    normalized = list(prompt_ids)
    if not normalized or normalized[0] != bos_id:
        normalized.insert(0, bos_id)

    bos_count = normalized.count(bos_id)
    if normalized[0] != bos_id or bos_count != 1:
        raise RuntimeError(
            f"chess rollout prompt must contain exactly one leading BOS token id {bos_id}; found {bos_count}"
        )
    return normalized


def leading_bos_evidence(prompt_ids: list[int], tokenizer) -> dict[str, int]:
    """Return auditable counts after enforcing the rollout prompt contract."""

    normalized = ensure_exactly_one_leading_bos(prompt_ids, tokenizer)
    bos_id = int(tokenizer.bos_token_id)
    return {
        "chess_prompt_bos_token_id": bos_id,
        "chess_prompt_first_token_id": int(normalized[0]),
        "chess_prompt_bos_count": normalized.count(bos_id),
        "chess_prompt_token_count": len(normalized),
    }
