from types import SimpleNamespace

import pytest

from chess_rl_miles.prompt_tokens import (
    ensure_exactly_one_leading_bos,
    leading_bos_evidence,
)


def test_prepends_bos_when_tokenizer_does_not_add_special_tokens():
    tokenizer = SimpleNamespace(bos_token_id=81)
    assert ensure_exactly_one_leading_bos([5, 6], tokenizer) == [81, 5, 6]


def test_preserves_one_existing_leading_bos():
    tokenizer = SimpleNamespace(bos_token_id=81)
    assert ensure_exactly_one_leading_bos([81, 5, 6], tokenizer) == [81, 5, 6]


def test_leading_bos_evidence_records_exact_token_contract():
    tokenizer = SimpleNamespace(bos_token_id=81)
    assert leading_bos_evidence([81, 5, 6], tokenizer) == {
        "chess_prompt_bos_token_id": 81,
        "chess_prompt_first_token_id": 81,
        "chess_prompt_bos_count": 1,
        "chess_prompt_token_count": 3,
    }


@pytest.mark.parametrize("prompt_ids", [[81, 81, 5], [5, 81, 6]])
def test_rejects_duplicate_or_nonleading_source_bos(prompt_ids):
    tokenizer = SimpleNamespace(bos_token_id=81)
    with pytest.raises(RuntimeError, match="exactly one leading BOS"):
        ensure_exactly_one_leading_bos(prompt_ids, tokenizer)


def test_rejects_tokenizer_without_bos():
    with pytest.raises(RuntimeError, match="no bos_token_id"):
        ensure_exactly_one_leading_bos([5, 6], SimpleNamespace(bos_token_id=None))
