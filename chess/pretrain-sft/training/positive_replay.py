"""Deterministic positive-rollout extraction for Exp 4.

The input is the JSONL written by :mod:`chess_rl_miles.io`.  Extraction is
strict and fail-closed for successful rows: a replay example is admitted only
when it has an exact binary success reward, completed status, valid chess
trajectory provenance, the required response structure, and lossless token
IDs/loss masks.

Selection is disk-backed because one RL-1500 leg contains more than three
million trajectories.  For each prompt group, the eligible candidate with the
lowest seed-keyed SHA-256 priority is selected.  Cryptographic hash ranking is
a deterministic uniform choice over the siblings without depending on input
file traversal order.  Exact prompt/response pairs are deduplicated only after
the per-group selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = 1
EXPECTED_TOKEN_ARTIFACT_SCHEMA = 1
DEFAULT_RESPONSE_LIMIT = 2_560
DEFAULT_CONTEXT_LIMIT = 3_072
DEFAULT_VOCAB_SIZE = 85
THINK_END_TOKEN = "</T>"
CALL_ENV_TOKEN = "<call_env>"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_ids_sha256(
    prompt_token_ids: Sequence[int],
    response_token_ids: Sequence[int],
) -> str:
    # This intentionally matches chess_rl_miles.io._token_ids_sha256.
    payload = json.dumps(
        [list(prompt_token_ids), list(response_token_ids)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def prompt_response_sha256(prompt: str, response: str) -> str:
    return sha256_bytes(canonical_json([prompt, response]).encode("utf-8"))


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(path))
    )


def resolve_jsonl_inputs(inputs: Sequence[os.PathLike[str] | str]) -> list[Path]:
    """Resolve files/directories to one naturally sorted, duplicate-free list."""
    resolved: dict[str, Path] = {}
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            candidates = path.rglob("*.jsonl")
        elif path.is_file():
            candidates = (path,)
        else:
            raise FileNotFoundError(path)
        for candidate in candidates:
            candidate = candidate.resolve()
            resolved[str(candidate)] = candidate
    paths = sorted(resolved.values(), key=_natural_key)
    if not paths:
        raise ValueError("no rollout JSONL files were resolved")
    return paths


@dataclass(frozen=True)
class ExtractionConfig:
    run_id: str
    policy_checkpoint: str
    filter_setting: str
    extraction_seed: int = 42
    response_limit: int = DEFAULT_RESPONSE_LIMIT
    context_limit: int = DEFAULT_CONTEXT_LIMIT
    vocab_size: int = DEFAULT_VOCAB_SIZE
    max_rl_step: int | None = 1_500
    require_all_attempts_scope: bool = True

    def validate(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not self.policy_checkpoint.strip():
            raise ValueError("policy_checkpoint is required")
        if self.filter_setting not in {"U", "D"}:
            raise ValueError("filter_setting must be U or D")
        if self.response_limit <= 0 or self.context_limit <= 0:
            raise ValueError("token limits must be positive")
        if self.response_limit > self.context_limit:
            raise ValueError("response_limit cannot exceed context_limit")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")


@dataclass(frozen=True)
class ValidationResult:
    record: dict[str, Any] | None
    rejection_reason: str | None

    @property
    def accepted(self) -> bool:
        return self.record is not None


def _numeric_one(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and float(value) == 1.0
    )


def _int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        result.append(int(item))
    return result


def _split_moves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part for part in re.split(r"[\s,]+", value.strip()) if part]
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    return []


def _trajectory_is_legal(row: Mapping[str, Any]) -> bool:
    """Validate the immutable puzzle line and the selected model moves.

    ``extra_info.Moves`` starts from ``extra_info.FEN`` and alternates the
    opponent/environment move with the model move.  Successful multi-turn
    rewards expose the parsed model moves in ``extracted_moves``.  Replaying
    the complete line catches malformed or illegal trajectories rather than
    trusting the first-move-only convenience metric.
    """
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    fen = metadata.get("FEN") or metadata.get("fen")
    full_moves = _split_moves(metadata.get("Moves") or metadata.get("moves"))
    extracted_moves = _split_moves(row.get("extracted_moves"))
    if not isinstance(fen, str) or not fen.strip():
        return False
    if len(full_moves) < 2 or len(full_moves) % 2 != 0:
        return False
    if extracted_moves != full_moves[1::2]:
        return False

    try:
        import chess

        board = chess.Board(fen)
        for uci in full_moves:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                return False
            board.push(move)
    except (ImportError, TypeError, ValueError):
        return False
    return True


def validate_positive_row(
    row: Mapping[str, Any],
    config: ExtractionConfig,
) -> ValidationResult:
    """Validate and normalize one rollout row.

    Ordinary failures are returned as explicit rejection reasons so the
    extraction manifest can account for every input row.  Corrupt JSON itself
    is treated as an exception by the caller rather than a QC rejection.
    """
    score = row.get("score")
    if not _numeric_one(score):
        return ValidationResult(None, "score_not_one")
    if row.get("status") != "completed":
        return ValidationResult(None, "status_not_completed")
    if (
        config.require_all_attempts_scope
        and row.get("sampling_scope")
        != "all_completed_attempts_before_dynamic_filter"
    ):
        return ValidationResult(None, "not_all_attempts_positive_stream")

    prompt = row.get("input")
    response = row.get("output")
    if not isinstance(prompt, str) or not isinstance(response, str):
        return ValidationResult(None, "invalid_prompt_or_response")
    if response.count(THINK_END_TOKEN) != 1:
        return ValidationResult(None, "think_end_count_not_one")
    if CALL_ENV_TOKEN not in response:
        return ValidationResult(None, "missing_call_env")

    group_index = row.get("group_index")
    sample_index = row.get("sample_index")
    if (
        isinstance(group_index, bool)
        or not isinstance(group_index, int)
        or group_index < 0
        or isinstance(sample_index, bool)
        or not isinstance(sample_index, int)
        or sample_index < 0
    ):
        return ValidationResult(None, "missing_sample_identity")

    if row.get("token_artifact_schema") != EXPECTED_TOKEN_ARTIFACT_SCHEMA:
        return ValidationResult(None, "missing_token_artifact")
    prompt_ids = _int_list(row.get("prompt_token_ids"))
    response_ids = _int_list(row.get("response_token_ids"))
    response_mask = _int_list(row.get("response_loss_mask"))
    if prompt_ids is None or response_ids is None or response_mask is None:
        return ValidationResult(None, "invalid_token_artifact")
    if not prompt_ids or not response_ids:
        return ValidationResult(None, "empty_token_artifact")
    if len(response_ids) != len(response_mask):
        return ValidationResult(None, "response_mask_length_mismatch")
    if int(row.get("response_length", -1)) != len(response_ids):
        return ValidationResult(None, "response_length_mismatch")
    if any(value not in (0, 1) for value in response_mask):
        return ValidationResult(None, "nonbinary_response_mask")
    if not any(response_mask):
        return ValidationResult(None, "no_model_owned_tokens")
    if any(
        token_id < 0 or token_id >= config.vocab_size
        for token_id in (*prompt_ids, *response_ids)
    ):
        return ValidationResult(None, "token_id_out_of_range")
    if len(response_ids) > config.response_limit:
        return ValidationResult(None, "response_limit_exceeded")
    if len(prompt_ids) + len(response_ids) > config.context_limit:
        return ValidationResult(None, "context_limit_exceeded")
    expected_token_hash = token_ids_sha256(prompt_ids, response_ids)
    if row.get("token_ids_sha256") != expected_token_hash:
        return ValidationResult(None, "token_hash_mismatch")

    step = row.get("step")
    rollout_id = row.get("rollout_id")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or isinstance(rollout_id, bool)
        or not isinstance(rollout_id, int)
    ):
        return ValidationResult(None, "invalid_rl_step")
    if config.max_rl_step is not None and not 0 <= step <= config.max_rl_step:
        return ValidationResult(None, "rl_step_out_of_range")
    weight_versions = row.get("weight_versions")
    if not isinstance(weight_versions, list):
        return ValidationResult(None, "invalid_weight_version_provenance")
    if not _trajectory_is_legal(row):
        return ValidationResult(None, "illegal_trajectory")

    metadata = row.get("metadata")
    reward = row.get("reward")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    reward = dict(reward) if isinstance(reward, Mapping) else {"score": score}
    pair_hash = prompt_response_sha256(prompt, response)
    record = {
        "schema_version": SCHEMA_VERSION,
        "prompt": prompt,
        "response": response,
        "prompt_token_ids": prompt_ids,
        "response_token_ids": response_ids,
        "response_loss_mask": response_mask,
        "token_ids_sha256": expected_token_hash,
        "prompt_response_sha256": pair_hash,
        "source_run_id": config.run_id,
        "policy_checkpoint": config.policy_checkpoint,
        "filter_setting": config.filter_setting,
        "rl_step": step,
        "rl_rollout_id": rollout_id,
        "group_index": group_index,
        "sample_index": sample_index,
        "weight_versions": list(weight_versions),
        "session_id": row.get("session_id"),
        "label": row.get("label"),
        "difficulty": row.get(
            "difficulty",
            metadata.get("difficulty", metadata.get("Rating")),
        ),
        "rating": metadata.get("Rating"),
        "puzzle_id": metadata.get("PuzzleId", metadata.get("uid")),
        "metadata": metadata,
        "reward": reward,
    }
    return ValidationResult(record, None)


def _selection_priority(
    *,
    seed: int,
    run_id: str,
    group_index: int,
    sample_index: int,
    source_file_sha256: str,
    source_line: int,
    source_row_sha256: str,
) -> str:
    identity = [
        int(seed),
        run_id,
        int(group_index),
        int(sample_index),
        source_file_sha256,
        int(source_line),
        source_row_sha256,
    ]
    return sha256_bytes(canonical_json(identity).encode("utf-8"))


def _open_selection_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE selected_groups (
            run_id TEXT NOT NULL,
            group_index INTEGER NOT NULL,
            priority TEXT NOT NULL,
            candidate_count INTEGER NOT NULL,
            record_json TEXT NOT NULL,
            PRIMARY KEY (run_id, group_index)
        )
        """
    )
    return connection


def _upsert_candidate(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    group_index: int,
    priority: str,
    record: Mapping[str, Any],
) -> None:
    connection.execute(
        """
        INSERT INTO selected_groups (
            run_id, group_index, priority, candidate_count, record_json
        ) VALUES (?, ?, ?, 1, ?)
        ON CONFLICT(run_id, group_index) DO UPDATE SET
            candidate_count = selected_groups.candidate_count + 1,
            record_json = CASE
                WHEN excluded.priority < selected_groups.priority
                THEN excluded.record_json
                ELSE selected_groups.record_json
            END,
            priority = CASE
                WHEN excluded.priority < selected_groups.priority
                THEN excluded.priority
                ELSE selected_groups.priority
            END
        """,
        (
            run_id,
            int(group_index),
            priority,
            canonical_json(record),
        ),
    )


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any], str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"rollout row must be an object at {path}:{line_number}"
                )
            yield line_number, row, sha256_bytes(
                canonical_json(row).encode("utf-8")
            )


def extract_positive_replay(
    inputs: Sequence[os.PathLike[str] | str],
    *,
    output_path: os.PathLike[str] | str,
    manifest_path: os.PathLike[str] | str | None,
    config: ExtractionConfig,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract a deterministic one-positive-per-prompt replay corpus."""
    config.validate()
    paths = resolve_jsonl_inputs(inputs)
    output = Path(output_path).expanduser().resolve()
    manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else output.with_suffix(output.suffix + ".manifest.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if output == manifest:
        raise ValueError("output_path and manifest_path must differ")
    existing = [path for path in (output, manifest) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite an existing replay artifact: "
            + ", ".join(str(path) for path in existing)
        )

    counters: Counter[str] = Counter()
    source_manifests: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="positive-replay-",
        dir=output.parent,
    ) as temp_dir:
        db_path = Path(temp_dir) / "selection.sqlite3"
        connection = _open_selection_db(db_path)
        try:
            for source_path in paths:
                source_sha = sha256_file(source_path)
                source_rows = 0
                for line_number, row, source_row_sha in _iter_jsonl(source_path):
                    source_rows += 1
                    counters["input_rows"] += 1
                    result = validate_positive_row(row, config)
                    if not result.accepted:
                        counters[f"rejected/{result.rejection_reason}"] += 1
                        continue
                    counters["eligible_rows"] += 1
                    record = dict(result.record or {})
                    record["source"] = {
                        "path": str(source_path),
                        "file_sha256": source_sha,
                        "line": line_number,
                        "row_sha256": source_row_sha,
                    }
                    priority = _selection_priority(
                        seed=config.extraction_seed,
                        run_id=config.run_id,
                        group_index=int(record["group_index"]),
                        sample_index=int(record["sample_index"]),
                        source_file_sha256=source_sha,
                        source_line=line_number,
                        source_row_sha256=source_row_sha,
                    )
                    record["selection_priority"] = priority
                    _upsert_candidate(
                        connection,
                        run_id=config.run_id,
                        group_index=int(record["group_index"]),
                        priority=priority,
                        record=record,
                    )
                connection.commit()
                source_manifests.append(
                    {
                        "path": str(source_path),
                        "bytes": source_path.stat().st_size,
                        "sha256": source_sha,
                        "rows": source_rows,
                    }
                )

            group_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM selected_groups"
                ).fetchone()[0]
            )
            counters["eligible_groups"] = group_count

            temp_output = Path(temp_dir) / "positive_replay.jsonl"
            output_digest = hashlib.sha256()
            seen_pairs: set[str] = set()
            written_rows = 0
            duplicate_rows = 0
            with temp_output.open("wb") as handle:
                cursor = connection.execute(
                    """
                    SELECT priority, candidate_count, record_json
                    FROM selected_groups
                    ORDER BY run_id ASC, group_index ASC
                    """
                )
                for priority, candidate_count, record_json in cursor:
                    record = json.loads(record_json)
                    pair_hash = str(record["prompt_response_sha256"])
                    if pair_hash in seen_pairs:
                        duplicate_rows += 1
                        continue
                    seen_pairs.add(pair_hash)
                    record["selection_priority"] = priority
                    record["eligible_siblings_in_group"] = int(candidate_count)
                    encoded = (canonical_json(record) + "\n").encode("utf-8")
                    handle.write(encoded)
                    output_digest.update(encoded)
                    written_rows += 1
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_output, output)
        finally:
            connection.close()

    counters["selected_before_exact_dedupe"] = group_count
    counters["exact_prompt_response_duplicates_dropped"] = duplicate_rows
    counters["output_rows"] = written_rows
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "exp4_positive_rollout_replay",
        "config": asdict(config),
        "selection": {
            "method": "minimum_seed_keyed_sha256_priority_per_prompt_group",
            "uniform_over": "eligible_sibling_trajectories",
            "deduplication": "exact_prompt_response_sha256_after_group_selection",
        },
        "sources": source_manifests,
        "counters": dict(sorted(counters.items())),
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "rows": written_rows,
            "sha256": output_digest.hexdigest(),
        },
    }
    temp_manifest = manifest.with_suffix(manifest.suffix + ".tmp")
    with temp_manifest.open("w", encoding="utf-8") as handle:
        json.dump(
            manifest_payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_manifest, manifest)
    return manifest_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract deterministic Exp 4 positive replay from Miles JSONL"
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Rollout JSONL file or directory; repeat for multiple inputs",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--filter-setting", choices=("U", "D"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--response-limit", type=int, default=DEFAULT_RESPONSE_LIMIT)
    parser.add_argument("--context-limit", type=int, default=DEFAULT_CONTEXT_LIMIT)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--max-rl-step", type=int, default=1_500)
    parser.add_argument(
        "--allow-accepted-only-input",
        action="store_true",
        help=(
            "Allow the post-filter training JSONL for legacy/unfiltered "
            "canaries. Production Exp 4 must use all_attempts_positive."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing output/manifest.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = extract_positive_replay(
        args.input,
        output_path=args.output,
        manifest_path=args.manifest,
        config=ExtractionConfig(
            run_id=args.run_id,
            policy_checkpoint=args.policy_checkpoint,
            filter_setting=args.filter_setting,
            extraction_seed=args.seed,
            response_limit=args.response_limit,
            context_limit=args.context_limit,
            vocab_size=args.vocab_size,
            max_rl_step=args.max_rl_step,
            require_all_attempts_scope=not args.allow_accepted_only_input,
        ),
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
