"""Fail-closed OLMo loader/training smoke for matched D0/D1 transfer corpora.

This is deliberately separate from :mod:`train_inner_mix`; running it cannot
touch or resume an active pretraining job.  It validates the matched pair and
both serialized corpora, builds OLMo composable NumPy sources, collates one
natural instance with one mask-bearing D0 instance and one mask-bearing D1
instance, then performs one optimizer step with a tiny OLMo2 transformer.

The smoke is an engineering gate, not a research result.  It does not train or
select a policy and it does not write checkpoints.

Example::

    modal run transfer_loader_smoke.py \
      --pair-root /checkpoints/transfer_data/pilot_v1/built_smoke_v2 \
      --natural-glob '/tokenized/3/part_*.npy'
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


MANIFEST_SCHEMA_VERSION = "math-transfer-manifest-v1"
CORPUS_SCHEMA_VERSION = "math-transfer-corpus-v1"
MATCH_SCHEMA_VERSION = "math-transfer-match-v1"
PAIR_ARMS = ("D0", "D1")
REQUIRED_CORPUS_FILES = {
    "token_ids_00000.npy",
    "labels_mask_00000.npy",
    "selected.jsonl",
}
DEFAULT_MAX_PROCESSED_TOKEN_RELATIVE_DELTA = 0.001


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_self_identifying_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing manifest: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON manifest: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    content_id = document.get("content_id")
    core = {key: value for key, value in document.items() if key != "content_id"}
    expected = _sha256_bytes(_canonical_json_bytes(core))
    if content_id != expected:
        raise ValueError(
            f"manifest content_id mismatch for {path}: {content_id!r} != {expected!r}"
        )
    if document.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported manifest schema for {path}: {document.get('schema_version')!r}"
        )
    return document


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _descriptor_map(manifest: Mapping[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    descriptors = manifest.get("files")
    if not isinstance(descriptors, list):
        raise ValueError(f"corpus manifest files must be a list: {root / 'manifest.json'}")
    out: dict[str, dict[str, Any]] = {}
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise ValueError("corpus file descriptor must be an object")
        name = str(descriptor.get("path", ""))
        if not name or Path(name).name != name:
            raise ValueError(f"unsafe or invalid declared corpus path: {name!r}")
        if name in out:
            raise ValueError(f"duplicate corpus file descriptor: {name}")
        out[name] = dict(descriptor)
    if set(out) != REQUIRED_CORPUS_FILES:
        raise ValueError(
            f"corpus must declare exactly {sorted(REQUIRED_CORPUS_FILES)}, got {sorted(out)}"
        )
    for name, descriptor in out.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing declared corpus file: {path}")
        if path.stat().st_size != int(descriptor.get("bytes", -1)):
            raise ValueError(f"declared byte count mismatch: {path}")
        if _sha256_file(path) != descriptor.get("sha256"):
            raise ValueError(f"declared checksum mismatch: {path}")
    return out


def _stratum_totals(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for row in rows:
        label = str(row["difficulty_bin"])
        current = totals.setdefault(
            label, {"document_count": 0, "assistant_tokens": 0, "processed_tokens": 0}
        )
        current["document_count"] += 1
        current["assistant_tokens"] += int(row["assistant_token_count"])
        current["processed_tokens"] += int(row["processed_token_count"])
    return totals


def validate_transfer_corpus(
    corpus_dir: str | os.PathLike[str], *, expected_arm: str
) -> dict[str, Any]:
    """Validate one raw uint32/bool corpus down to every document mask."""

    if expected_arm not in PAIR_ARMS:
        raise ValueError(f"expected_arm must be one of {PAIR_ARMS}, got {expected_arm!r}")
    if sys.byteorder != "little":
        raise RuntimeError("raw uint32 transfer artifacts are only gated on little-endian hosts")

    root = Path(corpus_dir)
    manifest_path = root / "manifest.json"
    manifest = _load_self_identifying_manifest(manifest_path)
    if manifest.get("artifact_kind") != "assistant_masked_transfer_corpus":
        raise ValueError(f"not an assistant-masked transfer corpus: {manifest_path}")
    if manifest.get("corpus_schema_version") != CORPUS_SCHEMA_VERSION:
        raise ValueError(f"unsupported corpus schema: {manifest_path}")
    if manifest.get("arm") != expected_arm:
        raise ValueError(
            f"corpus arm mismatch: expected {expected_arm}, got {manifest.get('arm')!r}"
        )
    if not manifest.get("prompt_manifest_sha256"):
        raise ValueError(f"missing prompt manifest identity: {manifest_path}")

    descriptors = _descriptor_map(manifest, root)
    if descriptors["token_ids_00000.npy"].get("format") != "raw_headerless_uint32":
        raise ValueError("token artifact must declare raw_headerless_uint32")
    if descriptors["labels_mask_00000.npy"].get("format") != "raw_headerless_bool":
        raise ValueError("mask artifact must declare raw_headerless_bool")

    processed_tokens = int(manifest.get("processed_tokens", -1))
    document_count = int(manifest.get("document_count", -1))
    assistant_tokens = int(manifest.get("loss_bearing_assistant_tokens", -1))
    masked_prompt_tokens = int(manifest.get("masked_prompt_tokens", -1))
    vocab_size = int(manifest.get("vocab_size", -1))
    eos_token_id = int(manifest.get("eos_token_id", -1))
    if min(processed_tokens, document_count, assistant_tokens, masked_prompt_tokens) <= 0:
        raise ValueError(f"corpus counts must all be positive: {manifest_path}")
    if assistant_tokens + masked_prompt_tokens != processed_tokens:
        raise ValueError(f"assistant/prompt token counts do not sum to total: {manifest_path}")
    if not 0 <= eos_token_id < vocab_size:
        raise ValueError(f"invalid EOS/vocabulary declaration: {manifest_path}")

    token_path = root / "token_ids_00000.npy"
    mask_path = root / "labels_mask_00000.npy"
    selected_path = root / "selected.jsonl"
    if token_path.stat().st_size != processed_tokens * np.dtype("<u4").itemsize:
        raise ValueError(f"raw uint32 byte count does not equal processed_tokens: {token_path}")
    if mask_path.stat().st_size != processed_tokens:
        raise ValueError(f"raw bool byte count does not equal processed_tokens: {mask_path}")
    if token_path.read_bytes()[:6] == b"\x93NUMPY":
        raise ValueError(f"token file has a NumPy header but raw bytes are required: {token_path}")
    if mask_path.read_bytes()[:6] == b"\x93NUMPY":
        raise ValueError(f"mask file has a NumPy header but raw bytes are required: {mask_path}")

    tokens = np.fromfile(token_path, dtype="<u4")
    mask_bytes = np.fromfile(mask_path, dtype=np.uint8)
    if tokens.size != processed_tokens or mask_bytes.size != processed_tokens:
        raise ValueError("raw transfer arrays have unexpected lengths")
    invalid_mask_values = np.unique(mask_bytes[(mask_bytes != 0) & (mask_bytes != 1)])
    if invalid_mask_values.size:
        raise ValueError(
            f"raw bool mask contains non-canonical bytes: {invalid_mask_values.tolist()}"
        )
    masks = mask_bytes.astype(np.bool_, copy=False)
    if tokens.size and int(tokens.max()) >= vocab_size:
        raise ValueError(f"token ID outside declared vocabulary in {token_path}")

    rows = _read_jsonl(selected_path)
    if len(rows) != document_count:
        raise ValueError(f"selected row count does not equal document_count: {selected_path}")
    if int(descriptors["selected.jsonl"].get("rows", -1)) != len(rows):
        raise ValueError(f"selected descriptor row count mismatch: {selected_path}")

    cursor = 0
    seen_candidates: set[str] = set()
    counted_assistant = 0
    for index, row in enumerate(rows):
        required = {
            "arm",
            "candidate_id",
            "difficulty_bin",
            "token_offset_start",
            "token_offset_end",
            "prompt_token_count",
            "assistant_token_count",
            "processed_token_count",
        }
        missing = required - set(row)
        if missing:
            raise ValueError(f"selected row {index} missing fields: {sorted(missing)}")
        if row["arm"] != expected_arm:
            raise ValueError(f"selected row {index} has wrong arm: {row['arm']!r}")
        candidate_id = str(row["candidate_id"])
        if candidate_id in seen_candidates:
            raise ValueError(f"duplicate selected candidate_id: {candidate_id}")
        seen_candidates.add(candidate_id)

        start = int(row["token_offset_start"])
        end = int(row["token_offset_end"])
        prompt_count = int(row["prompt_token_count"])
        response_count = int(row["assistant_token_count"])
        row_count = int(row["processed_token_count"])
        if start != cursor or end <= start or end - start != row_count:
            raise ValueError(f"non-contiguous or inconsistent offsets in selected row {index}")
        if prompt_count <= 0 or response_count <= 0 or prompt_count + response_count != row_count:
            raise ValueError(f"invalid prompt/assistant counts in selected row {index}")
        if int(tokens[end - 1]) != eos_token_id:
            raise ValueError(f"selected document {index} does not terminate with declared EOS")
        if masks[start : start + prompt_count].any():
            raise ValueError(f"selected document {index} exposes prompt tokens to loss")
        if not masks[start + prompt_count : end].all():
            raise ValueError(f"selected document {index} masks an assistant/EOS token")
        cursor = end
        counted_assistant += response_count

    if cursor != processed_tokens:
        raise ValueError("selected offsets do not cover the entire raw corpus")
    if counted_assistant != assistant_tokens or int(masks.sum()) != assistant_tokens:
        raise ValueError("assistant-token count disagrees with the raw loss mask")

    return {
        "arm": expected_arm,
        "content_id": manifest["content_id"],
        "manifest_sha256": _sha256_file(manifest_path),
        "processed_tokens": processed_tokens,
        "assistant_tokens": assistant_tokens,
        "masked_prompt_tokens": masked_prompt_tokens,
        "document_count": document_count,
        "vocab_size": vocab_size,
        "eos_token_id": eos_token_id,
        "token_path": str(token_path),
        "mask_path": str(mask_path),
        "rows": rows,
        "strata": _stratum_totals(rows),
    }


def validate_matched_pair(
    pair_root: str | os.PathLike[str],
    *,
    max_processed_token_relative_delta: float = DEFAULT_MAX_PROCESSED_TOKEN_RELATIVE_DELTA,
) -> dict[str, Any]:
    """Validate pair identity plus D0/D1 equality/tolerance from raw corpora."""

    if not 0 <= max_processed_token_relative_delta <= 1:
        raise ValueError("max_processed_token_relative_delta must be in [0, 1]")
    root = Path(pair_root)
    pair_path = root / "pair_manifest.json"
    pair = _load_self_identifying_manifest(pair_path)
    if pair.get("artifact_kind") != "matched_D0_D1_pair":
        raise ValueError(f"not a matched D0/D1 pair: {pair_path}")
    match = pair.get("match")
    if not isinstance(match, dict) or match.get("schema_version") != MATCH_SCHEMA_VERSION:
        raise ValueError(f"missing or unsupported match summary: {pair_path}")
    if tuple(match.get("arms", ())) != PAIR_ARMS:
        raise ValueError(f"match arms must be exactly {PAIR_ARMS}")

    declared_tolerance = float(match.get("max_processed_token_relative_delta", math.inf))
    if declared_tolerance > max_processed_token_relative_delta:
        raise ValueError(
            "pair was selected under a looser processed-token tolerance: "
            f"{declared_tolerance:.6f} > {max_processed_token_relative_delta:.6f}"
        )

    corpora = {
        arm: validate_transfer_corpus(root / arm, expected_arm=arm) for arm in PAIR_ARMS
    }
    declared_corpora = pair.get("corpora")
    if not isinstance(declared_corpora, dict):
        raise ValueError("pair manifest has no corpus identities")
    for arm in PAIR_ARMS:
        if declared_corpora.get(arm) != corpora[arm]["content_id"]:
            raise ValueError(f"pair manifest references a different {arm} corpus")

    assistant_counts = {arm: corpora[arm]["assistant_tokens"] for arm in PAIR_ARMS}
    if len(set(assistant_counts.values())) != 1:
        raise ValueError(f"D0/D1 assistant-token totals differ: {assistant_counts}")
    if int(match.get("assistant_tokens_per_arm", -1)) != assistant_counts["D0"]:
        raise ValueError("match summary assistant-token total disagrees with raw corpora")

    document_counts = {arm: corpora[arm]["document_count"] for arm in PAIR_ARMS}
    if len(set(document_counts.values())) != 1:
        raise ValueError(f"D0/D1 document totals differ: {document_counts}")
    if match.get("selected_documents") != document_counts:
        raise ValueError("match summary document totals disagree with raw corpora")

    processed_counts = {arm: corpora[arm]["processed_tokens"] for arm in PAIR_ARMS}
    if match.get("processed_tokens") != processed_counts:
        raise ValueError("match summary processed-token totals disagree with raw corpora")
    relative_delta = abs(processed_counts["D0"] - processed_counts["D1"]) / min(
        processed_counts.values()
    )
    if relative_delta > max_processed_token_relative_delta:
        raise ValueError(
            "processed-token mismatch exceeds gate tolerance: "
            f"D0={processed_counts['D0']:,}, D1={processed_counts['D1']:,}, "
            f"relative_delta={relative_delta:.6f} > "
            f"{max_processed_token_relative_delta:.6f}"
        )
    if not math.isclose(
        float(match.get("processed_token_relative_delta", math.inf)),
        relative_delta,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("reported processed-token relative delta is not reproducible")

    declared_strata = match.get("strata")
    if not isinstance(declared_strata, dict) or not declared_strata:
        raise ValueError("match summary has no matched strata")
    strata_by_arm = {arm: corpora[arm]["strata"] for arm in PAIR_ARMS}
    if set(strata_by_arm["D0"]) != set(strata_by_arm["D1"]):
        raise ValueError("D0/D1 matched difficulty strata differ")
    if set(declared_strata) != set(strata_by_arm["D0"]):
        raise ValueError("match summary strata differ from raw selected metadata")
    for label in sorted(strata_by_arm["D0"]):
        totals = {arm: strata_by_arm[arm][label] for arm in PAIR_ARMS}
        declared = declared_strata[label]
        if declared.get("status", "matched") != "matched":
            raise ValueError(f"match summary stratum {label!r} is not matched")
        document_counts = {
            arm: int(totals[arm]["document_count"]) for arm in PAIR_ARMS
        }
        if len(set(document_counts.values())) != 1:
            raise ValueError(
                f"D0/D1 document counts differ in stratum {label!r}: "
                f"{document_counts}"
            )
        if int(declared.get("document_count_per_arm", -1)) != document_counts["D0"]:
            raise ValueError(
                f"match summary stratum {label!r} has wrong document_count_per_arm"
            )

        for metric in ("assistant_tokens", "processed_tokens"):
            actual = {arm: int(totals[arm][metric]) for arm in PAIR_ARMS}
            declared_by_arm = declared.get(metric)
            declared_per_arm = declared.get(f"{metric}_per_arm")
            if declared_per_arm is not None:
                if len(set(actual.values())) != 1 or int(declared_per_arm) != actual["D0"]:
                    raise ValueError(
                        f"match summary stratum {label!r} has wrong {metric}_per_arm"
                    )
            elif declared_by_arm != actual:
                raise ValueError(
                    f"match summary stratum {label!r} has wrong {metric} totals"
                )

    return {
        "pair_content_id": pair["content_id"],
        "pair_manifest_sha256": _sha256_file(pair_path),
        "processed_token_relative_delta": relative_delta,
        "max_processed_token_relative_delta": max_processed_token_relative_delta,
        "corpora": corpora,
    }


def _resolve_one_natural_path(pattern: str) -> str:
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(f"natural data glob matched no files: {pattern!r}")
    path = Path(matches[0])
    if not path.is_file() or path.stat().st_size < np.dtype("<u4").itemsize:
        raise ValueError(f"natural token file is missing or empty: {path}")
    if path.stat().st_size % np.dtype("<u4").itemsize:
        raise ValueError(f"natural token file is not raw uint32-aligned: {path}")
    if path.read_bytes()[:6] == b"\x93NUMPY":
        raise ValueError(f"natural token file has a NumPy header: {path}")
    return str(path)


def run_one_step_smoke(
    *,
    pair_root: str,
    natural_glob: str,
    sequence_length: int = 128,
    seed: int = 1337,
    max_processed_token_relative_delta: float = DEFAULT_MAX_PROCESSED_TOKEN_RELATIVE_DELTA,
) -> dict[str, Any]:
    """Run one tiny OLMo optimizer step over natural + D0 + D1 instances."""

    if sequence_length < 8:
        raise ValueError("sequence_length must be at least 8")
    pair_summary = validate_matched_pair(
        pair_root,
        max_processed_token_relative_delta=max_processed_token_relative_delta,
    )
    natural_path = _resolve_one_natural_path(natural_glob)

    # Keep all heavyweight imports out of the local artifact-validation path.
    import torch

    from olmo_core.data import TokenizerConfig
    from olmo_core.data.composable import (
        ComposableDataLoaderConfig,
        ConcatAndChunkInstanceSource,
    )
    from olmo_core.data.composable.sliced_instance_source import SlicedInstanceSource
    from olmo_core.data.types import NumpyDatasetDType
    from olmo_core.data.utils import get_labels
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.utils import seed_all

    if not torch.cuda.is_available():
        raise RuntimeError("the Modal one-step smoke requires a CUDA GPU")
    seed_all(seed)
    tokenizer = TokenizerConfig.dolma2()
    padded_vocab_size = tokenizer.padded_vocab_size()
    for arm in PAIR_ARMS:
        if pair_summary["corpora"][arm]["vocab_size"] > padded_vocab_size:
            raise ValueError(f"{arm} corpus vocabulary exceeds OLMo tokenizer vocabulary")

    with tempfile.TemporaryDirectory(prefix="transfer-loader-smoke-") as work_dir:
        def source(path: str, mask_path: str | None, label: str):
            return ConcatAndChunkInstanceSource.Config.from_npy(
                path,
                tokenizer=tokenizer,
                sequence_length=sequence_length,
                max_sequence_length=sequence_length,
                dtype=NumpyDatasetDType.uint32,
                label_mask_paths=None if mask_path is None else [mask_path],
                expand_glob=False,
                label=label,
            ).build(work_dir)

        natural_source = source(natural_path, None, "natural")
        synthetic_sources = {
            arm: source(
                pair_summary["corpora"][arm]["token_path"],
                pair_summary["corpora"][arm]["mask_path"],
                arm,
            )
            for arm in PAIR_ARMS
        }
        if len(natural_source) < 1:
            raise ValueError("natural source produced no complete smoke instance")

        def mixed_mask_index(arm: str) -> int:
            source_value = synthetic_sources[arm]
            if len(source_value) < 1:
                raise ValueError(
                    f"{arm} corpus produced no complete {sequence_length}-token instance"
                )
            for index in range(len(source_value)):
                item = source_value[index]
                mask = np.asarray(item.get("label_mask"), dtype=np.bool_)
                # Labels are shifted left; require both masked and live targets after shift.
                if mask.shape == (sequence_length,) and mask[1:].any() and (~mask[1:]).any():
                    return index
            raise ValueError(
                f"{arm} has no instance containing both masked prompt and live assistant targets"
            )

        selected_indices = {arm: mixed_mask_index(arm) for arm in PAIR_ARMS}
        expected_items = [natural_source[0]] + [
            synthetic_sources[arm][selected_indices[arm]] for arm in PAIR_ARMS
        ]
        if "label_mask" in expected_items[0]:
            raise AssertionError("natural source unexpectedly supplied a label mask")

        sliced_sources = [
            SlicedInstanceSource(
                natural_source, slice(0, 1), seed=None, work_dir=work_dir
            )
        ]
        sliced_sources.extend(
            SlicedInstanceSource(
                synthetic_sources[arm],
                slice(selected_indices[arm], selected_indices[arm] + 1),
                seed=None,
                work_dir=work_dir,
            )
            for arm in PAIR_ARMS
        )
        loader = ComposableDataLoaderConfig(
            tokenizer=tokenizer,
            global_batch_size=len(sliced_sources) * sequence_length,
            seed=seed,
            work_dir=work_dir,
            shuffle=False,
            num_threads=0,
            num_workers=0,
            target_device_type="cpu",
            display_source_visualization=False,
        ).build(*sliced_sources)
        loader.reshuffle(epoch=1)
        batch = next(iter(loader))

        expected_shape = (3, sequence_length)
        if tuple(batch["input_ids"].shape) != expected_shape:
            raise AssertionError(
                f"mixed input batch has shape {tuple(batch['input_ids'].shape)}, "
                f"expected {expected_shape}"
            )
        if "label_mask" not in batch or tuple(batch["label_mask"].shape) != expected_shape:
            raise AssertionError(
                "mixed collator did not promote an all-True mask for the unmasked natural row"
            )
        if not bool(batch["label_mask"][0].all()):
            raise AssertionError("natural row was not fully enabled by the mixed collator")
        for row_index, arm in enumerate(PAIR_ARMS, start=1):
            expected_mask = torch.as_tensor(expected_items[row_index]["label_mask"]).bool()
            if not torch.equal(batch["label_mask"][row_index], expected_mask):
                raise AssertionError(f"{arm} label mask changed during OLMo collation")

        labels = get_labels(batch)
        expected_live_targets = torch.nn.functional.pad(
            batch["label_mask"][:, 1:], (0, 1), value=False
        )
        if not torch.equal(labels != -100, expected_live_targets):
            raise AssertionError("OLMo shifted labels do not exactly follow the collated mask")
        for row_index, arm in enumerate(PAIR_ARMS, start=1):
            if not bool(expected_live_targets[row_index].any()):
                raise AssertionError(f"{arm} contributes no live target to the smoke loss")
            if not bool((~expected_live_targets[row_index, :-1]).any()):
                raise AssertionError(f"{arm} contributes no masked prompt target to the smoke")

        device = torch.device("cuda")
        model_config = TransformerConfig.olmo2_1M(
            vocab_size=padded_vocab_size,
            attn_backend=AttentionBackendName.torch,
        )
        model = model_config.build(init_device="meta")
        model.init_weights(
            max_seq_len=sequence_length,
            max_local_microbatch_size=3 * sequence_length,
            device=device,
        )
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["input_ids"].to(device),
            labels=labels.to(device),
            ignore_index=-100,
            loss_reduction="mean",
            return_logits=False,
        )
        loss = output.loss
        if not bool(torch.isfinite(loss)):
            raise AssertionError(f"non-finite one-step loss: {loss.item()}")
        loss.backward()
        grad_square_sum = torch.zeros((), dtype=torch.float64, device=device)
        for parameter in model.parameters():
            if parameter.grad is not None:
                grad_square_sum += parameter.grad.detach().double().square().sum()
        grad_norm = grad_square_sum.sqrt()
        if not bool(torch.isfinite(grad_norm)) or float(grad_norm) <= 0:
            raise AssertionError(f"invalid one-step gradient norm: {float(grad_norm)}")
        optimizer.step()
        torch.cuda.synchronize()

        return {
            "gate": "PASS",
            "pair_content_id": pair_summary["pair_content_id"],
            "pair_manifest_sha256": pair_summary["pair_manifest_sha256"],
            "processed_token_relative_delta": pair_summary[
                "processed_token_relative_delta"
            ],
            "natural_path": natural_path,
            "sequence_length": sequence_length,
            "batch_rows": ["natural", "D0", "D1"],
            "selected_synthetic_instance_indices": selected_indices,
            "live_target_tokens": [
                int(expected_live_targets[index].sum()) for index in range(3)
            ],
            "masked_target_tokens": [
                int((~expected_live_targets[index]).sum()) for index in range(3)
            ],
            "model": "TransformerConfig.olmo2_1M",
            "model_parameters": model_config.num_params,
            "loss": float(loss.detach()),
            "gradient_norm": float(grad_norm.detach()),
            "optimizer_step_completed": True,
        }


# Optional Modal wrapper.  Pure validation/tests do not require torch or OLMo.
try:
    import modal
except ImportError:  # pragma: no cover
    modal = None


if modal is not None:
    _local = Path(__file__).resolve().parent
    _olmo_core = _local.parent / "OLMo-core"
    _remote_olmo_core = "/root/OLMo-core"
    _image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
        )
        .apt_install("build-essential", "git", "curl")
        .pip_install("torch==2.8.0", "numpy>=1.26")
        .add_local_dir(
            str(_olmo_core),
            remote_path=_remote_olmo_core,
            copy=True,
            ignore=[
                ".git",
                ".git/**",
                ".venv/**",
                "__pycache__/**",
                "build/**",
                "dist/**",
                "doc/**",
                "scratch/**",
            ],
        )
        .run_commands(f"cd {_remote_olmo_core} && pip install -e .")
    )
    app = modal.App("math-transfer-loader-smoke", image=_image)
    checkpoint_volume = modal.Volume.from_name(
        "olmo-core-checkpoints-v2", create_if_missing=True, version=2
    )
    tokenized_volume = modal.Volume.from_name(
        "math-pretraining-tokenized", create_if_missing=True
    )

    @app.function(
        gpu="H100:1",
        cpu=4,
        memory=16 * 1024,
        timeout=60 * 20,
        volumes={
            "/checkpoints": checkpoint_volume,
            "/tokenized": tokenized_volume,
        },
    )
    def one_step_remote(
        pair_root: str,
        natural_glob: str = "/tokenized/3/part_*.npy",
        sequence_length: int = 128,
        seed: int = 1337,
        max_processed_token_relative_delta: float = (
            DEFAULT_MAX_PROCESSED_TOKEN_RELATIVE_DELTA
        ),
    ) -> dict[str, Any]:
        checkpoint_volume.reload()
        tokenized_volume.reload()
        result = run_one_step_smoke(
            pair_root=pair_root,
            natural_glob=natural_glob,
            sequence_length=sequence_length,
            seed=seed,
            max_processed_token_relative_delta=max_processed_token_relative_delta,
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return result

    @app.local_entrypoint()
    def main(
        pair_root: str,
        natural_glob: str = "/tokenized/3/part_*.npy",
        sequence_length: int = 128,
        seed: int = 1337,
        max_processed_token_relative_delta: float = (
            DEFAULT_MAX_PROCESSED_TOKEN_RELATIVE_DELTA
        ),
    ) -> None:
        result = one_step_remote.remote(
            pair_root=pair_root,
            natural_glob=natural_glob,
            sequence_length=sequence_length,
            seed=seed,
            max_processed_token_relative_delta=max_processed_token_relative_delta,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
