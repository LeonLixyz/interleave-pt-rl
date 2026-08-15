"""Publish complete interleaved pretraining endpoints as immutable HF repos.

The three registered v1 endpoints are already complete.  Each publication:

* authenticates the exact training config, final trainer state, model/tokenizer
  inventory, and evaluator checkpoint fingerprint on the Modal Volume;
* carries both self-hashed official endpoint evaluation results;
* creates the entire usable endpoint plus provenance in one Hub commit; and
* verifies every remote file by size and SHA-256 (using the LFS object identity
  when available, otherwise downloading the file).

The ``publish_bound_v2r2_endpoint`` function exposes the same fail-closed path
for a future full v2r2 P1 or Exp2 endpoint.  Its caller must provide every
content identity before publication; no incomplete or unbound endpoint is
accepted.

Usage:
  modal run --detach \
    modal_scripts/upload_interleaved_pretrain_endpoints.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import modal


APP_NAME = "chess-interleave-pretrain-hf-publisher"
SCHEMA_VERSION = 1
HF_ORG = "Pre-to-Post-2"
V1_VERSION = "mix10b_sft90k_3072_v1_20260730"
V2R2_VERSION = "mix10b_sft90k_3072_v2r2_staged_gate_20260730"
PUBLICATION_CONTRACT = "complete-only-atomic-hf-endpoint-v1"
CHECKPOINT_MOUNT = Path("/checkpoints")
EVAL_MOUNT = Path("/eval-results")
HUB_GITATTRIBUTES_SHA256 = (
    "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"
)
HUB_GITATTRIBUTES_BYTES = 1_519
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")

FINAL_FILES = {
    "config.json",
    "generation_config.json",
    "interleaved_training_state.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.py",
    "tokenizer_config.json",
    "vocab.json",
}
MODEL_IDENTITY = {
    "model_type": "qwen3",
    "vocab_size": 85,
    "max_position_embeddings": 3_072,
    "hidden_size": 512,
    "head_dim": 128,
    "num_hidden_layers": 12,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "intermediate_size": 1_536,
    "tie_word_embeddings": True,
}

V1_ENDPOINTS: dict[str, dict[str, object]] = {
    "p1": {
        "stage": "p1",
        "repo_id": "Pre-to-Post-2/pretrain_interleave_47m_v1_p1_5b",
        "run_root": (
            "/checkpoints/interleave_50m/pretrain/"
            f"{V1_VERSION}/p1_shared"
        ),
        "experiment_version": V1_VERSION,
        "data_artifact_version": "mix10b_sft90k_v1",
        "expected_step": 9_920,
        "expected_arc_steps": [9_920],
        "expected_manifest_hash": (
            "6e2bbd62283df234e8ea10c1d27d871e4ce59991785c58b13de3d6b721feaf5a"
        ),
        "expected_config_sha256": (
            "9d84021cf2ee23799513db97a838309793cd5b27dfa6d58e4387ed1970bb8094"
        ),
        "expected_final_state_sha256": (
            "fd67f518b8d35db42a0f35a808d279fb376596c91c140bf9029bf9a8b2195c22"
        ),
        "expected_checkpoint_fingerprint": (
            "0f402347123cf8e7524d3e31ff3a60d5bd8c86b4d81e1fc1c5f7d28d276be503"
        ),
        "source_tree_sha256": (
            "98db54b40e6af5bbbbca526b890c5cf19a96924c08c0c3e92cf0ea7edc6aba49"
        ),
        "evaluation": {
            "endpoint_id": "p1",
            "namespace": "endpoint_v1",
            "components": {
                "losses": {
                    "directory": "losses_30a200d92436",
                    "result_hash": (
                        "5b43578c5670e66ccd600e3f3c795250ec03b051d503c8210165f40862b767db"
                    ),
                },
                "chess": {
                    "directory": "chess_cea01f20b708",
                    "result_hash": (
                        "7140794edd3b8b907e1be4a6a952dbf579a27a600194140833f7a0ceb22cffbc"
                    ),
                },
            },
        },
    },
    "exp2": {
        "stage": "exp2",
        "repo_id": (
            "Pre-to-Post-2/pretrain_interleave_47m_v1_exp2_monolithic_10b"
        ),
        "run_root": (
            "/checkpoints/interleave_50m/pretrain/"
            f"{V1_VERSION}/exp2_monolithic"
        ),
        "experiment_version": V1_VERSION,
        "data_artifact_version": "mix10b_sft90k_v1",
        "expected_step": 19_840,
        "expected_arc_steps": [19_840],
        "expected_manifest_hash": (
            "7be36dcbda5d31c4c554e270e5f1fbacf21f03020077f9f02ab3845e319e3a0e"
        ),
        "expected_config_sha256": (
            "8d7d7d22385a2afdfc3a4def774638234bbbe21ec9fe448570b6fb0460fce099"
        ),
        "expected_final_state_sha256": (
            "daf6616f064e522d43e1119baa9a80fd0a20a0bb5afc460f3b44abe90081097a"
        ),
        "expected_checkpoint_fingerprint": (
            "22ce8af7277d0c2fb1e1e603fb686f6a947cdd79bfd94aaa01fecdff86079a0b"
        ),
        "source_tree_sha256": (
            "98db54b40e6af5bbbbca526b890c5cf19a96924c08c0c3e92cf0ea7edc6aba49"
        ),
        "evaluation": {
            "endpoint_id": "e2-final",
            "namespace": "endpoint_v1",
            "components": {
                "losses": {
                    "directory": "losses_30a200d92436",
                    "result_hash": (
                        "ef2fe9842062f75288d58825ab5b2b850bd9227923b66855476799acfb3c2caa"
                    ),
                },
                "chess": {
                    "directory": "chess_cea01f20b708",
                    "result_hash": (
                        "fedebe38a74e3a530ae58ce5686e1672063979b2d2c3c1d57e9156af7269cd1d"
                    ),
                },
            },
        },
    },
    "exp3": {
        "stage": "exp3_p2",
        "repo_id": (
            "Pre-to-Post-2/pretrain_interleave_47m_v1_exp3_two_cosine_10b"
        ),
        "run_root": (
            "/checkpoints/interleave_50m/pretrain/"
            f"{V1_VERSION}/p2/"
            "exp3-two-cosine-control-from-p1-from-d8315ae0645b"
        ),
        "experiment_version": V1_VERSION,
        "data_artifact_version": "mix10b_sft90k_v1",
        "expected_step": 9_920,
        "expected_arc_steps": [9_920],
        "expected_manifest_hash": (
            "4144157fa4a5155f30d67af0344231a2179af2af6ea3135af18ea28e74083752"
        ),
        "expected_config_sha256": (
            "815c59943318767ff1965e5566ce954e9d24e4ff4c11032d3e90bec43ae381af"
        ),
        "expected_final_state_sha256": (
            "aa1da771d5cda7690d464aaa28b1169d6f9f8e6653909217ec180d691a1ccaa9"
        ),
        "expected_checkpoint_fingerprint": (
            "d7f6be3ced127707f365b9aec7da72c07894f630726c25fd1349dccdc5a26efc"
        ),
        "source_tree_sha256": (
            "98db54b40e6af5bbbbca526b890c5cf19a96924c08c0c3e92cf0ea7edc6aba49"
        ),
        "evaluation": {
            "endpoint_id": "e3-p2",
            "namespace": "endpoint_v1",
            "components": {
                "losses": {
                    "directory": "losses_30a200d92436",
                    "result_hash": (
                        "94bc0f9de3722bb9d90b1ca2da777d64674be301e0c9a7f220f309dcacb53266"
                    ),
                },
                "chess": {
                    "directory": "chess_cea01f20b708",
                    "result_hash": (
                        "0b1f63b3ec80226c4eb75c5e10cacaf569568f9f1ddb3409ee2dde8184cce9a8"
                    ),
                },
            },
        },
    },
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface-hub==0.35.3",
        "pyyaml==6.0.2",
    )
)
checkpoint_volume = modal.Volume.from_name(
    "rl-reasoning-checkpoints", create_if_missing=False
)
eval_volume = modal.Volume.from_name(
    "chess-rl-eval-results-r6", create_if_missing=False
)
app = modal.App(
    APP_NAME,
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        str(CHECKPOINT_MOUNT): checkpoint_volume,
        str(EVAL_MOUNT): eval_volume,
    },
)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON document is not an object: {path}")
    return value


def _require_sha(value: object, *, label: str) -> str:
    text = str(value)
    if not SHA256_RE.fullmatch(text):
        raise ValueError(f"{label} is not a SHA-256: {value!r}")
    return text


def _safe_absolute_path(value: object, *, prefix: str, label: str) -> Path:
    text = str(value)
    pure = PurePosixPath(text)
    if (
        not pure.is_absolute()
        or ".." in pure.parts
        or not text.startswith(prefix)
    ):
        raise ValueError(f"invalid {label}: {text!r}")
    return Path(text)


def _nested(payload: Mapping[str, object], *keys: str) -> object:
    value: object = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise RuntimeError(f"config lacks {'.'.join(keys)}")
        value = value[key]
    return value


def _validate_bound_spec(
    value: Mapping[str, object],
    *,
    v2r2: bool,
) -> dict[str, object]:
    spec = dict(value)
    required = {
        "stage",
        "repo_id",
        "run_root",
        "experiment_version",
        "data_artifact_version",
        "expected_step",
        "expected_arc_steps",
        "expected_manifest_hash",
        "expected_config_sha256",
        "expected_final_state_sha256",
        "expected_checkpoint_fingerprint",
        "source_tree_sha256",
    }
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"endpoint spec lacks {missing}")
    stage = str(spec["stage"])
    if v2r2 and stage not in {"p1", "exp2"}:
        raise ValueError("v2r2 publisher accepts only full P1 or Exp2")
    if not v2r2 and stage not in {"p1", "exp2", "exp3_p2"}:
        raise ValueError(f"invalid v1 endpoint stage: {stage}")
    version = str(spec["experiment_version"])
    expected_version = V2R2_VERSION if v2r2 else V1_VERSION
    if version != expected_version:
        raise ValueError(
            f"endpoint version mismatch: {version} != {expected_version}"
        )
    repo_id = str(spec["repo_id"])
    required_repo_prefix = (
        f"{HF_ORG}/pretrain_interleave_47m_v2r2_"
        if v2r2
        else f"{HF_ORG}/pretrain_interleave_47m_v1_"
    )
    if not repo_id.startswith(required_repo_prefix):
        raise ValueError(f"invalid immutable endpoint repo: {repo_id}")
    run_root = _safe_absolute_path(
        spec["run_root"],
        prefix=(
            "/checkpoints/interleave_50m/pretrain/"
            f"{expected_version}/"
        ),
        label="run_root",
    )
    spec["run_root"] = str(run_root)
    expected_step = int(spec["expected_step"])
    expected_by_stage = {"p1": 9_920, "exp2": 19_840}
    if v2r2 and expected_step != expected_by_stage[stage]:
        raise ValueError("v2r2 full endpoint step count is invalid")
    arc_steps = spec["expected_arc_steps"]
    if (
        not isinstance(arc_steps, list)
        or not arc_steps
        or not all(isinstance(step, int) and step > 0 for step in arc_steps)
        or arc_steps[-1] != expected_step
    ):
        raise ValueError("endpoint arc schedule is invalid")
    for key in (
        "expected_manifest_hash",
        "expected_config_sha256",
        "expected_final_state_sha256",
        "expected_checkpoint_fingerprint",
        "source_tree_sha256",
    ):
        spec[key] = _require_sha(spec[key], label=key)
    return spec


def _final_records(final: Path) -> list[dict[str, object]]:
    paths = sorted(path for path in final.rglob("*") if path.is_file())
    relative_names = {
        path.relative_to(final).as_posix() for path in paths
    }
    if relative_names != FINAL_FILES:
        raise RuntimeError(
            f"final endpoint inventory differs: "
            f"missing={sorted(FINAL_FILES - relative_names)} "
            f"extra={sorted(relative_names - FINAL_FILES)}"
        )
    records = [
        {
            "path": path.relative_to(final).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "source": "final_endpoint",
        }
        for path in paths
    ]
    if any(int(record["bytes"]) <= 0 for record in records):
        raise RuntimeError("final endpoint contains an empty file")
    return records


def _directory_manifest_sha256(
    records: list[dict[str, object]],
) -> str:
    rows = [
        f"{record['path']}\t{record['bytes']}\t{record['sha256']}\n"
        for record in records
        if record["source"] == "final_endpoint"
    ]
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def _checkpoint_fingerprint(final: Path) -> str:
    digest = hashlib.sha256()
    for relative in sorted(FINAL_FILES):
        path = final / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_evaluation_result(
    payload: Mapping[str, object],
    *,
    component: str,
    endpoint_id: str,
    checkpoint_fingerprint: str,
    expected_result_hash: str,
) -> None:
    expected = {
        "schema": "interleaved-endpoint-result-v1",
        "schema_version": 1,
        "namespace": "endpoint_v1",
        "endpoint_id": endpoint_id,
        "component": component,
        "checkpoint_sha256": checkpoint_fingerprint,
        "experiment_version": V1_VERSION,
        "state": "complete",
        "result_hash": expected_result_hash,
    }
    mismatches = {
        key: {"observed": payload.get(key), "expected": item}
        for key, item in expected.items()
        if payload.get(key) != item
    }
    if mismatches:
        raise RuntimeError(
            f"official endpoint evaluation drift ({component}): {mismatches}"
        )
    unhashed = {
        str(key): item
        for key, item in payload.items()
        if key != "result_hash"
    }
    if _canonical_sha256(unhashed) != expected_result_hash:
        raise RuntimeError(
            f"official endpoint evaluation self-hash mismatch: {component}"
        )


def _authenticate_endpoint(
    spec_value: Mapping[str, object],
    *,
    v2r2: bool,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, Path],
]:
    import yaml

    spec = _validate_bound_spec(spec_value, v2r2=v2r2)
    run_root = Path(str(spec["run_root"]))
    final = run_root / "final"
    config_path = run_root / "config.yaml"
    latest_state_path = run_root / "latest" / "trainer_state.json"
    final_state_path = final / "interleaved_training_state.json"
    for path in (config_path, latest_state_path, final_state_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"complete endpoint contract lacks {path}"
            )
    if _sha256_file(config_path) != spec["expected_config_sha256"]:
        raise RuntimeError("immutable training config SHA-256 mismatch")
    if _sha256_file(final_state_path) != spec[
        "expected_final_state_sha256"
    ]:
        raise RuntimeError("immutable final trainer-state SHA-256 mismatch")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise RuntimeError("training config is not an object")
    expected_config = {
        ("training", "output_dir"): str(run_root),
        ("training", "total_steps"): int(spec["expected_step"]),
        ("training", "arc_steps"): spec["expected_arc_steps"],
        ("training", "gradient_accumulation_steps"): 1,
        ("training", "local_batch_size"): 21,
        ("data", "expected_manifest_hash"): (
            spec["expected_manifest_hash"]
        ),
        ("data", "sequence_length"): 3_072,
        ("provenance", "experiment_version"): (
            spec["experiment_version"]
        ),
        ("provenance", "data_artifact_version"): (
            spec["data_artifact_version"]
        ),
        ("provenance", "source_tree_sha256"): (
            spec["source_tree_sha256"]
        ),
        ("provenance", "source_repo"): (
            "chess-pre-to-post/pretrain_v1_20b"
        ),
        ("provenance", "source_revision"): (
            "07dd1b7090ca5f0fb05ef624c26b20bff19483c8"
        ),
        ("provenance", "sft_repo"): (
            "chess-pre-to-post/sft_v1_200m_90k"
        ),
        ("provenance", "sft_revision"): (
            "97f60746dd253b4e130beeb5e66f39e9d42ef25c"
        ),
    }
    for keys, expected in expected_config.items():
        observed = _nested(config, *keys)
        if observed != expected:
            raise RuntimeError(
                f"training config drift at {'.'.join(keys)}: "
                f"{observed!r} != {expected!r}"
            )

    latest_state = _load_json(latest_state_path)
    final_state = _load_json(final_state_path)
    if latest_state != final_state:
        raise RuntimeError(
            "final trainer state differs from resumable latest state"
        )
    expected_state = {
        "global_step": int(spec["expected_step"]),
        "manifest_cursor": int(spec["expected_step"]),
        "manifest_hash": spec["expected_manifest_hash"],
        "arc_steps": spec["expected_arc_steps"],
        "gradient_accumulation_steps": 1,
        "local_batch_size": 21,
        "world_size": 8,
        "attention_backend": "sdpa",
        "torch_compile_mode": "none",
    }
    for key, expected in expected_state.items():
        if final_state.get(key) != expected:
            raise RuntimeError(
                f"final trainer-state drift at {key}: "
                f"{final_state.get(key)!r} != {expected!r}"
            )

    model_config = _load_json(final / "config.json")
    model_mismatches = {
        key: {"observed": model_config.get(key), "expected": expected}
        for key, expected in MODEL_IDENTITY.items()
        if model_config.get(key) != expected
    }
    if model_mismatches:
        raise RuntimeError(
            f"final model architecture drift: {model_mismatches}"
        )
    tokenizer_config = _load_json(final / "tokenizer_config.json")
    auto_map = tokenizer_config.get("auto_map")
    if (
        not isinstance(auto_map, Mapping)
        or auto_map.get("AutoTokenizer")
        != ["tokenizer.HFTokenizerWrapper", None]
    ):
        raise RuntimeError(
            "final tokenizer is not the self-contained wrapper"
        )

    records = _final_records(final)
    checkpoint_fingerprint = _checkpoint_fingerprint(final)
    if checkpoint_fingerprint != spec["expected_checkpoint_fingerprint"]:
        raise RuntimeError(
            "complete endpoint evaluator fingerprint mismatch: "
            f"{checkpoint_fingerprint} != "
            f"{spec['expected_checkpoint_fingerprint']}"
        )
    local_files = {
        record["path"]: final / str(record["path"])
        for record in records
    }

    config_record = {
        "path": "provenance/training_config.yaml",
        "bytes": config_path.stat().st_size,
        "sha256": _sha256_file(config_path),
        "source": "training_provenance",
    }
    records.append(config_record)
    local_files[str(config_record["path"])] = config_path

    evaluation_identity: dict[str, object] | None = None
    evaluation = spec.get("evaluation")
    if evaluation is not None:
        if v2r2 or not isinstance(evaluation, Mapping):
            raise ValueError("invalid endpoint evaluation binding")
        endpoint_id = str(evaluation.get("endpoint_id"))
        namespace = str(evaluation.get("namespace"))
        components = evaluation.get("components")
        if namespace != "endpoint_v1" or not isinstance(
            components, Mapping
        ):
            raise ValueError("invalid v1 endpoint evaluation spec")
        result_hashes: dict[str, str] = {}
        for component in ("losses", "chess"):
            component_spec = components.get(component)
            if not isinstance(component_spec, Mapping):
                raise ValueError(
                    f"missing endpoint evaluation component: {component}"
                )
            directory = str(component_spec.get("directory"))
            expected_result_hash = _require_sha(
                component_spec.get("result_hash"),
                label=f"{component}.result_hash",
            )
            source = (
                EVAL_MOUNT
                / namespace
                / endpoint_id
                / checkpoint_fingerprint
                / directory
                / "_SUCCESS.json"
            )
            if not source.is_file():
                raise FileNotFoundError(
                    f"official endpoint result is incomplete: {source}"
                )
            payload = _load_json(source)
            _validate_evaluation_result(
                payload,
                component=component,
                endpoint_id=endpoint_id,
                checkpoint_fingerprint=checkpoint_fingerprint,
                expected_result_hash=expected_result_hash,
            )
            remote_path = f"evaluation/{component}_SUCCESS.json"
            record = {
                "path": remote_path,
                "bytes": source.stat().st_size,
                "sha256": _sha256_file(source),
                "source": "official_endpoint_evaluation",
            }
            records.append(record)
            local_files[remote_path] = source
            result_hashes[component] = expected_result_hash
        evaluation_identity = {
            "namespace": namespace,
            "endpoint_id": endpoint_id,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "result_hashes": result_hashes,
        }

    manifest_core: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "interleaved_pretrain_hf_endpoint",
        "publication_contract": PUBLICATION_CONTRACT,
        "repo_id": spec["repo_id"],
        "stage": spec["stage"],
        "source": {
            "volume": "rl-reasoning-checkpoints",
            "run_root": str(run_root),
            "config_sha256": spec["expected_config_sha256"],
            "final_state_sha256": spec[
                "expected_final_state_sha256"
            ],
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "final_directory_manifest_sha256": (
                _directory_manifest_sha256(records)
            ),
        },
        "training_identity": {
            "experiment_version": spec["experiment_version"],
            "data_artifact_version": spec["data_artifact_version"],
            "source_tree_sha256": spec["source_tree_sha256"],
            "manifest_hash": spec["expected_manifest_hash"],
            "global_step": spec["expected_step"],
            "arc_steps": spec["expected_arc_steps"],
        },
        "evaluation": evaluation_identity,
        "files": records,
    }
    manifest = {
        **manifest_core,
        "manifest_sha256": _canonical_sha256(manifest_core),
    }
    return spec, manifest, local_files


def _repo_items(api, repo_id: str) -> dict[str, object]:
    try:
        return {
            str(item.path): item
            for item in api.list_repo_tree(
                repo_id=repo_id,
                repo_type="model",
                recursive=True,
                expand=True,
            )
            # A recursive root listing also emits RepoFolder entries.  They
            # are structural views, not repository files or commit payloads.
            if getattr(item, "size", None) is not None
        }
    except Exception as exc:
        text = str(exc)
        if "404" in text or "not found" in text.lower():
            return {}
        raise


def _download_path(repo_id: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            token=os.environ["HF_TOKEN"],
        )
    )


def _verify_remote(
    api,
    *,
    repo_id: str,
    manifest: Mapping[str, object],
) -> str:
    items = _repo_items(api, repo_id)
    expected_paths = {
        str(record["path"])
        for record in manifest["files"]
        if isinstance(record, Mapping)
    }
    expected_paths.update({"endpoint_manifest.json", ".gitattributes"})
    if set(items) != expected_paths:
        raise RuntimeError(
            f"remote endpoint inventory differs at {repo_id}: "
            f"missing={sorted(expected_paths - set(items))} "
            f"extra={sorted(set(items) - expected_paths)}"
        )

    attributes = _download_path(repo_id, ".gitattributes")
    if (
        attributes.stat().st_size != HUB_GITATTRIBUTES_BYTES
        or _sha256_file(attributes) != HUB_GITATTRIBUTES_SHA256
    ):
        raise RuntimeError("Hub .gitattributes identity drifted")

    remote_manifest_path = _download_path(
        repo_id, "endpoint_manifest.json"
    )
    remote_manifest = _load_json(remote_manifest_path)
    if remote_manifest != dict(manifest):
        raise RuntimeError("remote endpoint manifest differs")
    recorded_manifest_hash = remote_manifest.get("manifest_sha256")
    unhashed = {
        str(key): value
        for key, value in remote_manifest.items()
        if key != "manifest_sha256"
    }
    if recorded_manifest_hash != _canonical_sha256(unhashed):
        raise RuntimeError("remote endpoint manifest self-hash mismatch")

    by_path = {
        str(record["path"]): record
        for record in manifest["files"]
        if isinstance(record, Mapping)
    }
    for path, record in by_path.items():
        expected_size = int(record["bytes"])
        expected_sha = str(record["sha256"])
        item = items[path]
        if int(getattr(item, "size", -1)) != expected_size:
            raise RuntimeError(
                f"remote endpoint file size differs: {repo_id}/{path}"
            )
        lfs = getattr(item, "lfs", None)
        if lfs is not None:
            observed_sha = str(
                lfs.get("sha256", "")
                if isinstance(lfs, Mapping)
                else getattr(lfs, "sha256", "")
            )
            observed_size = int(
                lfs.get("size", -1)
                if isinstance(lfs, Mapping)
                else getattr(lfs, "size", -1)
            )
            if (
                observed_sha != expected_sha
                or observed_size != expected_size
            ):
                raise RuntimeError(
                    f"remote endpoint LFS identity differs: "
                    f"{repo_id}/{path}"
                )
            continue
        downloaded = _download_path(repo_id, path)
        if (
            downloaded.stat().st_size != expected_size
            or _sha256_file(downloaded) != expected_sha
        ):
            raise RuntimeError(
                f"remote endpoint content differs: {repo_id}/{path}"
            )
    head = str(api.model_info(repo_id=repo_id).sha)
    if not COMMIT_RE.fullmatch(head):
        raise RuntimeError(f"invalid HF endpoint HEAD: {head!r}")
    return head


def _publish_authenticated_endpoint(
    spec_value: Mapping[str, object],
    *,
    v2r2: bool,
) -> dict[str, object]:
    from huggingface_hub import CommitOperationAdd, HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required before any repository operation"
        )
    checkpoint_volume.reload()
    eval_volume.reload()
    spec, manifest, local_files = _authenticate_endpoint(
        spec_value,
        v2r2=v2r2,
    )
    repo_id = str(spec["repo_id"])
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    paths = set(_repo_items(api, repo_id))
    if "endpoint_manifest.json" in paths:
        head = _verify_remote(
            api,
            repo_id=repo_id,
            manifest=manifest,
        )
        return {
            "status": "verified_existing",
            "repo_id": repo_id,
            "commit": head,
            "checkpoint_fingerprint": spec[
                "expected_checkpoint_fingerprint"
            ],
            "manifest_sha256": manifest["manifest_sha256"],
            "file_count": len(manifest["files"]),
        }
    unexpected = paths - {".gitattributes"}
    if unexpected:
        raise RuntimeError(
            f"refusing partial/nonempty endpoint repo {repo_id}: "
            f"{sorted(unexpected)}"
        )
    if ".gitattributes" not in paths:
        raise RuntimeError(
            f"new Hub repo lacks authenticated .gitattributes: {repo_id}"
        )
    attributes = _download_path(repo_id, ".gitattributes")
    if (
        attributes.stat().st_size != HUB_GITATTRIBUTES_BYTES
        or _sha256_file(attributes) != HUB_GITATTRIBUTES_SHA256
    ):
        raise RuntimeError("Hub .gitattributes identity drifted")

    with tempfile.TemporaryDirectory(
        prefix="interleave-pretrain-hf-"
    ) as temporary:
        manifest_path = Path(temporary) / "endpoint_manifest.json"
        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        operations = [
            CommitOperationAdd(
                path_in_repo=remote_path,
                path_or_fileobj=str(local_path),
            )
            for remote_path, local_path in sorted(local_files.items())
        ]
        operations.append(
            CommitOperationAdd(
                path_in_repo="endpoint_manifest.json",
                path_or_fileobj=str(manifest_path),
            )
        )
        head = str(api.model_info(repo_id=repo_id).sha)
        commit = api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=operations,
            commit_message=(
                f"Publish complete immutable interleaved endpoint "
                f"{spec['stage']}"
            ),
            parent_commit=head,
        )
        commit_id = str(getattr(commit, "oid", "") or "")
        if not COMMIT_RE.fullmatch(commit_id):
            raise RuntimeError(
                f"HF returned an invalid endpoint commit: {commit_id!r}"
            )
    verified_head = _verify_remote(
        api,
        repo_id=repo_id,
        manifest=manifest,
    )
    if verified_head != commit_id:
        raise RuntimeError(
            f"endpoint HEAD moved during verification: "
            f"{verified_head} != {commit_id}"
        )
    return {
        "status": "published_and_verified",
        "repo_id": repo_id,
        "commit": commit_id,
        "checkpoint_fingerprint": spec[
            "expected_checkpoint_fingerprint"
        ],
        "manifest_sha256": manifest["manifest_sha256"],
        "file_count": len(manifest["files"]),
    }


@app.function(
    cpu=8.0,
    memory=32 * 1024,
    timeout=2 * 60 * 60,
    retries=modal.Retries(initial_delay=10.0, max_retries=1),
    max_containers=3,
)
def publish_registered_v1_endpoint(
    endpoint_key: str,
) -> dict[str, object]:
    if endpoint_key not in V1_ENDPOINTS:
        raise ValueError(f"unregistered v1 endpoint: {endpoint_key}")
    result = _publish_authenticated_endpoint(
        V1_ENDPOINTS[endpoint_key],
        v2r2=False,
    )
    print(
        f"[pretrain-hf] {endpoint_key}: "
        f"{json.dumps(result, sort_keys=True)}",
        flush=True,
    )
    return result


@app.function(
    cpu=8.0,
    memory=32 * 1024,
    timeout=2 * 60 * 60,
    retries=modal.Retries(initial_delay=10.0, max_retries=1),
)
def publish_bound_v2r2_endpoint(
    bound_spec: dict[str, object],
) -> dict[str, object]:
    """Publish a future full v2r2 endpoint with predeclared content hashes."""

    if "evaluation" in bound_spec:
        raise ValueError(
            "v2r2 training publication does not accept relabeled v1 evaluation"
        )
    result = _publish_authenticated_endpoint(bound_spec, v2r2=True)
    print(
        f"[pretrain-hf] v2r2: {json.dumps(result, sort_keys=True)}",
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    submitted = {
        key: {
            "call_id": call.object_id,
            "repo_id": spec["repo_id"],
        }
        for key, spec in sorted(V1_ENDPOINTS.items())
        for call in [publish_registered_v1_endpoint.spawn(key)]
    }
    print(json.dumps(submitted, indent=2, sort_keys=True), flush=True)
