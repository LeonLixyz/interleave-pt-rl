"""Fail-closed, deterministic provenance for Chess-RL/Miles runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".modal-launch-records",
        ".modal-launch-recovery",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".launch-recovery",
        "__pycache__",
        "wandb",
        "node_modules",
        ".vinext",
        ".wrangler",
        "dashboard",
        "tests",
        ".claude",
    }
)
EXCLUDED_FILENAMES = frozenset({".DS_Store"})
RUNTIME_PACKAGES = (
    "modal",
    "torch",
    "transformers",
    "ray",
    "sglang",
    "safetensors",
    "flash_attn",
    "wandb",
)


def source_path_is_excluded(relative: Path) -> bool:
    """Shared exclusion policy for hashed source and Modal source mounts."""

    relative = Path(relative)
    return (
        any(
            part in EXCLUDED_PARTS or part.endswith(".egg-info")
            for part in relative.parts
        )
        or relative.name in EXCLUDED_FILENAMES
        or relative.name == ".env"
        or relative.name.startswith(".env.")
        or relative.suffix.lower() in {".pyc", ".pyo", ".pem", ".key"}
        or relative.name.endswith("~")
    )


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    )
    temporary.replace(path)


def source_tree_identity(
    root: Path,
    *,
    excluded_relatives: Iterable[str] = (),
) -> dict[str, Any]:
    """Hash every file permitted into the corresponding Modal source mount.

    ``excluded_relatives`` is reserved for a tiny out-of-band contract
    binding module whose contents necessarily contain the digest of the
    contract that authenticates the rest of the source tree.  Exclusions are
    explicit, normalized relative paths; directories and glob patterns are
    deliberately unsupported.
    """

    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Source tree does not exist: {root}")
    excluded = frozenset(
        Path(str(relative)).as_posix() for relative in excluded_relatives
    )
    invalid_exclusions = sorted(
        relative
        for relative in excluded
        if (
            not relative
            or relative == "."
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        )
    )
    if invalid_exclusions:
        raise ValueError(
            "source-tree exclusions must be safe relative file paths: "
            + ", ".join(invalid_exclusions)
        )
    rows: list[str] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.as_posix() in excluded:
            continue
        if source_path_is_excluded(relative):
            continue
        size = path.stat().st_size
        digest = _sha256_file(path)
        rows.append(f"{relative.as_posix()}\t{size}\t{digest}\n")
        total_bytes += size
    if not rows:
        raise RuntimeError(f"No code/config files found under source tree: {root}")
    manifest_sha256 = hashlib.sha256("".join(rows).encode()).hexdigest()
    result = {
        "logical_root": str(root),
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "manifest_format": "relative_path<TAB>bytes<TAB>sha256<NEWLINE>",
        "manifest_sha256": manifest_sha256,
    }
    if excluded:
        result["excluded_relatives"] = sorted(excluded)
    return result


def directory_identity(
    root: Path,
    *,
    logical_path: str | None = None,
) -> dict[str, Any]:
    """Hash every regular file in an immutable HF checkpoint directory."""

    if not root.is_dir():
        raise FileNotFoundError(f"Identity directory does not exist: {root}")
    files: list[dict[str, Any]] = []
    manifest_rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(
            part in EXCLUDED_PARTS for part in path.relative_to(root).parts
        ):
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = _sha256_file(path)
        files.append({"path": relative, "bytes": size, "sha256": digest})
        manifest_rows.append(f"{relative}\t{size}\t{digest}\n")
    if not files:
        raise RuntimeError(f"No files found under identity directory: {root}")
    return {
        "logical_path": logical_path or str(root),
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "manifest_sha256": hashlib.sha256(
            "".join(manifest_rows).encode()
        ).hexdigest(),
        "files": files,
    }


def runtime_identity(
    *,
    image: str,
    packages: Iterable[str] = RUNTIME_PACKAGES,
) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    installed_packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            installed_packages[name.lower()] = distribution.version
    installed_packages = dict(sorted(installed_packages.items()))
    return {
        "image": image,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "installed_packages": installed_packages,
        "installed_packages_sha256": hashlib.sha256(
            _canonical_bytes(installed_packages)
        ).hexdigest(),
    }


def write_run_provenance(
    *,
    run_root: Path,
    identity: Mapping[str, Any],
    command: Sequence[str],
) -> dict[str, Any]:
    """Write immutable run identity plus an idempotent exact launch record.

    A retry may add ``--load`` to the exact command, so each unique command gets
    its own append-only launch document. The root identity cannot change: a
    source, data, origin-model, or semantic drift fails before training.
    """

    identity_payload = dict(identity)
    identity_sha256 = hashlib.sha256(
        _canonical_bytes(identity_payload)
    ).hexdigest()
    command_list = [str(item) for item in command]
    command_sha256 = hashlib.sha256(
        json.dumps(command_list, separators=(",", ":")).encode()
    ).hexdigest()
    now = datetime.now(timezone.utc).isoformat()

    root_manifest = run_root / "run_provenance.json"
    if root_manifest.exists():
        existing = json.loads(root_manifest.read_text())
        if existing.get("identity_sha256") != identity_sha256:
            raise RuntimeError(
                "Run provenance mismatch: refusing to continue a run root "
                f"created by different code/data/model semantics: {run_root}"
            )
    else:
        existing = {
            "schema_version": SCHEMA_VERSION,
            "created_at": now,
            "identity_sha256": identity_sha256,
            "identity": identity_payload,
            "initial_command_sha256": command_sha256,
            "initial_command": command_list,
        }
        _atomic_json(root_manifest, existing)

    launch_manifest = (
        run_root
        / "provenance"
        / f"launch_{command_sha256[:16]}.json"
    )
    launch_value = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now,
        "identity_sha256": identity_sha256,
        "command_sha256": command_sha256,
        "command": command_list,
    }
    if launch_manifest.exists():
        recorded = json.loads(launch_manifest.read_text())
        if (
            recorded.get("identity_sha256") != identity_sha256
            or recorded.get("command_sha256") != command_sha256
            or recorded.get("command") != command_list
        ):
            raise RuntimeError(f"Corrupt launch provenance: {launch_manifest}")
    else:
        _atomic_json(launch_manifest, launch_value)

    return {
        "root_manifest": str(root_manifest),
        "launch_manifest": str(launch_manifest),
        "identity_sha256": identity_sha256,
        "command_sha256": command_sha256,
    }
