"""Reproducibility manifest for one consolidation run (experiment_manifest.json).

Collects what is needed to reproduce or audit a run: package versions, git commit,
base model identity, resolved configuration, seed, devices, pre/post metrics and the
adapter location. Everything here works without the `train` extra installed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

from ..utils.ids import utc_now_iso

MANIFEST_FILENAME = "experiment_manifest.json"
MANIFEST_SCHEMA_VERSION = "experiment-manifest-v1"
TRACKED_PACKAGES = ("intra-model-memval-lab", "pydantic", "torch", "transformers", "peft", "accelerate")


def package_versions(packages: Iterable[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_state(root: Path) -> dict[str, Any]:
    """Current commit and whether the working tree has uncommitted changes (None when unknown)."""
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain") if commit else None
    return {"commit": commit or None, "dirty": None if status is None else bool(status)}


def hash_model_dir(directory: Path, *, chunk_size: int = 1 << 20) -> str:
    """SHA-256 over relative paths and contents of every file under a local model directory."""
    base = Path(directory)
    digest = hashlib.sha256()
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        digest.update(path.relative_to(base).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    return digest.hexdigest()


def describe_base_model(
    model_id: str,
    *,
    requested_revision: str | None = None,
    resolved_revision: str | None = None,
) -> dict[str, Any]:
    """Identify the base model: hub revision for remote ids, content hash for local directories."""
    local = Path(model_id)
    return {
        "id": model_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "local_sha256": hash_model_dir(local) if local.is_dir() else None,
    }


def build_experiment_manifest(
    *,
    run_id: str,
    status: str,
    repo_root: Path,
    base_model: dict[str, Any],
    seed: int,
    devices: dict[str, Any],
    config: dict[str, Any],
    data: dict[str, Any],
    metrics: dict[str, Any],
    adapter: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "created_at_utc": utc_now_iso(),
        "versions": package_versions(),
        "git": git_state(repo_root),
        "base_model": base_model,
        "seed": seed,
        "devices": devices,
        "config": config,
        "data": data,
        "metrics": metrics,
        "adapter": adapter,
        "decision": decision,
        "error": error,
    }


def manifest_to_json_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")
