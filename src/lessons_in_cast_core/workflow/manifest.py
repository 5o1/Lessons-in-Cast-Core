"""Atomic lifecycle management for pipeline run manifests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import ArtifactLayout


def start_run_manifest(
    layout: ArtifactLayout,
    *,
    dialogue_path: Path,
    dialogue_hash: str,
) -> None:
    """Start a run while preserving matching extraction provenance."""

    extraction: dict[str, Any] | None = None
    if layout.run_manifest.exists():
        with layout.run_manifest.open("r", encoding="utf-8") as source:
            previous = json.load(source)
        candidate = previous.get("extraction")
        if (
            isinstance(candidate, dict)
            and candidate.get("dialogue_path") == str(dialogue_path.resolve())
            and candidate.get("dialogue_sha256") == dialogue_hash
        ):
            extraction = candidate
    started_at = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "started_at": started_at,
        "updated_at": started_at,
    }
    if extraction is not None:
        manifest["extraction"] = extraction
    _write_run_manifest(layout, manifest)


def update_run_manifest(
    layout: ArtifactLayout,
    updates: dict[str, Any],
) -> None:
    """Merge updates into the current manifest and write it atomically."""

    current: dict[str, Any] = {}
    if layout.run_manifest.exists():
        with layout.run_manifest.open("r", encoding="utf-8") as source:
            current = json.load(source)
    current.setdefault("schema_version", 1)
    current.update(updates)
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_run_manifest(layout, current)


def _write_run_manifest(
    layout: ArtifactLayout,
    manifest: dict[str, Any],
) -> None:
    layout.run_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = layout.run_manifest.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(manifest, output, ensure_ascii=False, sort_keys=True, indent=2)
        output.write("\n")
    temporary.replace(layout.run_manifest)
