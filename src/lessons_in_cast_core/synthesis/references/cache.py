"""Content-addressed preparation of generated voice-reference artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ...hashing import content_hash, file_hash
from .builder import (
    SUPPORTED_AUDIO_SUFFIXES,
    ReferenceBuildError,
    ReferenceBuildResult,
    ReferenceBuildSettings,
    build_reference_from_directory,
)


REFERENCE_CACHE_SCHEMA_VERSION = 1


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(f"{output_path.suffix}.manifest.json")


def _source_state(
    input_directory: Path,
    output_path: Path,
) -> list[dict[str, str]]:
    directory = input_directory.resolve()
    if not directory.is_dir():
        raise ReferenceBuildError(f"Sample directory is missing: {directory}")
    output = output_path.resolve()
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
        and path.resolve() != output
    )
    if not paths:
        raise ReferenceBuildError(
            f"No supported audio samples were found below {directory}"
        )
    return [
        {
            "path": str(path.relative_to(directory)),
            "sha256": file_hash(path),
        }
        for path in paths
    ]


def ensure_reference_from_directory(
    *,
    pipeline_id: str,
    input_directory: Path,
    output_path: Path,
    settings: ReferenceBuildSettings,
    builder: Callable[
        [Path, Path, ReferenceBuildSettings], ReferenceBuildResult
    ] | None = None,
) -> tuple[ReferenceBuildResult, bool]:
    """Build a reference only when its sources or processing settings changed."""

    directory = input_directory.resolve()
    output = output_path.resolve()
    identity: dict[str, Any] = {
        "schema_version": REFERENCE_CACHE_SCHEMA_VERSION,
        "pipeline_id": pipeline_id,
        "input_directory": str(directory),
        "settings": asdict(settings),
        "sources": _source_state(directory, output),
    }
    fingerprint = content_hash(identity)
    manifest_path = _manifest_path(output)
    if output.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("schema_version")
                == REFERENCE_CACHE_SCHEMA_VERSION
                and manifest.get("fingerprint") == fingerprint
                and manifest.get("output_sha256") == file_hash(output)
            ):
                result = ReferenceBuildResult.from_dict(manifest["result"])
                return result, True
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    build = builder or build_reference_from_directory
    result = build(directory, output, settings)
    manifest = {
        "schema_version": REFERENCE_CACHE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "identity": identity,
        "output_sha256": file_hash(result.output_path),
        "result": result.to_dict(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    manifest_path.chmod(0o644)
    return result, False
