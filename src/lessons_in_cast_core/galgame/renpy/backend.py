"""Ren'Py implementation of the galgame backend contract."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from ...dialogue.types import DialogueRecord
from ...synthesis.types import AudioQualityResult, RenderTask
from ..api import (
    DialogueExtractionRequest,
    DialogueExtractionResult,
    GalgameBackend,
    GalgameInstallationResult,
)
from .dialogue import TabDialogueReader
from .extraction import SubprocessDialogueExtractor
from .installation import RenPyVoiceInstaller
from .integration import RenPyVoiceScriptWriter


class RenPyBackend(GalgameBackend):
    """Extract and install voice content for Ren'Py games."""

    def __init__(self, *, extraction_timeout_seconds: float = 600.0) -> None:
        self._timeout = extraction_timeout_seconds
        self._extractor = SubprocessDialogueExtractor(
            timeout_seconds=extraction_timeout_seconds
        )
        self._script_writer = RenPyVoiceScriptWriter()
        self._installer = RenPyVoiceInstaller()

    @property
    def backend_id(self) -> str:
        return "renpy"

    @property
    def configuration(self) -> Mapping[str, Any]:
        return {
            "backend": self.backend_id,
            "extraction_timeout_seconds": self._timeout,
        }

    @property
    def integration_filename(self) -> str:
        return "lessons_in_cast_voice.rpy"

    def extract_dialogue(
        self,
        request: DialogueExtractionRequest,
    ) -> DialogueExtractionResult:
        return self._extractor.extract(request)

    def write_integration(
        self,
        destination: Path,
        *,
        entries: Iterable[tuple[str, str]],
    ) -> Path:
        return self._script_writer.write(destination, entries=entries)

    def install_voice_bundle(
        self,
        destination_root: Path,
        *,
        artifact_root: Path,
        integration_artifact: Path,
        voice_manifest: Path,
        artifacts: Iterable[tuple[RenderTask, AudioQualityResult]],
    ) -> GalgameInstallationResult:
        return self._installer.install(
            destination_root,
            artifact_root=artifact_root,
            voice_script=integration_artifact,
            voice_manifest=voice_manifest,
            artifacts=artifacts,
        )

    def read_dialogue(
        self,
        source: Path,
        *,
        allowed_sources: Iterable[Path | str] | None = None,
        source_root: Path | None = None,
    ) -> Iterator[DialogueRecord]:
        return TabDialogueReader(
            allowed_sources,
            source_root=source_root,
        ).read(source)

    def voice_virtual_path(
        self,
        source_filename: str,
        identifier: str,
        audio_format: str,
    ) -> str:
        source = PurePosixPath(source_filename.replace("\\", "/"))
        if source.is_absolute() or any(
            part in {"", ".", ".."} for part in source.parts
        ):
            raise ValueError(f"Unsafe Ren'Py source path: {source_filename!r}")
        parts = source.parts[1:] if source.parts[0] == "game" else source.parts
        if not parts:
            raise ValueError(f"Empty Ren'Py source path: {source_filename!r}")
        relative = PurePosixPath(*parts)
        if relative.suffix.lower() in {".rpy", ".rpym", ".rpyc"}:
            relative = relative.with_suffix("")
        return str(
            PurePosixPath("voice")
            / relative
            / f"{identifier}.{audio_format.lstrip('.')}"
        )
