"""Backend-neutral contracts for visual-novel game engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..dialogue.types import DialogueRecord
from ..synthesis.types import AudioQualityResult, RenderTask


class DialogueExtractionError(RuntimeError):
    """Raised when a game backend cannot produce a dialogue table."""


@dataclass(frozen=True, slots=True)
class DialogueExtractionRequest:
    """Backend-neutral inputs for dialogue extraction."""

    release_path: Path
    output_path: Path
    source_paths: Sequence[Path] = ()
    language: str | None = None
    executable_path: Path | None = None


@dataclass(frozen=True, slots=True)
class DialogueExtractionResult:
    """Metadata returned by a game backend's dialogue extractor."""

    dialogue_path: Path
    source_count: int
    command: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GalgameInstallationResult:
    """Result of installing generated voice artifacts into a release bundle."""

    content_root: Path
    audio_count: int
    audio_archive: Path
    patch_path: Path


class DialogueExtractor(Protocol):
    """Extract a normalized dialogue artifact."""

    def extract(
        self,
        request: DialogueExtractionRequest,
    ) -> DialogueExtractionResult: ...


class GalgameBackend(ABC):
    """Contract implemented by each supported visual-novel engine."""

    @property
    @abstractmethod
    def backend_id(self) -> str:
        """Return the stable backend identifier used in project configuration."""

    @property
    @abstractmethod
    def configuration(self) -> Mapping[str, Any]:
        """Return settings that affect extraction or release integration."""

    @property
    @abstractmethod
    def integration_filename(self) -> str:
        """Return the backend-specific integration artifact filename."""

    @abstractmethod
    def extract_dialogue(
        self,
        request: DialogueExtractionRequest,
    ) -> DialogueExtractionResult:
        """Extract dialogue from one game release."""

    @abstractmethod
    def read_dialogue(
        self,
        source: Path,
        *,
        allowed_sources: Iterable[Path | str] | None = None,
        source_root: Path | None = None,
    ) -> Iterator[DialogueRecord]:
        """Normalize one backend-specific export into dialogue records."""

    @abstractmethod
    def voice_virtual_path(
        self,
        source_filename: str,
        identifier: str,
        audio_format: str,
    ) -> str:
        """Map a source line to the audio path expected by the engine."""

    @abstractmethod
    def write_integration(
        self,
        destination: Path,
        *,
        entries: Iterable[tuple[str, str]],
    ) -> Path:
        """Write the backend-specific dialogue-to-audio resolver."""

    @abstractmethod
    def install_voice_bundle(
        self,
        destination_root: Path,
        *,
        artifact_root: Path,
        integration_artifact: Path,
        voice_manifest: Path,
        artifacts: Iterable[tuple[RenderTask, AudioQualityResult]],
    ) -> GalgameInstallationResult:
        """Install generated artifacts into a backend-loadable directory tree."""
