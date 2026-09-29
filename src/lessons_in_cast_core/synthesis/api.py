"""Interfaces for replaceable speech synthesis backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ..performance import SpeechAdaptation
from .types import TtsJob


class SpeechSynthesizer(Protocol):
    """Generate the raw audio file for one immutable TTS job."""

    @property
    def name(self) -> str: ...

    @property
    def configuration(self) -> dict[str, Any]: ...

    def adapt(self, job: TtsJob) -> SpeechAdaptation: ...

    def synthesize(self, job: TtsJob, artifact_root: Path) -> Path: ...


class AudioEffectProcessor(Protocol):
    """Apply line-level effects or produce an effect-only output."""

    def process(
        self,
        source: Path | None,
        destination: Path,
        effects: tuple[str, ...],
    ) -> Path: ...
