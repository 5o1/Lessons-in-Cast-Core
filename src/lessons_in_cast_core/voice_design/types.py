"""Backend-neutral inputs and artifacts for character voice design."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VoiceDesignRequest:
    """Keep the spoken script separate from voice or delivery instructions.

    Without a reference, the instruction describes a new voice. With a
    reference, the supported changes depend on the backend; VoxCPM2 controls
    delivery while aiming to preserve the reference timbre.
    """

    text: str
    instruction: str = ""
    reference_audio: Path | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Voice-design text must not be empty")
        if not isinstance(self.instruction, str):
            raise ValueError("Voice-design instruction must be a string")
        if self.reference_audio is None and not self.instruction.strip():
            raise ValueError("Designing a new voice requires an instruction")
        if self.reference_audio is not None and not isinstance(self.reference_audio, Path):
            raise ValueError("reference_audio must be a Path")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")


@dataclass(frozen=True, slots=True)
class VoiceDesignResult:
    """One reference-quality audition and its reproducibility record."""

    audio_path: Path
    metadata_path: Path
    sample_rate: int
    duration_seconds: float
