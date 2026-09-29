"""Backend-neutral interface for character voice pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...performance import SpeechAdaptation
from ..references.builder import ReferenceBuildResult
from ..types import TtsJob

@dataclass(frozen=True, slots=True)
class ReferenceBuildRequest:
    """Explicit source and destination for a generated voice reference."""

    input_directory: Path
    output_path: Path


class VoicePipeline(ABC):
    """Backend-neutral contract used by the main dialogue pipeline."""

    @property
    @abstractmethod
    def pipeline_id(self) -> str:
        """Return a stable identifier for cache fingerprints and diagnostics."""

    @property
    @abstractmethod
    def character_id(self) -> str:
        """Return the character handled by this pipeline."""

    @property
    @abstractmethod
    def configuration(self) -> dict[str, Any]:
        """Return all settings that can affect rendered output."""

    def prepare(self) -> tuple[Path, ...]:
        """Prepare generated dependencies and return their paths."""

        return ()

    def override_reference_audio(self, path: Path) -> None:
        """Override this instance's reference, or reject an unsupported capability.

        Implementations must not modify persistent profile configuration/assets.
        Call before prepare(); the effective configuration must reflect the override.
        """

        raise NotImplementedError(f"{self.pipeline_id} does not support reference overrides")

    def list_voice_tags(self) -> tuple[str, ...]:
        """Discover this profile's usable voice names without preparing/loading a model.

        Unsupported profiles raise, distinguishing unavailable capability from
        a supported profile whose reference directory is empty.
        """
        raise NotImplementedError(f"{self.pipeline_id} does not support local-reference voice tags")

    def adapt(self, job: TtsJob) -> SpeechAdaptation:
        """Compile a job for diagnostics before rendering."""

        if job.segments or job.voice is not None or job.arbitrary_emotion is not None:
            raise NotImplementedError(f"{self.pipeline_id} must implement inline speech lowering")

        return SpeechAdaptation(
            job_id=job.id,
            dialogue_id=job.dialogue_id,
            backend=self.pipeline_id,
            text=job.text,
            emotion=job.emotion,
            parameters={"performance": job.performance.to_dict()},
        )

    @abstractmethod
    def render(self, job: TtsJob, artifact_root: Path) -> Path:
        """Render one immutable speech job, honoring all segments or rejecting them.

        Backend adapters may support inline controls natively or split/merge
        takes. They must never ignore segment emotion or voice selection.
        """

    def close(self) -> None:
        """Release any loaded models or subprocesses."""


class ReferenceVoicePipeline(VoicePipeline, ABC):
    """Optional capability implemented by pipelines that build references."""

    @property
    @abstractmethod
    def default_reference_request(self) -> ReferenceBuildRequest:
        """Return the configured source and generated-reference destination."""

    @abstractmethod
    def build_reference(
        self,
        request: ReferenceBuildRequest,
    ) -> ReferenceBuildResult:
        """Build a persistent reference from a directory of source samples."""
