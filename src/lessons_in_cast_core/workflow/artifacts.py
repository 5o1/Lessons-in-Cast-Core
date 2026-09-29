"""Paths and cleanup rules for one dialogue-pipeline run."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactLayout:
    """Canonical paths owned by one pipeline artifact root."""

    root: Path

    @property
    def polish(self) -> ArtifactLayout:
        return ArtifactLayout(self.root / "polish")

    @property
    def dialogue_tab(self) -> Path:
        return self.root / "dialogue.tab"

    @property
    def raw_dialogue(self) -> Path:
        return self.root / "raw.jsonl"

    @property
    def annotation_requests(self) -> Path:
        return self.root / "annotation_requests.jsonl"

    @property
    def source_audit(self) -> Path:
        return self.root / "source_audit.json"

    @property
    def annotation_responses(self) -> Path:
        return self.root / "annotation_responses.jsonl"

    @property
    def validated(self) -> Path:
        return self.root / "validated.jsonl"

    @property
    def review_required(self) -> Path:
        return self.root / "review_required.jsonl"

    @property
    def retryable(self) -> Path:
        return self.root / "retryable.jsonl"

    @property
    def retry_requests(self) -> Path:
        return self.root / "retry_requests.jsonl"

    @property
    def retry_responses(self) -> Path:
        return self.root / "retry_responses.jsonl"

    @property
    def retry_validated(self) -> Path:
        return self.root / "retry_validated.jsonl"

    @property
    def rejected(self) -> Path:
        return self.root / "rejected.jsonl"

    @property
    def validation_issues(self) -> Path:
        return self.root / "validation_issues.jsonl"

    @property
    def retry_validation_issues(self) -> Path:
        return self.root / "retry_validation_issues.jsonl"

    @property
    def tts_jobs(self) -> Path:
        return self.root / "tts_jobs.jsonl"

    @property
    def render_tasks(self) -> Path:
        return self.root / "render_tasks.jsonl"

    @property
    def synthesis_issues(self) -> Path:
        return self.root / "synthesis_issues.jsonl"

    @property
    def synthesis_adaptations(self) -> Path:
        return self.root / "synthesis_adaptations.jsonl"

    @property
    def audio_quality(self) -> Path:
        return self.root / "audio_quality.jsonl"

    @property
    def audio_effects(self) -> Path:
        """Effect provenance retained even when disposable WAVs are removed."""
        return self.root / "audio_effects.jsonl"

    @property
    def voice_manifest(self) -> Path:
        return self.root / "voice_manifest.json"

    @property
    def galgame_artifacts(self) -> Path:
        return self.root / "galgame"

    @property
    def run_manifest(self) -> Path:
        return self.root / "run_manifest.json"

    @property
    def release_bundle(self) -> Path:
        return self.root / "release_bundle"

    @property
    def release_patch(self) -> Path:
        return self.root / "lessons_in_cast_voice_patch.zip"

    def reset_generated(self) -> None:
        """Remove only pipeline-owned paths from an earlier run."""

        files = (
            self.dialogue_tab,
            self.raw_dialogue,
            self.annotation_requests,
            self.source_audit,
            self.annotation_responses,
            self.validated,
            self.review_required,
            self.retryable,
            self.retry_requests,
            self.retry_responses,
            self.retry_validated,
            self.rejected,
            self.validation_issues,
            self.retry_validation_issues,
            self.tts_jobs,
            self.render_tasks,
            self.synthesis_issues,
            self.synthesis_adaptations,
            self.audio_quality,
            self.voice_manifest,
            self.run_manifest,
            self.audio_effects,
            self.release_patch,
        )
        for path in files:
            path.unlink(missing_ok=True)
        for directory in (
            self.root / "audio",
            self.root / "voice",
            self.root / "codex",
            self.root / "api",
            self.polish.root,
            self.galgame_artifacts,
            self.release_bundle,
        ):
            if directory.exists():
                shutil.rmtree(directory)

    def discard_intermediates(self) -> None:
        """Remove pipeline-owned intermediates while preserving final output."""

        files = (
            self.dialogue_tab,
            self.raw_dialogue,
            self.annotation_requests,
            self.source_audit,
            self.annotation_responses,
            self.validated,
            self.review_required,
            self.retryable,
            self.retry_requests,
            self.retry_responses,
            self.retry_validated,
            self.rejected,
            self.validation_issues,
            self.retry_validation_issues,
            self.tts_jobs,
            self.render_tasks,
            self.synthesis_issues,
            self.synthesis_adaptations,
            self.audio_quality,
            self.voice_manifest,
        )
        for path in files:
            path.unlink(missing_ok=True)
        for directory in (
            self.root / "audio",
            self.root / "voice",
            self.root / "codex",
            self.root / "api",
            self.polish.root,
            self.galgame_artifacts,
        ):
            if directory.exists():
                shutil.rmtree(directory)
