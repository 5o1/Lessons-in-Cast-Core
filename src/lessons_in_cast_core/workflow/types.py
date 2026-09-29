"""Requests and results shared by dialogue workflow stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..dialogue.scopes import DialogueScope

from .artifacts import ArtifactLayout


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    artifact_root: Path
    dialogue_tab_path: Path
    allowed_sources: tuple[Path, ...] = ()
    prompt_version: str = "codex-v1"
    overrides_path: Path | None = None
    dialogue_scope: DialogueScope | None = None
    source_root: Path | None = None


@dataclass(frozen=True, slots=True)
class PipelineResult:
    artifacts: ArtifactLayout
    dialogue_count: int
    batch_count: int
    accepted_count: int
    review_count: int
    retryable_count: int
    rejected_count: int
    tts_job_count: int
    rendered_count: int


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    accepted_count: int
    review_required_count: int
    retryable_count: int
    rejected_count: int
    retry_request_count: int
    batch_issue_count: int

    @property
    def total(self) -> int:
        return (
            self.accepted_count
            + self.review_required_count
            + self.retryable_count
            + self.rejected_count
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "accepted": self.accepted_count,
            "review_required": self.review_required_count,
            "retryable": self.retryable_count,
            "rejected": self.rejected_count,
            "retry_requests": self.retry_request_count,
            "batch_issues": self.batch_issue_count,
        }
