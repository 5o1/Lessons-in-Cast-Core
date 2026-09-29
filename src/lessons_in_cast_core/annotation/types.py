"""Annotation, validation, and provenance types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..performance import SpeechPerformance


class DialogueAction(StrEnum):
    SPEAK = "speak"
    OMIT = "omit"
    SFX_ONLY = "sfx_only"
    SPEAK_WITH_EFFECT = "speak_with_effect"


class ValidationStatus(StrEnum):
    ACCEPTED = "accepted"
    RETRYABLE = "retryable"
    REVIEW_REQUIRED = "review_required"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class Annotation:
    id: str
    action: DialogueAction
    spoken_text: str
    emotion: str | None
    delivery: dict[str, str]
    effects: tuple[str, ...]
    confidence: float | None
    review_required: bool
    reason: str | None = None
    performance: SpeechPerformance = SpeechPerformance()
    keyframe_effects: list[dict] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "id": self.id,
            "action": self.action.value,
            "spoken_text": self.spoken_text,
            "emotion": self.emotion,
            "delivery": self.delivery,
            "effects": list(self.effects),
            "confidence": self.confidence,
            "review_required": self.review_required,
            "reason": self.reason,
            "performance": self.performance.to_dict(),
        }
        if self.emotion is None:
            result.pop("emotion")
        if self.keyframe_effects is not None:
            result["keyframe_effects"] = self.keyframe_effects
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Annotation:
        return cls(
            id=value["id"],
            action=DialogueAction(value["action"]),
            spoken_text=value["spoken_text"],
            emotion=value.get("emotion"),
            delivery=dict(value.get("delivery", {})),
            effects=tuple(value.get("effects", ())),
            confidence=value.get("confidence"),
            review_required=value.get("review_required", False),
            reason=value.get("reason"),
            performance=SpeechPerformance.from_dict(value.get("performance")),
            keyframe_effects=value.get("keyframe_effects"),
        )


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    severity: str
    dialogue_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "dialogue_id": self.dialogue_id,
        }


@dataclass(frozen=True, slots=True)
class ValidatedAnnotation:
    dialogue_id: str
    batch_id: str
    status: ValidationStatus
    annotation: Annotation | None
    issues: tuple[ValidationIssue, ...]
    input_hash: str
    prompt_version: str
    annotator_config_hash: str
    processed_at: str
    source: str = "model"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dialogue_id": self.dialogue_id,
            "batch_id": self.batch_id,
            "status": self.status.value,
            "annotation": self.annotation.to_dict() if self.annotation else None,
            "issues": [issue.to_dict() for issue in self.issues],
            "provenance": {
                "input_hash": self.input_hash,
                "prompt_version": self.prompt_version,
                "annotator_config_hash": self.annotator_config_hash,
                "processed_at": self.processed_at,
                "source": self.source,
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ValidatedAnnotation:
        provenance = value["provenance"]
        return cls(
            dialogue_id=value["dialogue_id"],
            batch_id=value["batch_id"],
            status=ValidationStatus(value["status"]),
            annotation=(
                Annotation.from_dict(value["annotation"])
                if value.get("annotation") is not None
                else None
            ),
            issues=tuple(
                ValidationIssue(
                    code=item["code"],
                    message=item["message"],
                    severity=item["severity"],
                    dialogue_id=item.get("dialogue_id"),
                )
                for item in value.get("issues", ())
            ),
            input_hash=provenance["input_hash"],
            prompt_version=provenance["prompt_version"],
            annotator_config_hash=provenance["annotator_config_hash"],
            processed_at=provenance["processed_at"],
            source=provenance.get("source", "model"),
        )


@dataclass(frozen=True, slots=True)
class BatchValidationResult:
    batch_id: str
    records: tuple[ValidatedAnnotation, ...]
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def accepted(self) -> tuple[ValidatedAnnotation, ...]:
        return tuple(
            item for item in self.records if item.status is ValidationStatus.ACCEPTED
        )
