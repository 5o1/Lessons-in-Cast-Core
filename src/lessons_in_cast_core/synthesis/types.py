"""Speech jobs, line-level render tasks, and audio validation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..annotation import DialogueAction
from ..performance import SpeechPerformance
from ..speech_markup import SpeechSegment


@dataclass(frozen=True, slots=True)
class TtsJob:
    id: str
    dialogue_id: str
    character_id: str
    text: str
    emotion: str | None
    delivery: dict[str, str]
    output_path: str
    cache_key: str
    voice_profile: str = ""
    performance: SpeechPerformance = SpeechPerformance()
    segments: tuple[SpeechSegment, ...] = ()
    voice: str | None = None
    arbitrary_emotion: str | None = None

    def __post_init__(self):
        if self.arbitrary_emotion is not None:
            if self.emotion is not None or self.segments:
                raise ValueError("arbitrary_emotion cannot overlap preset emotions or segment controls")
            if not isinstance(self.arbitrary_emotion, str) or not self.arbitrary_emotion.strip():
                raise ValueError("arbitrary_emotion requires a nonempty description")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "id": self.id,
            "dialogue_id": self.dialogue_id,
            "character_id": self.character_id,
            "text": self.text,
            "emotion": self.emotion,
            "delivery": self.delivery,
            "output_path": self.output_path,
            "cache_key": self.cache_key,
            "voice_profile": self.voice_profile,
            "performance": self.performance.to_dict(),
            "segments": [segment.to_dict() for segment in self.segments],
            "voice": self.voice,
            "arbitrary_emotion": self.arbitrary_emotion,
        }
        if self.segments:
            result.pop("emotion")
            result.pop("voice")
            result.pop("arbitrary_emotion")
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TtsJob:
        return cls(
            **{
                **value,
                "voice_profile": value.get("voice_profile", ""),
                "emotion": value.get("emotion"),
                "performance": SpeechPerformance.from_dict(
                    value.get("performance")
                ),
                "segments": tuple(SpeechSegment(**item) for item in value.get("segments", ())),
            }
        )


@dataclass(frozen=True, slots=True)
class RenderTask:
    dialogue_id: str
    identifier: str
    action: DialogueAction
    component_job_ids: tuple[str, ...]
    render_mode: str
    effects: tuple[str, ...]
    output_path: str
    virtual_path: str
    keyframe_program: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dialogue_id": self.dialogue_id,
            "identifier": self.identifier,
            "action": self.action.value,
            "component_job_ids": list(self.component_job_ids),
            "render_mode": self.render_mode,
            "effects": list(self.effects),
            "output_path": self.output_path,
            "virtual_path": self.virtual_path,
            "keyframe_program": self.keyframe_program,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RenderTask:
        return cls(
            dialogue_id=value["dialogue_id"],
            identifier=value["identifier"],
            action=DialogueAction(value["action"]),
            component_job_ids=tuple(value["component_job_ids"]),
            render_mode=value["render_mode"],
            effects=tuple(value["effects"]),
            output_path=value["output_path"],
            virtual_path=value["virtual_path"],
            keyframe_program=value.get("keyframe_program"),
        )


@dataclass(frozen=True, slots=True)
class SynthesisIssue:
    dialogue_id: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "dialogue_id": self.dialogue_id,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class SynthesisPlan:
    jobs: tuple[TtsJob, ...]
    render_tasks: tuple[RenderTask, ...]
    issues: tuple[SynthesisIssue, ...]


@dataclass(frozen=True, slots=True)
class AudioQualityResult:
    dialogue_id: str
    path: str
    valid: bool
    duration_seconds: float | None
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dialogue_id": self.dialogue_id,
            "path": self.path,
            "valid": self.valid,
            "duration_seconds": self.duration_seconds,
            "issues": list(self.issues),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AudioQualityResult:
        return cls(
            dialogue_id=value["dialogue_id"],
            path=value["path"],
            valid=value["valid"],
            duration_seconds=value.get("duration_seconds"),
            issues=tuple(value.get("issues", ())),
        )
