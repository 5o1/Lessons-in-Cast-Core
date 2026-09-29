"""Convert validated semantic annotations into cached speech jobs."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, replace
from pathlib import PurePosixPath
from typing import Any

from ..annotation import DialogueAction, ValidatedAnnotation, ValidationStatus
from ..characters import CharacterDefinition
from ..config import AudioConfig
from ..dialogue import DialogueRecord
from ..hashing import content_hash
from .types import RenderTask, SynthesisIssue, SynthesisPlan, TtsJob
from ..keyframes.program import compile_keyframes, extract_anchors
from ..speech_markup import parse_emotion_markup, emotion_markup


_UNSAFE_PATH = re.compile(r"[^A-Za-z0-9_.-]+")
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]+")


def _safe_component(value: str, fallback: str) -> str:
    cleaned = _UNSAFE_PATH.sub("_", value).strip("._")
    return cleaned or fallback


class SynthesisPlanner:
    """Create member-level TTS jobs and line-level render tasks."""

    def __init__(
        self,
        characters: dict[str, CharacterDefinition],
        *,
        audio_config: AudioConfig | None = None,
        synthesizer_configuration: dict[str, Any] | None = None,
        virtual_path_resolver: Callable[[str, str, str], str] | None = None,
        voice_route_available: Callable[[str, str | None], bool] | None = None,
        direction_context: dict[str, Any] | None = None,
    ) -> None:
        self._characters = characters
        self._audio_config = audio_config or AudioConfig()
        self._audio_format = self._audio_config.format.lstrip(".")
        self._intermediate_format = (
            self._audio_config.intermediate_format.lstrip(".")
        )
        self._synthesizer_configuration = synthesizer_configuration or {
            "adapter": "unconfigured"
        }
        self._virtual_path_resolver = virtual_path_resolver or (
            lambda _source, identifier, audio_format: (
                f"voice/{identifier}.{audio_format.lstrip('.')}"
            )
        )
        self._voice_route_available = voice_route_available
        self._direction_context = direction_context or {}

    def plan(
        self,
        records_by_id: dict[str, DialogueRecord],
        validated: list[ValidatedAnnotation],
    ) -> SynthesisPlan:
        jobs: list[TtsJob] = []
        renders: list[RenderTask] = []
        issues: list[SynthesisIssue] = []
        for item in self.iter_plan(
            (records_by_id[result.dialogue_id], result) for result in validated
        ):
            if isinstance(item, TtsJob):
                jobs.append(item)
            elif isinstance(item, RenderTask):
                renders.append(item)
            else:
                issues.append(item)
        return SynthesisPlan(tuple(jobs), tuple(renders), tuple(issues))

    def iter_plan(
        self,
        records_and_results: Iterable[tuple[DialogueRecord, ValidatedAnnotation]],
    ) -> Iterator[TtsJob | RenderTask | SynthesisIssue]:
        """Yield a plan incrementally while retaining only deduplication keys."""

        job_id_by_cache_key: dict[str, str] = {}
        identifiers: dict[str, str] = {}

        for record, result in records_and_results:
            if record.id != result.dialogue_id:
                raise ValueError(
                    f"Dialogue/result ID mismatch: {record.id!r} != "
                    f"{result.dialogue_id!r}"
                )
            if result.status is not ValidationStatus.ACCEPTED:
                continue
            annotation = result.annotation
            if annotation is None:
                continue
            if not _SAFE_IDENTIFIER.fullmatch(record.identifier):
                yield SynthesisIssue(
                    result.dialogue_id,
                    "unsafe_identifier",
                    f"Dialogue identifier {record.identifier!r} is not path-safe.",
                )
                continue
            previous_dialogue = identifiers.get(record.identifier)
            if previous_dialogue is not None and previous_dialogue != record.id:
                yield SynthesisIssue(
                    result.dialogue_id,
                    "duplicate_identifier",
                    f"Dialogue identifier {record.identifier!r} is not unique.",
                )
                continue
            identifiers[record.identifier] = record.id
            character_id = record.character or "narrator"
            character = self._characters.get(character_id)
            if character is None:
                yield SynthesisIssue(
                    result.dialogue_id,
                    "unknown_character",
                    f"Character {character_id!r} is not configured.",
                )
                continue
            character = character.resolve(
                record.filename,
                record.label,
                record.scene,
            )
            render_overrides = self._direction_context.get(record.id, {}).get("render", {})

            if annotation.action is DialogueAction.OMIT:
                continue

            try:
                virtual_path = self._virtual_path_resolver(
                    record.filename,
                    record.identifier,
                    self._audio_format,
                )
            except ValueError as exc:
                yield SynthesisIssue(
                    result.dialogue_id,
                    "unsafe_source_path",
                    str(exc),
                )
                continue

            component_ids: list[str] = []
            if annotation.action in {
                DialogueAction.SPEAK,
                DialogueAction.SPEAK_WITH_EFFECT,
            }:
                for member_id in character.synthesis_members:
                    member = self._characters[member_id].resolve(
                        record.filename,
                        record.label,
                        record.scene,
                    )
                    override = render_overrides.get(member_id, {})
                    if "default_voice_profile" in override:
                        member = replace(member, default_voice_profile=override["default_voice_profile"])
                    if (
                        self._voice_route_available is not None
                        and not self._voice_route_available(
                            member.id,
                            member.default_voice_profile or None,
                        )
                    ):
                        yield SynthesisIssue(
                            result.dialogue_id,
                            "missing_voice_profile",
                            f"Character {member.id!r} has no available voice "
                            "profile for this source context; speech was skipped.",
                        )
                        continue
                    from ..performance import SpeechPerformance
                    performance = {**render_overrides.get(character_id, {}).get("performance", {}),
                                   **override.get("performance", {})}
                    directed = replace(annotation, performance=SpeechPerformance.from_dict(
                        {**annotation.performance.to_dict(), **performance})) if performance else annotation
                    job = self._create_job(record, directed, member)
                    existing_id = job_id_by_cache_key.get(job.cache_key)
                    if existing_id is None:
                        job_id_by_cache_key[job.cache_key] = job.id
                        yield job
                        component_ids.append(job.id)
                    else:
                        component_ids.append(existing_id)

            if (
                annotation.action
                in {DialogueAction.SPEAK, DialogueAction.SPEAK_WITH_EFFECT}
                and not component_ids
            ):
                continue
            render_mode = character.render_mode or "single"
            output_name = f"{record.identifier}.{self._audio_format}"
            yield RenderTask(
                dialogue_id=record.id,
                identifier=record.identifier,
                action=annotation.action,
                component_job_ids=tuple(component_ids),
                render_mode=render_mode,
                effects=annotation.effects,
                output_path=str(PurePosixPath("voice") / output_name),
                virtual_path=virtual_path,
                keyframe_program=(compile_keyframes(
                    "".join(s.text for s in parse_emotion_markup(annotation.spoken_text)),
                    annotation.keyframe_effects,
                ) if annotation.keyframe_effects else None),
            )

    def _create_job(
        self,
        record: DialogueRecord,
        annotation: object,
        member: CharacterDefinition,
    ) -> TtsJob:
        from ..annotation import Annotation

        assert isinstance(annotation, Annotation)
        from ..speech_markup import parse_emotion_markup

        segments = (parse_emotion_markup(annotation.spoken_text)
                    if "<emotion" in annotation.spoken_text or "<arbitrary_emotion" in annotation.spoken_text else ())
        text = "".join(segment.text for segment in segments) if segments else annotation.spoken_text
        cache_text = annotation.spoken_text
        if annotation.keyframe_effects is not None:
            text = compile_keyframes(text, annotation.keyframe_effects)["text"]
            segments = tuple(replace(s, text=extract_anchors(s.text)[0]) for s in segments)
            cache_text = emotion_markup(segments) if segments else text
        identity = {
            "character": member.id,
            "text": cache_text,
            "emotion": annotation.emotion,
            "delivery": annotation.delivery,
            "performance": annotation.performance.to_dict(),
            "audio": asdict(self._audio_config),
            "synthesizer": self._synthesizer_configuration,
            "voice_profile": member.default_voice_profile,
        }
        cache_key = content_hash(identity)
        member_path = _safe_component(member.id, "character")
        output_path = str(
            PurePosixPath("audio/raw")
            / member_path
            / f"{cache_key}.{self._intermediate_format}"
        )
        return TtsJob(
            id=content_hash({"dialogue_id": record.id, **identity})[:24],
            dialogue_id=record.id,
            character_id=member.id,
            text=text,
            emotion=None if segments else annotation.emotion,
            delivery=annotation.delivery,
            output_path=output_path,
            cache_key=cache_key,
            voice_profile=member.default_voice_profile,
            performance=annotation.performance,
            segments=segments,
        )
