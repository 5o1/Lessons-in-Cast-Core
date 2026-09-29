"""Independent validation gate for untrusted semantic annotations."""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

from ..config import AnnotationConfig
from ..dialogue import DialogueBatch, DialogueRecord
from ..hashing import content_hash
from ..performance import PerformanceCueKind, SpeechPerformance, VocalMode
from ..keyframes.program import compile_keyframes, extract_anchors
from ..speech_markup import parse_emotion_markup
from .types import (
    Annotation,
    BatchValidationResult,
    DialogueAction,
    ValidatedAnnotation,
    ValidationIssue,
    ValidationStatus,
)


_PLACEHOLDER = re.compile(r"\[[^\[\]]+\]")
_TEXT_TAG = re.compile(r"\{[^{}]+\}")
_LEXICAL_SPEECH = re.compile(r"[^\W_]", re.UNICODE)
_ALLOWED_FIELDS = {
    "id",
    "action",
    "spoken_text",
    "emotion",
    "delivery",
    "effects",
    "confidence",
    "review_required",
    "reason",
    "performance",
    "keyframe_effects",
}


class AnnotationValidator:
    """Validate structure, batch identity, cross-fields, and text risks."""

    def __init__(self, config: AnnotationConfig) -> None:
        self._config = config

    def validate_batch(
        self,
        batch: DialogueBatch,
        response: dict[str, Any],
        *,
        prompt_version: str,
        annotator_configuration: dict[str, Any],
        processed_at: str | None = None,
        schema_version: int = 2,
        stage: str = "polish",
        cleaned_annotations: dict | None = None,
        keyframe_required: tuple[str, ...] | list[str] = (),
    ) -> BatchValidationResult:
        timestamp = processed_at or datetime.now(timezone.utc).isoformat()
        input_hash = content_hash(batch.to_dict())
        config_hash = content_hash(annotator_configuration)
        target_by_id = {item.id: item for item in batch.targets}
        context_ids = {
            item.id
            for item in (
                *batch.context_before,
                *batch.context_interleaved,
                *batch.context_after,
            )
        }
        batch_issues: list[ValidationIssue] = []

        if not isinstance(response, dict):
            return self._all_retryable(
                batch,
                "response_type",
                "Annotation response must be a JSON object.",
                input_hash,
                prompt_version,
                config_hash,
                timestamp,
            )
        unknown_response_fields = set(response) - {"batch_id", "annotations"}
        if unknown_response_fields:
            return self._all_retryable(
                batch,
                "unknown_response_fields",
                f"Unknown response fields: {sorted(unknown_response_fields)!r}",
                input_hash,
                prompt_version,
                config_hash,
                timestamp,
            )
        if response.get("batch_id") != batch.id:
            return self._all_retryable(
                batch,
                "batch_id_mismatch",
                "Annotation response batch_id does not match the request.",
                input_hash,
                prompt_version,
                config_hash,
                timestamp,
            )
        raw_annotations = response.get("annotations")
        if not isinstance(raw_annotations, list):
            return self._all_retryable(
                batch,
                "annotations_type",
                "Annotation response annotations must be an array.",
                input_hash,
                prompt_version,
                config_hash,
                timestamp,
            )

        raw_ids = [
            item.get("id")
            for item in raw_annotations
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
        id_counts = Counter(raw_ids)
        for raw_id in raw_ids:
            if raw_id in context_ids:
                batch_issues.append(
                    ValidationIssue(
                        "context_output",
                        "The model returned an annotation for a context-only record.",
                        "error",
                        raw_id,
                    )
                )
            elif raw_id not in target_by_id:
                batch_issues.append(
                    ValidationIssue(
                        "unknown_id",
                        "The model returned an unknown dialogue ID.",
                        "error",
                        raw_id,
                    )
                )

        raw_by_id: dict[str, dict[str, Any]] = {}
        for item in raw_annotations:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                if item["id"] in target_by_id and item["id"] not in raw_by_id:
                    raw_by_id[item["id"]] = item

        validated: list[ValidatedAnnotation] = []
        for target in batch.targets:
            issues: list[ValidationIssue] = []
            if id_counts[target.id] == 0:
                issues.append(
                    ValidationIssue(
                        "missing_target",
                        "The model did not return this target.",
                        "error",
                        target.id,
                    )
                )
                annotation = None
            elif id_counts[target.id] > 1:
                issues.append(
                    ValidationIssue(
                        "duplicate_target",
                        "The model returned this target more than once.",
                        "error",
                        target.id,
                    )
                )
                annotation = None
            else:
                annotation, annotation_issues = self._validate_annotation(
                    target,
                    raw_by_id[target.id],
                    schema_version=schema_version,
                    stage=stage,
                    keyframe_required=target.id in keyframe_required,
                )
                issues.extend(issue for issue in annotation_issues
                              if cleaned_annotations is None or issue.code != "high_impact_action")
                if annotation is not None and cleaned_annotations is not None:
                    original = cleaned_annotations.get(target.id)
                    if original is None:
                        issues.append(ValidationIssue("missing_cleaning_binding", "Polish has no accepted cleaning input.", "error", target.id))
                    else:
                        plain = "".join(s.text for s in parse_emotion_markup(annotation.spoken_text)) if annotation.spoken_text else ""
                        if annotation.keyframe_effects is not None:
                            plain = compile_keyframes(plain, annotation.keyframe_effects)["text"]
                        if plain != original["spoken_text"] or annotation.action.value != original["action"] or list(annotation.effects) != original["effects"]:
                            issues.append(ValidationIssue("polish_changed_cleaning", "Polish must preserve cleaned speech, action and effects exactly.", "error", target.id))
                        pauses = [cue.to_dict() for cue in annotation.performance.cues if cue.kind is PerformanceCueKind.PAUSE]
                        if pauses != original.get("performance", {}).get("cues", []):
                            issues.append(ValidationIssue("polish_changed_pauses", "Polish must preserve cleaning pause cues exactly.", "error", target.id))
                        if target.id in keyframe_required and annotation.keyframe_effects is None:
                            issues.append(ValidationIssue("missing_keyframe_effect", "This partially heard line requires a gain envelope.", "error", target.id))

            if annotation is None or any(issue.severity == "error" for issue in issues):
                status = ValidationStatus.RETRYABLE
            elif annotation.review_required or issues:
                status = ValidationStatus.REVIEW_REQUIRED
            else:
                status = ValidationStatus.ACCEPTED
            validated.append(
                ValidatedAnnotation(
                    dialogue_id=target.id,
                    batch_id=batch.id,
                    status=status,
                    annotation=annotation,
                    issues=tuple(issues),
                    input_hash=input_hash,
                    prompt_version=prompt_version,
                    annotator_config_hash=config_hash,
                    processed_at=timestamp,
                )
            )

        return BatchValidationResult(batch.id, tuple(validated), tuple(batch_issues))

    def _validate_annotation(
        self,
        target: DialogueRecord,
        raw: dict[str, Any],
        *, schema_version: int = 2, stage: str = "polish", keyframe_required: bool = False,
    ) -> tuple[Annotation | None, list[ValidationIssue]]:
        issues: list[ValidationIssue] = []

        def error(code: str, message: str) -> None:
            issues.append(ValidationIssue(code, message, "error", target.id))

        unknown = set(raw) - _ALLOWED_FIELDS
        if unknown:
            error("unknown_fields", f"Unknown fields: {sorted(unknown)!r}")
        required = {
            "id",
            "action",
            "spoken_text",
            "effects",
            "confidence",
            "review_required",
            "reason",
        }
        if stage != "cleaning":
            required.add("delivery")
        missing = required - set(raw)
        if missing:
            error("missing_fields", f"Missing fields: {sorted(missing)!r}")
            return None, issues
        if raw["id"] != target.id:
            error("id_mismatch", "Annotation ID does not match the target.")
        try:
            action = DialogueAction(raw["action"])
        except (TypeError, ValueError):
            error("invalid_action", "Action is not an allowed value.")
            return None, issues

        spoken_text = raw["spoken_text"]
        emotion = raw.get("emotion")
        delivery = raw.get("delivery", {})
        effects = raw["effects"]
        confidence = raw["confidence"]
        review_required = raw["review_required"]
        reason = raw["reason"]
        performance_raw = raw.get("performance")

        speaking = action in {DialogueAction.SPEAK, DialogueAction.SPEAK_WITH_EFFECT}
        markup = stage != "cleaning" and isinstance(spoken_text, str) and (schema_version >= 3 or "<emotion" in spoken_text or "<arbitrary_emotion" in spoken_text)
        if schema_version >= 3 and ({"emotion", "intensity"} & set(raw)):
            error("line_level_emotion", "Use a single semantic label inside each emotion tag, without numerical intensity.")
        segments = ()
        if markup and speaking:
            try:
                segments = parse_emotion_markup(spoken_text, self._config.allowed_emotions)
                spoken_text = "".join(segment.text for segment in segments)
            except ValueError as exc:
                error("emotion_markup", str(exc))

        if not isinstance(spoken_text, str):
            error("spoken_text_type", "spoken_text must be a string.")
        keyframe_effects = raw.get("keyframe_effects")
        if "keyframe_effects" in raw:
            if stage == "cleaning" or not speaking:
                error("keyframe_stage", "Keyframe effects are supported only on speaking polish annotations")
            elif not isinstance(keyframe_effects, list):
                error("keyframe_effects", "keyframe_effects must be an array")
            elif isinstance(spoken_text, str):
                try:
                    program = compile_keyframes(spoken_text, keyframe_effects)
                    spoken_text = program["text"]
                    if any(not extract_anchors(segment.text)[0].strip() for segment in segments):
                        error("keyframe_effects", "Markers must accompany speech, not occupy an empty emotion span")
                    if keyframe_required and any(
                            offset not in (0, len(spoken_text)) for offset in program["anchors"].values()):
                        error("keyframe_alignment", "Automatic fade-ins require start/end anchors, including trailing punctuation.")
                except ValueError as exc:
                    error("keyframe_effects", str(exc))
        if emotion is not None and not isinstance(emotion, str):
            error("emotion_type", "emotion must be a string or null.")
        elif emotion is not None and emotion not in self._config.allowed_emotions:
            error("invalid_emotion", f"Emotion {emotion!r} is not allowed.")
        if not isinstance(delivery, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in delivery.items()
        ):
            error("delivery_type", "delivery must be an object with string values.")
        if not isinstance(effects, list) or not all(
            isinstance(effect, str) for effect in effects
        ):
            error("effects_type", "effects must be an array of strings.")
        elif len(effects) != len(set(effects)):
            error("duplicate_effect", "effects cannot contain duplicates.")
        elif any(effect not in self._config.allowed_effects for effect in effects):
            error("invalid_effect", "effects contains an unsupported value.")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            error("invalid_confidence", "confidence must be null or between 0 and 1.")
        if not isinstance(review_required, bool):
            error("review_required_type", "review_required must be a boolean.")
        if reason is not None and not isinstance(reason, str):
            error("reason_type", "reason must be a string or null.")
        performance = self._validate_performance(
            performance_raw,
            spoken_text if isinstance(spoken_text, str) else "",
            error,
        )
        if stage == "cleaning":
            if any(key in raw for key in ("emotion", "intensity", "delivery")) or (isinstance(spoken_text, str) and re.search(r"<\s*/?\s*(emotion|voice)\b", spoken_text)):
                error("cleaning_acting_fields", "Cleaning cannot assign emotion, voice or delivery; use polish.")
            if any(value is not None for key, value in performance.to_dict().items() if key != "cues") or any(cue.kind is not PerformanceCueKind.PAUSE or cue.intensity is not None for cue in performance.cues):
                error("cleaning_performance", "Cleaning may add only pause cues, not acting controls.")
        if any(issue.severity == "error" for issue in issues):
            return None, issues

        assert isinstance(spoken_text, str)
        assert isinstance(delivery, dict)
        assert isinstance(effects, list)
        assert isinstance(review_required, bool)

        speaking = action in {DialogueAction.SPEAK, DialogueAction.SPEAK_WITH_EFFECT}
        if speaking and not spoken_text.strip():
            error("missing_spoken_text", "A speaking action requires spoken_text.")
        if not speaking and spoken_text:
            error("unexpected_spoken_text", "A non-speaking action requires empty spoken_text.")
        if speaking and stage != "cleaning" and not segments and emotion is None:
            error("missing_emotion", "A speaking action requires an emotion label.")
        if action is DialogueAction.SPEAK and effects:
            error("unexpected_effects", "Use speak_with_effect when effects are present.")
        if action is DialogueAction.SPEAK_WITH_EFFECT and not effects:
            error("missing_effects", "speak_with_effect requires at least one effect.")
        if action is DialogueAction.SFX_ONLY and not effects:
            error("missing_effects", "sfx_only requires at least one effect.")
        if action is DialogueAction.OMIT and effects:
            error("unexpected_effects", "omit cannot contain effects.")
        if not speaking and performance != SpeechPerformance():
            error(
                "unexpected_performance",
                "A non-speaking action cannot contain speech performance intent.",
            )
        if any(issue.severity == "error" for issue in issues):
            return None, issues

        self._add_text_risks(target, spoken_text, speaking, issues)
        if action in {DialogueAction.OMIT, DialogueAction.SFX_ONLY}:
            issues.append(
                ValidationIssue(
                    "high_impact_action",
                    "Omitting speech or replacing it with effects requires human review.",
                    "warning",
                    target.id,
                )
            )
        return (
            Annotation(
                id=target.id,
                action=action,
                spoken_text=raw["spoken_text"],
                emotion=emotion,
                delivery=delivery,
                effects=tuple(effects),
                confidence=float(confidence) if confidence is not None else None,
                review_required=review_required,
                reason=reason,
                performance=performance,
                keyframe_effects=keyframe_effects,
            ),
            issues,
        )

    @staticmethod
    def _validate_performance(
        raw: Any,
        spoken_text: str,
        error: Any,
    ) -> SpeechPerformance:
        """Validate schema-v2 performance while accepting omitted v1 data."""

        if raw is None:
            return SpeechPerformance()
        if not isinstance(raw, dict):
            error("performance_type", "performance must be an object.")
            return SpeechPerformance()
        allowed = {
            "direction",
            "vocal_mode",
            "speed",
            "pitch_semitones",
            "volume_gain_db",
            "energy",
            "brightness",
            "clarity",
            "breathiness",
            "cues",
        }
        unknown = set(raw) - allowed
        if unknown:
            error(
                "unknown_performance_fields",
                f"Unknown performance fields: {sorted(unknown)!r}",
            )
        direction = raw.get("direction")
        if direction is not None and not isinstance(direction, str):
            error("performance_direction_type", "direction must be a string or null.")
        mode = raw.get("vocal_mode")
        try:
            vocal_mode = VocalMode(mode) if mode is not None else None
        except (TypeError, ValueError):
            error("invalid_vocal_mode", "vocal_mode is not allowed.")
            vocal_mode = None

        ranges = {
            "speed": (0.0, 4.0, True),
            "pitch_semitones": (-48.0, 48.0, False),
            "volume_gain_db": (-60.0, 24.0, False),
            "energy": (-1.0, 1.0, False),
            "brightness": (-1.0, 1.0, False),
            "clarity": (-1.0, 1.0, False),
            "breathiness": (0.0, 1.0, False),
        }
        values: dict[str, float | None] = {}
        for field, (minimum, maximum, exclusive_minimum) in ranges.items():
            value = raw.get(field)
            if value is None:
                values[field] = None
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                error("performance_control_type", f"{field} must be a number or null.")
                values[field] = None
                continue
            number = float(value)
            valid_minimum = number > minimum if exclusive_minimum else number >= minimum
            if not valid_minimum or number > maximum:
                error("performance_control_range", f"{field} is outside its allowed range.")
                values[field] = None
            else:
                values[field] = number

        cues_raw = raw.get("cues", [])
        cues = []
        if not isinstance(cues_raw, list):
            error("performance_cues_type", "performance.cues must be an array.")
        else:
            previous_offset = -1
            for index, cue in enumerate(cues_raw):
                if not isinstance(cue, dict):
                    error("performance_cue_type", f"Cue {index} must be an object.")
                    continue
                if set(cue) - {"kind", "offset", "duration_seconds", "intensity"}:
                    error("unknown_performance_cue_fields", f"Cue {index} has unknown fields.")
                    continue
                try:
                    kind = PerformanceCueKind(cue.get("kind"))
                except (TypeError, ValueError):
                    error("invalid_performance_cue", f"Cue {index} kind is not allowed.")
                    continue
                offset = cue.get("offset")
                if isinstance(offset, bool) or not isinstance(offset, int):
                    error("performance_cue_offset", f"Cue {index} offset must be an integer.")
                    continue
                if offset < previous_offset or not 0 <= offset <= len(spoken_text):
                    error(
                        "performance_cue_offset",
                        f"Cue {index} offset must be ordered and within spoken_text.",
                    )
                    continue
                if (
                    0 < offset < len(spoken_text)
                    and spoken_text[offset - 1].isalpha()
                    and spoken_text[offset].isalpha()
                ):
                    error(
                        "performance_cue_word_split",
                        f"Cue {index} cannot split a word.",
                    )
                    continue
                previous_offset = offset
                duration = cue.get("duration_seconds")
                if duration is not None and (
                    isinstance(duration, bool)
                    or not isinstance(duration, (int, float))
                    or not 0 < duration <= 120
                ):
                    error("performance_cue_duration", f"Cue {index} duration is invalid.")
                    continue
                if kind is PerformanceCueKind.PAUSE and duration is None:
                    error("performance_pause_duration", f"Cue {index} pause needs a duration.")
                    continue
                intensity_value = cue.get("intensity")
                if intensity_value is not None and (
                    isinstance(intensity_value, bool)
                    or not isinstance(intensity_value, (int, float))
                    or not 0 <= intensity_value <= 1
                ):
                    error("performance_cue_intensity", f"Cue {index} intensity is invalid.")
                    continue
                cues.append(
                    {
                        "kind": kind.value,
                        "offset": offset,
                        "duration_seconds": (
                            float(duration) if duration is not None else None
                        ),
                        "intensity": (
                            float(intensity_value)
                            if intensity_value is not None
                            else None
                        ),
                    }
                )
        try:
            return SpeechPerformance.from_dict(
                {
                    "direction": direction if isinstance(direction, str) else None,
                    "vocal_mode": vocal_mode.value if vocal_mode else None,
                    **values,
                    "cues": cues,
                }
            )
        except (KeyError, TypeError, ValueError):
            error("invalid_performance", "performance could not be decoded.")
            return SpeechPerformance()

    def _add_text_risks(
        self,
        target: DialogueRecord,
        spoken_text: str,
        speaking: bool,
        issues: list[ValidationIssue],
    ) -> None:
        if not speaking:
            return
        if any(ord(character) < 32 and character not in "\t\n\r" for character in spoken_text):
            issues.append(
                ValidationIssue(
                    "control_character",
                    "spoken_text contains a control character.",
                    "warning",
                    target.id,
                )
            )
        if _TEXT_TAG.search(spoken_text):
            issues.append(
                ValidationIssue(
                    "text_tag",
                    "spoken_text may still contain a Ren'Py text tag.",
                    "warning",
                    target.id,
                )
            )
        if _LEXICAL_SPEECH.search(spoken_text) is None:
            issues.append(
                ValidationIssue(
                    "non_lexical_spoken_text",
                    "spoken_text contains no letters or numbers and may be a silent visual beat.",
                    "warning",
                    target.id,
                )
            )
        original_placeholders = Counter(_PLACEHOLDER.findall(target.dialogue))
        spoken_placeholders = Counter(_PLACEHOLDER.findall(spoken_text))
        if original_placeholders != spoken_placeholders:
            issues.append(
                ValidationIssue(
                    "placeholder_change",
                    "Square-bracket placeholders changed during cleaning.",
                    "warning",
                    target.id,
                )
            )
        original_lexical = "".join(
            character.casefold()
            for character in _TEXT_TAG.sub("", target.dialogue)
            if character.isalnum()
        )
        spoken_lexical = "".join(
            character.casefold() for character in spoken_text if character.isalnum()
        )
        if (
            len(original_lexical) >= 2
            and spoken_lexical
            and SequenceMatcher(None, original_lexical, spoken_lexical).ratio() < 0.2
        ):
            issues.append(
                ValidationIssue(
                    "lexical_mismatch",
                    "spoken_text has almost no lexical overlap with its source line.",
                    "error",
                    target.id,
                )
            )
        original_length = max(len(target.dialogue.strip()), 1)
        ratio = len(spoken_text.strip()) / original_length
        if (
            ratio < self._config.minimum_length_ratio
            or ratio > self._config.maximum_length_ratio
        ):
            issues.append(
                ValidationIssue(
                    "length_ratio",
                    f"Cleaned text length ratio {ratio:.2f} is outside the configured range.",
                    "warning",
                    target.id,
                )
            )

    def _all_retryable(
        self,
        batch: DialogueBatch,
        code: str,
        message: str,
        input_hash: str,
        prompt_version: str,
        config_hash: str,
        timestamp: str,
    ) -> BatchValidationResult:
        records = tuple(
            ValidatedAnnotation(
                dialogue_id=target.id,
                batch_id=batch.id,
                status=ValidationStatus.RETRYABLE,
                annotation=None,
                issues=(ValidationIssue(code, message, "error", target.id),),
                input_hash=input_hash,
                prompt_version=prompt_version,
                annotator_config_hash=config_hash,
                processed_at=timestamp,
            )
            for target in batch.targets
        )
        return BatchValidationResult(batch.id, records)
