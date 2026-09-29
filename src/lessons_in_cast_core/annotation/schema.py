"""JSON Schema construction for strict annotation responses."""

from __future__ import annotations

from typing import Any

from ..config import AnnotationConfig
from ..performance import PerformanceCueKind, VocalMode


def annotation_response_schema(
    config: AnnotationConfig | None = None,
    *, stage: str = "polish",
) -> dict[str, Any]:
    effect_schema: dict[str, Any] = {"type": "string"}
    if config is not None:
        effect_schema = {
            "type": "string",
            "enum": sorted(config.allowed_effects),
        }
    annotation = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "id",
            "action",
            "spoken_text",
            "delivery",
            "effects",
            "confidence",
            "review_required",
            "reason",
            "performance",
        ],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "action": {
                "type": "string",
                "enum": ["speak", "omit", "sfx_only", "speak_with_effect"],
            },
            "spoken_text": {
                "type": "string",
                "description": 'For speech, wrap every span in <emotion name="LABEL">text</emotion>. Each name is one preset semantic label; no combinations or numerical intensity. '
                               'Alternatively use <arbitrary_emotion description="AUDIBLE ACTING">text</arbitrary_emotion> for runtime-described emotion; never nest or overlap preset and arbitrary emotions. '
                               'Use adjacent spans for emotion changes. Optional <voice name="FILE_STEM"> wraps emotion spans to select a profile reference. '
                               'No nested voices or emotions. Escape literal &, < and >. '
                               'Use an empty string for non-speaking actions. Labels must come from allowed_emotions.',
            },
            "delivery": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "effects": {
                "type": "array",
                "items": effect_schema,
                "uniqueItems": True,
            },
            "confidence": {
                "type": ["number", "null"],
                "minimum": 0,
                "maximum": 1,
            },
            "review_required": {"type": "boolean"},
            "reason": {"type": ["string", "null"]},
            "performance": {
                "type": "object",
                "additionalProperties": False,
                "required": [
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
                ],
                "properties": {
                    "direction": {"type": ["string", "null"]},
                    "vocal_mode": {
                        "enum": [*[mode.value for mode in VocalMode], None],
                    },
                    "speed": {
                        "type": ["number", "null"],
                        "exclusiveMinimum": 0,
                        "maximum": 4,
                    },
                    "pitch_semitones": {
                        "type": ["number", "null"],
                        "minimum": -48,
                        "maximum": 48,
                    },
                    "volume_gain_db": {
                        "type": ["number", "null"],
                        "minimum": -60,
                        "maximum": 24,
                    },
                    "energy": {
                        "type": ["number", "null"],
                        "minimum": -1,
                        "maximum": 1,
                    },
                    "brightness": {
                        "type": ["number", "null"],
                        "minimum": -1,
                        "maximum": 1,
                    },
                    "clarity": {
                        "type": ["number", "null"],
                        "minimum": -1,
                        "maximum": 1,
                    },
                    "breathiness": {
                        "type": ["number", "null"],
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "cues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "kind",
                                "offset",
                                "duration_seconds",
                                "intensity",
                            ],
                            "properties": {
                                "kind": {
                                    "type": "string",
                                    "enum": [
                                        kind.value for kind in PerformanceCueKind
                                    ],
                                },
                                "offset": {
                                    "type": "integer",
                                    "minimum": 0,
                                },
                                "duration_seconds": {
                                    "type": ["number", "null"],
                                    "exclusiveMinimum": 0,
                                    "maximum": 120,
                                },
                                "intensity": {
                                    "type": ["number", "null"],
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                            },
                        },
                    },
                },
            },
        },
    }
    if stage == "polish":
        from ..keyframes.program import keyframe_effects_schema
        annotation["properties"]["keyframe_effects"] = keyframe_effects_schema()
    if stage == "cleaning":
        annotation["required"].remove("delivery")
        del annotation["properties"]["delivery"]
        annotation["properties"]["spoken_text"] = {"type": "string", "description": "Plain cleaned speech, without emotion or voice markup."}
        performance = annotation["properties"]["performance"]
        cues = performance["properties"]["cues"]
        cues["items"]["properties"]["kind"] = {"const": "pause"}
        cues["items"]["properties"]["intensity"] = {"type": "null"}
        performance["required"] = ["cues"]
        performance["properties"] = {"cues": cues}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Lessons in Cast annotation response",
        "type": "object",
        "additionalProperties": False,
        "required": ["batch_id", "annotations"],
        "properties": {
            "batch_id": {"type": "string", "minLength": 1},
            "annotations": {
                "type": "array",
                "items": annotation,
            },
        },
    }
