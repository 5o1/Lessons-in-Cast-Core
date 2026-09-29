from __future__ import annotations

import copy
import unittest

from lessons_in_cast_core.annotation import (
    AnnotationValidator,
    DialogueAction,
    ValidationStatus,
    apply_overrides,
    build_annotation_request,
)
from lessons_in_cast_core.config import AnnotationConfig
from lessons_in_cast_core.dialogue import DialogueBatch

from .fakes import MockDialogueAnnotator
from .helpers import record


class AnnotationValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AnnotationConfig(
            allowed_emotions=frozenset({"neutral", "surprised"}),
            allowed_effects=frozenset({"glitch"}),
            maximum_length_ratio=4.0,
            minimum_length_ratio=0.15,
        )
        self.validator = AnnotationValidator(self.config)
        self.context = record(0)
        self.target = record(1, dialogue="Wait?!")
        self.batch = DialogueBatch("batch", (self.context,), (self.target,), ())

    def _response(self) -> dict[str, object]:
        request = build_annotation_request(self.batch)
        return MockDialogueAnnotator().annotate(request)

    def test_accepts_valid_response(self) -> None:
        result = self.validator.validate_batch(
            self.batch,
            self._response(),
            prompt_version="1",
            annotator_configuration={"adapter": "mock"},
        )
        self.assertEqual(result.records[0].status, ValidationStatus.ACCEPTED)

    def test_request_contains_a_strict_response_schema(self) -> None:
        request = build_annotation_request(self.batch, annotation_config=self.config)
        schema = request["response_schema"]
        self.assertFalse(schema["additionalProperties"])
        annotation = schema["properties"]["annotations"]["items"]
        self.assertFalse(annotation["additionalProperties"])
        self.assertNotIn("emotion", annotation["properties"])
        self.assertIn("neutral", request["allowed_emotions"])
        self.assertIn("<emotion", annotation["properties"]["spoken_text"]["description"])
        self.assertIn("performance", annotation["required"])
        self.assertIn(
            "pause",
            annotation["properties"]["performance"]["properties"]["cues"]
            ["items"]["properties"]["kind"]["enum"],
        )

    def test_accepts_precise_portable_performance_cues(self) -> None:
        response = self._response()
        response["annotations"][0]["performance"] = {
            "direction": "A startled realization, then guarded calm.",
            "vocal_mode": "normal",
            "speed": 0.9,
            "pitch_semitones": 1.5,
            "volume_gain_db": -2.0,
            "energy": 0.4,
            "brightness": 0.2,
            "clarity": 0.1,
            "breathiness": 0.15,
            "cues": [
                {
                    "kind": "pause",
                    "offset": 4,
                    "duration_seconds": 0.35,
                    "intensity": None,
                }
            ],
        }
        result = self.validator.validate_batch(
            self.batch,
            response,
            prompt_version="2",
            annotator_configuration={},
        )
        annotation = result.records[0].annotation
        self.assertEqual(result.records[0].status, ValidationStatus.ACCEPTED)
        self.assertIsNotNone(annotation)
        assert annotation is not None
        self.assertEqual(annotation.performance.cues[0].offset, 4)

    def test_rejects_performance_cue_outside_cleaned_text(self) -> None:
        response = self._response()
        response["annotations"][0]["performance"] = {
            "cues": [
                {
                    "kind": "pause",
                    "offset": 99,
                    "duration_seconds": 0.2,
                    "intensity": None,
                }
            ]
        }
        result = self.validator.validate_batch(
            self.batch,
            response,
            prompt_version="2",
            annotator_configuration={},
        )
        self.assertEqual(result.records[0].status, ValidationStatus.RETRYABLE)
        self.assertIn(
            "performance_cue_offset",
            {issue.code for issue in result.records[0].issues},
        )

    def test_rejects_pause_that_splits_a_word(self) -> None:
        response = self._response()
        response["annotations"][0]["performance"] = {
            "cues": [{"kind": "pause", "offset": 2, "duration_seconds": 0.2, "intensity": None}]
        }
        result = self.validator.validate_batch(
            self.batch, response, prompt_version="2", annotator_configuration={}
        )
        self.assertEqual(result.records[0].status, ValidationStatus.RETRYABLE)
        self.assertIn("performance_cue_word_split", {issue.code for issue in result.records[0].issues})

    def test_missing_target_is_retryable(self) -> None:
        result = self.validator.validate_batch(
            self.batch,
            {"batch_id": "batch", "annotations": []},
            prompt_version="1",
            annotator_configuration={},
        )
        self.assertEqual(result.records[0].status, ValidationStatus.RETRYABLE)
        self.assertEqual(result.records[0].issues[0].code, "missing_target")

    def test_context_output_is_reported_without_corrupting_target(self) -> None:
        response = self._response()
        response["annotations"].append(
            {
                **response["annotations"][0],
                "id": self.context.id,
            }
        )
        result = self.validator.validate_batch(
            self.batch,
            response,
            prompt_version="1",
            annotator_configuration={},
        )
        self.assertEqual(result.records[0].status, ValidationStatus.ACCEPTED)
        self.assertEqual(result.issues[0].code, "context_output")

    def test_cross_field_violation_is_retryable(self) -> None:
        response = self._response()
        response["annotations"][0]["effects"] = ["glitch"]
        result = self.validator.validate_batch(
            self.batch,
            response,
            prompt_version="1",
            annotator_configuration={},
        )
        self.assertEqual(result.records[0].status, ValidationStatus.RETRYABLE)
        self.assertIsNone(result.records[0].annotation)

    def test_text_risk_requires_review(self) -> None:
        response = self._response()
        response["annotations"][0]["spoken_text"] = "{i}Wait?!{/i}"
        result = self.validator.validate_batch(
            self.batch,
            response,
            prompt_version="1",
            annotator_configuration={},
        )
        self.assertEqual(result.records[0].status, ValidationStatus.REVIEW_REQUIRED)

    def test_gross_lexical_mismatch_is_retryable(self) -> None:
        target = record(1, dialogue="...sei?")
        batch = DialogueBatch("batch", (), (target,), ())
        response = self._response()
        response["annotations"][0].update(id=target.id, spoken_text="What?")
        result = self.validator.validate_batch(
            batch, response, prompt_version="1", annotator_configuration={}
        )
        self.assertEqual(result.records[0].status, ValidationStatus.RETRYABLE)
        self.assertIn("lexical_mismatch", {issue.code for issue in result.records[0].issues})

    def test_punctuation_only_speech_requires_review(self) -> None:
        response = self._response()
        response["annotations"][0]["spoken_text"] = "........."
        result = self.validator.validate_batch(
            self.batch,
            response,
            prompt_version="1",
            annotator_configuration={},
        )

        self.assertEqual(result.records[0].status, ValidationStatus.REVIEW_REQUIRED)
        self.assertIn(
            "non_lexical_spoken_text",
            {issue.code for issue in result.records[0].issues},
        )

    def test_approved_override_accepts_risk_and_does_not_mutate_input(self) -> None:
        response = self._response()
        validated = list(
            self.validator.validate_batch(
                self.batch,
                response,
                prompt_version="1",
                annotator_configuration={},
            ).records
        )
        overrides = {
            self.target.id: {
                "spoken_text": "Wait?! Wait?! Wait?! Wait?! Wait?!",
                "approved": True,
            }
        }
        original = copy.deepcopy(overrides)
        applied = apply_overrides(
            validated,
            {self.target.id: self.target},
            overrides,
            self.validator,
        )
        self.assertEqual(applied[0].status, ValidationStatus.ACCEPTED)
        self.assertEqual(applied[0].source, "manual")
        self.assertEqual(overrides, original)

    def test_text_override_drops_stale_pause_offsets(self) -> None:
        response = self._response()
        response["annotations"][0]["performance"] = {
            "cues": [{"kind": "pause", "offset": 4, "duration_seconds": 0.2, "intensity": None}]
        }
        validated = list(self.validator.validate_batch(
            self.batch, response, prompt_version="1", annotator_configuration={}
        ).records)
        applied = apply_overrides(
            validated,
            {self.target.id: self.target},
            {self.target.id: {"spoken_text": "Teacher?", "approved": True}},
            self.validator,
        )
        self.assertEqual(applied[0].annotation.performance.cues, ())

    def test_manual_rejection(self) -> None:
        validated = list(
            self.validator.validate_batch(
                self.batch,
                self._response(),
                prompt_version="1",
                annotator_configuration={},
            ).records
        )
        applied = apply_overrides(
            validated,
            {self.target.id: self.target},
            {self.target.id: {"status": "rejected"}},
            self.validator,
        )
        self.assertEqual(applied[0].status, ValidationStatus.REJECTED)

    def test_omit_requires_human_review(self) -> None:
        response = self._response()
        response["annotations"][0].update(
            {
                "action": "omit",
                "spoken_text": "",
                "emotion": None,
            }
        )
        result = self.validator.validate_batch(
            self.batch,
            response,
            prompt_version="1",
            annotator_configuration={},
        )
        self.assertEqual(result.records[0].status, ValidationStatus.REVIEW_REQUIRED)
        self.assertEqual(result.records[0].issues[0].code, "high_impact_action")


if __name__ == "__main__":
    unittest.main()
