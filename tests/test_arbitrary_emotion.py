"""Mutually exclusive inline arbitrary emotions and runtime backend lowering."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import Mock

from lessons_in_cast_core.config import AudioConfig
from lessons_in_cast_core.speech_markup import SpeechSegment, emotion_markup, parse_emotion_markup
from lessons_in_cast_core.synthesis.types import TtsJob
from lessons_in_cast_core.synthesis.segmentation import split_emotion_job
from lessons_in_cast_core.synthesis.backends.index_tts.adapter import IndexTtsSubprocessSynthesizer
from lessons_in_cast_core.synthesis.backends.index_tts.emotion_preparation import AXES
from lessons_in_cast_core.synthesis.backends.index_tts.qwen_emotion import resolve_arbitrary_emotion


class ArbitraryEmotionTests(unittest.TestCase):
    def job(self, **changes):
        job = TtsJob("job", "line", "a", "I'm fine.", None, {}, "output.wav", "key",
                     arbitrary_emotion="Trying to hide tears with a smile.")
        return replace(job, **changes)

    def test_markup_roundtrip_and_voice_wrapper(self):
        text = '<voice name="soft"><arbitrary_emotion description="Warm &amp; hesitant.">I am fine.</arbitrary_emotion> <emotion name="calm">Really.</emotion></voice>'
        spans = parse_emotion_markup(text, {"calm"})
        self.assertEqual(spans[0].arbitrary_emotion, "Warm & hesitant.")
        self.assertIsNone(spans[0].emotion)
        self.assertEqual(spans[0].voice, "soft")
        self.assertEqual(parse_emotion_markup(emotion_markup(spans), {"calm"}), spans)

    def test_overlap_and_invalid_attributes_always_fail(self):
        for text in [
            '<emotion name="calm"><arbitrary_emotion description="Warm">Hi</arbitrary_emotion></emotion>',
            '<arbitrary_emotion description="Warm"><emotion name="calm">Hi</emotion></arbitrary_emotion>',
            '<arbitrary_emotion name="calm" description="Warm">Hi</arbitrary_emotion>',
            '<emotion name="calm" description="Warm">Hi</emotion>',
            '<arbitrary_emotion description=" ">Hi</arbitrary_emotion>',
            '<arbitrary_emotion>Hi</arbitrary_emotion>',
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_emotion_markup(text, {"calm"})
        with self.assertRaises(ValueError):
            self.job(emotion="calm")
        with self.assertRaises(ValueError):
            SpeechSegment("Hi", "calm", arbitrary_emotion="Warm")

    def test_splitting_keeps_different_descriptions_separate_and_hashes_them(self):
        segments = (SpeechSegment("Hi.", None, arbitrary_emotion="Warm"),
                    SpeechSegment("Bye.", None, arbitrary_emotion="Cold"))
        job = self.job(text="Hi.Bye.", arbitrary_emotion=None, segments=segments)
        parts = split_emotion_job(job)
        self.assertEqual(len(parts), 2)
        self.assertEqual([p.arbitrary_emotion for p in parts], ["Warm", "Cold"])
        self.assertEqual(TtsJob.from_dict(parts[0].to_dict()), parts[0])
        self.assertNotEqual(parts[0].cache_key, parts[1].cache_key)
        same = replace(job, segments=(segments[0], replace(segments[1], arbitrary_emotion="Warm")))
        self.assertEqual(len(split_emotion_job(same)), 1)

    def test_adapter_defers_qwen_until_render_and_keeps_description_out_of_speech(self):
        backend = IndexTtsSubprocessSynthesizer(repository_root=Path.cwd(), python_executable=Path("python"),
                    source_root=Path("external"), model_path=Path("models"), references={}, audio_config=AudioConfig())
        backend._exchange = Mock(side_effect=AssertionError("adapt must not start inference"))
        result = backend.adapt(self.job())
        self.assertEqual(result.text, "I'm fine.")
        self.assertIsNone(result.parameters["emotion_vector"])
        self.assertEqual(result.parameters["arbitrary_emotion"], self.job().arbitrary_emotion)
        backend._exchange.assert_not_called()

    def test_runtime_uses_multiple_axes_without_registering_presets(self):
        engine = Mock()
        engine.raw_output = "test"
        engine.inference.return_value = dict(zip(AXES, [0, 0, .4, .2, 0, .1, 0, .1]))
        resolution = resolve_arbitrary_emotion("Tears and fear", engine)
        engine.inference.assert_called_once_with("Tears and fear")
        self.assertEqual(resolution["vector"], [0, 0, .4, .2, 0, .09375, 0, .05625])
        self.assertEqual(len(resolution["vector"]), 8)

    def test_polish_validation_and_planner_preserve_dynamic_description(self):
        from lessons_in_cast_core.annotation import AnnotationValidator, ValidationStatus
        from lessons_in_cast_core.config import AnnotationConfig
        from lessons_in_cast_core.dialogue import DialogueBatch
        from lessons_in_cast_core.characters import CharacterDefinition
        from lessons_in_cast_core.synthesis.planner import SynthesisPlanner
        from .helpers import record
        target = record(1, dialogue="I'm fine.")
        markup = '<arbitrary_emotion description="Hiding tears.">I\'m fine.</arbitrary_emotion>'
        response = {"batch_id": "batch", "annotations": [{"id": target.id, "action": "speak",
            "spoken_text": markup, "delivery": {}, "effects": [], "confidence": 1,
            "review_required": False, "reason": None, "performance": {}}]}
        validator = AnnotationValidator(AnnotationConfig(frozenset({"calm"}), frozenset()))
        def validate():
            return validator.validate_batch(DialogueBatch("batch", (), (target,), ()), response,
                prompt_version="polish-v2", annotator_configuration={}, schema_version=5).records[0]
        result = validate()
        self.assertEqual(result.status, ValidationStatus.ACCEPTED, result.issues)
        character = CharacterDefinition("a", "Alice", "individual", (), None, "", 1, False, "profiles/a.py")
        plan = SynthesisPlanner({"a": character}).plan({target.id: target}, [result])
        self.assertEqual(plan.jobs[0].text, "I'm fine.")
        self.assertEqual(plan.jobs[0].segments[0].arbitrary_emotion, "Hiding tears.")
        response["annotations"][0]["spoken_text"] = '<emotion name="calm">' + markup + '</emotion>'
        self.assertEqual(validate().status, ValidationStatus.RETRYABLE)


if __name__ == "__main__":
    unittest.main()
