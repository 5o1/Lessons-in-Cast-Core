"""Inline speech validation, adapter lowering, reference routing and PCM joins."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import wave

from lessons_in_cast_core.annotation import AnnotationValidator, ValidationStatus
from lessons_in_cast_core.config import AnnotationConfig, AudioConfig
from lessons_in_cast_core.dialogue import DialogueBatch
from lessons_in_cast_core.performance import PerformanceCue, PerformanceCueKind, SpeechPerformance
from lessons_in_cast_core.speech_markup import SpeechSegment, emotion_markup, parse_emotion_markup
from lessons_in_cast_core.synthesis.backends.index_tts.adapter import IndexTtsSubprocessSynthesizer, index_emotion_vector
from lessons_in_cast_core.synthesis.planner import SynthesisPlanner
from lessons_in_cast_core.synthesis.references.voices import list_voice_references, resolve_voice_reference
from lessons_in_cast_core.synthesis.segmentation import split_emotion_job, concatenate_wav
from lessons_in_cast_core.synthesis.types import TtsJob
from lessons_in_cast_core.characters import CharacterDefinition
from .helpers import record


def write_wave(path, value=1, rate=24000, frames=24):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(int(value).to_bytes(2, "little", signed=True) * frames)


class InlineSpeechTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.default = self.root / "references/default.wav"
        write_wave(self.default)
        self.backend = IndexTtsSubprocessSynthesizer(repository_root=self.root,
            python_executable=self.root / "python", source_root=self.root, model_path=self.root,
            references={"a": (self.default,)}, audio_config=AudioConfig(format="wav"))
        self.segments = (SpeechSegment("I'm afraid. ", "afraid"),
                         SpeechSegment("But I'll try. I'm here.", "calm"))
        self.job = TtsJob("job", "line", "a", "".join(s.text for s in self.segments),
                          "neutral", {}, "audio/result.wav", "key", segments=self.segments)

    def test_roundtrip_entities_and_dynamic_voice(self):
        spans = (SpeechSegment("A & B < C. ", "calm", "My custom voice"), self.segments[1])
        self.assertEqual(parse_emotion_markup(emotion_markup(spans), {"calm"}), spans)

    def test_malformed_uncovered_unknown_nested_and_unsafe_markup_rejected(self):
        invalid = [
            "Plain text.", '<emotion name="sad">Oops.',
            '<emotion name="sad" intensity="nan">Hi.</emotion>',
            '<emotion name="sad" intensity="1.1">Hi.</emotion>',
            '<emotion name="invented">Hi.</emotion>',
            '<emotion name="sad"> </emotion>',
            '<emotion name="sad">Hi.</emotion> Unlabelled.',
            '<emotion name="sad"><emotion name="sad">Hi.</emotion></emotion>',
            '<voice name="../bad"><emotion name="sad">Hi.</emotion></voice>',
            '<voice name="sad">Unlabelled.</voice>',
            '<voice name="sad"><voice name="other"><emotion name="sad">Hi.</emotion></voice></voice>',
            '<!DOCTYPE speech><emotion name="sad">Hi.</emotion>',
            '<emotion name="sad" extra="x">Hi.</emotion>',
        ]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_emotion_markup(text, {"sad"})

    def validate(self, markup, **fields):
        target = record(1, dialogue=self.job.text)
        batch = DialogueBatch("batch", (), (target,), ())
        response = {"batch_id": "batch", "annotations": [{"id": target.id, "action": "speak",
            "spoken_text": markup, "delivery": {}, "effects": [], "confidence": 1,
            "review_required": False, "reason": None, "performance": {}, **fields}]}
        validator = AnnotationValidator(AnnotationConfig(frozenset({"calm", "afraid"}), frozenset(), 4, .15))
        return target, validator.validate_batch(batch, response, prompt_version="v4",
            annotator_configuration={}, schema_version=4).records[0]

    def test_validator_and_planner_keep_line_identity_but_strip_tags(self):
        target, result = self.validate(emotion_markup(self.segments))
        self.assertEqual(result.status, ValidationStatus.ACCEPTED, result.issues)
        character = CharacterDefinition("a", "Alice", "individual", (), None, "", 1, False, "profiles/a.py")
        plan = SynthesisPlanner({"a": character}).plan({target.id: target}, [result])
        self.assertFalse(plan.issues)
        self.assertEqual(len(plan.jobs), 1)
        self.assertEqual(plan.jobs[0].text, self.job.text)
        self.assertEqual(plan.jobs[0].segments, self.segments)
        self.assertEqual(plan.render_tasks[0].identifier, target.identifier)
        self.assertEqual(TtsJob.from_dict(plan.jobs[0].to_dict()), plan.jobs[0])

    def test_v4_rejects_plain_text_and_line_level_emotions(self):
        for text, fields in [(self.job.text, {}), (emotion_markup(self.segments), {"emotion": "calm", "intensity": .2})]:
            self.assertEqual(self.validate(text, **fields)[1].status, ValidationStatus.RETRYABLE)

    def test_cue_offsets_use_decoded_text_and_boundary_has_one_owner(self):
        boundary = len(self.segments[0].text)
        cues = (PerformanceCue(PerformanceCueKind.PAUSE, boundary, .4),
                PerformanceCue(PerformanceCueKind.EXHALE, len(self.job.text)))
        parts = split_emotion_job(replace(self.job, performance=SpeechPerformance(cues=cues)))
        self.assertEqual(parts[0].performance.cues, ())
        self.assertEqual([c.offset for c in parts[1].performance.cues], [0, len(self.segments[1].text)])
        _, validated = self.validate(emotion_markup(self.segments), performance={"cues": [{"kind": "pause", "offset": len(self.job.text)+1, "duration_seconds": .4}]})
        self.assertEqual(validated.status, ValidationStatus.RETRYABLE)

    def test_adjacent_equal_controls_merge_but_voice_change_splits(self):
        spans = (SpeechSegment("One. ", "calm"), SpeechSegment("Two. ", "calm"),
                 SpeechSegment("Three.", "calm", "custom"))
        parts = split_emotion_job(replace(self.job, text="".join(s.text for s in spans), segments=spans))
        self.assertEqual([p.text for p in parts], ["One. Two. ", "Three."])
        self.assertEqual(parts[1].voice, "custom")

    def test_dynamic_reference_discovery_ignores_non_audio_and_subdirectories(self):
        write_wave(self.default.parent / "tearful.WAV")
        (self.default.parent / "notes.txt").touch()
        write_wave(self.default.parent / "sources/hidden.wav")
        self.assertEqual(set(list_voice_references(self.default)), {"default", "tearful"})
        self.assertEqual(resolve_voice_reference(self.default, "tearful"), self.default.parent / "tearful.WAV")
        with self.assertRaises(ValueError):
            resolve_voice_reference(self.default, "missing")
        (self.default.parent / "tearful.flac").touch()
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            list_voice_references(self.default)

    def test_index_emotion_and_voice_requests_then_ordered_pcm_merge(self):
        alternate = self.default.parent / "tearful.wav"
        write_wave(alternate, 5)
        spans = (replace(self.segments[0], voice="tearful"), self.segments[1])
        job = replace(self.job, segments=spans)
        requests = []
        def exchange(request):
            requests.append(request)
            write_wave(Path(request["output"]), len(requests))
            return {"ok": True}
        self.backend._exchange = exchange
        adaptation = self.backend.adapt(job)
        self.assertEqual(len(adaptation.parameters["segments"]), 2)
        output = self.backend.synthesize(job, self.root)
        self.assertEqual([r["text"] for r in requests], [s.text for s in spans])
        self.assertEqual(requests[0]["references"], [str(alternate)])
        self.assertEqual(requests[1]["references"], [str(self.default)])
        self.assertEqual(requests[0]["emotion_vector"], index_emotion_vector("afraid"))
        self.assertEqual(requests[1]["emotion_vector"], index_emotion_vector("calm"))
        with wave.open(str(output)) as source:
            self.assertEqual(source.getnframes(), 48)
            self.assertEqual(source.readframes(48), b"\1\0"*24 + b"\2\0"*24)
        self.backend.synthesize(job, self.root)
        self.assertEqual(len(requests), 2)

    def test_missing_voice_stops_before_any_generation(self):
        self.backend._exchange = Mock()
        job = replace(self.job, segments=(self.segments[0], replace(self.segments[1], voice="missing")))
        with self.assertRaisesRegex(ValueError, "no matching"):
            self.backend.synthesize(job, self.root)
        self.backend._exchange.assert_not_called()
        self.assertFalse((self.root / job.output_path).exists())

    def test_reference_contents_change_configuration_fingerprint(self):
        before = self.backend.configuration
        write_wave(self.default.parent / "custom.wav", 4)
        self.assertNotEqual(before, self.backend.configuration)
        before = self.backend.configuration
        write_wave(self.default.parent / "custom.wav", 9)
        self.assertNotEqual(before, self.backend.configuration)

    def test_failed_segment_does_not_publish_line_and_resume_reuses_completed_segment(self):
        calls = []
        def exchange(request):
            calls.append(request)
            if len(calls) == 2:
                return {"ok": False, "error": "fixture failure"}
            write_wave(Path(request["output"]))
            return {"ok": True}
        self.backend._exchange = exchange
        with self.assertRaises(RuntimeError):
            self.backend.synthesize(self.job, self.root)
        self.assertFalse((self.root / self.job.output_path).exists())
        self.backend.synthesize(self.job, self.root)
        self.assertEqual(len(calls), 3)

    def test_join_rejects_mismatched_formats_without_overwriting_existing_file(self):
        other = self.root / "other.wav"
        write_wave(other, rate=48000)
        output = self.root / "existing.wav"
        write_wave(output, 7)
        before = output.read_bytes()
        with self.assertRaises(ValueError):
            concatenate_wav((self.default, other), output)
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
