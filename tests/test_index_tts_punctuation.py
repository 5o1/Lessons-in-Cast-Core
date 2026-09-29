"""IndexTTS-only pause-marker lowering; semantic acting remains unchanged."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from lessons_in_cast_core.config import AudioConfig
from lessons_in_cast_core.performance import PerformanceCue, PerformanceCueKind
from lessons_in_cast_core.speech_markup import SpeechSegment
from lessons_in_cast_core.synthesis.backends.index_tts.adapter import IndexTtsSubprocessSynthesizer
from lessons_in_cast_core.synthesis.backends.index_tts.text import lower_punctuation
from lessons_in_cast_core.synthesis.types import TtsJob


class IndexPunctuationTests(unittest.TestCase):
    def test_single_repeated_mixed_and_full_width(self):
        cases = {
            "Stop! Listen!": "Stop!, Listen!,",
            "Why? Really?": "Why?, Really?,",
            "Stop!!! Why???": "Stop!, Why?,",
            "What?! Really!?": "What?!, Really!?,",
            "What!?!?": "What!?,",
            "Why？！ Stop！！": "Why?!, Stop!,",
            "What? ! ?": "What?!,",
            "Stop!Listen": "Stop!, Listen",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(lower_punctuation(text, (), "comma")[0], expected)
                self.assertEqual(lower_punctuation(expected, (), "comma")[0], expected)

    def test_existing_pauses_are_not_duplicated(self):
        for text in ("Why?, Really!.", "What?... Really!…", "Why? , Really! ."):
            self.assertEqual(lower_punctuation(text, ())[0], text)

    def test_period_and_native_modes(self):
        self.assertEqual(lower_punctuation("It's Alice, Teacher! Alice!", ())[0], "It's Alice, Teacher!. Alice!.")
        self.assertEqual(lower_punctuation("Why?! Stop!!!", (), "period")[0], "Why?!. Stop!.")
        self.assertEqual(lower_punctuation("Why?! Stop!!!", (), "native")[0], "Why?! Stop!!!")
        with self.assertRaises(ValueError):
            lower_punctuation("Why?", (), "unsupported")

    def test_punctuation_only_requires_upstream_interpretation(self):
        for text in ("!", "?", "?!", "…?!", " \t"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "punctuation-only"):
                lower_punctuation(text, ())

    def test_pronunciation_not_corrupted(self):
        text = "<What?!|W AH1 T> <|SPECIAL_TOKEN_1|>W AH1 T<|SPECIAL_TOKEN_1|>?"
        self.assertEqual(lower_punctuation(text, ())[0], text + ".")

    def test_cue_offsets_precede_mark_normalization(self):
        text = "What!!! Are you okay?"
        cues = (PerformanceCue(PerformanceCueKind.PAUSE, 7, .4),
                PerformanceCue(PerformanceCueKind.PAUSE, len(text), .4))
        result, notes = lower_punctuation(text, cues)
        self.assertEqual(result, "What!. Are you okay?.")
        self.assertEqual(sum("reuse" in note.strategy for note in notes), 2)
        with self.assertRaises(ValueError):
            lower_punctuation(text, (PerformanceCue(PerformanceCueKind.PAUSE, 100),))

    def test_adapter_preserves_emotion_and_voice_and_fingerprints_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = IndexTtsSubprocessSynthesizer(repository_root=root, python_executable=root / "python",
                source_root=root, model_path=root, references={}, audio_config=AudioConfig())
            job = TtsJob("test", "line", "a", "What?!", "calm", {}, "audio.wav", "key")
            self.assertEqual(backend.adapt(job).text, "What?!.")
            self.assertEqual(backend.configuration["expressive_pause"], "period")
            backend.set_expressive_pause("comma")
            comma = backend.adapt(job)
            old_config = backend.configuration
            backend.set_expressive_pause("native")
            native = backend.adapt(job)
            self.assertEqual(comma.text, "What?!,")
            self.assertEqual(native.text, job.text)
            self.assertEqual(comma.parameters["emotion_vector"], native.parameters["emotion_vector"])
            self.assertEqual(job.text, "What?!")
            self.assertNotEqual(old_config, backend.configuration)
            backend.set_expressive_pause("comma")
            inline = replace(job, text="Hey! Why?", emotion=None,
                segments=(SpeechSegment("Hey! ", "happy"), SpeechSegment("Why?", "sad")))
            parts = backend.adapt(inline).parameters["segments"]
            self.assertEqual([part["text"] for part in parts], ["Hey!, ", "Why?,"])
            self.assertEqual([part["emotion"] for part in parts], ["happy", "sad"])
            arbitrary = backend.adapt(replace(job, emotion=None, arbitrary_emotion="Quiet uncertainty."))
            self.assertEqual(arbitrary.parameters["arbitrary_emotion"], "Quiet uncertainty.")
            self.assertIsNone(arbitrary.parameters["emotion_vector"])
