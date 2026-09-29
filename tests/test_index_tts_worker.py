"""Frontend auditing must not consume the segments needed for inference."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from lessons_in_cast_core.synthesis.backends.index_tts import worker


class IndexTtsWorkerTests(unittest.TestCase):
    def run_inference(self, make_segments, *, fail=False):
        expected = ["Why?.", " Alice!."]
        split = Mock(side_effect=lambda *_: make_segments(expected))
        inference_speech = Mock()
        tts = SimpleNamespace(split_text_by_tokens=split, _token_len=len,
                              gpt=SimpleNamespace(inference_speech=inference_speech))
        consumed = []

        def infer(**kwargs):
            for prefix in ("", "[en]"):
                segments = tts.split_text_by_tokens(kwargs["text"], 64, prefix)
                self.assertIsInstance(segments, list)
                consumed.append(list(segments))
            if fail:
                raise RuntimeError("Simulated inference failure")
            Path(kwargs["output_path"]).write_bytes(b"mock PCM output")

        tts.infer = infer
        torch = SimpleNamespace(manual_seed=Mock(), cuda=SimpleNamespace(is_available=lambda: False))
        numpy = SimpleNamespace(random=SimpleNamespace(seed=Mock()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output.wav"
            request = {
                "seed": 1, "output": str(output), "do_sample": False,
                "num_beams": 1, "repetition_penalty": 1.0, "length_penalty": 1.0,
                "max_mel_tokens": 100, "text": "Why?. Alice!.", "emotion_vector": None,
                "emotion_alpha": 0.65, "use_random_emotion": False,
                "interval_silence_ms": 200, "max_text_tokens_per_segment": 64,
                "duration_factor": 1.0, "text_normalization": True,
                "sample_rate": 48000, "channels": 1,
            }
            with patch.dict("sys.modules", {"numpy": numpy, "torch": torch}), \
                 patch.object(worker, "_prepare_reference", return_value=root / "reference.wav"), \
                 patch.object(worker, "_convert_audio"):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "Simulated inference failure"):
                        worker._infer(tts, request, "en", root, {})
                    self.assertFalse(output.exists())
                    result = None
                else:
                    result = worker._infer(tts, request, "en", root, {})
                    self.assertTrue(output.is_file())
                self.assertFalse(output.with_suffix(".index-tts.tmp.wav").exists())
            self.assertEqual(consumed, [expected, expected])
            self.assertIs(tts.split_text_by_tokens, split)
            self.assertIs(tts.gpt.inference_speech, inference_speech)
            self.assertEqual(split.call_count, 2)
        return result, expected

    def test_audit_preserves_list_iterator_and_generator_segments(self):
        for name, factory in (("list", list), ("iterator", iter),
                              ("generator", lambda items: (item for item in items))):
            with self.subTest(kind=name):
                result, expected = self.run_inference(factory)
                self.assertEqual(len(result["frontend_calls"]), 2)
                for call, prefix in zip(result["frontend_calls"], ("", "[en]")):
                    self.assertEqual(call["segments"], expected)
                    self.assertEqual(call["token_counts"], [len(prefix + part) for part in expected])
                    self.assertEqual(call["language_prefix"], prefix)
                    self.assertEqual(call["normalized_text"], "Why?. Alice!.")

    def test_inference_failure_restores_original_methods(self):
        self.run_inference(lambda items: (item for item in items), fail=True)
