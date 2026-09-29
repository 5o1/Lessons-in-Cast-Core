from __future__ import annotations

import io
import json
import tempfile
import unittest
import wave
from pathlib import Path

from lessons_in_cast_core.config import AudioConfig
from lessons_in_cast_core.performance import (
    PerformanceCue,
    PerformanceCueKind,
    SpeechPerformance,
)
from lessons_in_cast_core.pronunciations import (
    LexicalPitchPoint,
    LexicalProsody,
    LexicalProsodyUnit,
    SelectedPronunciation,
)
from lessons_in_cast_core.synthesis import MiniMaxSpeechHttpSynthesizer, TtsJob


class _Response:
    def __init__(self, value: dict[str, object]) -> None:
        self._data = json.dumps(value).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._data


class _Opener:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        return _Response(self.value)


def _wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(44100)
        target.writeframes(b"\0\0" * 4410)
    return output.getvalue()


def _job() -> TtsJob:
    return TtsJob(
        id="job",
        dialogue_id="line",
        character_id="a",
        text="Hello Alice.",
        emotion="affectionate",
        delivery={},
        output_path="audio/raw/a/job.wav",
        cache_key="cache",
        performance=SpeechPerformance(
            speed=0.9,
            energy=0.4,
            cues=(
                PerformanceCue(
                    PerformanceCueKind.PAUSE,
                    5,
                    duration_seconds=0.35,
                ),
                PerformanceCue(PerformanceCueKind.SIGH, 6),
            ),
        ),
    )


class MiniMaxSpeechTests(unittest.TestCase):
    def _adapter(self, opener=None) -> MiniMaxSpeechHttpSynthesizer:
        rule = SelectedPronunciation(
            term="Alice",
            pronunciation="ˈæ.lɪs",
            system="ipa",
            preferred_system="ipa",
            language="en",
            case_sensitive=False,
            whole_word=True,
            prosody=LexicalProsody(
                units=(
                    LexicalProsodyUnit(
                        "al",
                        (("ipa", "ˈæ"), ("arpabet", "AE1")),
                        (
                            LexicalPitchPoint(0.0, -3.0),
                            LexicalPitchPoint(0.5, 3.0),
                            LexicalPitchPoint(1.0, -1.0),
                        ),
                    ),
                    LexicalProsodyUnit(
                        "iss",
                        (("ipa", "lɪs"), ("arpabet", "L IH0 S")),
                        (
                            LexicalPitchPoint(0.0, 1.0),
                            LexicalPitchPoint(1.0, -2.0),
                        ),
                    ),
                ),
            ),
        )
        return MiniMaxSpeechHttpSynthesizer(
            model="speech-2.8-hd",
            voice_id="English_test_voice",
            audio_config=AudioConfig(sample_rate=44100),
            pronunciations=(rule,),
            api_key="secret",
            opener=opener,
        )

    def test_compiles_portable_events_emotion_and_lexical_pitch(self) -> None:
        adaptation = self._adapter().adapt(_job())
        self.assertIn("<#0.35#>", adaptation.text)
        self.assertIn("(sighs)", adaptation.text)
        self.assertEqual(adaptation.emotion, "happy")
        self.assertEqual(
            adaptation.parameters["pronunciation_dict"]["tone"],
            ["Alice/(ˈæ˨˦˧.lɪs˧˨)"],
        )
        self.assertEqual(
            adaptation.parameters["voice_setting"]["speed"],
            0.9,
        )

    def test_decodes_hex_audio_without_exposing_key_in_configuration(self) -> None:
        opener = _Opener(
            {
                "data": {"audio": _wav_bytes().hex(), "status": 2},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }
        )
        adapter = self._adapter(opener)
        self.assertNotIn("secret", json.dumps(adapter.configuration))
        with tempfile.TemporaryDirectory() as directory:
            path = adapter.synthesize(_job(), Path(directory))
            self.assertTrue(path.is_file())
            with wave.open(str(path), "rb") as source:
                self.assertEqual(source.getframerate(), 44100)
                self.assertGreater(source.getnframes(), 0)
        request = opener.requests[0][0]
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "speech-2.8-hd")


if __name__ == "__main__":
    unittest.main()
