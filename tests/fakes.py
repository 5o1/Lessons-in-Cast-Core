"""Deterministic test doubles for annotation and speech synthesis."""

from __future__ import annotations

import wave
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lessons_in_cast_core.config import AudioConfig
from lessons_in_cast_core.hashing import content_hash
from lessons_in_cast_core.jsonl import read_jsonl, write_jsonl
from lessons_in_cast_core.synthesis.types import TtsJob
from lessons_in_cast_core.speech_markup import SpeechSegment, emotion_markup


class MockDialogueAnnotator:
    @property
    def configuration(self) -> dict[str, Any]:
        return {"adapter": "mock", "version": 1}

    def annotate(self, request: dict[str, Any]) -> dict[str, Any]:
        batch = request["batch"]
        return {
            "batch_id": batch["batch_id"],
            "annotations": [
                {
                    "id": item["id"],
                    "action": "speak",
                    "spoken_text": item["dialogue"],
                    "emotion": "neutral",
                    "delivery": {},
                    "effects": [],
                    "confidence": 1.0,
                    "review_required": False,
                    "reason": "Deterministic mock annotation.",
                }
                for item in batch["targets"]
            ],
        }

def write_mock_responses(requests_path: Path, responses_path: Path) -> int:
    annotator = MockDialogueAnnotator()
    envelopes = []
    for request in read_jsonl(requests_path):
        response = annotator.annotate(request)
        if request.get("stage") == "cleaning":
            for annotation in response["annotations"]:
                annotation.pop("emotion")
                annotation.pop("delivery")
                annotation["performance"] = {"cues": []}
        elif request.get("schema_version", 1) >= 3:
            for annotation in response["annotations"]:
                annotation["spoken_text"] = emotion_markup((SpeechSegment(annotation["spoken_text"],
                    annotation.pop("emotion")),))
                original = request.get("cleaned_annotations", {}).get(annotation["id"])
                if original is not None:
                    annotation["performance"] = original["performance"]
                    annotation["effects"] = original["effects"]
                    annotation["action"] = original["action"]
                    if original["action"] in {"omit", "sfx_only"}:
                        annotation["spoken_text"] = ""
        envelopes.append(
            {
                "request_hash": content_hash(request),
                "batch_id": request["batch"]["batch_id"],
                "prompt_version": request["prompt_version"],
                "annotator_configuration": annotator.configuration,
                "generated_at": "2026-09-05T00:00:00+00:00",
                "response": response,
            }
        )
    return write_jsonl(envelopes, responses_path)



class SilenceSynthesizer:
    def __init__(self, config: AudioConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "silence"

    @property
    def configuration(self) -> dict[str, Any]:
        return {"adapter": self.name, "audio": asdict(self._config)}

    def synthesize(self, job: TtsJob, artifact_root: Path) -> Path:
        destination = artifact_root / job.output_path
        if destination.is_file():
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        duration = min(max(len(job.text) / 12.0, 0.1), 10.0)
        frame_count = round(duration * self._config.sample_rate)
        frame = b"\x00" * self._config.sample_width * self._config.channels
        with wave.open(str(destination), "wb") as output:
            output.setnchannels(self._config.channels)
            output.setsampwidth(self._config.sample_width)
            output.setframerate(self._config.sample_rate)
            output.writeframes(frame * frame_count)
        return destination
