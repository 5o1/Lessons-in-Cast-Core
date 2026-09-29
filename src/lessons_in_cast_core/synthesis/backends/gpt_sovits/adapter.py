"""HTTP adapter for a locally running GPT-SoVITS API v2 server."""

from __future__ import annotations

import json
import re
import tempfile
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ....performance import (
    AdaptationFidelity,
    FeatureAdaptation,
    SpeechAdaptation,
    approximate_cues_with_punctuation,
    resolve_legacy_delivery,
)
from ...types import TtsJob


@dataclass(frozen=True, slots=True)
class GptSoVitsReference:
    audio_path: Path
    prompt_text: str = ""
    prompt_language: str = "ja"


class GptSoVitsHttpSynthesizer:
    """Generate WAV files through GPT-SoVITS' version-2 HTTP API."""

    def __init__(
        self,
        endpoint: str,
        default_reference: GptSoVitsReference,
        *,
        references_by_emotion: Mapping[str, GptSoVitsReference] | None = None,
        text_language: str = "en",
        seed: int = 233333,
        base_speed: float = 1.0,
        timeout_seconds: float = 300.0,
        pronunciations: Mapping[str, str] | None = None,
    ) -> None:
        if base_speed <= 0:
            raise ValueError("GPT-SoVITS base_speed must be positive")
        self._endpoint = endpoint.rstrip("/") + "/tts"
        self._default_reference = default_reference
        self._references_by_emotion = dict(references_by_emotion or {})
        self._text_language = text_language
        self._seed = seed
        self._timeout_seconds = timeout_seconds
        self._base_speed = float(base_speed)
        self._pronunciations = dict(pronunciations or {})
        if any(
            not source or not replacement
            for source, replacement in self._pronunciations.items()
        ):
            raise ValueError(
                "Pronunciation entries must have non-empty source and replacement text"
            )

    @property
    def name(self) -> str:
        return "gpt-sovits-http-v2"

    @property
    def configuration(self) -> dict[str, Any]:
        def reference(value: GptSoVitsReference) -> dict[str, str]:
            return {
                "audio_path": str(value.audio_path.resolve()),
                "prompt_text": value.prompt_text,
                "prompt_language": value.prompt_language,
            }

        return {
            "adapter": self.name,
            "endpoint": self._endpoint,
            "text_language": self._text_language,
            "seed": self._seed,
            "pronunciations": dict(sorted(self._pronunciations.items())),
            "base_speed": self._base_speed,
            "default_reference": reference(self._default_reference),
            "references_by_emotion": {
                emotion: reference(value)
                for emotion, value in sorted(self._references_by_emotion.items())
            },
        }

    def synthesize(self, job: TtsJob, artifact_root: Path) -> Path:
        destination = artifact_root / job.output_path
        if destination.is_file():
            return destination
        if job.segments:
            from ...segmentation import synthesize_segments
            self.adapt(job)
            return synthesize_segments(job, artifact_root, self.synthesize)
        adaptation = self.adapt(job)
        reference = self._references_by_emotion.get(
            job.emotion,
            self._default_reference,
        )
        payload = {
            "text": adaptation.text,
            "text_lang": self._text_language,
            "ref_audio_path": str(reference.audio_path.resolve()),
            "prompt_text": reference.prompt_text,
            "prompt_lang": reference.prompt_language,
            "text_split_method": "cut5",
            "batch_size": 1,
            "speed_factor": adaptation.parameters["speed_factor"],
            "seed": self._seed,
            "media_type": "wav",
            "streaming_mode": False,
            "parallel_infer": True,
            "repetition_penalty": 1.35,
        }
        request = urllib.request.Request(
            self._endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=self._timeout_seconds) as response:
                data = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"GPT-SoVITS rejected {job.dialogue_id}: HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"GPT-SoVITS request failed for {job.dialogue_id}: {exc.reason}"
            ) from exc

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            suffix=".wav",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(data)
        try:
            with wave.open(str(temporary), "rb") as source:
                if source.getnframes() < 1:
                    raise RuntimeError(
                        f"GPT-SoVITS returned empty WAV for {job.dialogue_id}"
                    )
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    def adapt(self, job: TtsJob) -> SpeechAdaptation:
        if job.arbitrary_emotion is not None:
            raise ValueError("GPT-SoVITS does not yet implement arbitrary_emotion")
        if job.segments:
            from ...segmentation import adapt_segments
            return adapt_segments(job, self.adapt)
        if job.voice is not None:
            raise NotImplementedError("GPT-SoVITS voice switching also requires a matching reference transcript; file-only voice tags are not supported")
        performance, legacy_notes = resolve_legacy_delivery(
            job.performance,
            job.delivery,
        )
        text, cue_notes = approximate_cues_with_punctuation(
            job.text,
            performance.cues,
        )
        text = apply_pronunciations(text, self._pronunciations)
        notes = [*legacy_notes, *cue_notes]
        if job.emotion != "neutral" and job.emotion not in self._references_by_emotion:
            notes.append(FeatureAdaptation("emotion", AdaptationFidelity.DROPPED,
                f"no reference configured for semantic label {job.emotion!r}; using the default reference"))
        for field, value in {
            "direction": performance.direction,
            "vocal_mode": performance.vocal_mode,
            "pitch_semitones": performance.pitch_semitones,
            "volume_gain_db": performance.volume_gain_db,
            "energy": performance.energy,
            "brightness": performance.brightness,
            "clarity": performance.clarity,
            "breathiness": performance.breathiness,
        }.items():
            if value is not None:
                notes.append(
                    FeatureAdaptation(
                        f"performance.{field}",
                        AdaptationFidelity.DROPPED,
                        "GPT-SoVITS HTTP v2 exposes no matching stable control",
                    )
                )
        return SpeechAdaptation(
            job_id=job.id,
            dialogue_id=job.dialogue_id,
            backend=self.name,
            text=text,
            emotion=job.emotion,
            parameters={
                "speed_factor": self._base_speed * (performance.speed or 1.0),
                "reference_emotion": (
                    job.emotion
                    if job.emotion in self._references_by_emotion
                    else "default"
                ),
            },
            features=tuple(notes),
        )


def apply_pronunciations(text: str, pronunciations: Mapping[str, str]) -> str:
    """Apply synthesis-only aliases without changing the source dialogue."""
    result = text
    for source, replacement in sorted(
        pronunciations.items(), key=lambda item: len(item[0]), reverse=True
    ):
        pattern = rf"(?<!\w){re.escape(source)}(?!\w)"
        result = re.sub(pattern, replacement, result)
    return result
