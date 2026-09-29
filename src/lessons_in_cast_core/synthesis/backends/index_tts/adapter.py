"""Persistent subprocess adapter for the official IndexTTS 2.5 environment."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from ....config import AudioConfig
from ....paths import SOURCE_ROOT
from ....performance import (
    AdaptationFidelity,
    FeatureAdaptation,
    SpeechAdaptation,
    resolve_legacy_delivery,
)
from ....pronunciations import SelectedPronunciation
from ...types import TtsJob
from ...segmentation import adapt_segments, synthesize_segments
from ...references.voices import resolve_voice_reference, voice_reference_fingerprint


from .emotions import EMOTION_VECTORS
from .text import VERSION as TEXT_FRONTEND_VERSION, lower_punctuation

_AXIS_ORDER = ("joy", "anger", "sadness", "fear", "disgust", "depression", "surprise", "calm")
_EMOTION_BIAS = (0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625)
_MAX_EMOTION_SUM = 0.8


def normalize_index_emotion_vector(vector: list[float]) -> list[float]:
    """Apply the bias and 0.8 total-strength cap used by the official WebUI."""

    if len(vector) != len(_AXIS_ORDER):
        raise ValueError("IndexTTS emotion vectors must contain eight values")
    biased = [
        max(float(value), 0.0) * bias for value, bias in zip(vector, _EMOTION_BIAS)
    ]
    total = sum(biased)
    if total > _MAX_EMOTION_SUM:
        scale = _MAX_EMOTION_SUM / total
        biased = [value * scale for value in biased]
    return [round(value, 6) for value in biased]


def index_emotion_vector(emotion: str) -> list[float]:
    """Compile one semantic label to its complete normalized model vector."""

    if emotion not in EMOTION_VECTORS:
        raise ValueError(f"No IndexTTS emotion mapping is defined for {emotion!r}")
    return normalize_index_emotion_vector(list(EMOTION_VECTORS[emotion]))


def apply_index_pronunciations(
    text: str, pronunciations: Mapping[str, str],
    rules: Sequence[SelectedPronunciation] = (),
) -> str:
    """Add official word-and-ARPABET pronunciation annotations."""

    configured = {rule.term: rule for rule in rules}
    patterns = []
    for word, phonemes in sorted(
        pronunciations.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if not word or not phonemes:
            raise ValueError("Pronunciation entries cannot be empty")
        rule = configured.get(word)
        pattern = re.escape(word)
        if rule is None or rule.whole_word:
            pattern = rf"(?<!\w){pattern}(?!\w)"
        if rule is None or not rule.case_sensitive:
            pattern = f"(?i:{pattern})"
        patterns.append((pattern, phonemes))
    if not patterns:
        return text
    # A single substitution prevents nested annotations for overlapping names.
    # Existing frontend annotations must remain opaque and idempotent.
    protected = r"<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>|<[^|>\n]+\|[^>\n]+>"
    combined = re.compile("|".join(f"({pattern})" for pattern, _ in patterns))
    def annotate(piece):
        return combined.sub(
            lambda match: f"<{match.group(0)}|{patterns[match.lastindex - 1][1]}>", piece
        )
    result, offset = [], 0
    for match in re.finditer(protected, text):
        result.extend((annotate(text[offset:match.start()]), match.group()))
        offset = match.end()
    return "".join(result) + annotate(text[offset:])


class IndexTtsSubprocessSynthesizer:
    """Keep one GPU-loaded IndexTTS worker alive across all TTS jobs."""

    def __init__(
        self,
        *,
        repository_root: Path,
        python_executable: Path,
        source_root: Path,
        model_path: Path,
        references: Mapping[str, Sequence[Path]],
        audio_config: AudioConfig,
        base_speed: float = 1.0,
        language: str = "EN",
        seed: int = 233333,
        use_bf16: bool = True,
        prepare_references: bool = True,
        trim_reference_silence: bool = True,
        reference_trim_top_db: float = 40.0,
        reference_trim_padding_ms: int = 150,
        reference_gate_hold_ms: int = 500,
        minimum_reference_speech_seconds: float = 3.0,
        minimum_reference_duration_seconds: float = 15.0,
        emotion_alpha: float = 0.65,
        use_random_emotion: bool = False,
        do_sample: bool = False,
        top_p: float = 0.8,
        top_k: int = 30,
        temperature: float = 0.8,
        num_beams: int = 3,
        repetition_penalty: float = 10.0,
        length_penalty: float = 0.0,
        max_mel_tokens: int = 1500,
        interval_silence_ms: int = 200,
        max_text_tokens_per_segment: int = 120,
        text_normalization: bool = True,
        pronunciations: Mapping[str, str] | None = None,
        pronunciation_rules: Sequence[SelectedPronunciation] | None = None,
        emotion_vectors: Mapping[str, Sequence[float]] | None = None,
        emotion_mapping_id: str | None = None,
        expressive_pause: str = "period",
    ) -> None:
        if (
            audio_config.format.lstrip(".").lower() != "wav"
            or audio_config.sample_width != 2
        ):
            raise ValueError("IndexTTS adapter currently requires 16-bit WAV output")
        if base_speed <= 0:
            raise ValueError("IndexTTS base_speed must be positive")
        if expressive_pause not in ("native", "comma", "period"):
            raise ValueError("IndexTTS expressive_pause must be native, comma or period")
        self._expressive_pause = expressive_pause
        self._root = repository_root.resolve()
        # Resolving this symlink would bypass the virtual environment and use
        # uv's base interpreter without the environment's installed packages.
        self._python = python_executable.absolute()
        self._source = source_root.resolve()
        self._model = model_path.resolve()
        self._base_speed = float(base_speed)
        self._references = {
            key: tuple(value.resolve() for value in values)
            for key, values in references.items()
        }
        self._audio = audio_config
        self._language = language
        self._seed = seed
        self._use_bf16 = use_bf16
        self._reference_processing = {
            "prepare_references": prepare_references,
            "trim_reference_silence": trim_reference_silence,
            "reference_trim_top_db": reference_trim_top_db,
            "reference_trim_padding_ms": reference_trim_padding_ms,
            "reference_gate_hold_ms": reference_gate_hold_ms,
            "minimum_reference_speech_seconds": minimum_reference_speech_seconds,
            "minimum_reference_duration_seconds": minimum_reference_duration_seconds,
        }
        self._inference = {
            "emotion_alpha": emotion_alpha,
            "use_random_emotion": use_random_emotion,
            "do_sample": do_sample,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "length_penalty": length_penalty,
            "max_mel_tokens": max_mel_tokens,
            "interval_silence_ms": interval_silence_ms,
            "max_text_tokens_per_segment": max_text_tokens_per_segment,
            "text_normalization": text_normalization,
        }
        self._pronunciations = dict(pronunciations or {})
        self._pronunciation_rules = tuple(pronunciation_rules or ())
        self._emotion_vectors = {name: normalize_index_emotion_vector(list(values))
                                 for name, values in (EMOTION_VECTORS if emotion_vectors is None else emotion_vectors).items()}
        self._emotion_mapping_id = emotion_mapping_id or "legacy-manual-presets"
        self._process: subprocess.Popen[str] | None = None

    @property
    def name(self) -> str:
        return "index-tts-2.5"

    @property
    def configuration(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "model_path": str(self._model),
            "language": self._language,
            "base_speed": self._base_speed,
            "seed": self._seed,
            "use_bf16": self._use_bf16,
            "reference_processing": dict(self._reference_processing),
            "inference": dict(self._inference),
            "references": {
                key: [str(path) for path in paths]
                for key, paths in sorted(self._references.items())
            },
            "pronunciations": dict(sorted(self._pronunciations.items())),
            "pronunciation_rules": [
                rule.to_dict() for rule in self._pronunciation_rules
            ],
            "inline_speech_version": 3,
            "arbitrary_emotion_version": 1,
            "text_frontend_version": TEXT_FRONTEND_VERSION,
            "expressive_pause": self._expressive_pause,
            "emotion_vectors": self._emotion_vectors,
            "emotion_mapping_id": self._emotion_mapping_id,
            "voice_references": {key: voice_reference_fingerprint(paths[0])
                                 for key, paths in self._references.items() if paths},
        }

    def set_references(
        self, character_id: str, references: Sequence[Path]
    ) -> None:
        """Replace one character's references without restarting the worker."""

        resolved = tuple(reference.resolve() for reference in references)
        if not resolved:
            raise ValueError("IndexTTS requires at least one reference")
        self._references[character_id] = resolved

    def set_pronunciations(self, pronunciations: Mapping[str, str]) -> None:
        """Replace pronunciation annotations without restarting the worker."""
        self._pronunciations = dict(pronunciations)

    def set_expressive_pause(self, mode: str) -> None:
        """Select an IndexTTS-only punctuation fallback before planning a take."""
        if mode not in ("native", "comma", "period"):
            raise ValueError("IndexTTS expressive_pause must be native, comma or period")
        self._expressive_pause = mode

    def synthesize(self, job: TtsJob, artifact_root: Path) -> Path:
        destination = (artifact_root / job.output_path).resolve()
        if destination.is_file():
            return destination
        if job.segments:
            self.adapt(job)
            return synthesize_segments(job, artifact_root, self.synthesize)
        references = self._references.get(job.character_id)
        if references is None:
            raise RuntimeError(
                f"No IndexTTS reference is configured for character "
                f"{job.character_id!r}"
            )
        if job.voice is not None:
            references = (self._voice_reference(job),)
        for reference in references:
            if not reference.is_file():
                raise FileNotFoundError(
                    f"IndexTTS reference audio is missing: {reference}"
                )
        destination.parent.mkdir(parents=True, exist_ok=True)
        adaptation = self.adapt(job)
        response = self._exchange(
            {
                "id": job.id,
                "references": [str(reference) for reference in references],
                "text": adaptation.text,
                "output": str(destination),
                "emotion_vector": adaptation.parameters["emotion_vector"],
                "arbitrary_emotion": job.arbitrary_emotion,
                "emotion_energy": job.performance.energy,
                "duration_factor": adaptation.parameters["duration_factor"],
                "seed": self._seed,
                "sample_rate": self._audio.sample_rate,
                "channels": self._audio.channels,
                **self._reference_processing,
                **self._inference,
            }
        )
        if response.get("ok") is not True:
            raise RuntimeError(
                f"IndexTTS failed for {job.dialogue_id}: "
                f"{response.get('error', 'unknown worker error')}"
            )
        if not destination.is_file():
            raise RuntimeError(f"IndexTTS did not create {destination}")
        if response.get("text_frontend") is not None:
            destination.with_suffix(".frontend.json").write_text(
                json.dumps(response["text_frontend"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if job.arbitrary_emotion is not None:
            destination.with_suffix(".emotion.json").write_text(
                json.dumps({"description": job.arbitrary_emotion, "resolution": response.get("emotion_resolution")}, indent=2) + "\n",
                encoding="utf-8",
            )
        return destination

    def adapt(self, job: TtsJob) -> SpeechAdaptation:
        if job.segments:
            return adapt_segments(job, self.adapt)
        performance, legacy_notes = resolve_legacy_delivery(
            job.performance,
            job.delivery,
        )
        text, cue_notes = lower_punctuation(
            job.text,
            performance.cues,
            self._expressive_pause,
        )
        text = apply_index_pronunciations(text, self._pronunciations, self._pronunciation_rules)
        notes = [*legacy_notes, *cue_notes]
        voice_reference = self._voice_reference(job) if job.voice is not None else None
        speed = self._base_speed * (performance.speed or 1.0)
        if job.arbitrary_emotion is not None:
            emotion_vector = None
            notes.append(FeatureAdaptation("arbitrary_emotion", AdaptationFidelity.APPROXIMATED,
                                          "resolve the acting description through local QwenEmotion at synthesis time"))
        elif job.emotion not in self._emotion_vectors:
            raise ValueError(f"No prepared IndexTTS emotion mapping for {job.emotion!r}")
        else:
            emotion_vector = list(self._emotion_vectors[job.emotion])
        if performance.energy is not None and emotion_vector is not None:
            scale = 1.0 + performance.energy * 0.5
            emotion_vector = [value * scale for value in emotion_vector]
            total = sum(emotion_vector)
            if total > _MAX_EMOTION_SUM:
                emotion_vector = [value * _MAX_EMOTION_SUM / total for value in emotion_vector]
            emotion_vector = [round(value, 6) for value in emotion_vector]
            notes.append(
                FeatureAdaptation(
                    "performance.energy",
                    AdaptationFidelity.APPROXIMATED,
                    "scaled the IndexTTS emotion-vector strength",
                )
            )
        for field, value in {
            "direction": performance.direction,
            "pitch_semitones": performance.pitch_semitones,
            "volume_gain_db": performance.volume_gain_db,
            "brightness": performance.brightness,
            "clarity": performance.clarity,
            "breathiness": performance.breathiness,
            "vocal_mode": performance.vocal_mode,
        }.items():
            if value is not None:
                notes.append(
                    FeatureAdaptation(
                        f"performance.{field}",
                        AdaptationFidelity.DROPPED,
                        "the current IndexTTS worker exposes no matching control",
                    )
                )
        for rule in self._pronunciation_rules:
            if rule.prosody.has_pitch_contour:
                notes.append(
                    FeatureAdaptation(
                        f"pronunciation.{rule.term}.prosody.units",
                        AdaptationFidelity.DROPPED,
                        "retained ARPABET segment identity and lexical stress, but "
                        "dropped the independent aligned F0 feature because "
                        "IndexTTS exposes no word-local pitch control",
                    )
                )
            if (
                rule.prosody.duration_scale is not None
                or any(
                    unit.duration_scale is not None
                    for unit in rule.prosody.units
                )
            ):
                notes.append(
                    FeatureAdaptation(
                        f"pronunciation.{rule.term}.prosody.duration",
                        AdaptationFidelity.DROPPED,
                        "IndexTTS has no per-word or per-unit duration control",
                    )
                )
        return SpeechAdaptation(
            job_id=job.id,
            dialogue_id=job.dialogue_id,
            backend=self.name,
            text=text,
            emotion=job.emotion,
            parameters={
                "expressive_pause": self._expressive_pause,
                "voice": job.voice,
                "reference_audio": str(voice_reference) if voice_reference else None,
                "emotion_vector": emotion_vector,
                "arbitrary_emotion": job.arbitrary_emotion,
                "duration_factor": 1.0 / speed,
            },
            features=tuple(notes),
        )

    def _voice_reference(self, job: TtsJob) -> Path:
        references = self._references.get(job.character_id, ())
        if len(references) != 1:
            raise ValueError("Voice tags require exactly one default reference audio")
        return resolve_voice_reference(references[0], job.voice)

    def _exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        process = self._ensure_process()
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(
                f"IndexTTS worker stopped unexpectedly (exit {process.poll()})"
            )
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError("IndexTTS worker returned a non-object response")
        return response

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        for required in (self._python, self._source, self._model):
            if not required.exists():
                raise FileNotFoundError(f"IndexTTS dependency is missing: {required}")
        environment = os.environ.copy()
        package_root = str(SOURCE_ROOT)
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            package_root
            if not existing
            else os.pathsep.join((package_root, existing))
        )
        command = [
            str(self._python),
            "-m",
            "lessons_in_cast_core.synthesis.backends.index_tts.worker",
            "--source-root",
            str(self._source),
            "--model-path",
            str(self._model),
            "--language", self._language,
        ]
        if self._use_bf16:
            command.append("--use-bf16")
        self._process = subprocess.Popen(
            command,
            cwd=self._root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        return self._process

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)

    def __enter__(self) -> "IndexTtsSubprocessSynthesizer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
