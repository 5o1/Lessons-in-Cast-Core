"""Synchronous HTTP adapter for MiniMax Speech 2.x models."""

from __future__ import annotations

import binascii
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from ....config import AudioConfig
from ....performance import (
    AdaptationFidelity,
    FeatureAdaptation,
    PerformanceCue,
    PerformanceCueKind,
    SpeechAdaptation,
    VocalMode,
    insert_performance_markers,
    punctuation_for_cue,
    resolve_legacy_delivery,
)
from ....pronunciations import (
    LexicalPitchPoint,
    SelectedPronunciation,
    pronunciation_matches,
)
from ...types import TtsJob
from .emotion_lowering import lower_arbitrary_job, validate_rules


_MODELS = {
    "speech-2.8-hd",
    "speech-2.8-turbo",
    "speech-2.6-hd",
    "speech-2.6-turbo",
    "speech-02-hd",
    "speech-02-turbo",
    "speech-01-hd",
    "speech-01-turbo",
}
_SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100)
_MINIMAX_TAGS = {
    PerformanceCueKind.LAUGH: "laughs",
    PerformanceCueKind.CHUCKLE: "chuckle",
    PerformanceCueKind.COUGH: "coughs",
    PerformanceCueKind.THROAT_CLEAR: "clear-throat",
    PerformanceCueKind.GROAN: "groans",
    PerformanceCueKind.BREATH: "breath",
    PerformanceCueKind.PANT: "pant",
    PerformanceCueKind.INHALE: "inhale",
    PerformanceCueKind.EXHALE: "exhale",
    PerformanceCueKind.GASP: "gasps",
    PerformanceCueKind.SNIFF: "sniffs",
    PerformanceCueKind.SIGH: "sighs",
    PerformanceCueKind.SNORT: "snorts",
    PerformanceCueKind.BURP: "burps",
    PerformanceCueKind.LIP_SMACK: "lip-smacking",
    PerformanceCueKind.HUM: "humming",
    PerformanceCueKind.HISS: "hissing",
    PerformanceCueKind.HESITATION: "emm",
    PerformanceCueKind.SNEEZE: "sneezes",
}
_EMOTIONS = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "afraid": "fearful",
    "disgusted": "disgusted",
    "surprised": "surprised",
    "calm": "calm",
    "neutral": "calm",
}
_COMPOSITE_EMOTIONS = {
    "tearful_resolve": "sad",
    "restrained_grief": "sad",
    "emotional_breakdown": "fearful",
    "tearful_remorse": "sad",
    "playful_affection": "happy",
    "guarded_composure": "calm",
    "firm_boundary": "angry",
    "cheerful_pressure": "happy",
    "tentative_hope": "calm",
    "possessive_care": "calm",
    "affectionate": "happy",
    "excited": "surprised",
    "embarrassed": "fearful",
    "confused": "surprised",
    "sarcastic": "disgusted",
    "distressed": "sad",
}
_RETRYABLE_STATUS_CODES = {1001, 1002, 1039}
_IPA_TONE_LEVELS = (
    (-4.0, "˩"),
    (-2.0, "˨"),
    (2.0, "˧"),
    (4.0, "˦"),
    (float("inf"), "˥"),
)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def _nearest_sample_rate(value: int) -> int:
    return min(_SAMPLE_RATES, key=lambda candidate: abs(candidate - value))


def _ipa_pitch_contour(contour: tuple[LexicalPitchPoint, ...]) -> str:
    """Quantize one core F0 contour to the IPA levels MiniMax accepts."""

    symbols: list[str] = []
    for point in contour:
        symbol = next(
            marker
            for upper_bound, marker in _IPA_TONE_LEVELS
            if point.semitones <= upper_bound
        )
        if not symbols or symbols[-1] != symbol:
            symbols.append(symbol)
    return "".join(symbols)


def _minimax_pronunciation(
    rule: SelectedPronunciation,
) -> tuple[str, tuple[FeatureAdaptation, ...]]:
    value = rule.pronunciation
    notes: list[FeatureAdaptation] = []
    if rule.system == "ipa":
        aligned_ipa = tuple(
            unit.for_system("ipa") for unit in rule.prosody.units
        )
        if rule.prosody.units and all(aligned_ipa):
            compiled_units: list[str] = []
            for ipa_unit, prosody_unit in zip(
                aligned_ipa,
                rule.prosody.units,
            ):
                assert ipa_unit is not None
                contour = _ipa_pitch_contour(prosody_unit.pitch_contour)
                duration = prosody_unit.duration_scale
                length = "ː" if duration is not None and duration > 1.15 else ""
                compiled_units.append(f"{ipa_unit}{length}{contour}")
            value = ".".join(compiled_units)
            if rule.prosody.has_pitch_contour:
                notes.append(
                    FeatureAdaptation(
                        f"pronunciation.{rule.term}.prosody.units",
                        AdaptationFidelity.APPROXIMATED,
                        "quantized each aligned unit's F0 targets to IPA tone letters; "
                        "the core contour remains more precise than MiniMax input",
                    )
                )
            if any(
                unit.duration_scale is not None
                for unit in rule.prosody.units
            ):
                notes.append(
                    FeatureAdaptation(
                        f"pronunciation.{rule.term}.prosody.unit_duration",
                        AdaptationFidelity.APPROXIMATED,
                        "quantized unit durations to IPA length marks",
                    )
                )
        elif rule.prosody.units:
            notes.append(
                FeatureAdaptation(
                    f"pronunciation.{rule.term}.prosody.units",
                    AdaptationFidelity.DROPPED,
                    "one or more lexical units have no aligned IPA form",
                )
            )
        value = f"({value})"
        if (
            rule.prosody.duration_scale is not None
            and rule.prosody.duration_scale > 1.15
        ):
            value = value[:-1] + "ː)"
            notes.append(
                FeatureAdaptation(
                    f"pronunciation.{rule.term}.duration_scale",
                    AdaptationFidelity.APPROXIMATED,
                    "represented a longer lexical duration with an IPA length mark",
                )
            )
        elif rule.prosody.duration_scale is not None:
            notes.append(
                FeatureAdaptation(
                    f"pronunciation.{rule.term}.duration_scale",
                    AdaptationFidelity.DROPPED,
                    "MiniMax has no precise per-word duration control",
                )
            )
    elif rule.prosody.has_pitch_contour:
        notes.append(
            FeatureAdaptation(
                f"pronunciation.{rule.term}.prosody.units",
                AdaptationFidelity.DROPPED,
                f"the selected {rule.system} representation cannot carry aligned F0 targets",
            )
        )
    if rule.case_sensitive or not rule.whole_word:
        notes.append(
            FeatureAdaptation(
                f"pronunciation.{rule.term}.matching",
                AdaptationFidelity.APPROXIMATED,
                "MiniMax pronunciation_dict does not expose match flags",
            )
        )
    return f"{rule.term}/{value}", tuple(notes)


class MiniMaxSpeechHttpSynthesizer:
    """Generate WAV intermediates through MiniMax's non-streaming T2A API."""

    def __init__(
        self,
        *,
        model: str,
        voice_id: str,
        audio_config: AudioConfig,
        pronunciations: Iterable[SelectedPronunciation] = (),
        endpoint: str = "https://api.minimax.io/v1/t2a_v2",
        api_key: str | None = None,
        api_key_environment: str = "MINIMAX_API_KEY",
        language_boost: str = "auto",
        text_normalization: bool = True,
        base_speed: float = 1.0,
        base_volume: float = 1.0,
        base_pitch_semitones: int = 0,
        voice_brightness: float = 0.0,
        voice_energy: float = 0.0,
        voice_clarity: float = 0.0,
        sound_effect: str | None = None,
        timeout_seconds: float = 300.0,
        maximum_retries: int = 4,
        retry_backoff_seconds: float = 2.0,
        opener: Any | None = None,
        arbitrary_emotions: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if model not in _MODELS:
            raise ValueError(f"Unsupported MiniMax speech model: {model!r}")
        if not voice_id:
            raise ValueError("MiniMax voice_id cannot be empty")
        if audio_config.intermediate_format.lstrip(".").lower() != "wav":
            raise ValueError("MiniMax adapter requires WAV pipeline intermediates")
        if audio_config.sample_width != 2 or audio_config.channels not in {1, 2}:
            raise ValueError("MiniMax adapter requires 16-bit mono or stereo output")
        self._model = model
        self._voice_id = voice_id
        self._audio = audio_config
        self._pronunciations = tuple(pronunciations)
        self._endpoint = endpoint
        self._api_key = api_key
        self._api_key_environment = api_key_environment
        self._language_boost = language_boost
        self._text_normalization = text_normalization
        self._base_speed = base_speed
        self._base_volume = base_volume
        self._base_pitch = base_pitch_semitones
        self._voice_brightness = voice_brightness
        self._voice_energy = voice_energy
        self._voice_clarity = voice_clarity
        self._sound_effect = sound_effect
        self._timeout = timeout_seconds
        self._maximum_retries = maximum_retries
        self._retry_backoff = retry_backoff_seconds
        self._opener = opener or urllib.request.build_opener()
        self._arbitrary_emotions = validate_rules(arbitrary_emotions if arbitrary_emotions is not None else {})

    @property
    def name(self) -> str:
        return "minimax-speech-http-v2"

    @property
    def configuration(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "model": self._model,
            "voice_id": self._voice_id,
            "endpoint": self._endpoint,
            "api_key_environment": self._api_key_environment,
            "language_boost": self._language_boost,
            "text_normalization": self._text_normalization,
            "base_speed": self._base_speed,
            "base_volume": self._base_volume,
            "base_pitch_semitones": self._base_pitch,
            "voice_brightness": self._voice_brightness,
            "voice_energy": self._voice_energy,
            "voice_clarity": self._voice_clarity,
            "sound_effect": self._sound_effect,
            "timeout_seconds": self._timeout,
            "maximum_retries": self._maximum_retries,
            "retry_backoff_seconds": self._retry_backoff,
            "pronunciations": [rule.to_dict() for rule in self._pronunciations],
            "arbitrary_emotions": validate_rules(self._arbitrary_emotions),
            "arbitrary_emotion_lowering_version": 1,
        }

    def _compile_cues(
        self,
        text: str,
        cues: tuple[PerformanceCue, ...],
    ) -> tuple[str, tuple[FeatureAdaptation, ...]]:
        markers: list[tuple[int, str]] = []
        notes: list[FeatureAdaptation] = []
        supports_tags = self._model.startswith("speech-2.8-")
        previous_pause_offset: int | None = None
        for index, cue in enumerate(cues):
            if cue.kind is PerformanceCueKind.PAUSE:
                if (
                    cue.offset in {0, len(text)}
                    or cue.offset == previous_pause_offset
                ):
                    marker = punctuation_for_cue(cue)
                    markers.append((cue.offset, marker))
                    notes.append(
                        FeatureAdaptation(
                            f"performance.cues[{index}]",
                            AdaptationFidelity.APPROXIMATED,
                            f"rendered an invalid boundary/consecutive pause as {marker.strip()!r}",
                        )
                    )
                    continue
                duration = _clamp(cue.duration_seconds or 0.25, 0.01, 99.99)
                markers.append((cue.offset, f"<#%0.2f#>" % duration))
                previous_pause_offset = cue.offset
                fidelity = (
                    AdaptationFidelity.EXACT
                    if duration == cue.duration_seconds
                    else AdaptationFidelity.APPROXIMATED
                )
                notes.append(
                    FeatureAdaptation(
                        f"performance.cues[{index}]",
                        fidelity,
                        "compiled the pause to a MiniMax duration marker",
                    )
                )
                continue
            tag = _MINIMAX_TAGS.get(cue.kind) if supports_tags else None
            if tag is None:
                marker = punctuation_for_cue(cue)
                markers.append((cue.offset, marker))
                notes.append(
                    FeatureAdaptation(
                        f"performance.cues[{index}]",
                        AdaptationFidelity.APPROXIMATED,
                        f"rendered unsupported {cue.kind.value} as {marker.strip()!r}",
                    )
                )
                continue
            markers.append((cue.offset, f"({tag})"))
            notes.append(
                FeatureAdaptation(
                    f"performance.cues[{index}]",
                    AdaptationFidelity.EXACT,
                    f"compiled {cue.kind.value} to MiniMax sound tag ({tag})",
                )
            )
            if cue.intensity is not None or cue.duration_seconds is not None:
                notes.append(
                    FeatureAdaptation(
                        f"performance.cues[{index}].shape",
                        AdaptationFidelity.DROPPED,
                        "MiniMax sound tags do not expose event intensity or duration",
                    )
                )
        compiled = insert_performance_markers(text, markers)
        return compiled, tuple(notes)

    def adapt(self, job: TtsJob) -> SpeechAdaptation:
        if job.arbitrary_emotion is not None:
            lowered, note = lower_arbitrary_job(job, self._arbitrary_emotions)
            adaptation = self.adapt(lowered)
            return replace(adaptation, features=(note, *adaptation.features))
        if job.segments:
            from ...segmentation import adapt_segments
            return adapt_segments(job, self.adapt)
        if job.voice is not None:
            raise NotImplementedError("MiniMax Speech uses remote voice IDs; local reference voice tags require a reference-capable backend")
        performance, legacy_notes = resolve_legacy_delivery(
            job.performance,
            job.delivery,
        )
        text, cue_notes = self._compile_cues(job.text, performance.cues)
        notes = [*legacy_notes, *cue_notes]

        speed = self._base_speed * (performance.speed or 1.0)
        clamped_speed = _clamp(speed, 0.5, 2.0)
        if speed != clamped_speed:
            notes.append(
                FeatureAdaptation(
                    "performance.speed",
                    AdaptationFidelity.APPROXIMATED,
                    f"clamped speed {speed:.3f} to MiniMax range 0.5..2.0",
                )
            )
        pitch = self._base_pitch + (performance.pitch_semitones or 0.0)
        rounded_pitch = round(_clamp(pitch, -12, 12))
        if pitch != rounded_pitch:
            notes.append(
                FeatureAdaptation(
                    "performance.pitch_semitones",
                    AdaptationFidelity.APPROXIMATED,
                    "rounded or clamped pitch to MiniMax integer semitones",
                )
            )
        gain = math.pow(10.0, (performance.volume_gain_db or 0.0) / 20.0)
        volume = _clamp(self._base_volume * gain, 0.01, 10.0)

        emotion = _EMOTIONS.get(job.emotion)
        if emotion is None:
            if job.emotion not in _COMPOSITE_EMOTIONS:
                raise ValueError(f"No MiniMax emotion mapping for {job.emotion!r}")
            emotion = _COMPOSITE_EMOTIONS[job.emotion]
            notes.append(
                FeatureAdaptation(
                    "emotion",
                    AdaptationFidelity.APPROXIMATED,
                    f"mapped core emotion {job.emotion!r} to MiniMax {emotion!r}",
                )
            )

        brightness = _clamp(
            self._voice_brightness + (performance.brightness or 0.0), -1, 1
        )
        energy = _clamp(
            self._voice_energy + (performance.energy or 0.0), -1, 1
        )
        clarity = _clamp(
            self._voice_clarity + (performance.clarity or 0.0), -1, 1
        )
        if performance.brightness is not None:
            notes.append(
                FeatureAdaptation(
                    "performance.brightness",
                    AdaptationFidelity.APPROXIMATED,
                    "mapped brightness to MiniMax Deepen/Brighten",
                )
            )
        if performance.energy is not None:
            notes.append(
                FeatureAdaptation(
                    "performance.energy",
                    AdaptationFidelity.APPROXIMATED,
                    "mapped positive energy to MiniMax Stronger/Softer",
                )
            )
        if performance.clarity is not None:
            notes.append(
                FeatureAdaptation(
                    "performance.clarity",
                    AdaptationFidelity.APPROXIMATED,
                    "mapped clarity to MiniMax Nasal/Crisp",
                )
            )
        if performance.direction:
            notes.append(
                FeatureAdaptation(
                    "performance.direction",
                    AdaptationFidelity.DROPPED,
                    "MiniMax T2A has no per-utterance free-form direction field",
                )
            )
        if performance.breathiness is not None:
            energy = _clamp(energy - performance.breathiness * 0.35, -1, 1)
            volume = _clamp(volume * (1 - performance.breathiness * 0.2), 0.01, 10)
            notes.append(
                FeatureAdaptation(
                    "performance.breathiness",
                    AdaptationFidelity.APPROXIMATED,
                    "softened energy and volume because no breathiness control exists",
                )
            )
        if performance.vocal_mode is VocalMode.WHISPER:
            if self._model.startswith("speech-2.6-"):
                emotion = "whisper"
                notes.append(
                    FeatureAdaptation(
                        "performance.vocal_mode",
                        AdaptationFidelity.EXACT,
                        "used MiniMax speech-2.6 whisper emotion",
                    )
                )
            else:
                energy = min(energy, 0.45)
                volume = min(volume, self._base_volume * 0.65)
                notes.append(
                    FeatureAdaptation(
                        "performance.vocal_mode",
                        AdaptationFidelity.APPROXIMATED,
                        "Speech 2.8 lacks whisper mode; softened energy and volume",
                    )
                )
        elif performance.vocal_mode in {VocalMode.SHOUT, VocalMode.SING}:
            strategy = (
                "raised energy and volume"
                if performance.vocal_mode is VocalMode.SHOUT
                else "retained text; MiniMax T2A has no singing mode"
            )
            if performance.vocal_mode is VocalMode.SHOUT:
                energy = max(energy, 0.6)
                volume = min(volume * 1.25, 10)
            notes.append(
                FeatureAdaptation(
                    "performance.vocal_mode",
                    AdaptationFidelity.APPROXIMATED
                    if performance.vocal_mode is VocalMode.SHOUT
                    else AdaptationFidelity.DROPPED,
                    strategy,
                )
            )

        pronunciation_tone: list[str] = []
        for rule in self._pronunciations:
            if not pronunciation_matches(job.text, rule):
                continue
            compiled, pronunciation_notes = _minimax_pronunciation(rule)
            pronunciation_tone.append(compiled)
            notes.extend(pronunciation_notes)
        voice_modify: dict[str, Any] = {
            "pitch": round(brightness * 100),
            "intensity": round(-energy * 100),
            "timbre": round(clarity * 100),
        }
        if self._sound_effect is not None:
            voice_modify["sound_effects"] = self._sound_effect
        voice_setting = {
            "voice_id": self._voice_id,
            "speed": round(clamped_speed, 4),
            "vol": round(volume, 4),
            "pitch": rounded_pitch,
            "emotion": emotion,
            "text_normalization": self._text_normalization,
        }
        parameters: dict[str, Any] = {
            "model": self._model,
            "language_boost": self._language_boost,
            "voice_setting": voice_setting,
            "voice_modify": voice_modify,
            "audio_setting": {
                "sample_rate": _nearest_sample_rate(self._audio.sample_rate),
                "format": "wav",
                "channel": self._audio.channels,
            },
        }
        if pronunciation_tone:
            parameters["pronunciation_dict"] = {"tone": pronunciation_tone}
        return SpeechAdaptation(
            job_id=job.id,
            dialogue_id=job.dialogue_id,
            backend=self.name,
            text=text,
            emotion=emotion,
            parameters=parameters,
            features=tuple(notes),
        )

    def build_payload(self, job: TtsJob) -> dict[str, Any]:
        if job.segments:
            raise ValueError("Split inline speech before building a MiniMax payload")
        adaptation = self.adapt(job)
        return {
            "model": self._model,
            "text": adaptation.text,
            "stream": False,
            "output_format": "hex",
            **adaptation.parameters,
        }

    def synthesize(self, job: TtsJob, artifact_root: Path) -> Path:
        destination = (artifact_root / job.output_path).resolve()
        if destination.is_file():
            return destination
        if job.segments:
            from ...segmentation import synthesize_segments
            self.adapt(job)
            return synthesize_segments(job, artifact_root, self.synthesize)
        api_key = self._api_key or os.environ.get(self._api_key_environment)
        if not api_key:
            raise RuntimeError(
                f"MiniMax API key is missing; set {self._api_key_environment}"
            )
        payload = self.build_payload(job)
        # Persist the provider response before any local conversion. Retrying a
        # failed conversion must not buy another generation of the same take.
        cache = destination.with_suffix(".minimax")
        cache.mkdir(parents=True, exist_ok=True)
        response_path = cache / "response.json"
        identity = {"endpoint": self._endpoint, "payload": payload}
        if response_path.exists():
            cached = json.loads(response_path.read_text(encoding="utf-8"))
            if cached.get("request") != identity:
                raise ValueError("MiniMax response cache inputs changed; use a new output path")
            response = cached["response"]
        else:
            response = self._request(payload, api_key, job.dialogue_id)
            # Store only successful responses as reusable audio. Provider errors
            # remain inspectable but do not permanently poison a funding retry.
            target = response_path if (response.get("base_resp") or {}).get("status_code") == 0 else cache / "error.json"
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                staged = Path(temporary) / "response.json"
                staged.write_text(json.dumps({"request": identity, "response": response}, ensure_ascii=False) + "\n", encoding="utf-8")
                staged.replace(target)
        base = response.get("base_resp") or {}
        status_code = base.get("status_code")
        if status_code != 0:
            raise RuntimeError(
                f"MiniMax rejected {job.dialogue_id}: status {status_code}: "
                f"{base.get('status_msg', 'unknown error')}"
            )
        data = response.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("audio"), str):
            raise RuntimeError(
                f"MiniMax returned no audio for {job.dialogue_id}; "
                f"trace_id={response.get('trace_id')}"
            )
        try:
            audio = bytes.fromhex(data["audio"])
        except (ValueError, binascii.Error) as exc:
            raise RuntimeError(
                f"MiniMax returned invalid hex audio for {job.dialogue_id}"
            ) from exc
        raw = cache / "source.wav"
        raw.write_bytes(audio)
        # Publish only a complete converted WAV, retaining the original on error.
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            normalized = Path(directory) / "normalized.wav"
            self._normalize_wav(raw, normalized, job.dialogue_id)
            normalized.replace(destination)
        return destination

    def _request(
        self,
        payload: dict[str, Any],
        api_key: str,
        dialogue_id: str,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            self._endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(self._maximum_retries + 1):
            try:
                with self._opener.open(request, timeout=self._timeout) as source:
                    response = json.loads(source.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429 or exc.code >= 500:
                    if attempt < self._maximum_retries:
                        time.sleep(self._retry_backoff * (2**attempt))
                        continue
                raise RuntimeError(
                    f"MiniMax request failed for {dialogue_id}: HTTP "
                    f"{exc.code}: {detail}"
                ) from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt < self._maximum_retries:
                    time.sleep(self._retry_backoff * (2**attempt))
                    continue
                raise RuntimeError(
                    f"MiniMax request failed for {dialogue_id}: {exc}"
                ) from exc
            if not isinstance(response, dict):
                raise RuntimeError("MiniMax returned a non-object response")
            status = (response.get("base_resp") or {}).get("status_code")
            if status in _RETRYABLE_STATUS_CODES and attempt < self._maximum_retries:
                time.sleep(self._retry_backoff * (2**attempt))
                continue
            return response
        raise AssertionError("unreachable MiniMax retry loop")

    def _normalize_wav(
        self,
        source: Path,
        destination: Path,
        dialogue_id: str,
    ) -> None:
        try:
            with wave.open(str(source), "rb") as audio:
                valid = audio.getnframes() > 0
                matching = (
                    audio.getframerate() == self._audio.sample_rate
                    and audio.getnchannels() == self._audio.channels
                    and audio.getsampwidth() == self._audio.sample_width
                )
        except (OSError, EOFError, wave.Error) as exc:
            raise RuntimeError(
                f"MiniMax returned invalid WAV for {dialogue_id}"
            ) from exc
        if not valid:
            raise RuntimeError(f"MiniMax returned empty WAV for {dialogue_id}")
        if matching:
            shutil.copy2(source, destination)
            return
        temporary = destination.with_suffix(".normalizing.wav")
        command = [
            self._audio.ffmpeg_executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ac",
            str(self._audio.channels),
            "-ar",
            str(self._audio.sample_rate),
            "-c:a",
            "pcm_s16le",
            str(temporary),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        if completed.returncode:
            temporary.unlink(missing_ok=True)
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"MiniMax WAV normalization failed for {dialogue_id}: {detail}"
            )
        temporary.replace(destination)
