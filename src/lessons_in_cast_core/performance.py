"""Backend-neutral speech performance intent and adaptation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Mapping


class PerformanceCueKind(StrEnum):
    """Audible events that may occur at a text boundary."""

    PAUSE = "pause"
    LAUGH = "laugh"
    CHUCKLE = "chuckle"
    COUGH = "cough"
    THROAT_CLEAR = "throat_clear"
    GROAN = "groan"
    BREATH = "breath"
    PANT = "pant"
    INHALE = "inhale"
    EXHALE = "exhale"
    GASP = "gasp"
    SNIFF = "sniff"
    SIGH = "sigh"
    SNORT = "snort"
    BURP = "burp"
    LIP_SMACK = "lip_smack"
    HUM = "hum"
    HISS = "hiss"
    HESITATION = "hesitation"
    SNEEZE = "sneeze"
    WHISTLE = "whistle"
    CRY = "cry"
    APPLAUSE = "applause"


class VocalMode(StrEnum):
    NORMAL = "normal"
    WHISPER = "whisper"
    SHOUT = "shout"
    SING = "sing"


@dataclass(frozen=True, slots=True)
class PerformanceCue:
    """One event inserted before ``spoken_text[offset]``."""

    kind: PerformanceCueKind
    offset: int
    duration_seconds: float | None = None
    intensity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "offset": self.offset,
            "duration_seconds": self.duration_seconds,
            "intensity": self.intensity,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PerformanceCue:
        return cls(
            kind=PerformanceCueKind(value["kind"]),
            offset=value["offset"],
            duration_seconds=value.get("duration_seconds"),
            intensity=value.get("intensity"),
        )


@dataclass(frozen=True, slots=True)
class SpeechPerformance:
    """Portable actor direction expressed independently of any TTS vendor.

    Numeric controls use common physical or normalized units. ``speed`` is a
    multiplier, ``pitch_semitones`` is a musical interval,
    ``volume_gain_db`` is relative gain, and the remaining voice qualities are
    normalized values. Backends may approximate unsupported controls and must
    describe that lowering in an adaptation report.
    """

    direction: str | None = None
    vocal_mode: VocalMode | None = None
    speed: float | None = None
    pitch_semitones: float | None = None
    volume_gain_db: float | None = None
    energy: float | None = None
    brightness: float | None = None
    clarity: float | None = None
    breathiness: float | None = None
    cues: tuple[PerformanceCue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "vocal_mode": self.vocal_mode.value if self.vocal_mode else None,
            "speed": self.speed,
            "pitch_semitones": self.pitch_semitones,
            "volume_gain_db": self.volume_gain_db,
            "energy": self.energy,
            "brightness": self.brightness,
            "clarity": self.clarity,
            "breathiness": self.breathiness,
            "cues": [cue.to_dict() for cue in self.cues],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> SpeechPerformance:
        if value is None:
            return cls()
        mode = value.get("vocal_mode")
        return cls(
            direction=value.get("direction"),
            vocal_mode=VocalMode(mode) if mode is not None else None,
            speed=value.get("speed"),
            pitch_semitones=value.get("pitch_semitones"),
            volume_gain_db=value.get("volume_gain_db"),
            energy=value.get("energy"),
            brightness=value.get("brightness"),
            clarity=value.get("clarity"),
            breathiness=value.get("breathiness"),
            cues=tuple(
                PerformanceCue.from_dict(item)
                for item in value.get("cues", ())
            ),
        )


class AdaptationFidelity(StrEnum):
    EXACT = "exact"
    APPROXIMATED = "approximated"
    DROPPED = "dropped"


@dataclass(frozen=True, slots=True)
class FeatureAdaptation:
    feature: str
    fidelity: AdaptationFidelity
    strategy: str

    def to_dict(self) -> dict[str, str]:
        return {
            "feature": self.feature,
            "fidelity": self.fidelity.value,
            "strategy": self.strategy,
        }


@dataclass(frozen=True, slots=True)
class SpeechAdaptation:
    """Inspectable result of compiling a core job for one backend."""

    job_id: str
    dialogue_id: str
    backend: str
    text: str
    emotion: str | None
    parameters: dict[str, Any]
    features: tuple[FeatureAdaptation, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "dialogue_id": self.dialogue_id,
            "backend": self.backend,
            "text": self.text,
            "emotion": self.emotion,
            "parameters": self.parameters,
            "features": [feature.to_dict() for feature in self.features],
        }


_PACE_SPEEDS = {
    "very slow": 0.72,
    "slow": 0.84,
    "measured": 0.9,
    "hesitant": 0.9,
    "gentle": 0.94,
    "natural": 1.0,
    "normal": 1.0,
    "brisk": 1.1,
    "fast": 1.18,
    "rapid": 1.25,
}
_VOLUME_GAINS = {
    "very soft": -9.0,
    "soft": -5.0,
    "gentle": -3.0,
    "quiet": -5.0,
    "normal": 0.0,
    "firm": 1.5,
    "raised": 3.0,
    "loud": 5.0,
}


def resolve_legacy_delivery(
    performance: SpeechPerformance,
    delivery: Mapping[str, str],
) -> tuple[SpeechPerformance, tuple[FeatureAdaptation, ...]]:
    """Recover portable controls from the original free-form delivery map."""

    result = performance
    notes: list[FeatureAdaptation] = []
    if result.speed is None:
        pace = delivery.get("pace", "").strip().lower()
        if pace in _PACE_SPEEDS:
            result = replace(result, speed=_PACE_SPEEDS[pace])
            notes.append(
                FeatureAdaptation(
                    "delivery.pace",
                    AdaptationFidelity.APPROXIMATED,
                    "mapped the legacy pace label to a portable speed multiplier",
                )
            )
    if result.volume_gain_db is None:
        volume = delivery.get("volume", "").strip().lower()
        if volume in _VOLUME_GAINS:
            result = replace(result, volume_gain_db=_VOLUME_GAINS[volume])
            notes.append(
                FeatureAdaptation(
                    "delivery.volume",
                    AdaptationFidelity.APPROXIMATED,
                    "mapped the legacy volume label to relative dB gain",
                )
            )
    if result.vocal_mode is None and any(
        "whisper" in str(value).lower()
        for key, value in delivery.items()
        if key in {"whispering", "vocal_effort", "tone", "volume"}
    ):
        result = replace(result, vocal_mode=VocalMode.WHISPER)
        notes.append(
            FeatureAdaptation(
                "delivery.whispering",
                AdaptationFidelity.APPROXIMATED,
                "converted a legacy whisper cue to the portable vocal mode",
            )
        )
    return result, tuple(notes)


def insert_performance_markers(
    text: str,
    markers: list[tuple[int, str]],
) -> str:
    """Insert already-compiled markers without shifting later offsets."""

    result = text
    for offset, marker in sorted(markers, key=lambda item: item[0], reverse=True):
        if not 0 <= offset <= len(text):
            raise ValueError(
                f"Performance cue offset {offset} is outside text length {len(text)}"
            )
        result = result[:offset] + marker + result[offset:]
    return result


def punctuation_for_cue(cue: PerformanceCue) -> str:
    """Provide a conservative fallback for a backend without event tags."""

    if cue.kind is PerformanceCueKind.PAUSE:
        duration = cue.duration_seconds or 0.25
        if duration <= 0.18:
            return ", "
        if duration <= 0.7:
            return "... "
        return ". "
    if cue.kind in {
        PerformanceCueKind.BREATH,
        PerformanceCueKind.INHALE,
        PerformanceCueKind.EXHALE,
        PerformanceCueKind.PANT,
    }:
        return ", "
    return "... "


def approximate_cues_with_punctuation(
    text: str,
    cues: tuple[PerformanceCue, ...],
) -> tuple[str, tuple[FeatureAdaptation, ...]]:
    """Lower events into punctuation understood by most text frontends."""

    markers: list[tuple[int, str]] = []
    notes: list[FeatureAdaptation] = []
    for index, cue in enumerate(cues):
        marker = punctuation_for_cue(cue)
        markers.append((cue.offset, marker))
        notes.append(
            FeatureAdaptation(
                f"performance.cues[{index}]",
                AdaptationFidelity.APPROXIMATED,
                f"rendered {cue.kind.value} as punctuation {marker.strip()!r}",
            )
        )
    return insert_performance_markers(text, markers), tuple(notes)
