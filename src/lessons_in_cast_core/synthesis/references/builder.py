"""Deterministic construction of model-ready voice reference audio."""

from __future__ import annotations

import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SUPPORTED_AUDIO_SUFFIXES = frozenset(
    {".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}
)


class ReferenceBuildError(ValueError):
    """Raised when sample audio cannot produce a valid reference."""


@dataclass(frozen=True, slots=True)
class ReferenceBuildSettings:
    top_db: float = 40.0
    padding_ms: int = 150
    gate_hold_ms: int = 500
    minimum_speech_seconds: float = 3.0
    minimum_duration_seconds: float = 15.0
    trim_silence: bool = True


@dataclass(frozen=True, slots=True)
class ReferenceBuildResult:
    output_path: Path
    selected_paths: tuple[Path, ...]
    source_duration_seconds: float
    voiced_seconds: float
    prepared_duration_seconds: float
    sample_rate: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "selected_paths": [str(path) for path in self.selected_paths],
            "source_duration_seconds": round(self.source_duration_seconds, 6),
            "voiced_seconds": round(self.voiced_seconds, 6),
            "prepared_duration_seconds": round(
                self.prepared_duration_seconds, 6
            ),
            "sample_rate": self.sample_rate,
        }


    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReferenceBuildResult":
        return cls(
            output_path=Path(value["output_path"]),
            selected_paths=tuple(
                Path(path) for path in value["selected_paths"]
            ),
            source_duration_seconds=float(value["source_duration_seconds"]),
            voiced_seconds=float(value["voiced_seconds"]),
            prepared_duration_seconds=float(
                value["prepared_duration_seconds"]
            ),
            sample_rate=int(value["sample_rate"]),
        )


@dataclass(frozen=True, slots=True)
class _Candidate:
    path: Path
    voiced_seconds: float
    duration_seconds: float

    @property
    def voiced_ratio(self) -> float:
        if self.duration_seconds <= 0:
            return 0.0
        return self.voiced_seconds / self.duration_seconds


def _load_sources(sources: Sequence[Path]) -> tuple[Any, int, float]:
    import librosa
    import numpy

    if not sources:
        raise ReferenceBuildError("At least one reference source is required")
    clips = []
    sample_rate: int | None = None
    source_duration = 0.0
    for source in sources:
        path = source.resolve()
        if not path.is_file():
            raise ReferenceBuildError(f"Reference source is missing: {path}")
        clip, clip_rate = librosa.load(path, sr=None, mono=True)
        if clip.size == 0:
            raise ReferenceBuildError(f"Reference source is empty: {path}")
        source_duration += len(clip) / clip_rate
        if sample_rate is None:
            sample_rate = int(clip_rate)
        elif clip_rate != sample_rate:
            clip = librosa.resample(
                clip,
                orig_sr=clip_rate,
                target_sr=sample_rate,
            )
        clips.append(clip)
    assert sample_rate is not None
    return numpy.concatenate(clips), sample_rate, source_duration


def _apply_silence_gate(
    audio: Any,
    sample_rate: int,
    settings: ReferenceBuildSettings,
) -> tuple[Any, float]:
    import librosa
    import numpy

    intervals = librosa.effects.split(audio, top_db=settings.top_db)
    if not len(intervals):
        raise ReferenceBuildError("Reference audio contains no detected speech")
    voiced_seconds = (
        sum(int(end) - int(start) for start, end in intervals) / sample_rate
    )
    if not settings.trim_silence:
        return audio, voiced_seconds

    hold = round(sample_rate * settings.gate_hold_ms / 1000)
    open_intervals: list[list[int]] = []
    for start, end in intervals:
        extended_end = min(len(audio), int(end) + hold)
        if open_intervals and int(start) <= open_intervals[-1][1]:
            open_intervals[-1][1] = max(
                open_intervals[-1][1],
                extended_end,
            )
        else:
            open_intervals.append([int(start), extended_end])

    padding = round(sample_rate * settings.padding_ms / 1000)
    open_intervals[0][0] = max(0, open_intervals[0][0] - padding)
    open_intervals[-1][1] = min(
        len(audio),
        open_intervals[-1][1] + padding,
    )
    prepared = numpy.concatenate(
        [audio[start:end] for start, end in open_intervals]
    )
    return prepared, voiced_seconds


def _prepare(
    sources: Sequence[Path],
    settings: ReferenceBuildSettings,
) -> tuple[Any, int, float, float, float]:
    audio, sample_rate, source_duration = _load_sources(sources)
    prepared, voiced_seconds = _apply_silence_gate(
        audio,
        sample_rate,
        settings,
    )
    prepared_duration = len(prepared) / sample_rate
    return (
        prepared,
        sample_rate,
        source_duration,
        voiced_seconds,
        prepared_duration,
    )


def _validate_result(
    sources: Sequence[Path],
    voiced_seconds: float,
    prepared_duration: float,
    settings: ReferenceBuildSettings,
) -> None:
    if voiced_seconds < settings.minimum_speech_seconds:
        raise ReferenceBuildError(
            f"Reference has only {voiced_seconds:.2f}s of detected speech; "
            f"at least {settings.minimum_speech_seconds:.2f}s is required: "
            f"{tuple(sources)}"
        )
    if prepared_duration < settings.minimum_duration_seconds:
        warnings.warn(
            f"Reference is {prepared_duration:.2f}s after silence-gate "
            f"trimming; at least {settings.minimum_duration_seconds:.2f}s is "
            f"recommended: {tuple(sources)}",
            RuntimeWarning,
            stacklevel=2,
        )


def _write_pcm16_wav(output_path: Path, audio: Any, sample_rate: int) -> Path:
    import soundfile

    destination = output_path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}-",
        suffix=".tmp.wav",
        delete=False,
    ) as temporary_file:
        temporary = Path(temporary_file.name)
    try:
        soundfile.write(
            temporary,
            audio,
            sample_rate,
            subtype="PCM_16",
            format="WAV",
        )
        temporary.replace(destination)
        destination.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def prepare_reference_sources(
    sources: Sequence[Path],
    output_path: Path,
    settings: ReferenceBuildSettings,
) -> ReferenceBuildResult:
    """Prepare an explicitly ordered sequence of source clips."""

    resolved_sources = tuple(path.resolve() for path in sources)
    (
        prepared,
        sample_rate,
        source_duration,
        voiced_seconds,
        prepared_duration,
    ) = _prepare(resolved_sources, settings)
    _validate_result(
        resolved_sources,
        voiced_seconds,
        prepared_duration,
        settings,
    )
    destination = _write_pcm16_wav(output_path, prepared, sample_rate)
    return ReferenceBuildResult(
        output_path=destination,
        selected_paths=resolved_sources,
        source_duration_seconds=source_duration,
        voiced_seconds=voiced_seconds,
        prepared_duration_seconds=prepared_duration,
        sample_rate=sample_rate,
    )


def _inspect_candidate(
    path: Path,
    settings: ReferenceBuildSettings,
) -> _Candidate | None:
    import librosa

    try:
        audio, sample_rate = librosa.load(path, sr=None, mono=True)
    except Exception:
        return None
    if audio.size == 0:
        return None
    intervals = librosa.effects.split(audio, top_db=settings.top_db)
    if not len(intervals):
        return None
    voiced_seconds = (
        sum(int(end) - int(start) for start, end in intervals) / sample_rate
    )
    return _Candidate(
        path=path.resolve(),
        voiced_seconds=voiced_seconds,
        duration_seconds=len(audio) / sample_rate,
    )


def build_reference_from_directory(
    input_directory: Path,
    output_path: Path,
    settings: ReferenceBuildSettings,
) -> ReferenceBuildResult:
    """Select and combine the strongest samples until all constraints pass."""

    directory = input_directory.resolve()
    if not directory.is_dir():
        raise ReferenceBuildError(f"Sample directory is missing: {directory}")
    destination = output_path.resolve()
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
        and path.resolve() != destination
    )
    candidates = [
        candidate
        for path in paths
        if (candidate := _inspect_candidate(path, settings)) is not None
    ]
    candidates.sort(
        key=lambda item: (
            -item.voiced_seconds,
            -item.voiced_ratio,
            str(item.path),
        )
    )
    if not candidates:
        raise ReferenceBuildError(
            f"No usable audio samples were found below {directory}"
        )

    selected: list[Path] = []
    latest: tuple[Any, int, float, float, float] | None = None
    for candidate in candidates:
        selected.append(candidate.path)
        latest = _prepare(selected, settings)
        if (
            latest[3] >= settings.minimum_speech_seconds
            and latest[4] >= settings.minimum_duration_seconds
        ):
            break
    assert latest is not None
    _validate_result(selected, latest[3], latest[4], settings)
    destination = _write_pcm16_wav(destination, latest[0], latest[1])
    return ReferenceBuildResult(
        output_path=destination,
        selected_paths=tuple(selected),
        source_duration_seconds=latest[2],
        voiced_seconds=latest[3],
        prepared_duration_seconds=latest[4],
        sample_rate=latest[1],
    )
