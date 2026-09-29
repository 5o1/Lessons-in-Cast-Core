"""Load and validate configuration for one IndexTTS voice profile."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ....config import ConfigurationError
from ...references.builder import ReferenceBuildSettings


@dataclass(frozen=True, slots=True)
class IndexTtsPipelineConfig:
    """All backend-specific settings owned by an IndexTTS voice pipeline."""

    python_executable: str
    model_id: str
    base_speed: float
    source_root: str
    language: str
    seed: int
    use_bf16: bool
    emotion_alpha: float
    use_random_emotion: bool
    do_sample: bool
    top_p: float
    top_k: int
    temperature: float
    num_beams: int
    repetition_penalty: float
    length_penalty: float
    max_mel_tokens: int
    interval_silence_ms: int
    max_text_tokens_per_segment: int
    text_normalization: bool
    reference_source_directory: str
    reference_scope: str
    reference_sources: tuple[str, ...]
    reference_path: str
    reference_settings: ReferenceBuildSettings
    emotion_vectors_path: str = ""
    expressive_pause: str = "period"


def _load_table(path: Path, name: str) -> dict[str, Any]:
    try:
        with path.open("rb") as source:
            data = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"Unable to load {path}: {exc}") from exc
    table = data.get(name)
    if not isinstance(table, dict):
        raise ConfigurationError(f"{path}: [{name}] table is required")
    return table


def _typed(
    path: Path,
    table_name: str,
    table: dict[str, Any],
    name: str,
    expected: type | tuple[type, ...],
) -> Any:
    value = table.get(name)
    if (
        not isinstance(value, expected)
        or (isinstance(value, bool) and expected is not bool)
    ):
        raise ConfigurationError(
            f"{path}: {table_name}.{name} has an invalid type"
        )
    return value


def load_index_tts_pipeline_config(
    path: Path,
    *,
    repository_root: Path,
) -> IndexTtsPipelineConfig:
    """Load one project-specific IndexTTS pipeline configuration."""

    resolved = path if path.is_absolute() else repository_root / path
    resolved = resolved.resolve()
    backend = _load_table(resolved, "backend")
    expressive_pause = backend.get("expressive_pause", "period")
    if expressive_pause not in ("native", "comma", "period"):
        raise ConfigurationError("backend.expressive_pause must be native, comma or period")
    render = _load_table(resolved, "render")
    reference = _load_table(resolved, "reference")
    emotion_vectors_path = backend.get("emotion_vectors_path", "")
    if not isinstance(emotion_vectors_path, str):
        raise ConfigurationError("backend.emotion_vectors_path must be a repository-relative path")
    if Path(emotion_vectors_path).is_absolute() or ".." in Path(emotion_vectors_path).parts:
        raise ConfigurationError("backend.emotion_vectors_path must be a safe repository-relative path")

    text = lambda table_name, table, name: _typed(
        resolved, table_name, table, name, str
    )
    integer = lambda table_name, table, name: _typed(
        resolved, table_name, table, name, int
    )
    number = lambda table_name, table, name: float(
        _typed(resolved, table_name, table, name, (int, float))
    )
    boolean = lambda table_name, table, name: _typed(
        resolved, table_name, table, name, bool
    )
    raw_sources = reference.get("sources")
    if (
        not isinstance(raw_sources, list)
        or not raw_sources
        or not all(isinstance(value, str) and value for value in raw_sources)
    ):
        raise ConfigurationError(
            f"{resolved}: reference.sources must be a non-empty array of paths"
        )

    result = IndexTtsPipelineConfig(
        expressive_pause=expressive_pause,
        emotion_vectors_path=emotion_vectors_path,
        model_id=text("backend", backend, "model"),
        base_speed=number("render", render, "base_speed"),
        python_executable=text("backend", backend, "python_executable"),
        source_root=text("backend", backend, "source_root"),
        language=text("backend", backend, "language"),
        seed=integer("backend", backend, "seed"),
        use_bf16=boolean("backend", backend, "use_bf16"),
        emotion_alpha=number("backend", backend, "emotion_alpha"),
        use_random_emotion=boolean(
            "backend", backend, "use_random_emotion"
        ),
        do_sample=boolean("backend", backend, "do_sample"),
        top_p=number("backend", backend, "top_p"),
        top_k=integer("backend", backend, "top_k"),
        temperature=number("backend", backend, "temperature"),
        num_beams=integer("backend", backend, "num_beams"),
        repetition_penalty=number(
            "backend", backend, "repetition_penalty"
        ),
        length_penalty=number("backend", backend, "length_penalty"),
        max_mel_tokens=integer("backend", backend, "max_mel_tokens"),
        interval_silence_ms=integer(
            "backend", backend, "interval_silence_ms"
        ),
        max_text_tokens_per_segment=integer(
            "backend", backend, "max_text_tokens_per_segment"
        ),
        text_normalization=boolean(
            "backend", backend, "text_normalization"
        ),
        reference_scope=text("reference", reference, "scope"),
        reference_source_directory=text(
            "reference", reference, "source_directory"
        ),
        reference_sources=tuple(raw_sources),
        reference_path=text("reference", reference, "path"),
        reference_settings=ReferenceBuildSettings(
            top_db=number("reference", reference, "top_db"),
            padding_ms=integer("reference", reference, "padding_ms"),
            gate_hold_ms=integer("reference", reference, "gate_hold_ms"),
            minimum_speech_seconds=number(
                "reference", reference, "minimum_speech_seconds"
            ),
            minimum_duration_seconds=number(
                "reference", reference, "minimum_duration_seconds"
            ),
            trim_silence=boolean(
                "reference", reference, "trim_silence"
            ),
        ),
    )
    if not result.language:
        raise ConfigurationError(f"{resolved}: backend.language cannot be empty")
    if (
        not result.python_executable
        or not result.source_root
        or not result.model_id
    ):
        raise ConfigurationError(
            f"{resolved}: backend executable, source root, and model ID cannot be empty"
        )
    if not 0.0 <= result.emotion_alpha <= 1.0:
        raise ConfigurationError(
            f"{resolved}: backend.emotion_alpha must be between 0 and 1"
        )
    if result.base_speed <= 0:
        raise ConfigurationError(
            f"{resolved}: render.base_speed must be positive"
        )
    if not 0.0 < result.top_p <= 1.0:
        raise ConfigurationError(
            f"{resolved}: backend.top_p must be between 0 and 1"
        )
    if result.top_k < 1 or result.num_beams < 1:
        raise ConfigurationError(
            f"{resolved}: backend top_k and num_beams must be positive"
        )
    if result.temperature <= 0 or result.repetition_penalty <= 0:
        raise ConfigurationError(
            f"{resolved}: backend penalties and temperature must be positive"
        )
    if result.max_mel_tokens < 1 or result.max_text_tokens_per_segment < 1:
        raise ConfigurationError(
            f"{resolved}: backend token limits must be positive"
        )
    if result.interval_silence_ms < 0:
        raise ConfigurationError(
            f"{resolved}: backend.interval_silence_ms cannot be negative"
        )
    settings = result.reference_settings
    if result.reference_scope not in {"profile", "repository"}:
        raise ConfigurationError(
            f"{resolved}: reference.scope must be profile or repository"
        )
    for configured_path in (
        result.reference_source_directory,
        result.reference_path,
        *result.reference_sources,
    ):
        candidate = Path(configured_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ConfigurationError(
                f"{resolved}: reference paths must be safe relative paths"
            )
    if settings.top_db <= 0:
        raise ConfigurationError(
            f"{resolved}: reference.top_db must be positive"
        )
    if settings.padding_ms < 0 or settings.gate_hold_ms < 0:
        raise ConfigurationError(
            f"{resolved}: reference gate timings cannot be negative"
        )
    if settings.minimum_speech_seconds <= 0:
        raise ConfigurationError(
            f"{resolved}: reference.minimum_speech_seconds must be positive"
        )
    if settings.minimum_duration_seconds < 15:
        raise ConfigurationError(
            f"{resolved}: reference.minimum_duration_seconds must be at least 15"
        )
    return result
