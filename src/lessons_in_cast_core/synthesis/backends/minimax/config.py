"""Configuration for a MiniMax Speech voice profile."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ....config import ConfigurationError
from .emotion_lowering import validate_rules


_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SOUND_EFFECTS = {
    "spacious_echo",
    "auditorium_echo",
    "lofi_telephone",
    "robotic",
}


@dataclass(frozen=True, slots=True)
class MiniMaxPipelineConfig:
    model_id: str
    voice_id: str
    base_speed: float = 1.0
    base_volume: float = 1.0
    base_pitch_semitones: int = 0
    language_boost: str = "auto"
    text_normalization: bool = True
    endpoint: str = "https://api.minimax.io/v1/t2a_v2"
    api_key_environment: str = "MINIMAX_API_KEY"
    timeout_seconds: float = 300.0
    maximum_retries: int = 4
    retry_backoff_seconds: float = 2.0
    voice_brightness: float = 0.0
    voice_energy: float = 0.0
    voice_clarity: float = 0.0
    sound_effect: str | None = None
    arbitrary_emotions: dict[str, dict[str, Any]] = field(default_factory=dict)


def _table(data: dict[str, Any], name: str, path: Path) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path}: [{name}] table is required")
    return value


def _number(table: dict[str, Any], name: str, default: float) -> float:
    value = table.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(name)
    return float(value)


def load_minimax_pipeline_config(
    path: Path,
    *,
    repository_root: Path,
) -> MiniMaxPipelineConfig:
    resolved = (path if path.is_absolute() else repository_root / path).resolve()
    try:
        with resolved.open("rb") as source:
            data = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"Unable to load {resolved}: {exc}") from exc
    backend = _table(data, "backend", resolved)
    render = _table(data, "render", resolved)
    try:
        model_id = backend["model"]
        voice_id = backend["voice_id"]
        language_boost = backend.get("language_boost", "auto")
        endpoint = backend.get("endpoint", "https://api.minimax.io/v1/t2a_v2")
        api_key_environment = backend.get(
            "api_key_environment", "MINIMAX_API_KEY"
        )
        text_normalization = backend.get("text_normalization", True)
        maximum_retries = backend.get("maximum_retries", 4)
        sound_effect = backend.get("sound_effect")
        base_pitch = render.get("base_pitch_semitones", 0)
        result = MiniMaxPipelineConfig(
            model_id=model_id,
            voice_id=voice_id,
            base_speed=_number(render, "base_speed", 1.0),
            base_volume=_number(render, "base_volume", 1.0),
            base_pitch_semitones=base_pitch,
            language_boost=language_boost,
            text_normalization=text_normalization,
            endpoint=endpoint,
            api_key_environment=api_key_environment,
            timeout_seconds=_number(backend, "timeout_seconds", 300.0),
            maximum_retries=maximum_retries,
            retry_backoff_seconds=_number(
                backend, "retry_backoff_seconds", 2.0
            ),
            voice_brightness=_number(render, "voice_brightness", 0.0),
            voice_energy=_number(render, "voice_energy", 0.0),
            voice_clarity=_number(render, "voice_clarity", 0.0),
            sound_effect=sound_effect,
            arbitrary_emotions=validate_rules(data.get("arbitrary_emotions", {})),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{resolved}: MiniMax profile contains a missing or invalid field"
        ) from exc
    text_values = {
        "backend.model": result.model_id,
        "backend.voice_id": result.voice_id,
        "backend.language_boost": result.language_boost,
        "backend.endpoint": result.endpoint,
        "backend.api_key_environment": result.api_key_environment,
    }
    if any(
        not isinstance(value, str) or not value.strip()
        for value in text_values.values()
    ):
        raise ConfigurationError(
            f"{resolved}: MiniMax identifiers and endpoint must be non-empty strings"
        )
    if not isinstance(result.text_normalization, bool):
        raise ConfigurationError(
            f"{resolved}: backend.text_normalization must be a boolean"
        )
    if isinstance(result.base_pitch_semitones, bool) or not isinstance(
        result.base_pitch_semitones, int
    ):
        raise ConfigurationError(
            f"{resolved}: render.base_pitch_semitones must be an integer"
        )
    if isinstance(result.maximum_retries, bool) or not isinstance(
        result.maximum_retries, int
    ):
        raise ConfigurationError(
            f"{resolved}: backend.maximum_retries must be an integer"
        )
    if not _ENVIRONMENT_NAME.fullmatch(result.api_key_environment):
        raise ConfigurationError(
            f"{resolved}: backend.api_key_environment is not a safe variable name"
        )
    if not 0.5 <= result.base_speed <= 2:
        raise ConfigurationError(f"{resolved}: render.base_speed must be 0.5..2")
    if not 0 < result.base_volume <= 10:
        raise ConfigurationError(f"{resolved}: render.base_volume must be >0..10")
    if not -12 <= result.base_pitch_semitones <= 12:
        raise ConfigurationError(
            f"{resolved}: render.base_pitch_semitones must be -12..12"
        )
    if result.timeout_seconds <= 0 or result.retry_backoff_seconds < 0:
        raise ConfigurationError(
            f"{resolved}: MiniMax timeout must be positive and backoff non-negative"
        )
    if result.maximum_retries < 0:
        raise ConfigurationError(
            f"{resolved}: backend.maximum_retries cannot be negative"
        )
    for name, value in {
        "voice_brightness": result.voice_brightness,
        "voice_energy": result.voice_energy,
        "voice_clarity": result.voice_clarity,
    }.items():
        if not -1 <= value <= 1:
            raise ConfigurationError(f"{resolved}: render.{name} must be -1..1")
    if (
        result.sound_effect is not None
        and result.sound_effect not in _SOUND_EFFECTS
    ):
        raise ConfigurationError(
            f"{resolved}: backend.sound_effect is not supported by MiniMax"
        )
    return result
