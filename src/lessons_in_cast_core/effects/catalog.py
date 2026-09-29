"""Backend-neutral, validated effect presets and ordered chains."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import tomllib
from typing import Any


class EffectError(ValueError):
    """Invalid or unavailable effect processing; never silently drop a filter."""


# Defaults live in core; workspace TOML may override or add named presets.
DEFAULTS: dict[str, dict[str, float | int]] = {
    "fade_in": {"duration_seconds": 0.15},
    "fade_out": {"duration_seconds": 0.25},
    "telephone": {"low_hz": 300.0, "high_hz": 3400.0, "bits": 8},
    "monster": {"semitones": -6.0, "bass_db": 6.0, "gain_db": -6.0},
    "censor_beep": {"start_seconds": 0.0, "end_seconds": -1.0,
                    "duration_seconds": 1.0, "frequency_hz": 1000.0,
                    "level_db": -18.0, "edge_seconds": 0.005},
    "glitch": {"position": 0.35, "chunk_seconds": 0.16, "repeats": 4,
               "gap_seconds": 0.045, "hold_seconds": 0.8,
               "grain_seconds": 0.04, "dropout_seconds": 0.55,
               "gate_on_seconds": 0.065, "gate_off_seconds": 0.055,
               "edge_seconds": 0.004},
}

DESCRIPTIONS = {
    "fade_in": "Fade from silence at the beginning without trimming speech.",
    "fade_out": "Fade to silence at the end without trimming speech.",
    "telephone": "Band-limited, mono, low-resolution telephone voice.",
    "monster": "Lower pitch with bass weight and approximately unchanged tempo.",
    "censor_beep": "Replace speech with a beep, or generate a beep without TTS.",
    "glitch": "Repeat a fragment, hold a grain, stutter with dropouts, then resume.",
}


@dataclass(frozen=True)
class EffectSpec:
    kind: str
    parameters: dict[str, float | int] = field(default_factory=dict)

    def resolved(self) -> dict[str, float | int]:
        if self.kind not in DEFAULTS:
            raise EffectError(f"Unknown effect type: {self.kind!r}")
        unknown = self.parameters.keys() - DEFAULTS[self.kind].keys()
        if unknown:
            raise EffectError(f"{self.kind}: unknown parameters: {sorted(unknown)}")
        values = {**DEFAULTS[self.kind], **self.parameters}
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise EffectError(f"{self.kind}.{key} must be a finite number")
        for key in ("bits", "repeats"):
            if key in values and not isinstance(values[key], int):
                raise EffectError(f"{self.kind}.{key} must be an integer")
        for key, value in values.items():
            if key.endswith("_seconds") and key != "end_seconds" and not 0 <= value <= 120:
                raise EffectError(f"{self.kind}.{key} must be between 0 and 120")
        if self.kind in {"fade_in", "fade_out", "censor_beep"} and values["duration_seconds"] <= 0:
            raise EffectError("duration_seconds must be positive")
        if self.kind == "telephone":
            if not 0 < values["low_hz"] < values["high_hz"] < 4000 or not 2 <= values["bits"] <= 16:
                raise EffectError("telephone requires 0 < low_hz < high_hz < 4000 and 2..16 bits")
        if self.kind == "monster":
            if not -12 <= values["semitones"] <= 0 or not 0 <= values["bass_db"] <= 18 or not -24 <= values["gain_db"] <= 0:
                raise EffectError("monster requires -12..0 semitones, 0..18 bass_db, -24..0 gain_db")
        if self.kind == "censor_beep":
            end = values["end_seconds"]
            if end != -1 and not values["start_seconds"] < end <= 120:
                raise EffectError("end_seconds must be -1 (end of clip) or after start_seconds, at most 120")
            if not 20 <= values["frequency_hz"] <= 20000 or not -60 <= values["level_db"] <= -3:
                raise EffectError("beep requires 20..20000 Hz and -60..-3 dBFS")
        if self.kind == "glitch":
            if not 0 <= values["position"] < 1 or not 1 <= values["repeats"] <= 32:
                raise EffectError("glitch requires 0 <= position < 1 and 1..32 repeats")
            for key in ("chunk_seconds", "grain_seconds", "gate_on_seconds", "gate_off_seconds"):
                if values[key] <= 0:
                    raise EffectError(f"glitch.{key} must be positive")
        return values

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "parameters": self.resolved()}


@dataclass(frozen=True)
class EffectLibrary:
    presets: dict[str, EffectSpec] = field(default_factory=lambda: {key: EffectSpec(key) for key in DEFAULTS})
    chains: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def resolve(self, names: tuple[str, ...]) -> tuple[EffectSpec, ...]:
        result: list[EffectSpec] = []

        def visit(name: str, parents: tuple[str, ...]) -> None:
            if not isinstance(name, str) or not name:
                raise EffectError("Effect names must be non-empty strings")
            if name in parents:
                raise EffectError(f"Cyclic effect chain: {' -> '.join((*parents, name))}")
            if len(parents) > 16 or len(result) >= 64:
                raise EffectError("Effect chain is too large")
            if name in self.presets:
                self.presets[name].resolved()
                result.append(self.presets[name])
            elif name in self.chains:
                if not self.chains[name]:
                    raise EffectError(f"Empty effect chain: {name}")
                for member in self.chains[name]:
                    visit(member, (*parents, name))
            else:
                raise EffectError(f"Unknown effect preset or chain: {name!r}")

        for name in names:
            visit(name, ())
        return tuple(result)

    def to_dict(self) -> dict[str, Any]:
        return {"presets": {key: value.to_dict() for key, value in sorted(self.presets.items())},
                "chains": {key: list(value) for key, value in sorted(self.chains.items())}}


def load_effect_library(path: Path | None = None) -> EffectLibrary:
    if path is None:
        return EffectLibrary()
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise EffectError(f"Cannot load effects configuration {path}: {exc}") from exc
    if data.keys() - {"presets", "chains"}:
        raise EffectError("Effects configuration accepts only presets and chains")
    presets = dict(EffectLibrary().presets)
    raw_presets, raw_chains = data.get("presets", {}), data.get("chains", {})
    if not isinstance(raw_presets, dict) or not isinstance(raw_chains, dict):
        raise EffectError("presets and chains must be TOML tables")
    for name, value in raw_presets.items():
        if not name or not isinstance(value, dict):
            raise EffectError("Each preset must be a named table")
        kind = value.get("type", name)
        if not isinstance(kind, str):
            raise EffectError(f"{name}.type must be a string")
        spec = EffectSpec(kind, {key: item for key, item in value.items() if key != "type"})
        spec.resolved()
        presets[name] = spec
    chains = {}
    for name, members in raw_chains.items():
        if not isinstance(members, list) or not members or not all(isinstance(item, str) for item in members):
            raise EffectError(f"Chain {name!r} must be a non-empty array of names")
        if name in presets:
            raise EffectError(f"Chain {name!r} conflicts with a preset")
        chains[name] = tuple(members)
    library = EffectLibrary(presets, chains)
    for name in (*presets, *chains):
        library.resolve((name,))
    return library
