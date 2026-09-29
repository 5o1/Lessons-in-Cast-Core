"""Backend-neutral proper-noun pronunciation configuration."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConfigurationError, find_repository_root


_SYSTEM_ID = re.compile(r"[a-z][a-z0-9_-]*")


@dataclass(frozen=True, slots=True)
class LexicalPitchPoint:
    """Relative F0 target at one normalized position inside a prosodic unit."""

    position: float
    semitones: float

    def to_dict(self) -> dict[str, float]:
        return {"position": self.position, "semitones": self.semitones}


@dataclass(frozen=True, slots=True)
class LexicalProsody:
    """Pronunciation-aligned lexical F0 and timing instructions.

    Pitch is expressed in semitones relative to the local phrase baseline, so
    the contour follows the speaker's register instead of forcing an absolute
    frequency. Each unit has its own 0..1 time axis; this prevents targets from
    drifting when neighbouring syllables or phonemes have different lengths.
    """

    unit_kind: str = "syllable"
    pitch_reference: str = "local_baseline"
    interpolation: str = "linear"
    units: tuple["LexicalProsodyUnit", ...] = ()
    duration_scale: float | None = None

    @property
    def has_pitch_contour(self) -> bool:
        return any(unit.pitch_contour for unit in self.units)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_kind": self.unit_kind,
            "pitch_reference": self.pitch_reference,
            "interpolation": self.interpolation,
            "units": [unit.to_dict() for unit in self.units],
            "duration_scale": self.duration_scale,
        }


@dataclass(frozen=True, slots=True)
class LexicalProsodyUnit:
    """One aligned syllable, mora, or phoneme in a lexical contour."""

    label: str
    systems: tuple[tuple[str, str], ...] = ()
    pitch_contour: tuple[LexicalPitchPoint, ...] = ()
    duration_scale: float | None = None

    def for_system(self, system: str) -> str | None:
        return dict(self.systems).get(system)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "systems": dict(self.systems),
            "pitch_contour": [point.to_dict() for point in self.pitch_contour],
            "duration_scale": self.duration_scale,
        }


@dataclass(frozen=True, slots=True)
class ProperNounPronunciation:
    """Pronunciations available for one exact proper noun."""

    term: str
    systems: tuple[tuple[str, str], ...]
    language: str | None = None
    case_sensitive: bool = False
    whole_word: bool = True
    prosody: LexicalProsody = LexicalProsody()

    def for_system(self, system: str) -> str | None:
        return dict(self.systems).get(system)

    def to_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "language": self.language,
            "case_sensitive": self.case_sensitive,
            "whole_word": self.whole_word,
            "systems": dict(self.systems),
            "prosody": self.prosody.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SelectedPronunciation:
    """One lexicon entry lowered to a representation a backend accepts."""

    term: str
    pronunciation: str
    system: str
    preferred_system: str
    language: str | None
    case_sensitive: bool
    whole_word: bool
    prosody: LexicalProsody

    @property
    def exact(self) -> bool:
        return self.system == self.preferred_system

    def to_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "pronunciation": self.pronunciation,
            "system": self.system,
            "preferred_system": self.preferred_system,
            "exact": self.exact,
            "language": self.language,
            "case_sensitive": self.case_sensitive,
            "whole_word": self.whole_word,
            "prosody": self.prosody.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PronunciationLexicon:
    """Validated proper nouns keyed by backend pronunciation system."""

    entries: tuple[ProperNounPronunciation, ...]

    def for_system(self, system: str) -> dict[str, str]:
        if not _SYSTEM_ID.fullmatch(system):
            raise ValueError(f"Invalid pronunciation system: {system!r}")
        result: dict[str, str] = {}
        for entry in self.entries:
            pronunciation = entry.for_system(system)
            if pronunciation is not None:
                result[entry.term] = pronunciation
        return result

    def select(
        self,
        systems: tuple[str, ...],
        *,
        language: str | None = None,
    ) -> tuple[SelectedPronunciation, ...]:
        """Choose the first supported representation for every matching term."""

        if not systems:
            raise ValueError("At least one pronunciation system is required")
        for system in systems:
            if not _SYSTEM_ID.fullmatch(system):
                raise ValueError(f"Invalid pronunciation system: {system!r}")
        selected: list[SelectedPronunciation] = []
        for entry in self.entries:
            if (
                language is not None
                and entry.language is not None
                and entry.language.casefold() != language.casefold()
            ):
                continue
            forms = dict(entry.systems)
            for system in systems:
                pronunciation = forms.get(system)
                if pronunciation is None:
                    continue
                selected.append(
                    SelectedPronunciation(
                        term=entry.term,
                        pronunciation=pronunciation,
                        system=system,
                        preferred_system=systems[0],
                        language=entry.language,
                        case_sensitive=entry.case_sensitive,
                        whole_word=entry.whole_word,
                        prosody=entry.prosody,
                    )
                )
                break
        return tuple(selected)

    def to_dict(self) -> dict[str, Any]:
        return {
            entry.term: dict(entry.systems)
            for entry in self.entries
        }


def pronunciation_matches(text: str, rule: SelectedPronunciation) -> bool:
    """Apply the portable matching contract before backend compilation."""

    escaped = re.escape(rule.term)
    pattern = rf"(?<!\w){escaped}(?!\w)" if rule.whole_word else escaped
    flags = 0 if rule.case_sensitive else re.IGNORECASE
    return re.search(pattern, text, flags=flags) is not None


def _parse_duration_scale(
    value: Any,
    *,
    location: str,
) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0.25 <= float(value) <= 4
    ):
        raise ConfigurationError(f"{location} must be between 0.25 and 4")
    return float(value)


def _parse_pitch_contour(
    value: Any,
    *,
    location: str,
) -> tuple[LexicalPitchPoint, ...]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{location} must be an array")
    points: list[LexicalPitchPoint] = []
    previous_position = -1.0
    for index, raw_point in enumerate(value):
        if not isinstance(raw_point, dict) or set(raw_point) != {
            "position",
            "semitones",
        }:
            raise ConfigurationError(
                f"{location}[{index}] must contain position and semitones"
            )
        position = raw_point["position"]
        semitones = raw_point["semitones"]
        if (
            isinstance(position, bool)
            or not isinstance(position, (int, float))
            or not 0 <= float(position) <= 1
            or float(position) <= previous_position
            or isinstance(semitones, bool)
            or not isinstance(semitones, (int, float))
            or not -24 <= float(semitones) <= 24
        ):
            raise ConfigurationError(
                f"{location}[{index}] is outside the supported range or order"
            )
        previous_position = float(position)
        points.append(LexicalPitchPoint(float(position), float(semitones)))
    if points and (
        len(points) < 2
        or points[0].position != 0.0
        or points[-1].position != 1.0
    ):
        raise ConfigurationError(
            f"{location} must contain at least two targets and explicitly "
            "anchor positions 0.0 and 1.0"
        )
    return tuple(points)


def _parse_lexical_prosody(
    value: Any,
    *,
    location: str,
) -> LexicalProsody:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{location} must be a table")
    unknown = set(value) - {
        "unit_kind",
        "pitch_reference",
        "interpolation",
        "duration_scale",
        "units",
    }
    if unknown:
        raise ConfigurationError(
            f"{location} has unknown fields {sorted(unknown)!r}"
        )
    unit_kind = value.get("unit_kind", "syllable")
    if unit_kind not in {"syllable", "mora", "phoneme"}:
        raise ConfigurationError(
            f"{location}.unit_kind must be syllable, mora, or phoneme"
        )
    pitch_reference = value.get("pitch_reference", "local_baseline")
    if pitch_reference != "local_baseline":
        raise ConfigurationError(
            f"{location}.pitch_reference must be local_baseline"
        )
    interpolation = value.get("interpolation", "linear")
    if interpolation not in {"step", "linear", "smooth"}:
        raise ConfigurationError(
            f"{location}.interpolation must be step, linear, or smooth"
        )
    raw_units = value.get("units", [])
    if not isinstance(raw_units, list):
        raise ConfigurationError(f"{location}.units must be an array")
    units: list[LexicalProsodyUnit] = []
    for index, raw_unit in enumerate(raw_units):
        unit_location = f"{location}.units[{index}]"
        if not isinstance(raw_unit, dict):
            raise ConfigurationError(f"{unit_location} must be a table")
        unknown_unit = set(raw_unit) - {
            "label",
            "forms",
            "pitch_contour",
            "duration_scale",
        }
        if unknown_unit:
            raise ConfigurationError(
                f"{unit_location} has unknown fields {sorted(unknown_unit)!r}"
            )
        label = raw_unit.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ConfigurationError(
                f"{unit_location}.label must be a non-empty string"
            )
        raw_forms = raw_unit.get("forms")
        if not isinstance(raw_forms, dict) or not raw_forms:
            raise ConfigurationError(
                f"{unit_location}.forms must define at least one aligned form"
            )
        systems: list[tuple[str, str]] = []
        for system, form in raw_forms.items():
            if (
                not isinstance(system, str)
                or not _SYSTEM_ID.fullmatch(system)
                or not isinstance(form, str)
                or not form.strip()
            ):
                raise ConfigurationError(
                    f"{unit_location}.forms contains an invalid system or value"
                )
            systems.append((system, form.strip()))
        contour = _parse_pitch_contour(
            raw_unit.get("pitch_contour", []),
            location=f"{unit_location}.pitch_contour",
        )
        units.append(
            LexicalProsodyUnit(
                label=label.strip(),
                systems=tuple(sorted(systems)),
                pitch_contour=contour,
                duration_scale=_parse_duration_scale(
                    raw_unit.get("duration_scale"),
                    location=f"{unit_location}.duration_scale",
                ),
            )
        )
    return LexicalProsody(
        unit_kind=unit_kind,
        pitch_reference=pitch_reference,
        interpolation=interpolation,
        units=tuple(units),
        duration_scale=_parse_duration_scale(
            value.get("duration_scale"),
            location=f"{location}.duration_scale",
        ),
    )


def load_pronunciation_lexicon(
    path: Path = Path("configs/pronunciations.toml"),
    *,
    repository_root: Path | None = None,
) -> PronunciationLexicon:
    """Load the shared proper-noun pronunciation lexicon."""

    root = (repository_root or find_repository_root()).resolve()
    resolved = path if path.is_absolute() else root / path
    try:
        with resolved.open("rb") as source:
            data = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"Unable to load {resolved}: {exc}") from exc

    schema_version = data.get("schema_version")
    if schema_version not in {1, 2}:
        raise ConfigurationError(
            f"{resolved}: schema_version must be 1 or 2"
        )
    raw_entries = data.get("proper_nouns")
    if not isinstance(raw_entries, dict):
        raise ConfigurationError(
            f"{resolved}: [proper_nouns] table is required"
        )

    entries: list[ProperNounPronunciation] = []
    normalized_terms: set[str] = set()
    for term, raw_entry in raw_entries.items():
        if not isinstance(term, str) or not term.strip():
            raise ConfigurationError(
                f"{resolved}: proper-noun terms must be non-empty strings"
            )
        normalized = term.casefold()
        if normalized in normalized_terms:
            raise ConfigurationError(
                f"{resolved}: duplicate case-insensitive term {term!r}"
            )
        normalized_terms.add(normalized)
        if not isinstance(raw_entry, dict) or not raw_entry:
            raise ConfigurationError(
                f"{resolved}: proper_nouns.{term} must define a pronunciation"
            )

        if schema_version == 1:
            raw_systems = raw_entry
            language = None
            case_sensitive = False
            whole_word = True
            prosody = LexicalProsody()
        else:
            raw_systems = raw_entry.get("forms")
            language = raw_entry.get("language")
            case_sensitive = raw_entry.get("case_sensitive", False)
            whole_word = raw_entry.get("whole_word", True)
            raw_prosody = raw_entry.get("prosody", {})
            if language is not None and (
                not isinstance(language, str) or not language.strip()
            ):
                raise ConfigurationError(
                    f"{resolved}: proper_nouns.{term}.language must be a "
                    "non-empty string or omitted"
                )
            if not isinstance(case_sensitive, bool) or not isinstance(
                whole_word, bool
            ):
                raise ConfigurationError(
                    f"{resolved}: proper_nouns.{term} matching flags must be booleans"
                )
            prosody = _parse_lexical_prosody(
                raw_prosody,
                location=f"{resolved}: proper_nouns.{term}.prosody",
            )
        if not isinstance(raw_systems, dict) or not raw_systems:
            raise ConfigurationError(
                f"{resolved}: proper_nouns.{term}.forms must define at least "
                "one pronunciation"
            )

        systems: list[tuple[str, str]] = []
        for system, pronunciation in raw_systems.items():
            if (
                not isinstance(system, str)
                or not _SYSTEM_ID.fullmatch(system)
                or not isinstance(pronunciation, str)
                or not pronunciation.strip()
            ):
                raise ConfigurationError(
                    f"{resolved}: proper_nouns.{term} contains an invalid "
                    "pronunciation system or value"
                )
            systems.append((system, pronunciation.strip()))
        entries.append(
            ProperNounPronunciation(
                term=term,
                systems=tuple(sorted(systems)),
                language=language.strip() if language is not None else None,
                case_sensitive=case_sensitive,
                whole_word=whole_word,
                prosody=prosody,
            )
        )

    entries.sort(key=lambda entry: entry.term.casefold())
    return PronunciationLexicon(tuple(entries))
