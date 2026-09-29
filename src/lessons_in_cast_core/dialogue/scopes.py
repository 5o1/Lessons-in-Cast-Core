"""Named, reviewable dialogue ranges for demos and focused runs."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..config import ConfigurationError, find_repository_root
from .types import DialogueRecord


@dataclass(frozen=True, slots=True)
class SpeakerOverride:
    source_character: str
    target_character: str
    start_identifier: str
    end_identifier: str
    inclusive: bool = True


@dataclass(frozen=True, slots=True)
class DialogueScope:
    name: str
    description: str
    source: str
    start_identifier: str
    end_identifier: str
    inclusive: bool
    context_characters: tuple[str, ...]
    voice_characters: tuple[str, ...]
    context_only_characters: tuple[str, ...]
    speaker_overrides: tuple[SpeakerOverride, ...]

    def apply(self, records: Any) -> tuple[DialogueRecord, ...]:
        """Select this range, apply scoped speaker aliases, and resequence it."""

        selected: list[DialogueRecord] = []
        started = False
        ended = False
        for record in records:
            if record.filename != self.source:
                continue
            if not started and record.identifier == self.start_identifier:
                started = True
            if not started:
                continue
            is_end = record.identifier == self.end_identifier
            if not is_end or self.inclusive:
                selected.append(record)
            if is_end:
                ended = True
                break
        if not started:
            raise ValueError(
                f"Scope {self.name!r} start identifier was not found: "
                f"{self.start_identifier!r}"
            )
        if not ended:
            raise ValueError(
                f"Scope {self.name!r} end identifier was not found: "
                f"{self.end_identifier!r}"
            )

        for override in self.speaker_overrides:
            start = self._find_identifier(selected, override.start_identifier)
            end = self._find_identifier(selected, override.end_identifier)
            if end < start:
                raise ValueError(
                    f"Scope {self.name!r} speaker override ends before it starts"
                )
            stop = end + 1 if override.inclusive else end
            for index in range(start, stop):
                record = selected[index]
                if record.character == override.source_character:
                    selected[index] = replace(
                        record,
                        character=override.target_character,
                    )
        return tuple(
            replace(record, sequence=sequence)
            for sequence, record in enumerate(selected)
        )

    def _find_identifier(
        self,
        records: list[DialogueRecord],
        identifier: str,
    ) -> int:
        for index, record in enumerate(records):
            if record.identifier == identifier:
                return index
        raise ValueError(
            f"Scope {self.name!r} override boundary was not found: {identifier!r}"
        )


def load_dialogue_scopes(
    path: Path = Path("configs/demo_scopes.toml"),
    *,
    repository_root: Path | None = None,
) -> dict[str, DialogueScope]:
    root = (repository_root or find_repository_root()).resolve()
    resolved = path if path.is_absolute() else root / path
    if not resolved.exists():
        return {}
    try:
        with resolved.open("rb") as source:
            payload = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"Unable to load {resolved}: {exc}") from exc
    tables = payload.get("scopes", {})
    if not isinstance(tables, dict):
        raise ConfigurationError(f"{resolved}: scopes must be a table")
    scopes: dict[str, DialogueScope] = {}
    for name, value in tables.items():
        if not isinstance(value, dict):
            raise ConfigurationError(f"{resolved}: scopes.{name} must be a table")
        try:
            overrides = tuple(
                SpeakerOverride(**item)
                for item in value.get("speaker_overrides", ())
            )
            scope = DialogueScope(
                name=name,
                description=value.get("description", ""),
                source=value["source"],
                start_identifier=value["start_identifier"],
                end_identifier=value["end_identifier"],
                inclusive=value.get("inclusive", True),
                context_characters=tuple(value.get("context_characters", ())),
                voice_characters=tuple(value.get("voice_characters", ())),
                context_only_characters=tuple(
                    value.get("context_only_characters", ())
                ),
                speaker_overrides=overrides,
            )
        except (KeyError, TypeError) as exc:
            raise ConfigurationError(
                f"{resolved}: invalid scope {name!r}: {exc}"
            ) from exc
        if not scope.name or not scope.source:
            raise ConfigurationError(f"{resolved}: scope names and sources are required")
        if set(scope.voice_characters) & set(scope.context_only_characters):
            raise ConfigurationError(
                f"{resolved}: scope {name!r} voice and context-only characters overlap"
            )
        scopes[name] = scope
    return scopes
