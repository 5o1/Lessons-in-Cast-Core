"""Character mapping loader and consistency checks."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from .config import ConfigurationError, find_repository_root


@dataclass(frozen=True, slots=True)
class CharacterDefinition:
    id: str
    name: str
    type: str
    members: tuple[str, ...]
    render_mode: str | None
    definition_path: str
    definition_line: int
    built_in: bool
    default_voice_profile: str
    context: tuple[str, ...] = ()
    variants: tuple["CharacterDefinition", ...] = ()

    @property
    def synthesis_members(self) -> tuple[str, ...]:
        return self.members if self.type == "ensemble" else (self.id,)

    def resolve(
        self,
        filename: str,
        label: str = "",
        scene: str = "",
    ) -> "CharacterDefinition":
        """Resolve the most specific configuration for one source location."""

        location = (normalize_source_path(filename),)
        if label:
            location += (label,)
            if scene:
                location += (scene,)
        matches = (
            variant
            for variant in self.variants
            if location[: len(variant.context)] == variant.context
        )
        return max(matches, key=lambda item: len(item.context), default=self)

    def configured_variants(self) -> tuple["CharacterDefinition", ...]:
        """Return the global definition followed by contextual definitions."""

        return (self, *self.variants)


_CONFIG_FIELDS = {
    "name",
    "type",
    "members",
    "render_mode",
    "definition_path",
    "definition_line",
    "built_in",
    "default_voice_profile",
}


def normalize_source_path(value: str) -> str:
    """Normalize a configured source path without resolving it on disk."""

    return str(PurePosixPath(value.replace("\\", "/")))


def load_characters(
    path: Path = Path("configs/characters.toml"),
    *,
    repository_root: Path | None = None,
) -> dict[str, CharacterDefinition]:
    root = (repository_root or find_repository_root()).resolve()
    resolved_path = path if path.is_absolute() else root / path
    try:
        with resolved_path.open("rb") as source:
            data = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"Unable to load {resolved_path}: {exc}") from exc
    raw_characters = data.get("characters")
    if not isinstance(raw_characters, dict):
        raise ConfigurationError(f"{resolved_path}: [characters] table is required")

    result: dict[str, CharacterDefinition] = {}
    for character_id, raw in raw_characters.items():
        if not isinstance(raw, dict):
            raise ConfigurationError(
                f"{resolved_path}: characters.{character_id} must be a table"
            )
        result[character_id] = _parse_character(resolved_path, character_id, raw)

    for base in result.values():
        for character in base.configured_variants():
            _validate_character_relationships(resolved_path, character, result)
    return result


def _validate_character_relationships(
    source: Path,
    character: CharacterDefinition,
    characters: dict[str, CharacterDefinition],
) -> None:
    location = ".".join(character.context)
    suffix = f" at {location!r}" if location else ""
    if character.type != "ensemble":
        return
    if not character.members:
        raise ConfigurationError(
            f"{source}: ensemble {character.id!r}{suffix} has no members"
        )
    if len(character.members) != len(set(character.members)):
        raise ConfigurationError(
            f"{source}: ensemble {character.id!r}{suffix} has duplicate members"
        )
    if character.id in character.members:
        raise ConfigurationError(
            f"{source}: ensemble {character.id!r}{suffix} contains itself"
        )
    missing = [member for member in character.members if member not in characters]
    if missing:
        raise ConfigurationError(
            f"{source}: ensemble {character.id!r}{suffix} has unknown members "
            f"{missing!r}"
        )
    if character.render_mode is None:
        raise ConfigurationError(
            f"{source}: ensemble {character.id!r}{suffix} requires render_mode"
        )
    if character.render_mode != "unison":
        raise ConfigurationError(
            f"{source}: ensemble {character.id!r}{suffix} has unsupported "
            f"render_mode {character.render_mode!r}"
        )


def _parse_character(
    source: Path,
    character_id: str,
    raw: dict[str, Any],
) -> CharacterDefinition:
    base_values = {key: value for key, value in raw.items() if key in _CONFIG_FIELDS}
    base = _build_character(source, character_id, base_values, context=())
    variants: list[CharacterDefinition] = []
    for filename, child in raw.items():
        if filename in _CONFIG_FIELDS:
            continue
        if not isinstance(child, dict):
            raise ConfigurationError(
                f"{source}: characters.{character_id}.{filename} must be a table"
            )
        normalized = normalize_source_path(filename)
        source_path = PurePosixPath(normalized)
        if (
            not normalized
            or source_path.is_absolute()
            or ".." in source_path.parts
        ):
            raise ConfigurationError(
                f"{source}: characters.{character_id} has invalid source path "
                f"{filename!r}"
            )
        _parse_context_node(
            source,
            character_id,
            child,
            inherited=base_values,
            context=(normalized,),
            variants=variants,
        )
    return replace(base, variants=tuple(variants))


def _parse_context_node(
    source: Path,
    character_id: str,
    raw: dict[str, Any],
    *,
    inherited: dict[str, Any],
    context: tuple[str, ...],
    variants: list[CharacterDefinition],
) -> None:
    if len(context) > 3:
        raise ConfigurationError(
            f"{source}: characters.{character_id} context may contain only "
            "filename, label, and scene"
        )
    local = {key: value for key, value in raw.items() if key in _CONFIG_FIELDS}
    merged = {**inherited, **local}
    variants.append(_build_character(source, character_id, merged, context=context))
    children = {key: value for key, value in raw.items() if key not in _CONFIG_FIELDS}
    if len(context) == 3 and children:
        raise ConfigurationError(
            f"{source}: characters.{character_id}.{'.'.join(context)} has "
            "nested data below the scene level"
        )
    for key, child in children.items():
        if not key or not isinstance(child, dict):
            raise ConfigurationError(
                f"{source}: characters.{character_id}.{'.'.join((*context, key))} "
                "must be a non-empty table"
            )
        _parse_context_node(
            source,
            character_id,
            child,
            inherited=merged,
            context=(*context, key),
            variants=variants,
        )


def _build_character(
    source: Path,
    character_id: str,
    raw: dict[str, Any],
    *,
    context: tuple[str, ...],
) -> CharacterDefinition:
    character_type = raw.get("type", "individual")
    if character_type not in {"individual", "ensemble"}:
        raise ConfigurationError(
            f"{source}: characters.{character_id}.type is invalid"
        )
    members = raw.get("members", ())
    if not isinstance(members, (list, tuple)) or not all(
        isinstance(member, str) for member in members
    ):
        raise ConfigurationError(
            f"{source}: characters.{character_id}.members must be an array of strings"
        )
    if character_type == "individual" and members:
        raise ConfigurationError(
            f"{source}: individual character {character_id!r} cannot have members"
        )
    required_types: dict[str, type] = {
        "name": str,
        "definition_path": str,
        "definition_line": int,
        "built_in": bool,
        "default_voice_profile": str,
    }
    for field, expected in required_types.items():
        if (
            field not in raw
            or not isinstance(raw[field], expected)
            or (expected is int and isinstance(raw[field], bool))
        ):
            raise ConfigurationError(
                f"{source}: characters.{character_id}.{field} has an invalid type"
            )
    render_mode = raw.get("render_mode")
    if render_mode is not None and not isinstance(render_mode, str):
        raise ConfigurationError(
            f"{source}: characters.{character_id}.render_mode must be a string"
        )
    return CharacterDefinition(
        id=character_id,
        name=raw["name"],
        type=character_type,
        members=tuple(members),
        render_mode=render_mode,
        definition_path=raw["definition_path"],
        definition_line=raw["definition_line"],
        built_in=raw["built_in"],
        default_voice_profile=raw["default_voice_profile"],
        context=context,
    )
