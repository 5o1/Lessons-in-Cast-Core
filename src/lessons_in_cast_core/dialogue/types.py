"""Canonical types passed between dialogue pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DIALOGUE_SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class DialogueRecord:
    """One losslessly imported row from a galgame backend."""

    id: str
    sequence: int
    identifier: str
    character: str
    dialogue: str
    filename: str
    line_number: int
    source_statement: str
    label: str = ""
    scene: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DIALOGUE_SCHEMA_VERSION,
            "id": self.id,
            "sequence": self.sequence,
            "identifier": self.identifier,
            "character": self.character,
            "dialogue": self.dialogue,
            "filename": self.filename,
            "line_number": self.line_number,
            "source_statement": self.source_statement,
            "label": self.label,
            "scene": self.scene,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DialogueRecord:
        required = {
            "id": str,
            "sequence": int,
            "identifier": str,
            "character": str,
            "dialogue": str,
            "filename": str,
            "line_number": int,
            "source_statement": str,
            "label": str,
            "scene": str,
        }
        allowed = {"schema_version", *required}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Dialogue record has unknown fields: {sorted(unknown)!r}")
        for field, expected_type in required.items():
            if field not in value:
                raise ValueError(f"Dialogue record is missing {field!r}")
            if not isinstance(value[field], expected_type) or isinstance(value[field], bool):
                raise ValueError(f"Dialogue record field {field!r} has an invalid type")
        version = value.get("schema_version", DIALOGUE_SCHEMA_VERSION)
        if version != DIALOGUE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported dialogue schema version: {version!r}")
        return cls(**{field: value[field] for field in required})

    def model_view(self) -> dict[str, Any]:
        """Return only the fields that a semantic annotator needs."""

        return {
            "id": self.id,
            "character": self.character,
            "dialogue": self.dialogue,
            "filename": self.filename,
            "line_number": self.line_number,
            "label": self.label,
            "scene": self.scene,
        }
