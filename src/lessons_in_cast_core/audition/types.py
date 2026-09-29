"""Portable casting briefs and source-backed audition sides."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from ..dialogue.types import DialogueRecord


def slug(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value):
        raise ValueError(f"Expected a safe lowercase identifier: {value!r}")
    return value


def string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value


@dataclass(frozen=True)
class AuditionCase:
    id: str
    title: str
    state: str
    background: str
    addressee: str
    intention: str
    direction: str
    listen_for: tuple[str, ...]
    avoid: tuple[str, ...]
    source: DialogueRecord


@dataclass(frozen=True)
class AuditionProject:
    id: str
    character: str
    brief: str
    coverage: dict[str, Any]
    cases: tuple[AuditionCase, ...]
    document: dict[str, Any]


def load_project(path: Path) -> AuditionProject:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"version", "id", "character", "brief", "coverage", "cases"}:
        raise ValueError("Invalid audition project fields")
    if type(data["version"]) is not int or data["version"] != 2:
        raise ValueError("Unsupported audition project version")
    slug(data["id"])
    string(data["character"], "character")
    string(data["brief"], "brief")
    coverage = data["coverage"]
    if not isinstance(coverage, dict) or coverage.get("status") not in {"partial", "complete"}:
        raise ValueError("coverage.status must be partial or complete")
    string(coverage.get("notes"), "coverage.notes")
    if not isinstance(data["cases"], list) or not data["cases"]:
        raise ValueError("An audition needs at least one case")
    cases = []
    required = set(AuditionCase.__dataclass_fields__)
    for raw in data["cases"]:
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("Invalid audition case fields; cleaned text belongs to cleaning, acting labels to polish, never to casting inputs")
        slug(raw["id"])
        for field in ("title", "state", "background", "addressee", "intention", "direction"):
            string(raw[field], field)
        for field in ("listen_for", "avoid"):
            if not isinstance(raw[field], list) or not raw[field] or any(not isinstance(v, str) or not v.strip() for v in raw[field]):
                raise ValueError(f"{field} must contain nonempty listening criteria")
        source = DialogueRecord.from_dict(raw["source"])
        if source.line_number < 1 or not source.dialogue.strip():
            raise ValueError("A case needs a real source line and original dialogue")
        cases.append(AuditionCase(**{**raw, "listen_for": tuple(raw["listen_for"]), "avoid": tuple(raw["avoid"]),
                                    "source": source}))
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("Duplicate audition case IDs")
    return AuditionProject(data["id"], data["character"], data["brief"], coverage, tuple(cases), data)
