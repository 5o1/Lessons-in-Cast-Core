"""Mechanical integrity reports for imported dialogue."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from .types import DialogueRecord


_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]+")


def audit_dialogue(
    records: Iterable[DialogueRecord],
    *,
    known_characters: set[str],
) -> dict[str, Any]:
    """Report structural facts without interpreting or rewriting dialogue."""

    identifiers: Counter[str] = Counter()
    character_counts: Counter[str] = Counter()
    file_counts: Counter[str] = Counter()
    ids: set[str] = set()
    duplicate_ids: list[str] = []
    unsafe_identifiers: list[str] = []
    unknown_characters: Counter[str] = Counter()
    sequence_errors: list[dict[str, int]] = []
    count = 0
    for record in records:
        if record.id in ids and len(duplicate_ids) < 100:
            duplicate_ids.append(record.id)
        ids.add(record.id)
        identifiers[record.identifier] += 1
        character_id = record.character or "narrator"
        character_counts[character_id] += 1
        file_counts[record.filename] += 1
        if character_id not in known_characters:
            unknown_characters[character_id] += 1
        if not _SAFE_IDENTIFIER.fullmatch(record.identifier):
            unsafe_identifiers.append(record.identifier)
        if record.sequence != count and len(sequence_errors) < 100:
            sequence_errors.append(
                {"expected": count, "actual": record.sequence}
            )
        count += 1

    duplicate_identifiers = {
        identifier: occurrences
        for identifier, occurrences in identifiers.items()
        if occurrences > 1
    }
    valid = not (
        duplicate_ids
        or duplicate_identifiers
        or unsafe_identifiers
        or unknown_characters
        or sequence_errors
    )
    return {
        "schema_version": 1,
        "valid": valid,
        "record_count": count,
        "file_count": len(file_counts),
        "character_count": len(character_counts),
        "file_counts": dict(sorted(file_counts.items())),
        "character_counts": dict(sorted(character_counts.items())),
        "duplicate_ids": duplicate_ids,
        "duplicate_identifiers": duplicate_identifiers,
        "unsafe_identifiers": unsafe_identifiers[:100],
        "unknown_characters": dict(sorted(unknown_characters.items())),
        "sequence_errors": sequence_errors,
    }
