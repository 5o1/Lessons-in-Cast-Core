"""Deterministic adjacent-record windows for semantic annotation."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass
from itertools import groupby
from typing import Any

from ..config import BatchingConfig
from ..hashing import content_hash
from .types import DialogueRecord


BATCH_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class DialogueBatch:
    id: str
    context_before: tuple[DialogueRecord, ...]
    targets: tuple[DialogueRecord, ...]
    context_after: tuple[DialogueRecord, ...]
    context_interleaved: tuple[DialogueRecord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BATCH_SCHEMA_VERSION,
            "batch_id": self.id,
            "context_before": [item.model_view() for item in self.context_before],
            "targets": [item.model_view() for item in self.targets],
            "context_after": [item.model_view() for item in self.context_after],
            "context_interleaved": [
                item.model_view() for item in self.context_interleaved
            ],
        }

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        records_by_id: dict[str, DialogueRecord] | None = None,
    ) -> DialogueBatch:
        if value.get("schema_version", BATCH_SCHEMA_VERSION) != BATCH_SCHEMA_VERSION:
            raise ValueError("Unsupported batch schema version")
        batch_id = value.get("batch_id")
        if not isinstance(batch_id, str):
            raise ValueError("Batch is missing a string batch_id")

        def resolve(name: str) -> tuple[DialogueRecord, ...]:
            items = value.get(name)
            if not isinstance(items, list):
                raise ValueError(f"Batch field {name!r} must be an array")
            resolved: list[DialogueRecord] = []
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                    raise ValueError(f"Batch field {name!r} contains an invalid record")
                if records_by_id is None:
                    required = {
                        "character": str,
                        "dialogue": str,
                        "filename": str,
                        "line_number": int,
                    }
                    if any(
                        field not in item
                        or not isinstance(item[field], expected)
                        or (expected is int and isinstance(item[field], bool))
                        for field, expected in required.items()
                    ):
                        raise ValueError(
                            f"Batch field {name!r} contains invalid record fields"
                        )
                    if any(not isinstance(item.get(key, ""), str) for key in ("label", "scene")):
                        raise ValueError(f"Batch field {name!r} has invalid label or scene")
                    resolved.append(
                        DialogueRecord(
                            id=item["id"],
                            sequence=0,
                            identifier="",
                            character=item["character"],
                            dialogue=item["dialogue"],
                            filename=item["filename"],
                            line_number=item["line_number"],
                            source_statement="",
                            label=item.get("label", ""),
                            scene=item.get("scene", ""),
                        )
                    )
                else:
                    try:
                        resolved.append(records_by_id[item["id"]])
                    except KeyError as exc:
                        raise ValueError(
                            f"Batch references unknown dialogue ID {item['id']!r}"
                        ) from exc
            return tuple(resolved)

        return cls(
            id=batch_id,
            context_before=resolve("context_before"),
            targets=resolve("targets"),
            context_after=resolve("context_after"),
            context_interleaved=resolve("context_interleaved"),
        )


class DialogueBatchBuilder:
    """Build fixed-size windows without claiming semantic scene boundaries."""

    def __init__(self, config: BatchingConfig) -> None:
        self._config = config

    def build(
        self,
        records: Iterable[DialogueRecord],
        *,
        target_characters: Collection[str] | None = None,
    ) -> Iterator[DialogueBatch]:
        seen_ids: set[str] = set()
        checked_records = self._check_unique_ids(records, seen_ids)
        target_set = (
            frozenset(target_characters)
            if target_characters is not None
            else None
        )
        for _filename, segment in groupby(
            checked_records,
            key=lambda record: record.filename,
        ):
            yield from self._build_segment(segment, target_set)

    @staticmethod
    def _check_unique_ids(
        records: Iterable[DialogueRecord],
        seen_ids: set[str],
    ) -> Iterator[DialogueRecord]:
        for record in records:
            if record.id in seen_ids:
                raise ValueError(f"Duplicate dialogue ID: {record.id}")
            seen_ids.add(record.id)
            yield record

    def _build_segment(
        self,
        records: Iterable[DialogueRecord],
        target_characters: frozenset[str] | None,
    ) -> Iterator[DialogueBatch]:
        source = iter(records)
        buffer: list[DialogueRecord] = []
        before_history: list[DialogueRecord] = []
        source_exhausted = False
        while True:
            desired = self._config.target_size + self._config.context_after
            while len(buffer) < desired and not source_exhausted:
                try:
                    buffer.append(next(source))
                except StopIteration:
                    source_exhausted = True
            if not buffer:
                return

            end = min(self._config.target_size, len(buffer))
            while (
                end > 1
                and self._characters(buffer[:end]) > self._config.max_characters
            ):
                end -= 1

            before = (
                list(before_history[-self._config.context_before:])
                if self._config.context_before
                else []
            )
            window = list(buffer[:end])
            after = list(buffer[end:end + self._config.context_after])
            self._trim_context(before, window, after)
            if target_characters is None:
                targets = window
                interleaved: list[DialogueRecord] = []
            else:
                targets = [
                    item
                    for item in window
                    if (item.character or "narrator") in target_characters
                ]
                target_ids = {item.id for item in targets}
                interleaved = [
                    item for item in window if item.id not in target_ids
                ]
            identity = {
                "before": [item.id for item in before],
                "targets": [item.id for item in targets],
                "interleaved": [item.id for item in interleaved],
                "after": [item.id for item in after],
            }
            if targets:
                yield DialogueBatch(
                    id=content_hash(identity)[:24],
                    context_before=tuple(before),
                    targets=tuple(targets),
                    context_after=tuple(after),
                    context_interleaved=tuple(interleaved),
                )
            before_history.extend(window)
            if self._config.context_before == 0:
                before_history.clear()
            elif len(before_history) > self._config.context_before:
                del before_history[:-self._config.context_before]
            del buffer[:end]

    def _trim_context(
        self,
        before: list[DialogueRecord],
        targets: list[DialogueRecord],
        after: list[DialogueRecord],
    ) -> None:
        while (
            self._characters((*before, *targets, *after))
            > self._config.max_characters
            and (before or after)
        ):
            before_distance = len(before)
            after_distance = len(after)
            if before and (not after or before_distance >= after_distance):
                del before[0]
            else:
                after.pop()

    @staticmethod
    def _characters(records: Iterable[DialogueRecord]) -> int:
        return sum(len(record.dialogue) + len(record.character) for record in records)
