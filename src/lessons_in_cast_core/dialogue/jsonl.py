"""JSON Lines adapters for canonical dialogue records."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from ..jsonl import read_jsonl, write_jsonl
from .types import DialogueRecord


class JsonlDialogueReader:
    def read(self, source: Path) -> Iterator[DialogueRecord]:
        for value in read_jsonl(source):
            yield DialogueRecord.from_dict(value)


class JsonlDialogueWriter:
    def write(
        self,
        records: Iterable[DialogueRecord],
        destination: Path,
    ) -> int:
        return write_jsonl((record.to_dict() for record in records), destination)
