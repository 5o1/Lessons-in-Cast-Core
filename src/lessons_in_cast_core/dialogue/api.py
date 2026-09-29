"""Interfaces implemented by dialogue stream adapters and processors."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol

from .types import DialogueRecord


class DialogueReader(Protocol):
    """Read an external dialogue artifact as a lazy record stream."""

    def read(self, source: Path) -> Iterable[DialogueRecord]: ...


class DialogueProcessor(Protocol):
    """Transform, annotate, filter, or expand a dialogue stream."""

    def process(
        self,
        records: Iterable[DialogueRecord],
    ) -> Iterator[DialogueRecord]: ...


class DialogueWriter(Protocol):
    """Serialize a dialogue stream and return the number of written records."""

    def write(
        self,
        records: Iterable[DialogueRecord],
        destination: Path,
    ) -> int: ...
