"""Utilities for composing lazy dialogue stream transformations."""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

from .api import DialogueProcessor
from .types import DialogueRecord


def apply_processors(
    records: Iterable[DialogueRecord],
    processors: Sequence[DialogueProcessor],
) -> Iterator[DialogueRecord]:
    """Apply processors in declaration order without materializing the stream."""

    stream: Iterable[DialogueRecord] = records
    for processor in processors:
        stream = processor.process(stream)
    yield from stream
