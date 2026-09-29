"""Strict and atomic JSON Lines input/output."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


class JsonlError(ValueError):
    """Raised when a JSON Lines artifact cannot be decoded."""


class AtomicJsonlWriter:
    """Incrementally write a JSONL artifact and replace it only on success."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.count = 0
        self._temporary_name: str | None = None
        self._output: Any = None

    def __enter__(self) -> AtomicJsonlWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, self._temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        self._output = os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        )
        return self

    def write(self, record: Mapping[str, Any]) -> None:
        if self._output is None:
            raise RuntimeError("AtomicJsonlWriter is not open")
        self._output.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        self._output.write("\n")
        self.count += 1

    def __exit__(self, exception_type: object, *_: object) -> None:
        if self._output is None or self._temporary_name is None:
            return
        try:
            if exception_type is None:
                self._output.flush()
                os.fsync(self._output.fileno())
            self._output.close()
            if exception_type is None:
                os.replace(self._temporary_name, self.path)
            else:
                os.unlink(self._temporary_name)
        except FileNotFoundError:
            pass
        finally:
            self._output = None
            self._temporary_name = None


class JsonlIndex:
    """Index JSONL object offsets by a string field without retaining values."""

    def __init__(self, path: Path, key: str) -> None:
        self.path = path
        self.key = key
        self.offsets: dict[str, int] = {}
        self.duplicates: set[str] = set()
        self._source: Any = None

    def __enter__(self) -> JsonlIndex:
        self._source = self.path.open("rb")
        line_number = 0
        while True:
            offset = self._source.tell()
            line = self._source.readline()
            if not line:
                break
            line_number += 1
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise JsonlError(f"{self.path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict) or not isinstance(value.get(self.key), str):
                raise JsonlError(
                    f"{self.path}:{line_number}: expected string field {self.key!r}"
                )
            indexed_value = value[self.key]
            if indexed_value in self.offsets:
                self.duplicates.add(indexed_value)
            else:
                self.offsets[indexed_value] = offset
        return self

    def get(self, value: str) -> dict[str, Any] | None:
        if self._source is None:
            raise RuntimeError("JsonlIndex is not open")
        offset = self.offsets.get(value)
        if offset is None:
            return None
        self._source.seek(offset)
        item = json.loads(self._source.readline())
        if not isinstance(item, dict):
            raise JsonlError(f"{self.path}: indexed value is not an object")
        return item

    def __exit__(self, *_: object) -> None:
        if self._source is not None:
            self._source.close()
            self._source = None


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a UTF-8 JSON Lines file."""

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JsonlError(f"{path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise JsonlError(f"{path}:{line_number}: expected a JSON object")
            yield value


def write_jsonl(records: Iterable[Mapping[str, Any]], path: Path) -> int:
    """Atomically write JSON objects and return the record count."""

    with AtomicJsonlWriter(path) as output:
        for record in records:
            output.write(record)
    return output.count
