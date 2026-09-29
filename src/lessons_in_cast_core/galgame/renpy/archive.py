"""Deterministic writer for Ren'Py RPA-3.0 archives."""

from __future__ import annotations

import pickle
import zlib
from collections.abc import Iterable
from pathlib import Path, PurePosixPath


class RenPyArchiveError(ValueError):
    """Raised when an RPA member name or source is invalid."""


class RenPyArchiveWriter:
    """Pack files into an RPA-3.0 archive loadable by Ren'Py 8."""

    _HEADER_SIZE = 34
    _KEY = 0x4C494343

    def write(
        self,
        destination: Path,
        members: Iterable[tuple[str, Path]],
    ) -> tuple[str, ...]:
        normalized: list[tuple[str, Path]] = []
        seen: set[str] = set()
        for virtual_path, source in members:
            name = self._normalize_name(virtual_path)
            if name in seen:
                raise RenPyArchiveError(f"Duplicate RPA member: {name!r}")
            if not source.is_file():
                raise FileNotFoundError(f"RPA source file is missing: {source}")
            seen.add(name)
            normalized.append((name, source))
        normalized.sort(key=lambda item: item[0])

        destination.parent.mkdir(parents=True, exist_ok=True)
        index: dict[str, list[tuple[int, int]]] = {}
        offset = self._HEADER_SIZE
        with destination.open("wb") as archive:
            archive.write(b"\x00" * self._HEADER_SIZE)
            for name, source in normalized:
                length = source.stat().st_size
                index[name] = [(offset ^ self._KEY, length ^ self._KEY)]
                with source.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        archive.write(chunk)
                offset += length
            encoded_index = pickle.dumps(index, protocol=2)
            archive.write(zlib.compress(encoded_index, level=9))
            archive.seek(0)
            archive.write(
                f"RPA-3.0 {offset:016x} {self._KEY:08x}\n".encode("ascii")
            )
        return tuple(name for name, _source in normalized)

    @staticmethod
    def _normalize_name(value: str) -> str:
        path = PurePosixPath(value.replace("\\", "/"))
        if path.is_absolute() or not path.parts or any(
            part in {"", ".", ".."} for part in path.parts
        ):
            raise RenPyArchiveError(f"Unsafe RPA member path: {value!r}")
        return str(path)
