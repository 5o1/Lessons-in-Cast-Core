"""Lexical source context for Ren'Py dialogue rows."""

from __future__ import annotations

import re
import warnings
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


_LABEL = re.compile(
    r"^\s*label\s+([A-Za-z_][A-Za-z0-9_.]*)\s*(?:\([^)]*\))?\s*:"
)
_SCENE = re.compile(r"^\s*scene(?:\s+(.*?))?\s*$")


@dataclass(frozen=True, slots=True)
class RenPySourceContext:
    label: str = ""
    scene: str = ""
    scene_line: int = 0

    @property
    def path(self) -> tuple[str, ...]:
        """Return a path truncated at the first missing level."""

        if not self.label:
            return ()
        if not self.scene:
            return (self.label,)
        return (self.label, self.scene)


class RenPySourceContextIndex:
    """Find the nearest lexical label and scene above a source line."""

    def __init__(self, source_path: Path) -> None:
        labels: list[tuple[int, str]] = []
        scenes: list[tuple[int, str]] = []
        with source_path.open("r", encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, start=1):
                label = _LABEL.match(line)
                if label is not None:
                    labels.append((line_number, label.group(1)))
                scene = _SCENE.match(line)
                if scene is not None:
                    expression = (scene.group(1) or "").split("#", 1)[0].strip()
                    name = re.split(r"\s+(?:with|onlayer|at|as|zorder|behind)\b|:", expression)[0].strip()
                    scenes.append((line_number, "" if name.startswith("expression ") else name))
        self._labels = tuple(labels)
        self._label_lines = tuple(line for line, _ in labels)
        self._scenes = tuple(scenes)
        self._scene_lines = tuple(line for line, _ in scenes)

    def at(self, line_number: int) -> RenPySourceContext:
        label = self._nearest(self._labels, self._label_lines, line_number)
        if not label:
            return RenPySourceContext()
        label_index = bisect_right(self._label_lines, line_number) - 1
        scene_index = bisect_right(self._scene_lines, line_number) - 1
        if scene_index < 0 or self._scene_lines[scene_index] < self._label_lines[label_index]:
            return RenPySourceContext(label)
        scene_line, scene = self._scenes[scene_index]
        return RenPySourceContext(label, scene, scene_line)

    @staticmethod
    def _nearest(
        entries: tuple[tuple[int, str], ...],
        lines: tuple[int, ...],
        line_number: int,
    ) -> str:
        index = bisect_right(lines, line_number) - 1
        return entries[index][1] if index >= 0 else ""


class RenPyContextResolver:
    """Resolve source contexts while warning once for each recoverable gap."""

    def __init__(self, source_root: Path) -> None:
        self._source_root = source_root.resolve()
        self._indexes: dict[str, RenPySourceContextIndex | None] = {}
        self._warned: set[tuple[str, str]] = set()

    def resolve(self, filename: str, line_number: int) -> RenPySourceContext:
        index = self._indexes.get(filename)
        if filename not in self._indexes:
            index = self._load(filename)
            self._indexes[filename] = index
        if index is None:
            return RenPySourceContext()
        context = index.at(line_number)
        if not context.label:
            self._warn(
                filename,
                "label",
                f"{filename}: no preceding Ren'Py label for one or more dialogue "
                "lines; label and scene were left empty",
            )
            return RenPySourceContext()
        if not context.scene:
            self._warn(
                filename,
                "scene",
                f"{filename}: no preceding Ren'Py scene for one or more dialogue "
                "lines; scene was left empty",
            )
        return context

    def _load(self, filename: str) -> RenPySourceContextIndex | None:
        relative = PurePosixPath(filename.replace("\\", "/"))
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            self._warn(
                filename,
                "source",
                f"{filename}: unsafe Ren'Py source path; label and scene were left empty",
            )
            return None
        source_path = self._source_root.joinpath(*relative.parts)
        try:
            return RenPySourceContextIndex(source_path)
        except (OSError, UnicodeError) as exc:
            self._warn(
                filename,
                "source",
                f"{filename}: unable to read Ren'Py source ({exc}); label and scene "
                "were left empty",
            )
            return None

    def _warn(self, filename: str, kind: str, message: str) -> None:
        key = (filename, kind)
        if key in self._warned:
            return
        self._warned.add(key)
        warnings.warn(message, RuntimeWarning, stacklevel=3)
