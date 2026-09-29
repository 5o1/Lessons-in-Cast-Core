"""Create a new workspace from the templates shipped with the package."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath

DEFAULT_RELEASE = "game_releases/current"

# Template names that differ from their destination, so that packaging tools
# do not treat them as metadata of the core repository itself.
_RENAMED = {PurePosixPath("gitignore"): PurePosixPath(".gitignore")}
# Machine-local files seeded from a tracked template.
_SEEDED = {
    PurePosixPath("configs/model_sources.toml"): PurePosixPath("configs/model_sources.default.toml"),
}
_WORKSPACE_CONFIG = PurePosixPath("configs/workspace.toml")


@dataclass(frozen=True, slots=True)
class WorkspaceInitResult:
    root: Path
    created: tuple[PurePosixPath, ...]
    skipped: tuple[PurePosixPath, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "created": [str(path) for path in self.created],
            "skipped": [str(path) for path in self.skipped],
        }


def _template_root() -> Traversable:
    return files(__package__).joinpath("templates", "workspace")


def _walk(directory: Traversable, prefix: PurePosixPath = PurePosixPath()):
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        relative = prefix / entry.name
        if entry.is_dir():
            yield from _walk(entry, relative)
        elif entry.name != "__init__.py" and not entry.name.endswith((".pyc", ".pyo")):
            yield relative, entry


def workspace_files(release: str = DEFAULT_RELEASE) -> dict[PurePosixPath, bytes]:
    """Return every workspace file, keyed by its workspace-relative path."""

    result: dict[PurePosixPath, bytes] = {}
    for relative, entry in _walk(_template_root()):
        content = entry.read_bytes()
        if relative == _WORKSPACE_CONFIG:
            # A JSON string is also a valid TOML basic string.
            quoted = json.dumps(release, ensure_ascii=False)
            content = content.replace(b'"{release}"', quoted.encode("utf-8"))
        result[_RENAMED.get(relative, relative)] = content
    for destination, source in _SEEDED.items():
        result[destination] = result[source]
    return dict(sorted(result.items()))


def initialize_workspace(
    root: Path,
    *,
    release: str = DEFAULT_RELEASE,
    force: bool = False,
) -> WorkspaceInitResult:
    """Write the workspace templates below ``root``.

    Existing files are kept unless ``force`` is set, so running this again
    only adds files that are missing.
    """

    root = root.resolve()
    created: list[PurePosixPath] = []
    skipped: list[PurePosixPath] = []
    for relative, content in workspace_files(release).items():
        destination = root.joinpath(*relative.parts)
        if destination.exists() and not force:
            skipped.append(relative)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        created.append(relative)
    return WorkspaceInitResult(root, tuple(created), tuple(skipped))
