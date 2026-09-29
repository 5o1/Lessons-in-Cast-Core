"""Subprocess dialogue extractor used by the Ren'Py backend."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from ..api import (
    DialogueExtractionError,
    DialogueExtractionRequest,
    DialogueExtractionResult,
)


class SubprocessDialogueExtractor:
    """Invoke the game release's native Ren'Py launcher."""

    def __init__(self, *, timeout_seconds: float = 600.0) -> None:
        self._timeout_seconds = timeout_seconds

    def extract(
        self,
        request: DialogueExtractionRequest,
    ) -> DialogueExtractionResult:
        release_path = request.release_path.resolve()
        if not release_path.is_dir():
            raise DialogueExtractionError(
                f"Ren'Py release directory does not exist: {release_path}"
            )
        launcher = (
            self._launcher_command(release_path, request.executable_path.resolve())
            if request.executable_path is not None
            else self._find_launcher(release_path)
        )
        language = request.language if request.language is not None else "None"
        command = (*launcher, str(release_path), "dialogue", language)
        try:
            completed = subprocess.run(
                command,
                cwd=release_path,
                capture_output=True,
                text=True,
                encoding="utf-8", errors="replace",
                timeout=self._timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DialogueExtractionError(
                f"Unable to run Ren'Py dialogue extraction: {exc}"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise DialogueExtractionError(
                f"Ren'Py dialogue extraction failed with exit code "
                f"{completed.returncode}: {detail}"
            )

        generated = release_path / "dialogue.tab"
        if not generated.is_file():
            raise DialogueExtractionError(
                f"Ren'Py completed without producing {generated}"
            )
        output_path = request.output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if generated != output_path:
            shutil.copy2(generated, output_path)
        return DialogueExtractionResult(
            dialogue_path=output_path,
            source_count=len(request.source_paths),
            command=command,
        )

    @classmethod
    def _find_launcher(cls, release_path: Path) -> tuple[str, ...]:
        if os.name == "nt":
            # Windows releases ship Game.exe beside Game.py; the .exe is a GUI
            # program, so run Game.py with the bundled console interpreter.
            launchers = sorted(
                path
                for path in release_path.glob("*.py")
                if path.with_suffix(".exe").is_file()
            )
        else:
            launchers = sorted(release_path.glob("*.sh"))
        if not launchers:
            raise DialogueExtractionError(
                f"No Ren'Py launcher was found in {release_path}"
            )
        if len(launchers) > 1:
            names = ", ".join(path.name for path in launchers)
            raise DialogueExtractionError(
                f"Multiple Ren'Py launchers found; specify one explicitly: {names}"
            )
        return cls._launcher_command(release_path, launchers[0].resolve())

    @staticmethod
    def _launcher_command(release_path: Path, launcher: Path) -> tuple[str, ...]:
        if launcher.suffix.lower() != ".py":
            return (str(launcher),)
        interpreters = sorted(
            release_path.glob("lib/*windows*/python.exe"),
            key=lambda path: ("x86_64" not in path.parent.name, path.parent.name),
        )
        if not interpreters:
            raise DialogueExtractionError(
                f"No bundled Windows Python interpreter was found in {release_path / 'lib'}"
            )
        return (str(interpreters[0].resolve()), str(launcher))
