from __future__ import annotations

import os
import pickle
import tempfile
import zipfile
import zlib
import unittest
from pathlib import Path

from lessons_in_cast_core.annotation import DialogueAction
from lessons_in_cast_core.config import ConfigurationError
from lessons_in_cast_core.galgame import DialogueExtractionRequest, load_galgame_backend
from lessons_in_cast_core.galgame.renpy import (
    RenPyBackend,
    RenPyVoiceInstaller,
    RenPyVoiceScriptWriter,
    SubprocessDialogueExtractor,
)
from lessons_in_cast_core.synthesis import AudioQualityResult, RenderTask


class GalgameBackendTests(unittest.TestCase):
    def test_loader_selects_renpy_backend(self) -> None:
        backend = load_galgame_backend("renpy")
        self.assertIsInstance(backend, RenPyBackend)
        self.assertEqual(backend.backend_id, "renpy")

    def test_loader_rejects_unknown_backend(self) -> None:
        with self.assertRaises(ConfigurationError):
            load_galgame_backend("unknown")

    def test_renpy_backend_owns_virtual_voice_path_rules(self) -> None:
        backend = RenPyBackend()
        self.assertEqual(
            backend.voice_virtual_path("game/chapter/main.rpy", "line_123", "wav"),
            "voice/chapter/main/line_123.wav",
        )
        with self.assertRaises(ValueError):
            backend.voice_virtual_path("../outside.rpy", "line_123", "wav")

    def test_windows_launcher_runs_game_script_with_bundled_python(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory)
            for name in ("lib/py3-windows-i686/python.exe", "lib/py3-windows-x86_64/python.exe"):
                (release / name).parent.mkdir(parents=True)
                (release / name).write_bytes(b"")
            script = release / "Game.py"
            command = SubprocessDialogueExtractor._launcher_command(release, script)
            self.assertEqual(
                command,
                (str((release / "lib/py3-windows-x86_64/python.exe").resolve()), str(script)),
            )
            self.assertEqual(
                SubprocessDialogueExtractor._launcher_command(release, release / "game.sh"),
                (str(release / "game.sh"),),
            )

    @unittest.skipIf(os.name == "nt", "uses a POSIX shell launcher")
    def test_subprocess_extractor_passes_release_before_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release"
            release.mkdir()
            launcher = release / "game.sh"
            launcher.write_text(
                "#!/bin/sh\n"
                'test "$2" = "dialogue" || exit 7\n'
                'test "$3" = "None" || exit 8\n'
                'printf "Identifier\\tCharacter\\tDialogue\\tFilename\\tLine Number'
                '\\tRenPy Script\\n" > "$1/dialogue.tab"\n',
                encoding="utf-8",
            )
            launcher.chmod(launcher.stat().st_mode | 0o111)
            destination = root / "build" / "dialogue.tab"
            result = SubprocessDialogueExtractor().extract(
                DialogueExtractionRequest(
                    release_path=release,
                    output_path=destination,
                    executable_path=launcher,
                )
            )
            self.assertEqual(result.dialogue_path, destination)
            self.assertEqual(result.command[1], str(release.resolve()))
            self.assertTrue(destination.is_file())

    def test_auto_voice_script_uses_dialogue_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "voice.rpy"
            RenPyVoiceScriptWriter().write(
                destination,
                entries=[("line_123", "voice/part1/main/line_123.wav")],
            )
            self.assertIn(
                '"line_123": "voice/part1/main/line_123.wav"',
                destination.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "config.auto_voice = _lessons_in_cast_auto_voice",
                destination.read_text(encoding="utf-8"),
            )

    def test_installer_mirrors_source_path_and_removes_stale_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            source = artifacts / "voice" / "line_123.wav"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"audio")
            script = artifacts / "lessons_in_cast_voice.rpy"
            manifest = artifacts / "voice_manifest.json"
            script.write_text("init python:\n    pass\n", encoding="utf-8")
            manifest.write_text("{}\n", encoding="utf-8")
            task = RenderTask(
                dialogue_id="dialogue-1",
                identifier="line_123",
                action=DialogueAction.SPEAK,
                component_job_ids=("job-1",),
                render_mode="single",
                effects=(),
                output_path="voice/line_123.wav",
                virtual_path="voice/part1/main/line_123.wav",
            )
            quality = AudioQualityResult(
                dialogue_id="dialogue-1",
                path=str(source),
                valid=True,
                duration_seconds=1.0,
                issues=(),
            )
            stale = root / "release" / "game" / "voice" / "stale.wav"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"stale")

            result = RenPyVoiceInstaller().install(
                root / "release",
                artifact_root=artifacts,
                voice_script=script,
                voice_manifest=manifest,
                artifacts=[(task, quality)],
            )

            self.assertEqual(result.audio_count, 1)
            self.assertTrue(result.audio_archive.is_file())
            self.assertFalse((result.content_root / "voice").exists())
            with result.audio_archive.open("rb") as archive:
                header = archive.read(40)
                index_offset = int(header[8:24], 16)
                key = int(header[25:33], 16)
                archive.seek(index_offset)
                index = pickle.loads(zlib.decompress(archive.read()))
                offset, length = index[task.virtual_path][0]
                archive.seek(offset ^ key)
                self.assertEqual(archive.read(length ^ key), b"audio")
            with zipfile.ZipFile(result.patch_path) as package:
                self.assertIn("game/lessons_in_cast_voice.rpa", package.namelist())
            self.assertFalse(stale.exists())


if __name__ == "__main__":
    unittest.main()
