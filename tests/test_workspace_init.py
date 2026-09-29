from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from lessons_in_cast_core.characters import load_characters
from lessons_in_cast_core.cli import main
from lessons_in_cast_core.config import load_workspace_config
from lessons_in_cast_core.dialogue import load_dialogue_scopes
from lessons_in_cast_core.pronunciations import load_pronunciation_lexicon
from lessons_in_cast_core.workspace_init import initialize_workspace


def _run(*argv: str) -> dict:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = main(list(argv))
    assert status == 0
    return json.loads(output.getvalue())


class WorkspaceInitTests(unittest.TestCase):
    def test_initialized_workspace_passes_check_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = _run("--root", directory, "init")
            self.assertEqual(report["skipped"], [])
            for name in (".gitignore", "configs/pipeline.toml", "configs/model_sources.toml",
                         "prompts/codex_dialogue_cleanup_v4.md", "profiles/README.md"):
                self.assertIn(name, report["created"])
                self.assertTrue((root / name).is_file())
            self.assertFalse((root / "gitignore").exists())

            summary = _run("--root", directory, "check-config")
            self.assertEqual(summary["characters"], 1)
            self.assertEqual(summary["dialogue_scopes"], [])
            self.assertEqual(summary["voice_profiles"], [])
            self.assertEqual(summary["cleaning_backend"], "codex")
            self.assertIn("narrator", load_characters(repository_root=root))
            self.assertEqual(load_dialogue_scopes(repository_root=root), {})
            load_pronunciation_lexicon(repository_root=root)

    def test_existing_files_are_kept_unless_forced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            pipeline = root / "configs" / "pipeline.toml"
            pipeline.write_text("# mine\n", encoding="utf-8")

            result = initialize_workspace(root)
            self.assertIn("configs/pipeline.toml", map(str, result.skipped))
            self.assertEqual(pipeline.read_text(encoding="utf-8"), "# mine\n")
            self.assertEqual(initialize_workspace(root).created, ())

            initialize_workspace(root, force=True)
            self.assertNotEqual(pipeline.read_text(encoding="utf-8"), "# mine\n")

    def test_release_path_is_quoted_for_toml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_workspace(root, release='releases\\Game "1.0"')
            config = load_workspace_config(repository_root=root)
            self.assertEqual(config.release_path, (root / 'releases\\Game "1.0"').resolve())
