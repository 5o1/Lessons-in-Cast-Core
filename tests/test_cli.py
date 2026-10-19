from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from lessons_in_cast_core.cli import main


class UnifiedCliTests(unittest.TestCase):
    def test_nested_help_uses_full_command_path(self):
        for command in (["audition", "render"], ["voice-design"], ["effects", "render"]):
            with self.subTest(command=command):
                output = io.StringIO()
                with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as caught:
                    main([*command, "--help"])
                self.assertEqual(caught.exception.code, 0)
                self.assertIn("lessons-in-cast " + " ".join(command), output.getvalue())

    def test_effects_list_needs_no_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["--root", directory, "effects", "list"])
            self.assertEqual(code, 0)
            self.assertIn("descriptions", json.loads(output.getvalue()))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_effects_invalid_input_preserves_parser_error(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stderr(output), self.assertRaises(SystemExit) as caught:
                main(["effects", "render", "--output", str(Path(directory) / "output.wav"), "--effects", "unknown-test-effect"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("lessons-in-cast effects: error:", output.getvalue())
