from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lessons_in_cast_core.characters import load_characters
from lessons_in_cast_core.config import ConfigurationError


class ContextualCharacterTests(unittest.TestCase):
    def _write_config(self, root: Path, extra: str) -> Path:
        config = root / "characters.toml"
        config.write_text(
            "[characters.crowd2]\n"
            'name = "Crowd"\n'
            'definition_path = "game/definitions.rpy"\n'
            "definition_line = 1\n"
            "built_in = false\n"
            'default_voice_profile = "profiles/global.py"\n'
            + extra,
            encoding="utf-8",
        )
        return config

    def test_resolves_path_label_scene_with_prefix_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_config(
                root,
                "\n[characters.crowd2.\"game/events/chapter/a.rpy\"]\n"
                'default_voice_profile = "profiles/file.py"\n'
                "\n[characters.crowd2.\"game/events/chapter/a.rpy\".opening]\n"
                'default_voice_profile = "profiles/label.py"\n'
                "\n[characters.crowd2.\"game/events/chapter/a.rpy\".opening.room]\n"
                'default_voice_profile = "profiles/scene.py"\n',
            )
            character = load_characters(config, repository_root=root)["crowd2"]

            self.assertEqual(
                character.resolve("game/other/a.rpy").default_voice_profile,
                "profiles/global.py",
            )
            self.assertEqual(
                character.resolve(
                    "game\\events\\chapter\\a.rpy"
                ).default_voice_profile,
                "profiles/file.py",
            )
            self.assertEqual(
                character.resolve(
                    "game/events/chapter/a.rpy", "opening"
                ).default_voice_profile,
                "profiles/label.py",
            )
            self.assertEqual(
                character.resolve(
                    "game/events/chapter/a.rpy", "opening", "room"
                ).default_voice_profile,
                "profiles/scene.py",
            )

    def test_missing_label_truncates_before_scene(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_config(
                root,
                "\n[characters.crowd2.\"game/a.rpy\"]\n"
                'default_voice_profile = "profiles/file.py"\n'
                "\n[characters.crowd2.\"game/a.rpy\".opening.room]\n"
                'default_voice_profile = "profiles/scene.py"\n',
            )
            character = load_characters(config, repository_root=root)["crowd2"]

            resolved = character.resolve("game/a.rpy", "", "room")

            self.assertEqual(resolved.context, ("game/a.rpy",))
            self.assertEqual(resolved.default_voice_profile, "profiles/file.py")

    def test_rejects_context_below_scene_level(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_config(
                root,
                "\n[characters.crowd2.\"game/a.rpy\".opening.room.extra]\n"
                'default_voice_profile = "profiles/invalid.py"\n',
            )
            with self.assertRaisesRegex(ConfigurationError, "scene level"):
                load_characters(config, repository_root=root)


if __name__ == "__main__":
    unittest.main()
