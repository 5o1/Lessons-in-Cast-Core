from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lessons_in_cast_core.config import ConfigurationError
from lessons_in_cast_core.model_registry import load_model_registry


class ModelRegistryTests(unittest.TestCase):
    def test_loads_metadata_and_resolves_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "models.toml"
            config.write_text(
                "[models.shared]\n"
                'path = "models/shared"\n'
                'provider = "huggingface"\n'
                'repository = "owner/shared"\n'
                'revision = "abc123"\n'
                'license = "mit"\n',
                encoding="utf-8",
            )
            registry = load_model_registry(config, repository_root=root)
            definition = registry.require("shared")
            self.assertEqual(definition.revision, "abc123")
            self.assertEqual(
                registry.resolve_path("shared"),
                (root / "models/shared").resolve(),
            )

    def test_rejects_incomplete_entries_and_unknown_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "models.toml"
            config.write_text(
                "[models.incomplete]\n"
                'path = "models/incomplete"\n',
                encoding="utf-8",
            )
            with self.assertRaises(ConfigurationError):
                load_model_registry(config, repository_root=root)

            config.write_text("[models]\n", encoding="utf-8")
            registry = load_model_registry(config, repository_root=root)
            with self.assertRaises(ConfigurationError):
                registry.require("missing")


if __name__ == "__main__":
    unittest.main()
