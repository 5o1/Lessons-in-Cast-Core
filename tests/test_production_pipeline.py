from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lessons_in_cast_core.characters import CharacterDefinition
from lessons_in_cast_core.config import (
    AnnotationConfig,
    AudioConfig,
    BatchingConfig,
    PipelineConfig,
)
from lessons_in_cast_core.galgame.renpy import RenPyBackend
from lessons_in_cast_core.pipeline import ArtifactLayout, DialoguePipeline, PipelineRequest

from .fakes import SilenceSynthesizer, write_mock_responses
from .helpers import write_minimal_workspace


HEADER = "Identifier\tCharacter\tDialogue\tFilename\tLine Number\tRen'Py Script\n"


class ProductionPipelineTests(unittest.TestCase):
    def test_external_responses_run_retains_only_release_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dialogue_tab = root / "dialogue.tab"
            dialogue_tab.write_text(
                HEADER + 'one\ta\tHello.\tgame/chapter/main.rpy\t1\ta "[what]"\n',
                encoding="utf-8",
            )
            config = PipelineConfig(
                batching=BatchingConfig(target_size=1),
                annotation=AnnotationConfig(
                    allowed_emotions=frozenset({"neutral"}),
                    allowed_effects=frozenset(),
                ),
                audio=AudioConfig(),
                repository_root=root,
            )
            character = CharacterDefinition(
                id="a",
                name="Alice",
                type="individual",
                members=(),
                render_mode=None,
                definition_path="game/definitions.rpy",
                definition_line=1,
                built_in=False,
                default_voice_profile="",
            )
            annotation_root = root / "annotation"
            request = PipelineRequest(
                artifact_root=annotation_root,
                dialogue_tab_path=dialogue_tab,
                allowed_sources=(Path("game/chapter/main.rpy"),),
            )
            annotation_pipeline = DialoguePipeline(
                config=config,
                characters={"a": character},
                galgame_backend=RenPyBackend(),
            )
            annotation_pipeline.prepare(request)
            annotation_layout = ArtifactLayout(annotation_root)
            write_mock_responses(
                annotation_layout.annotation_requests,
                annotation_layout.annotation_responses,
            )

            production_root = root / "production"
            annotation_pipeline.validate(annotation_layout)
            write_minimal_workspace(root)
            annotation_pipeline.prepare_polish(annotation_layout)
            write_mock_responses(annotation_layout.polish.annotation_requests, annotation_layout.polish.annotation_responses)
            production_root.mkdir()
            (production_root / "unrelated.txt").write_text("keep", encoding="utf-8")
            production_pipeline = DialoguePipeline(
                config=config,
                characters={"a": character},
                synthesizer=SilenceSynthesizer(config.audio),
                galgame_backend=RenPyBackend(),
            )
            result = production_pipeline.run_from_responses(
                PipelineRequest(
                    artifact_root=production_root,
                    dialogue_tab_path=dialogue_tab,
                    allowed_sources=(Path("game/chapter/main.rpy"),),
                ),
                annotation_layout.annotation_responses,
                polish_responses_path=annotation_layout.polish.annotation_responses,
            )

            self.assertEqual(
                {path.name for path in production_root.iterdir()},
                {"release_bundle", "lessons_in_cast_voice_patch.zip", "run_manifest.json", "unrelated.txt"},
            )
            self.assertTrue(
                (
                    result.artifacts.release_bundle
                    / "game"
                    / "lessons_in_cast_voice.rpa"
                ).is_file()
            )
            self.assertTrue(result.artifacts.release_patch.is_file())


if __name__ == "__main__":
    unittest.main()
