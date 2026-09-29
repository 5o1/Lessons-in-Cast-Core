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
from lessons_in_cast_core.jsonl import read_jsonl, write_jsonl

from .fakes import SilenceSynthesizer, write_mock_responses
from .helpers import write_minimal_workspace


HEADER = (
    "Identifier\tCharacter\tDialogue\tFilename\tLine Number\tRen'Py Script\n"
)


class PipelineTests(unittest.TestCase):
    def test_pipeline_stages_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dialogue_tab = root / "dialogue.tab"
            dialogue_tab.write_text(
                HEADER
                + 'one\ta\tHello.\tgame/route_a.rpy\t1\ta "[what]"\n'
                + 'two\ta\tAgain.\tgame/route_a.rpy\t2\ta "[what]"\n',
                encoding="utf-8",
            )
            config = PipelineConfig(
                batching=BatchingConfig(
                    target_size=1, context_before=1, context_after=1
                ),
                annotation=AnnotationConfig(
                    allowed_emotions=frozenset({"neutral"}),
                    allowed_effects=frozenset({"fade_out"}),
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
            pipeline = DialoguePipeline(
                config=config,
                characters={"a": character},
                synthesizer=SilenceSynthesizer(config.audio),
                galgame_backend=RenPyBackend(),
            )
            artifacts = root / "build"
            request = PipelineRequest(
                artifact_root=artifacts,
                dialogue_tab_path=dialogue_tab,
                allowed_sources=(Path("game/route_a.rpy"),),
            )
            dialogue_count, batch_count = pipeline.prepare(request)
            layout = ArtifactLayout(artifacts)
            write_mock_responses(
                layout.annotation_requests,
                layout.annotation_responses,
            )
            responses = list(read_jsonl(layout.annotation_responses))
            responses[0]["response"]["annotations"][0].update(action="speak_with_effect", effects=["fade_out"])
            write_jsonl(responses, layout.annotation_responses)
            validated = pipeline.validate(layout)
            with self.assertRaisesRegex(ValueError, "No polish"):
                pipeline.plan_synthesis(layout)
            write_minimal_workspace(root)
            pipeline.prepare_polish(layout)
            write_mock_responses(layout.polish.annotation_requests, layout.polish.annotation_responses)
            pipeline.plan_synthesis(layout)
            _, rendered_count = pipeline.synthesize(layout)
            pipeline.build_release_bundle(layout)

            self.assertEqual(dialogue_count, 2)
            self.assertEqual(batch_count, 2)
            self.assertEqual(validated.accepted_count, 2)
            self.assertEqual(rendered_count, 2)
            self.assertTrue(layout.voice_manifest.is_file())
            self.assertTrue(layout.run_manifest.is_file())
            self.assertTrue(
                (
                    layout.galgame_artifacts / "lessons_in_cast_voice.rpy"
                ).is_file()
            )
            self.assertTrue(
                (
                    layout.release_bundle
                    / "game"
                    / "lessons_in_cast_voice.rpy"
                ).is_file()
            )
            self.assertTrue(
                (
                    layout.release_bundle
                    / "game"
                    / "lessons_in_cast_voice.rpa"
                ).is_file()
            )
            self.assertTrue(layout.release_patch.is_file())
            audits = list(read_jsonl(layout.audio_effects))
            self.assertEqual(len(audits), 1)
            self.assertEqual(audits[0]["steps"][0]["type"], "fade_out")
            layout.discard_intermediates()
            self.assertTrue(layout.audio_effects.is_file())
            self.assertFalse((layout.root / "voice").exists())

    def test_validation_exports_only_failed_targets_for_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dialogue_tab = root / "dialogue.tab"
            dialogue_tab.write_text(
                HEADER
                + 'one\ta\tHello.\tgame/route_a.rpy\t1\ta "[what]"\n'
                + 'two\ta\tAgain.\tgame/route_a.rpy\t2\ta "[what]"\n',
                encoding="utf-8",
            )
            config = PipelineConfig(
                batching=BatchingConfig(
                    target_size=2, context_before=1, context_after=1
                ),
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
            pipeline = DialoguePipeline(
                config=config,
                characters={"a": character},
                galgame_backend=RenPyBackend(),
            )
            artifacts = root / "build"
            request = PipelineRequest(
                artifact_root=artifacts,
                dialogue_tab_path=dialogue_tab,
                allowed_sources=(Path("game/route_a.rpy"),),
            )
            pipeline.prepare(request)
            artifact_layout = ArtifactLayout(request.artifact_root)
            write_mock_responses(
                artifact_layout.annotation_requests,
                artifact_layout.annotation_responses,
            )
            envelopes = list(read_jsonl(artifact_layout.annotation_responses))
            envelopes[0]["response"]["annotations"].pop()
            write_jsonl(envelopes, artifact_layout.annotation_responses)
            summary = pipeline.validate(artifact_layout)
            self.assertEqual(summary.retryable_count, 1)
            retry_requests = list(read_jsonl(artifact_layout.retry_requests))
            self.assertEqual(len(retry_requests), 1)
            self.assertEqual(len(retry_requests[0]["batch"]["targets"]), 1)
            write_mock_responses(
                artifact_layout.retry_requests,
                artifact_layout.retry_responses,
            )
            retried = pipeline.validate(artifact_layout, retry=True)
            self.assertEqual(retried.accepted_count, 2)
            self.assertEqual(retried.retryable_count, 0)
            self.assertFalse(list(read_jsonl(artifact_layout.retry_requests)))


if __name__ == "__main__":
    unittest.main()
