from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lessons_in_cast_core.characters import CharacterDefinition
from lessons_in_cast_core.cli import main
from lessons_in_cast_core.config import AnnotationConfig, AudioConfig, BatchingConfig, PipelineConfig
from lessons_in_cast_core.galgame.renpy import RenPyBackend
from lessons_in_cast_core.jsonl import read_jsonl, write_jsonl
from lessons_in_cast_core.pipeline import ArtifactLayout, DialoguePipeline, PipelineRequest
from lessons_in_cast_core.synthesis.audio import AudioRenderError

from .fakes import RecordingSynthesizer, write_mock_responses
from .helpers import write_minimal_workspace


class ProductionPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "dialogue.tab"
        self.source.write_text(
            "Identifier\tCharacter\tDialogue\tFilename\tLine Number\tRen'Py Script\n"
            'one\ta\tHello.\tgame/main.rpy\t3\ta "[what]"\n'
            'two\ta\tAgain.\tgame/main.rpy\t4\ta "[what]"\n', encoding="utf-8")
        (self.root / "game").mkdir()
        (self.root / "game/main.rpy").write_text('label opening:\n    scene bg room\n    a "Hello."\n    a "Again."\n', encoding="utf-8")
        self.config = PipelineConfig(batching=BatchingConfig(target_size=1),
            annotation=AnnotationConfig(allowed_emotions=frozenset({"neutral"}), allowed_effects=frozenset()),
            audio=AudioConfig(format="wav"), repository_root=self.root)
        self.character = CharacterDefinition("a", "Alice", "individual", (), None, "game/definitions.rpy", 1, False, "")
        self.backend = RecordingSynthesizer(self.config.audio)
        self.pipeline = DialoguePipeline(config=self.config, characters={"a": self.character},
            synthesizer=self.backend, galgame_backend=RenPyBackend())
        write_minimal_workspace(self.root)
        (self.root / "configs/workspace.toml").write_text("# workspace marker\n", encoding="utf-8")
        self.layout = ArtifactLayout(self.root / "annotation")
        self.pipeline.prepare(PipelineRequest(artifact_root=self.layout.root, dialogue_tab_path=self.source, source_root=self.root))
        write_mock_responses(self.layout.annotation_requests, self.layout.annotation_responses)
        self.pipeline.validate(self.layout)
        self.pipeline.prepare_polish(self.layout)
        write_mock_responses(self.layout.polish.annotation_requests, self.layout.polish.annotation_responses)

    def test_external_responses_run_retains_only_release_outputs(self):
        production = self.root / "production"
        production.mkdir()
        (production / "unrelated.txt").write_text("keep", encoding="utf-8")
        result = self.pipeline.run_from_responses(
            PipelineRequest(artifact_root=production, dialogue_tab_path=self.source, source_root=self.root),
            self.layout.annotation_responses, polish_responses_path=self.layout.polish.annotation_responses)
        self.assertEqual({p.name for p in production.iterdir()},
            {"release_bundle", "lessons_in_cast_voice_patch.zip", "run_manifest.json", "unrelated.txt"})
        self.assertEqual(result.rendered_count, 2)
        self.assertTrue((result.artifacts.release_bundle / "game/lessons_in_cast_voice.rpa").is_file())
        self.assertTrue(result.artifacts.release_patch.is_file())
        self.assertTrue(self.backend.closed)

    def test_in_place_inputs_survive_success_and_failure_with_either_retention_mode(self):
        for retain in (True, False):
            for fail in (False, True):
                with self.subTest(retain=retain, fail=fail):
                    layout = ArtifactLayout(self.root / f"production-{retain}-{fail}")
                    layout.polish.root.mkdir(parents=True)
                    sources = (self.source, self.layout.annotation_responses, self.layout.polish.annotation_responses)
                    inputs = (layout.dialogue_tab, layout.annotation_responses, layout.polish.annotation_responses)
                    for source, target in zip(sources, inputs):
                        shutil.copy2(source, target)
                    before = {path: path.read_bytes() for path in inputs}
                    self.backend.invalid_text = "Hello." if fail else None
                    call = lambda: self.pipeline.run_from_responses(
                        PipelineRequest(artifact_root=layout.root, dialogue_tab_path=layout.dialogue_tab, source_root=self.root),
                        layout.annotation_responses, polish_responses_path=layout.polish.annotation_responses,
                        cache_intermediates=retain)
                    if fail:
                        with self.assertRaises(AudioRenderError):
                            call()
                        self.assertFalse(layout.release_patch.exists())
                    else:
                        self.assertEqual(call().rendered_count, 2)
                    for path, original in before.items():
                        self.assertEqual(path.read_bytes(), original)

    def test_missing_input_does_not_clear_an_existing_run(self):
        layout = ArtifactLayout(self.root / "production")
        layout.root.mkdir()
        layout.release_patch.write_bytes(b"previous release")
        with self.assertRaises(FileNotFoundError):
            self.pipeline.run_from_responses(PipelineRequest(artifact_root=layout.root, dialogue_tab_path=self.source),
                self.root / "missing.jsonl", polish_responses_path=self.layout.polish.annotation_responses)
        self.assertEqual(layout.release_patch.read_bytes(), b"previous release")

    def test_existing_raw_audio_is_still_checked_by_the_backend(self):
        self.pipeline.synthesize(self.layout)
        raw = self.layout.root / next(read_jsonl(self.layout.tts_jobs))["output_path"]
        raw.write_bytes(b"corrupted cache")
        self.assertEqual(self.pipeline.synthesize(self.layout), (2, 2))
        self.assertEqual(len(self.backend.calls), 4)
        self.assertTrue(all(row["valid"] for row in read_jsonl(self.layout.audio_quality)))

    def test_partial_and_failed_synthesis_return_nonzero_and_cannot_bundle(self):
        for invalid in ("Hello.", "*"):
            with self.subTest(invalid=invalid):
                self.backend.invalid_text = invalid
                stderr = io.StringIO()
                with patch("lessons_in_cast_core.cli.load_pipeline_config", return_value=self.config), patch(
                        "lessons_in_cast_core.cli.load_workspace_config", return_value=SimpleNamespace(release_path=self.root)), patch(
                        "lessons_in_cast_core.cli.load_characters", return_value={"a": self.character}), patch(
                        "lessons_in_cast_core.cli.load_dialogue_sources", return_value=()), patch(
                        "lessons_in_cast_core.cli.load_dialogue_scopes", return_value={}), patch(
                        "lessons_in_cast_core.cli._voice_profile_synthesizer", return_value=self.backend), contextlib.redirect_stderr(stderr):
                    code = main(["--root", str(self.root), "--build-dir", str(self.layout.root), "synthesize"])
                self.assertEqual(code, 1)
                self.assertIn("Synthesis incomplete", stderr.getvalue())
                manifest = json.loads(self.layout.run_manifest.read_text(encoding="utf-8"))
                self.assertEqual(manifest["synthesis_status"], "failed" if invalid == "*" else "partial")
                self.assertEqual(manifest["audio_quality_failure_count"], 2 if invalid == "*" else 1)
                with self.assertRaises(AudioRenderError):
                    self.pipeline.build_release_bundle(self.layout)
                self.assertFalse(self.layout.release_patch.exists())
                self.assertTrue(self.backend.closed)

    def test_keyframes_use_explicit_workspace_from_another_working_directory(self):
        responses = list(read_jsonl(self.layout.polish.annotation_responses))
        row = responses[0]["response"]["annotations"][0]
        row["spoken_text"] = '<emotion name="neutral">{1}Hello.{2}</emotion>'
        row["keyframe_effects"] = [{"type": "gain_envelope", "interpolation": "linear",
                                   "keyframes": [{"anchor": "1", "gain": 0}, {"anchor": "2", "gain": 1}]}]
        write_jsonl(responses, self.layout.polish.annotation_responses)
        current = Path.cwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            try:
                os.chdir(elsewhere)
                self.assertEqual(self.pipeline.synthesize(self.layout), (2, 2))
            finally:
                os.chdir(current)
        self.assertTrue(list((self.layout.root / "alignment").glob("*.json")))
