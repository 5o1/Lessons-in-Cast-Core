from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lessons_in_cast_core.hashing import file_hash
from lessons_in_cast_core.model_registry import ModelDefinition, ModelRegistry
from lessons_in_cast_core.voice_design import (
    VoiceDesignRequest,
    VoiceDesignResult,
    VoxCPM2Settings,
    VoxCPM2VoiceDesigner,
)
from lessons_in_cast_core.voice_design.cli import main
from lessons_in_cast_core.voice_design.voxcpm2 import compile_voxcpm2_text


def _write_audio(path, _waveform=None, sample_rate=48000):
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(b"\x01\x00" * (sample_rate // 10))


class VoiceDesignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        # Resolve so Windows 8.3 short temp paths match the resolved paths the code records.
        self.root = Path(temporary.name).resolve()
        self.model_path = self.root / "models/test"
        self.model_path.mkdir(parents=True)
        (self.model_path / "config.json").write_text('{"architecture": "voxcpm2"}')
        for filename in ("model.safetensors", "audiovae.pth"):
            (self.model_path / filename).write_bytes(b"fake test weights")
        self.registry = ModelRegistry(self.root, {
            "voxcpm2": ModelDefinition(
                id="voxcpm2", path="models/test", provider="huggingface",
                repository="openbmb/VoxCPM2", revision="test-revision", license="apache-2.0",
            ),
        })
        self.model = Mock()
        self.model.tts_model.sample_rate = 48000
        self.factory = Mock(return_value=self.model)
        module = SimpleNamespace(VoxCPM=SimpleNamespace(from_pretrained=self.factory))
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {"voxcpm": module}).start()
        patch(
            "lessons_in_cast_core.voice_design.voxcpm2._write_waveform",
            side_effect=_write_audio,
        ).start()

    def test_import_does_not_load_optional_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; import lessons_in_cast_core.voice_design; "
             "assert not {'torch', 'voxcpm', 'soundfile', 'numpy'} & sys.modules.keys()"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_compiles_instruction_separately_from_script(self):
        request = VoiceDesignRequest("Hello,\n Alice!", "Warm voice,\n natural delivery")
        self.assertEqual(compile_voxcpm2_text(request), "(Warm voice, natural delivery)Hello, Alice!")
        self.assertEqual(request.text, "Hello,\n Alice!")
        clone = VoiceDesignRequest("Hello.", reference_audio=Path("reference.wav"))
        self.assertEqual(compile_voxcpm2_text(clone), "Hello.")

    def test_rejects_invalid_requests(self):
        for values in (
            {"text": " "}, {"text": "Hello", "instruction": " "},
            {"text": "Hello", "instruction": None},
            {"text": "Hello", "instruction": "Warm", "seed": -1},
            {"text": "Hello", "instruction": "Warm", "seed": True},
            {"text": "Hello", "instruction": "Warm", "seed": 2**32},
            {"text": "Hello", "reference_audio": "reference.wav"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                VoiceDesignRequest(**values)

    def test_rejects_invalid_settings(self):
        for values in (
            {"cfg_value": float("nan")}, {"cfg_value": float("inf")},
            {"cfg_value": 0}, {"cfg_value": True}, {"inference_timesteps": 0},
            {"inference_timesteps": 1.5}, {"max_length": 1}, {"device": ""},
            {"optimize": "false"}, {"retry_badcase": 1},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                VoxCPM2Settings(**values)

    def test_generation_loads_local_model_lazily_and_records_provenance(self):
        designer = VoxCPM2VoiceDesigner(self.registry)
        self.factory.assert_not_called()
        request = VoiceDesignRequest("Hello.", "Warm adult female voice", seed=17)
        result = designer.generate(request, self.root / "build/one.wav")
        self.factory.assert_called_once_with(
            str(self.model_path), local_files_only=True, load_denoiser=False,
            optimize=False, device="auto",
        )
        self.model.generate.assert_called_once_with(
            text="(Warm adult female voice)Hello.", reference_wav_path=None,
            cfg_value=2.0, inference_timesteps=10, max_len=4096, seed=17,
            normalize=False, denoise=False, retry_badcase=False,
        )
        self.assertEqual(result.sample_rate, 48000)
        self.assertAlmostEqual(result.duration_seconds, 0.1)
        metadata = json.loads(result.metadata_path.read_text())
        self.assertEqual(metadata["request"]["seed"], 17)
        self.assertEqual(metadata["mode"], "voice_design")
        self.assertEqual(metadata["configuration"]["model"]["revision"], "test-revision")
        self.assertEqual(metadata["audio"]["sha256"], file_hash(result.audio_path))
        designer.generate(request, self.root / "build/two.wav")
        self.assertEqual(self.factory.call_count, 1)
        designer.close()
        designer.generate(request, self.root / "build/three.wav")
        self.assertEqual(self.factory.call_count, 2)

    def test_reference_mode_and_settings_reach_official_api(self):
        reference = self.root / "reference.wav"
        _write_audio(reference)
        designer = VoxCPM2VoiceDesigner(self.registry, settings=VoxCPM2Settings(
            device="cpu", cfg_value=3.0, inference_timesteps=20,
            max_length=2048, retry_badcase=True,
        ))
        result = designer.generate(
            VoiceDesignRequest("Good morning.", "Quietly, without excitement", reference),
            self.root / "reference-style.wav",
        )
        parameters = self.model.generate.call_args.kwargs
        self.assertEqual(parameters["reference_wav_path"], str(reference))
        self.assertNotIn("prompt_wav_path", parameters)
        self.assertNotIn("prompt_text", parameters)
        self.assertEqual(parameters["cfg_value"], 3.0)
        self.assertEqual(parameters["inference_timesteps"], 20)
        self.assertEqual(parameters["max_len"], 2048)
        self.assertTrue(parameters["retry_badcase"])
        metadata = json.loads(result.metadata_path.read_text())
        self.assertEqual(metadata["mode"], "reference_style")
        self.assertEqual(metadata["request"]["reference_sha256"], file_hash(reference))

    def test_missing_reference_fails_before_model_load(self):
        with self.assertRaises(FileNotFoundError):
            VoxCPM2VoiceDesigner(self.registry).generate(
                VoiceDesignRequest("Hello", reference_audio=self.root / "missing.wav"),
                self.root / "output.wav",
            )
        self.factory.assert_not_called()

    def test_missing_weights_fail_without_download(self):
        (self.model_path / "model.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            VoxCPM2VoiceDesigner(self.registry)._load_model()
        self.factory.assert_not_called()

    def test_rejects_v1_weights(self):
        (self.model_path / "config.json").write_text('{"architecture": "voxcpm"}')
        with self.assertRaisesRegex(ValueError, "not a VoxCPM 1.x"):
            VoxCPM2VoiceDesigner(self.registry)._load_model()
        self.factory.assert_not_called()

    def test_missing_dependencies_have_actionable_error(self):
        with patch.dict(sys.modules, {"voxcpm": None}):
            with self.assertRaisesRegex(RuntimeError, "Conda environment"):
                VoxCPM2VoiceDesigner(self.registry)._load_model()

    def test_never_overwrites_an_audition_or_metadata(self):
        request = VoiceDesignRequest("Hello", "Warm")
        designer = VoxCPM2VoiceDesigner(self.registry)
        for filename in ("output.wav", "output.json"):
            existing = self.root / filename
            existing.write_text("preserved")
            with self.assertRaises(FileExistsError):
                designer.generate(request, self.root / "output.wav")
            self.assertEqual(existing.read_text(), "preserved")
            existing.unlink()
        self.factory.assert_not_called()

    def test_rejects_non_wav_destination(self):
        with self.assertRaises(ValueError):
            VoxCPM2VoiceDesigner(self.registry).generate(
                VoiceDesignRequest("Hello", "Warm"), self.root / "output.mp3"
            )
        self.factory.assert_not_called()

    def test_failure_leaves_no_published_artifacts(self):
        output = self.root / "build/output.wav"
        with patch(
            "lessons_in_cast_core.voice_design.voxcpm2._write_waveform",
            side_effect=RuntimeError("Invalid waveform"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Invalid waveform"):
                VoxCPM2VoiceDesigner(self.registry).generate(
                    VoiceDesignRequest("Hello", "Warm"), output,
                )
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".json").exists())
        self.assertEqual(list(output.parent.iterdir()), [])

    def test_cli_is_independent_of_game_configuration(self):
        output = self.root / "build/test.wav"
        result = VoiceDesignResult(output, output.with_suffix(".json"), 48000, 0.1)
        with patch("lessons_in_cast_core.voice_design.cli.load_model_registry", return_value=self.registry):
            with patch("lessons_in_cast_core.voice_design.cli.VoxCPM2VoiceDesigner") as factory:
                factory.return_value.generate.return_value = result
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = main([
                        "--root", str(self.root), "--text", "Hello",
                        "--reference-audio", "reference.wav", "--output", "build/test.wav",
                    ])
                self.assertEqual(code, 0)
                self.assertEqual(json.loads(stdout.getvalue())["audio_path"], str(output))
                request, destination = factory.return_value.generate.call_args.args
                self.assertEqual(request.reference_audio, self.root / "reference.wav")
                self.assertEqual(destination, output)
                factory.return_value.close.assert_called_once()

    def test_cli_reports_invalid_input_without_loading_backend(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main([
                "--root", str(self.root), "--text", " ",
                "--instruction", "Warm", "--output", "build/test.wav",
            ])
        self.assertEqual(code, 1)
        self.assertIn("text must not be empty", stderr.getvalue())
        self.factory.assert_not_called()
