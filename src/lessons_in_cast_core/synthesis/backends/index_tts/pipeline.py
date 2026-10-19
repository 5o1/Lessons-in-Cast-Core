"""Reusable IndexTTS implementation of the voice-profile interface."""

from __future__ import annotations

import json
import os
import subprocess
from abc import ABC
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .factory import create_index_tts_backend
from .config import (
    load_index_tts_pipeline_config,
)
from ....paths import SOURCE_ROOT
from ...profiles.context import VoiceProfileContext
from ...references.builder import ReferenceBuildResult
from ...types import TtsJob
from ...references.voices import reference_audio_files, list_voice_references
from ...profiles.api import (
    ReferenceBuildRequest,
    ReferenceVoicePipeline,
)


class IndexTtsPipeline(ReferenceVoicePipeline, ABC):
    """Base class for one character rendered through IndexTTS 2.5."""

    @property
    def configuration_path(self) -> str:
        """Return the profile-relative IndexTTS configuration path."""

        return "config.toml"

    def __init__(self, context: VoiceProfileContext) -> None:
        if context.character.id != self.character_id:
            raise ValueError(
                f"{self.pipeline_id} handles {self.character_id!r}, not "
                f"{context.character.id!r}"
            )
        self._context = context
        root = context.repository_root.resolve()
        self._config_path = context.resolve_resource(self.configuration_path)
        self._config = load_index_tts_pipeline_config(
            self._config_path,
            repository_root=root,
        )
        config = self._config
        self._model_definition = context.model_registry.require(config.model_id)
        self._model_path = context.model_registry.resolve_path(config.model_id)
        resolve = context.resolve_repository_path
        self._pronunciation_lexicon_path = resolve(
            "configs/pronunciations.toml"
        )
        reference_resolver = (
            context.resolve_resource
            if config.reference_scope == "profile"
            else resolve
        )
        self._reference_source = reference_resolver(
            config.reference_source_directory
        )
        self._reference_sources = tuple(
            reference_resolver(
                Path(config.reference_source_directory) / source
            )
            for source in config.reference_sources
        )
        self._reference_path = reference_resolver(config.reference_path)
        self._backend = create_index_tts_backend(
            root, config, context.project_config, context.model_registry,
            {self.character_id: (self._reference_path,)},
        )

    @property
    def configuration(self) -> dict[str, Any]:
        config = self._config
        return {
            "pipeline_id": self.pipeline_id,
            "character_id": self.character_id,
            "configuration_path": str(self._config_path),
            "model": self._model_definition.to_dict(),
            "backend": self._backend.configuration,
            "pronunciation_config_path": str(
                self._pronunciation_lexicon_path
            ),
            "reference_source_directory": str(self._reference_source),
            "reference_sources": [
                str(path) for path in self._reference_sources
            ],
            "reference_path": str(self._reference_path),
            "reference_settings": asdict(config.reference_settings),
        }


    @property
    def default_reference_request(self) -> ReferenceBuildRequest:
        return ReferenceBuildRequest(
            input_directory=self._reference_source,
            output_path=self._reference_path,
        )

    def prepare(self) -> tuple[Path, ...]:
        if not self._reference_path.is_file():
            source_arguments = [
                argument
                for source in self._reference_sources
                for argument in ("--source", str(source))
            ]
            self._run_reference_builder(
                source_arguments,
                self._reference_path,
            )
        return tuple(dict.fromkeys((self._reference_path, *reference_audio_files(self._reference_path))))

    def list_voice_tags(self) -> tuple[str, ...]:
        return tuple(list_voice_references(self._reference_path))

    def override_reference_audio(self, path: Path) -> None:
        reference = path.expanduser().resolve()
        if not reference.is_file():
            raise FileNotFoundError(f"Reference override is missing: {reference}")
        self._reference_path = reference
        self._backend.set_references(self.character_id, (reference,))

    def build_reference(
        self,
        request: ReferenceBuildRequest,
    ) -> ReferenceBuildResult:
        return self._run_reference_builder(
            ["--input-dir", str(request.input_directory.resolve())],
            request.output_path,
        )

    def _run_reference_builder(
        self,
        source_arguments: list[str],
        output_path: Path,
    ) -> ReferenceBuildResult:
        root = self._context.repository_root.resolve()
        config = self._config
        settings = config.reference_settings
        python_executable = (root / config.python_executable).absolute()
        if not python_executable.exists():
            raise FileNotFoundError(
                f"Voice-profile Python executable is missing: "
                f"{python_executable}"
            )
        environment = os.environ.copy()
        python_paths = (str(root), str(SOURCE_ROOT))
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            (*python_paths, *((existing,) if existing else ()))
        )
        command = [
            str(python_executable),
            "-m",
            "lessons_in_cast_core.synthesis.references.cli",
            *source_arguments,
            "--output",
            str(output_path.resolve()),
            "--top-db",
            str(settings.top_db),
            "--padding-ms",
            str(settings.padding_ms),
            "--gate-hold-ms",
            str(settings.gate_hold_ms),
            "--minimum-speech-seconds",
            str(settings.minimum_speech_seconds),
            "--minimum-duration-seconds",
            str(settings.minimum_duration_seconds),
        ]
        if not settings.trim_silence:
            command.append("--no-trim-silence")
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8", errors="replace",
            check=False,
        )
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"{self.pipeline_id} reference builder failed: {detail}"
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{self.pipeline_id} reference builder returned invalid JSON"
            ) from exc
        return ReferenceBuildResult.from_dict(result)

    def render(self, job: TtsJob, artifact_root: Path) -> Path:
        if job.character_id != self.character_id:
            raise ValueError(
                f"{self.pipeline_id} cannot render {job.character_id!r}"
            )
        self.prepare()
        return self._backend.synthesize(job, artifact_root)

    @property
    def last_cache_hit(self) -> bool:
        return self._backend.last_cache_hit

    def adapt(self, job: TtsJob):
        if job.character_id != self.character_id:
            raise ValueError(
                f"{self.pipeline_id} cannot adapt {job.character_id!r}"
            )
        return self._backend.adapt(job)

    def close(self) -> None:
        self._backend.close()
