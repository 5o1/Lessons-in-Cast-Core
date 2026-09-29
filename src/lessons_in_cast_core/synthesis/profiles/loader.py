"""Load user-defined voice profiles and route jobs through their pipelines."""

from __future__ import annotations

import hashlib
import importlib.util
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ...performance import SpeechAdaptation
from ...characters import CharacterDefinition
from ...config import ConfigurationError, PipelineConfig
from ...model_registry import ModelRegistry, load_model_registry
from ..types import TtsJob
from .api import VoicePipeline
from .context import VoiceProfileContext


def load_voice_profile(
    repository_root: Path,
    entrypoint_path: Path,
    character: CharacterDefinition,
    project_config: PipelineConfig,
    *,
    model_registry: ModelRegistry,
) -> VoicePipeline:
    """Load one profile entrypoint and validate its public contract."""

    root = repository_root.resolve()
    resolved = (
        entrypoint_path
        if entrypoint_path.is_absolute()
        else root / entrypoint_path
    )
    resolved = resolved.resolve()
    profiles_root = (root / "profiles").resolve()
    if not resolved.is_relative_to(profiles_root):
        raise ValueError(f"Voice profile must be inside {profiles_root}: {resolved}")
    if not resolved.is_file() or resolved.suffix != ".py":
        raise FileNotFoundError(f"Voice profile entrypoint is missing: {resolved}")

    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
    module_name = f"lessons_in_cast_core_voice_profile_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load voice profile: {resolved}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError(
            f"Unable to execute voice profile {resolved}: {exc}"
        ) from exc

    factory = getattr(module, "create_pipeline", None)
    if not callable(factory):
        raise TypeError(f"{resolved} must export create_pipeline(context)")
    context = VoiceProfileContext(
        repository_root=root,
        entrypoint=resolved,
        profile_root=resolved.parent,
        character=character,
        project_config=project_config,
        model_registry=model_registry,
    )
    pipeline = factory(context)
    if not isinstance(pipeline, VoicePipeline):
        raise TypeError(
            f"{resolved} create_pipeline() must return VoicePipeline"
        )
    if pipeline.character_id != character.id:
        raise ValueError(
            f"{resolved} handles {pipeline.character_id!r}, expected "
            f"{character.id!r}"
        )
    return pipeline


class NoVoiceProfilesError(ConfigurationError):
    """Raised when no character selects a voice profile."""


class VoiceProfileSynthesizer:
    """Route speech jobs by character ID and resolved profile entrypoint."""

    def __init__(
        self,
        pipelines: dict[tuple[str, str], VoicePipeline],
        default_profiles: dict[str, str],
    ) -> None:
        if not pipelines:
            raise NoVoiceProfilesError("At least one character voice profile is required")
        self._pipelines = dict(pipelines)
        self._default_profiles = dict(default_profiles)
        self._active_pipeline: VoicePipeline | None = None
        self._closed_pipeline_ids: set[int] = set()

    @property
    def name(self) -> str:
        return "character-voice-profiles"

    def _display_key(self, character_id: str, entrypoint: str) -> str:
        if self._default_profiles.get(character_id) == entrypoint:
            return character_id
        return f"{character_id}@{entrypoint}"

    @property
    def configuration(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "profiles": {
                self._display_key(character_id, entrypoint): {
                    "character_id": character_id,
                    "entrypoint": entrypoint,
                    "pipeline": pipeline.configuration,
                }
                for (character_id, entrypoint), pipeline in sorted(
                    self._pipelines.items()
                )
            },
        }

    def prepare(self) -> dict[str, tuple[Path, ...]]:
        """Prepare generated dependencies for every configured profile."""

        return {
            self._display_key(character_id, entrypoint): pipeline.prepare()
            for (character_id, entrypoint), pipeline in sorted(self._pipelines.items())
        }

    def supports(
        self,
        character_id: str,
        entrypoint: str | None = None,
    ) -> bool:
        """Return whether a resolved character/profile route can be rendered."""

        resolved = entrypoint or self._default_profiles.get(character_id, "")
        return (character_id, resolved) in self._pipelines

    def order_jobs(self, jobs: Iterable[TtsJob]) -> tuple[TtsJob, ...]:
        """Group jobs by profile so large models are loaded once per route."""

        return tuple(
            sorted(
                jobs,
                key=lambda job: (
                    job.voice_profile
                    or self._default_profiles.get(job.character_id, ""),
                    job.character_id,
                    job.id,
                ),
            )
        )

    def _pipeline_for(self, job: TtsJob) -> VoicePipeline:
        entrypoint = job.voice_profile or self._default_profiles.get(
            job.character_id,
            "",
        )
        pipeline = self._pipelines.get((job.character_id, entrypoint))
        if pipeline is None:
            raise RuntimeError(
                f"No voice profile is configured for {job.character_id!r} "
                f"at entrypoint {entrypoint!r}"
            )
        return pipeline

    def _activate(self, pipeline: VoicePipeline) -> None:
        if self._active_pipeline is pipeline:
            return
        if self._active_pipeline is not None:
            self._active_pipeline.close()
            self._closed_pipeline_ids.add(id(self._active_pipeline))
        self._active_pipeline = pipeline
        self._closed_pipeline_ids.discard(id(pipeline))

    def synthesize(self, job: TtsJob, artifact_root: Path) -> Path:
        pipeline = self._pipeline_for(job)
        self._activate(pipeline)
        return pipeline.render(job, artifact_root)

    def adapt(self, job: TtsJob) -> SpeechAdaptation:
        """Route backend lowering while retaining a serializable audit record."""

        pipeline = self._pipeline_for(job)
        return pipeline.adapt(job)

    def close(self) -> None:
        unique = {id(pipeline): pipeline for pipeline in self._pipelines.values()}
        for identity, pipeline in unique.items():
            if identity not in self._closed_pipeline_ids:
                pipeline.close()
        self._active_pipeline = None
        self._closed_pipeline_ids = set(unique)


def load_configured_voice_profiles(
    repository_root: Path,
    project_config: PipelineConfig,
    characters: dict[str, CharacterDefinition],
    *,
    model_registry: ModelRegistry | None = None,
) -> VoiceProfileSynthesizer:
    """Load every distinct global or contextual character voice profile."""

    models = model_registry
    if models is None:
        models = load_model_registry(repository_root=repository_root)
    pipelines: dict[tuple[str, str], VoicePipeline] = {}
    default_profiles: dict[str, str] = {}
    from ...kantoku import Kantoku
    directed = list(Kantoku(repository_root, project_config, characters).profile_variants())
    for character_id, base in characters.items():
        if base.default_voice_profile:
            default_profiles[character_id] = base.default_voice_profile
        for character in (*base.configured_variants(), *(v for v in directed if v.id == character_id)):
            entrypoint = character.default_voice_profile
            if not entrypoint:
                continue
            route = (character_id, entrypoint)
            if route in pipelines:
                continue
            pipelines[route] = load_voice_profile(
                repository_root,
                Path(entrypoint),
                character,
                project_config,
                model_registry=models,
            )
    return VoiceProfileSynthesizer(pipelines, default_profiles)
