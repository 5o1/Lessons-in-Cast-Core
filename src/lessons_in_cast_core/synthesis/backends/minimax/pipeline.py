"""Reusable MiniMax implementation of the voice-profile interface."""

from __future__ import annotations

from abc import ABC
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ....pronunciations import load_pronunciation_lexicon
from ...profiles.api import VoicePipeline
from ...profiles.context import VoiceProfileContext
from ...types import TtsJob
from ....performance import SpeechAdaptation
from .adapter import MiniMaxSpeechHttpSynthesizer
from .config import load_minimax_pipeline_config


class MiniMaxSpeechPipeline(VoicePipeline, ABC):
    """Base class for one character rendered through MiniMax Speech."""

    @property
    def configuration_path(self) -> str:
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
        self._config = load_minimax_pipeline_config(
            self._config_path,
            repository_root=root,
        )
        config = self._config
        self._model_definition = context.model_registry.require(config.model_id)
        self._pronunciation_lexicon_path = context.resolve_repository_path(
            "configs/pronunciations.toml"
        )
        language = {
            "English": "en",
            "Japanese": "ja",
            "Chinese": "zh",
            "Chinese,Yue": "yue",
        }.get(config.language_boost)
        self._pronunciations = load_pronunciation_lexicon(
            self._pronunciation_lexicon_path,
            repository_root=root,
        ).select(
            ("ipa", "kana", "romaji", "respelling"),
            language=language,
        )
        self._backend = MiniMaxSpeechHttpSynthesizer(
            model=self._model_definition.revision,
            voice_id=config.voice_id,
            audio_config=context.project_config.audio,
            pronunciations=self._pronunciations,
            endpoint=config.endpoint,
            api_key_environment=config.api_key_environment,
            language_boost=config.language_boost,
            text_normalization=config.text_normalization,
            base_speed=config.base_speed,
            base_volume=config.base_volume,
            base_pitch_semitones=config.base_pitch_semitones,
            voice_brightness=config.voice_brightness,
            voice_energy=config.voice_energy,
            voice_clarity=config.voice_clarity,
            sound_effect=config.sound_effect,
            timeout_seconds=config.timeout_seconds,
            maximum_retries=config.maximum_retries,
            arbitrary_emotions=config.arbitrary_emotions,
            retry_backoff_seconds=config.retry_backoff_seconds,
        )

    @property
    def configuration(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "character_id": self.character_id,
            "configuration_path": str(self._config_path),
            "model": self._model_definition.to_dict(),
            "profile": asdict(self._config),
            "backend": self._backend.configuration,
            "pronunciation_config_path": str(self._pronunciation_lexicon_path),
        }

    def adapt(self, job: TtsJob) -> SpeechAdaptation:
        if job.character_id != self.character_id:
            raise ValueError(
                f"{self.pipeline_id} cannot adapt {job.character_id!r}"
            )
        return self._backend.adapt(job)

    def render(self, job: TtsJob, artifact_root: Path) -> Path:
        if job.character_id != self.character_id:
            raise ValueError(
                f"{self.pipeline_id} cannot render {job.character_id!r}"
            )
        return self._backend.synthesize(job, artifact_root)
