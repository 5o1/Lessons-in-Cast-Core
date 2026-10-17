"""Build IndexTTS backends from the same profile settings for every caller."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from ....config import PipelineConfig
from ....model_registry import ModelRegistry
from ....pronunciations import load_pronunciation_lexicon
from ...profiles.context import resolve_repository_path
from .adapter import IndexTtsSubprocessSynthesizer
from .config import IndexTtsPipelineConfig


def create_index_tts_backend(
    root: Path, config: IndexTtsPipelineConfig, project_config: PipelineConfig,
    registry: ModelRegistry, references: Mapping[str, Sequence[Path]],
) -> IndexTtsSubprocessSynthesizer:
    """Use profile inference, pronunciation and emotion settings without loading a model."""
    root = root.resolve()

    def resolve(value: str) -> Path:
        return resolve_repository_path(root, value)

    lexicon = load_pronunciation_lexicon(repository_root=root)
    model = registry.require(config.model_id)

    settings = config.reference_settings
    emotion_vectors = emotion_mapping_id = None
    if config.emotion_vectors_path:
        from ....emotion_presets import load_emotion_catalog
        from .emotion_preparation import load_vector_cache
        emotion_vectors, emotion_mapping_id = load_vector_cache(
            resolve(config.emotion_vectors_path), load_emotion_catalog(root / "configs/emotions.toml"),
            model_id=config.model_id,
            model_definition=model.to_dict(), source_root=resolve(config.source_root),
        )
    return IndexTtsSubprocessSynthesizer(
        emotion_vectors=emotion_vectors,
        emotion_mapping_id=emotion_mapping_id,
        repository_root=root,
        python_executable=resolve(config.python_executable),
        source_root=resolve(config.source_root),
        model_path=registry.resolve_path(config.model_id),
        references=references,
        audio_config=replace(
            project_config.audio,
            format=project_config.audio.intermediate_format,
        ),
        base_speed=config.base_speed,
        language=config.language,
        seed=config.seed,
        use_bf16=config.use_bf16,
        prepare_references=False,
        trim_reference_silence=settings.trim_silence,
        reference_trim_top_db=settings.top_db,
        reference_trim_padding_ms=settings.padding_ms,
        reference_gate_hold_ms=settings.gate_hold_ms,
        minimum_reference_speech_seconds=settings.minimum_speech_seconds,
        minimum_reference_duration_seconds=settings.minimum_duration_seconds,
        emotion_alpha=config.emotion_alpha,
        use_random_emotion=config.use_random_emotion,
        do_sample=config.do_sample,
        top_p=config.top_p,
        top_k=config.top_k,
        temperature=config.temperature,
        num_beams=config.num_beams,
        repetition_penalty=config.repetition_penalty,
        length_penalty=config.length_penalty,
        max_mel_tokens=config.max_mel_tokens,
        interval_silence_ms=config.interval_silence_ms,
        max_text_tokens_per_segment=config.max_text_tokens_per_segment,
        text_normalization=config.text_normalization,
        expressive_pause=config.expressive_pause,
        pronunciations=lexicon.for_system("arpabet"),
        pronunciation_rules=lexicon.select(("arpabet", "respelling"), language="en" if config.language.upper() == "EN" else None),
    )
