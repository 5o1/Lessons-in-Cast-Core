"""Model-neutral speech synthesis planning and audio utilities."""

from ..performance import (
    AdaptationFidelity,
    FeatureAdaptation,
    PerformanceCue,
    PerformanceCueKind,
    SpeechAdaptation,
    SpeechPerformance,
    VocalMode,
)
from .api import AudioEffectProcessor, SpeechSynthesizer
from .audio import AudioQualityChecker, WaveRenderer
from .backends.gpt_sovits import GptSoVitsHttpSynthesizer, GptSoVitsReference
from .backends.index_tts import (
    IndexTtsPipeline,
    IndexTtsSubprocessSynthesizer,
    apply_index_pronunciations,
    index_emotion_vector,
    normalize_index_emotion_vector,
)
from .backends.minimax import (
    MiniMaxPipelineConfig,
    MiniMaxSpeechHttpSynthesizer,
    MiniMaxSpeechPipeline,
    load_minimax_pipeline_config,
)
from .planner import SynthesisPlanner
from .references import (
    ReferenceBuildError,
    ReferenceBuildResult,
    ReferenceBuildSettings,
    build_reference_from_directory,
    prepare_reference_sources,
)
from .profiles import (
    ReferenceBuildRequest,
    ReferenceVoicePipeline,
    VoicePipeline,
    VoiceProfileContext,
    VoiceProfileSynthesizer,
    load_configured_voice_profiles,
    load_voice_profile,
)
from .types import (
    AudioQualityResult,
    RenderTask,
    SynthesisIssue,
    SynthesisPlan,
    TtsJob,
)

__all__ = [
    "AdaptationFidelity",
    "AudioQualityChecker",
    "AudioQualityResult",
    "AudioEffectProcessor",
    "GptSoVitsHttpSynthesizer",
    "GptSoVitsReference",
    "FeatureAdaptation",
    "IndexTtsSubprocessSynthesizer",
    "IndexTtsPipeline",
    "apply_index_pronunciations",
    "index_emotion_vector",
    "normalize_index_emotion_vector",
    "MiniMaxPipelineConfig",
    "MiniMaxSpeechHttpSynthesizer",
    "MiniMaxSpeechPipeline",
    "load_minimax_pipeline_config",
    "PerformanceCue",
    "PerformanceCueKind",
    "prepare_reference_sources",
    "ReferenceBuildError",
    "ReferenceBuildRequest",
    "ReferenceVoicePipeline",
    "ReferenceBuildResult",
    "ReferenceBuildSettings",
    "build_reference_from_directory",
    "RenderTask",
    "SpeechSynthesizer",
    "SpeechAdaptation",
    "SpeechPerformance",
    "SynthesisIssue",
    "SynthesisPlan",
    "SynthesisPlanner",
    "TtsJob",
    "VoicePipeline",
    "VoiceProfileContext",
    "VoiceProfileSynthesizer",
    "VocalMode",
    "load_configured_voice_profiles",
    "load_voice_profile",
    "WaveRenderer",
]
