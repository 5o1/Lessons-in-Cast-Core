"""IndexTTS speech-synthesis backend and reusable profile pipeline."""

from .adapter import (
    IndexTtsSubprocessSynthesizer,
    apply_index_pronunciations,
    index_emotion_vector,
    normalize_index_emotion_vector,
)
from .pipeline import IndexTtsPipeline
from .factory import create_index_tts_backend

__all__ = [
    "IndexTtsPipeline",
    "create_index_tts_backend",
    "IndexTtsSubprocessSynthesizer",
    "apply_index_pronunciations",
    "index_emotion_vector",
    "normalize_index_emotion_vector",
]
