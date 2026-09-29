"""IndexTTS speech-synthesis backend and reusable profile pipeline."""

from .adapter import (
    IndexTtsSubprocessSynthesizer,
    apply_index_pronunciations,
    index_emotion_vector,
    normalize_index_emotion_vector,
)
from .pipeline import IndexTtsPipeline

__all__ = [
    "IndexTtsPipeline",
    "IndexTtsSubprocessSynthesizer",
    "apply_index_pronunciations",
    "index_emotion_vector",
    "normalize_index_emotion_vector",
]
