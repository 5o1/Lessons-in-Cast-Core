"""Backend-neutral galgame integration and built-in engine adapters."""

from .api import (
    DialogueExtractionError,
    DialogueExtractionRequest,
    DialogueExtractionResult,
    DialogueExtractor,
    GalgameBackend,
    GalgameInstallationResult,
)
from .loader import available_galgame_backends, load_galgame_backend
from .manifest import VoiceManifestWriter

__all__ = [
    "DialogueExtractionError",
    "DialogueExtractionRequest",
    "DialogueExtractionResult",
    "DialogueExtractor",
    "GalgameBackend",
    "GalgameInstallationResult",
    "VoiceManifestWriter",
    "available_galgame_backends",
    "load_galgame_backend",
]
