"""GPT-SoVITS speech-synthesis backend."""

from .adapter import GptSoVitsHttpSynthesizer, GptSoVitsReference

__all__ = ["GptSoVitsHttpSynthesizer", "GptSoVitsReference"]
