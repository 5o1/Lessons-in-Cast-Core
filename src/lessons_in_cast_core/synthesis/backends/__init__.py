"""Built-in speech synthesis backend adapters."""

from .minimax import MiniMaxSpeechHttpSynthesizer, MiniMaxSpeechPipeline

__all__ = ["MiniMaxSpeechHttpSynthesizer", "MiniMaxSpeechPipeline"]
