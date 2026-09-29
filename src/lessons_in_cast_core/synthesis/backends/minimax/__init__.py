"""MiniMax Speech HTTP backend and reusable profile pipeline."""

from .adapter import MiniMaxSpeechHttpSynthesizer
from .config import MiniMaxPipelineConfig, load_minimax_pipeline_config
from .pipeline import MiniMaxSpeechPipeline

__all__ = [
    "MiniMaxPipelineConfig",
    "MiniMaxSpeechHttpSynthesizer",
    "MiniMaxSpeechPipeline",
    "load_minimax_pipeline_config",
]
