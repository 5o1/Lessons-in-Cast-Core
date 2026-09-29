"""Polish-authored anchors compiled after polish, never sent to a TTS model."""

from .program import compile_keyframes, extract_anchors
from .renderer import KeyframeProcessor

__all__ = ["compile_keyframes", "extract_anchors", "KeyframeProcessor"]
