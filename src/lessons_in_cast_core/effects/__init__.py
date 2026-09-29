"""Backend-neutral audio effects applied after speech synthesis."""

from .catalog import DESCRIPTIONS, EffectError, EffectLibrary, EffectSpec, load_effect_library
from .processor import CoreEffectProcessor

__all__ = ["CoreEffectProcessor", "EffectError", "EffectLibrary", "EffectSpec", "load_effect_library", "DESCRIPTIONS"]
