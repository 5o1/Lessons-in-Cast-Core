"""Public contract for replaceable voice-design backends."""

from pathlib import Path
from typing import Any, Protocol

from .types import VoiceDesignRequest, VoiceDesignResult


class VoiceDesigner(Protocol):
    """Create reference auditions independently of dialogue synthesis jobs."""

    @property
    def name(self) -> str: ...

    @property
    def configuration(self) -> dict[str, Any]: ...

    def generate(
        self, request: VoiceDesignRequest, output_path: Path
    ) -> VoiceDesignResult: ...

    def close(self) -> None: ...
