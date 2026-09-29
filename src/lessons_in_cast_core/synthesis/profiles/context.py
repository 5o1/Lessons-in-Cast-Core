"""Workspace services exposed to user-defined voice profiles."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ...characters import CharacterDefinition
from ...config import PipelineConfig
from ...model_registry import ModelRegistry


@dataclass(frozen=True, slots=True)
class VoiceProfileContext:
    """Dependencies and safe path resolution supplied to a voice profile."""

    repository_root: Path
    entrypoint: Path
    profile_root: Path
    character: CharacterDefinition
    project_config: PipelineConfig
    model_registry: ModelRegistry

    def resolve_resource(self, relative_path: str | Path) -> Path:
        """Resolve a bundle-owned path without allowing it to escape the bundle."""

        path = Path(relative_path)
        if path.is_absolute():
            raise ValueError(f"Profile resource path must be relative: {path}")
        resolved = (self.profile_root / path).resolve()
        if not resolved.is_relative_to(self.profile_root):
            raise ValueError(f"Profile resource escapes its bundle: {path}")
        return resolved

    def resolve_repository_path(self, relative_path: str | Path) -> Path:
        """Resolve a workspace dependency while keeping it inside the repository."""

        path = Path(relative_path)
        if path.is_absolute():
            raise ValueError(f"Repository path must be relative: {path}")
        resolved = Path(os.path.abspath(self.repository_root / path))
        if not resolved.is_relative_to(self.repository_root):
            raise ValueError(f"Repository path escapes the workspace: {path}")
        return resolved
