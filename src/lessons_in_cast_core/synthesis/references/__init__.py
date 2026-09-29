"""Reference-audio construction and content-addressed caching."""

from .builder import (
    ReferenceBuildError,
    ReferenceBuildResult,
    ReferenceBuildSettings,
    build_reference_from_directory,
    prepare_reference_sources,
)
from .cache import ensure_reference_from_directory

__all__ = [
    "ReferenceBuildError",
    "ReferenceBuildResult",
    "ReferenceBuildSettings",
    "build_reference_from_directory",
    "ensure_reference_from_directory",
    "prepare_reference_sources",
]
