"""Construct configured galgame backend implementations."""

from __future__ import annotations

from collections.abc import Callable

from ..config import ConfigurationError
from .api import GalgameBackend


def _create_renpy_backend() -> GalgameBackend:
    from .renpy import RenPyBackend

    return RenPyBackend()


_BACKEND_FACTORIES: dict[str, Callable[[], GalgameBackend]] = {
    "renpy": _create_renpy_backend,
}


def available_galgame_backends() -> tuple[str, ...]:
    """Return the registered built-in backend IDs."""

    return tuple(sorted(_BACKEND_FACTORIES))


def load_galgame_backend(backend_id: str) -> GalgameBackend:
    """Create one built-in galgame backend by its stable identifier."""

    try:
        factory = _BACKEND_FACTORIES[backend_id]
    except KeyError as exc:
        raise ConfigurationError(
            f"Unsupported galgame backend: {backend_id!r}; "
            f"available: {available_galgame_backends()!r}"
        ) from exc
    return factory()
