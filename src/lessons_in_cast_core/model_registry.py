"""Runtime registry for local model artifacts and hosted model services."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import ConfigurationError, find_repository_root


_MODEL_ID = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    """One stable local or hosted model identity."""

    id: str
    path: str | None
    provider: str
    repository: str
    revision: str
    license: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "id": self.id,
            "path": self.path,
            "provider": self.provider,
            "repository": self.repository,
            "revision": self.revision,
            "license": self.license,
        }


class ModelRegistry:
    """Resolve stable model IDs to local artifacts or hosted identities."""

    def __init__(
        self,
        repository_root: Path,
        models: Mapping[str, ModelDefinition],
    ) -> None:
        self._repository_root = repository_root.resolve()
        self._models = dict(models)

    @property
    def configuration(self) -> dict[str, dict[str, str | None]]:
        return {
            model_id: definition.to_dict()
            for model_id, definition in sorted(self._models.items())
        }

    def require(self, model_id: str) -> ModelDefinition:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise ConfigurationError(
                f"Unknown model ID {model_id!r}; define it in "
                "configs/model_sources.toml"
            ) from exc

    def resolve_path(self, model_id: str) -> Path:
        definition = self.require(model_id)
        if not definition.path:
            raise ConfigurationError(
                f"Model {model_id!r} is hosted and has no local artifact path"
            )
        configured = Path(definition.path).expanduser()
        if configured.is_absolute():
            return configured.resolve()
        return (self._repository_root / configured).resolve()


def load_model_registry(
    path: Path = Path("configs/model_sources.toml"),
    *,
    repository_root: Path | None = None,
) -> ModelRegistry:
    """Load the machine-local model registry used by voice profiles."""

    root = (repository_root or find_repository_root()).resolve()
    resolved = path if path.is_absolute() else root / path
    try:
        with resolved.open("rb") as source:
            data = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"Unable to load {resolved}: {exc}") from exc

    raw_models = data.get("models")
    if not isinstance(raw_models, dict):
        raise ConfigurationError(f"{resolved}: [models] table is required")

    models: dict[str, ModelDefinition] = {}
    required = ("provider", "repository", "revision", "license")
    for model_id, raw in raw_models.items():
        if not isinstance(model_id, str) or not _MODEL_ID.fullmatch(model_id):
            raise ConfigurationError(f"{resolved}: invalid model ID {model_id!r}")
        if not isinstance(raw, dict):
            raise ConfigurationError(f"{resolved}: models.{model_id} must be a table")
        values: dict[str, str] = {}
        for field in required:
            value: Any = raw.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(
                    f"{resolved}: models.{model_id}.{field} must be a non-empty string"
                )
            values[field] = value
        path_value = raw.get("path")
        if path_value is not None and (
            not isinstance(path_value, str) or not path_value.strip()
        ):
            raise ConfigurationError(
                f"{resolved}: models.{model_id}.path must be a non-empty "
                "string or omitted"
            )
        models[model_id] = ModelDefinition(
            id=model_id,
            path=path_value,
            **values,
        )

    return ModelRegistry(root, models)
