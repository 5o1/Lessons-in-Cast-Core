"""Configuration loading and validation for pipeline workspaces."""

from __future__ import annotations

import tomllib
import re
from urllib.parse import urlsplit
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .effects import EffectError, EffectLibrary, load_effect_library


class ConfigurationError(ValueError):
    """Raised when a project configuration is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class WorkspaceConfig:
    repository_root: Path
    release_path: Path


@dataclass(frozen=True, slots=True)
class BatchingConfig:
    target_size: int = 50
    context_before: int = 12
    context_after: int = 12
    max_characters: int = 40_000


@dataclass(frozen=True, slots=True)
class AnnotationConfig:
    allowed_emotions: frozenset[str]
    allowed_effects: frozenset[str]
    maximum_length_ratio: float = 4.0
    minimum_length_ratio: float = 0.15
    emotion_presets: dict[str, tuple[str, str]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CodexConfig:
    batches_per_packet: int = 2
    source_files: tuple[str, ...] = ()
    prompt_path: str = "prompts/codex_dialogue_cleanup_v4.md"
    prompt_version: str = "cleaning-v4"
    polish_prompt_path: str = "prompts/codex_dialogue_polish.md"
    polish_prompt_version: str = "polish-v3"


@dataclass(frozen=True, slots=True)
class AudioConfig:
    format: str = "wav"
    intermediate_format: str = "wav"
    sample_rate: int = 44_100
    channels: int = 1
    sample_width: int = 2
    bitrate_kbps: int = 48
    ffmpeg_executable: str = "ffmpeg"
    ffprobe_executable: str = "ffprobe"
    minimum_duration_seconds: float = 0.05
    maximum_duration_seconds: float = 120.0


@dataclass(frozen=True, slots=True)
class GalgameConfig:
    backend: str = "renpy"


@dataclass(frozen=True, slots=True)
class AnnotationApiConfig:
    base_url: str = ""
    model: str = ""
    api_key_environment: str = "LLM_API_KEY"
    response_format: str = "json_schema"
    context_window_tokens: int = 65536
    max_completion_tokens: int = 16384
    completion_token_parameter: str = "max_completion_tokens"
    reasoning_split: bool | None = None
    max_repair_attempts: int = 2
    timeout_seconds: int = 180
    parallel_workers: int = 1


@dataclass(frozen=True, slots=True)
class AnnotationContextConfig:
    recent_lines: int = 24
    lookahead_lines: int = 12
    memory_tokens: int = 3000


@dataclass(frozen=True, slots=True)
class AnnotationStageConfig:
    backend: str = "codex"
    api: AnnotationApiConfig = AnnotationApiConfig()
    context: AnnotationContextConfig = AnnotationContextConfig()


@dataclass(frozen=True, slots=True)
class KantokuConfig:
    directory: str = "kantoku"


def _annotation_stage(value: dict, name: str) -> AnnotationStageConfig:
    try:
        result = AnnotationStageConfig(**{**value,
            "api": AnnotationApiConfig(**value.get("api", {})),
            "context": AnnotationContextConfig(**value.get("context", {}))})
    except (TypeError, AttributeError) as exc:
        raise ConfigurationError(f"Invalid {name} configuration: {exc}") from exc
    if result.backend not in ("codex", "api"):
        raise ConfigurationError(f"{name}.backend must be codex or api")
    api = result.api
    for key in ("base_url", "model", "api_key_environment"):
        if not isinstance(getattr(api, key), str):
            raise ConfigurationError(f"{name}.api.{key} must be a string")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api.api_key_environment):
        raise ConfigurationError(f"{name}.api.api_key_environment is invalid")
    if api.base_url:
        url = urlsplit(api.base_url)
        if url.scheme not in ("https", "http") or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ConfigurationError(f"{name}.api.base_url must be an HTTP(S) URL without credentials or query")
    if result.backend == "api" and (not api.base_url or not api.model.strip()):
        raise ConfigurationError(f"{name}.api requires base_url and model")
    if api.response_format not in ("json_schema", "json_object"):
        raise ConfigurationError(f"{name}.api.response_format must be json_schema or json_object")
    if api.completion_token_parameter not in ("max_tokens", "max_completion_tokens"):
        raise ConfigurationError(f"{name}.api.completion_token_parameter is invalid")
    if api.reasoning_split is not None and type(api.reasoning_split) is not bool:
        raise ConfigurationError(f"{name}.api.reasoning_split must be a boolean")
    for key, minimum in (("context_window_tokens", 1), ("max_completion_tokens", 1),
                         ("max_repair_attempts", 0), ("timeout_seconds", 1),
                         ("parallel_workers", 1)):
        number = getattr(api, key)
        if type(number) is not int or number < minimum:
            raise ConfigurationError(f"{name}.api.{key} must be an integer >= {minimum}")
    for key in ("recent_lines", "lookahead_lines", "memory_tokens"):
        number = getattr(result.context, key)
        if type(number) is not int or number < 0:
            raise ConfigurationError(f"{name}.context.{key} must be a nonnegative integer")
    if api.max_completion_tokens + result.context.memory_tokens + 1024 >= api.context_window_tokens:
        raise ConfigurationError(f"{name}: context window must leave room for input and output")
    return result


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    batching: BatchingConfig
    annotation: AnnotationConfig
    audio: AudioConfig
    codex: CodexConfig = CodexConfig()
    galgame: GalgameConfig = GalgameConfig()
    cleaning: AnnotationStageConfig = AnnotationStageConfig()
    polish: AnnotationStageConfig = AnnotationStageConfig()
    kantoku: KantokuConfig = KantokuConfig()
    repository_root: Path | None = None
    effects: EffectLibrary = field(default_factory=EffectLibrary)


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as source:
            return tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"Unable to load {path}: {exc}") from exc


def find_repository_root(start: Path | None = None) -> Path:
    """Find the nearest directory containing the workspace configuration."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "configs" / "workspace.toml").is_file():
            return candidate
    raise ConfigurationError("Could not find configs/workspace.toml")


def load_workspace_config(
    path: Path = Path("configs/workspace.toml"),
    *,
    repository_root: Path | None = None,
) -> WorkspaceConfig:
    root = (repository_root or find_repository_root()).resolve()
    resolved_path = path if path.is_absolute() else root / path
    data = _load_toml(resolved_path)
    try:
        configured_release = Path(data["workspace"]["current_game_release"])
    except (KeyError, TypeError) as exc:
        raise ConfigurationError(
            f"{resolved_path}: workspace.current_game_release is required"
        ) from exc
    release_path = (
        configured_release
        if configured_release.is_absolute()
        else root / configured_release
    )
    return WorkspaceConfig(root, release_path.resolve())


def load_dialogue_sources(
    path: Path = Path("configs/dialogue_sources.toml"),
    *,
    repository_root: Path | None = None,
    include_excluded: bool = False,
) -> tuple[Path, ...]:
    root = (repository_root or find_repository_root()).resolve()
    resolved_path = path if path.is_absolute() else root / path
    data = _load_toml(resolved_path)
    sources = data.get("sources")
    if not isinstance(sources, dict):
        raise ConfigurationError(f"{resolved_path}: [sources] table is required")
    result: list[Path] = []
    for group, values in sources.items():
        if group == "excluded_by_default" and not include_excluded:
            continue
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ConfigurationError(
                f"{resolved_path}: sources.{group} must be an array"
            )
        result.extend(Path(value) for value in values)
    if len(result) != len(set(result)):
        raise ConfigurationError(f"{resolved_path}: duplicate dialogue source paths")
    return tuple(result)


def load_pipeline_config(
    path: Path = Path("configs/pipeline.toml"),
    *,
    repository_root: Path | None = None,
) -> PipelineConfig:
    root = (repository_root or find_repository_root()).resolve()
    resolved_path = path if path.is_absolute() else root / path
    data = _load_toml(resolved_path)
    batching = data.get("batching", {})
    from .emotion_presets import load_emotion_catalog
    emotion_catalog = load_emotion_catalog(root / "configs/emotions.toml")
    annotation = data.get("annotation", {})
    audio = data.get("audio", {})
    codex = data.get("codex", {})
    galgame = data.get("galgame", {})
    effects_config = data.get("effects", {})
    if not isinstance(effects_config, dict) or effects_config.keys() - {"config_path"}:
        raise ConfigurationError("effects must contain only config_path")
    effects_path = effects_config.get("config_path")
    if effects_path is not None and (not isinstance(effects_path, str) or not effects_path):
        raise ConfigurationError("effects.config_path must be a non-empty path")
    try:
        effects = load_effect_library(root / effects_path if effects_path else None)
    except EffectError as exc:
        raise ConfigurationError(str(exc)) from exc
    codex_source_files = codex.get("source_files", [])
    if not isinstance(codex_source_files, list) or not all(
        isinstance(value, str) and value for value in codex_source_files
    ):
        raise ConfigurationError("codex.source_files must be an array of paths")
    if len(codex_source_files) != len(set(codex_source_files)):
        raise ConfigurationError("codex.source_files contains duplicate paths")
    result = PipelineConfig(
        batching=BatchingConfig(**batching),
        annotation=AnnotationConfig(
            emotion_presets=emotion_catalog,
            allowed_emotions=frozenset(annotation.get("allowed_emotions", ())),
            allowed_effects=frozenset(annotation.get("allowed_effects", ())),
            maximum_length_ratio=annotation.get("maximum_length_ratio", 4.0),
            minimum_length_ratio=annotation.get("minimum_length_ratio", 0.15),
        ),
        audio=AudioConfig(**audio),
        codex=CodexConfig(
            batches_per_packet=codex.get("batches_per_packet", 2),
            source_files=tuple(codex_source_files),
            prompt_path=codex.get(
                "prompt_path", "prompts/codex_dialogue_cleanup_v4.md"
            ),
            prompt_version=codex.get("prompt_version", "cleaning-v4"),
            polish_prompt_path=codex.get("polish_prompt_path", "prompts/codex_dialogue_polish.md"),
            polish_prompt_version=codex.get("polish_prompt_version", "polish-v3"),
        ),
        galgame=GalgameConfig(backend=galgame.get("backend", "renpy")),
        cleaning=_annotation_stage(data.get("cleaning", {}), "cleaning"),
        polish=_annotation_stage(data.get("polish", {}), "polish"),
        kantoku=KantokuConfig(**data.get("kantoku", {})),
        repository_root=root,
        effects=effects,
    )
    directory = result.kantoku.directory
    if not isinstance(directory, str) or not directory.strip() or Path(directory).is_absolute() or ".." in Path(directory).parts:
        raise ConfigurationError("kantoku.directory must be a safe relative path")
    if result.batching.target_size < 1:
        raise ConfigurationError("batching.target_size must be positive")
    if result.batching.context_before < 0 or result.batching.context_after < 0:
        raise ConfigurationError("batching context sizes cannot be negative")
    if result.batching.max_characters < 1:
        raise ConfigurationError("batching.max_characters must be positive")
    if result.annotation.minimum_length_ratio < 0:
        raise ConfigurationError("annotation.minimum_length_ratio cannot be negative")
    if result.annotation.maximum_length_ratio < result.annotation.minimum_length_ratio:
        raise ConfigurationError("annotation length ratio range is invalid")
    if result.codex.batches_per_packet < 1:
        raise ConfigurationError("codex.batches_per_packet must be positive")
    for field, value in {
        "codex.prompt_path": result.codex.prompt_path,
        "codex.prompt_version": result.codex.prompt_version,
        "codex.polish_prompt_path": result.codex.polish_prompt_path,
        "codex.polish_prompt_version": result.codex.polish_prompt_version,
    }.items():
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f"{field} must be a non-empty string")
    prompt_path = Path(result.codex.prompt_path)
    if prompt_path.is_absolute() or ".." in prompt_path.parts:
        raise ConfigurationError("codex.prompt_path must be a safe relative path")
    if Path(result.codex.polish_prompt_path).is_absolute() or ".." in Path(result.codex.polish_prompt_path).parts:
        raise ConfigurationError("codex.polish_prompt_path must be a safe relative path")
    if not result.galgame.backend.strip():
        raise ConfigurationError("galgame.backend cannot be empty")
    if not result.annotation.allowed_emotions:
        raise ConfigurationError("annotation.allowed_emotions cannot be empty")
    from .emotions import emotion_definitions
    emotion_definitions(result.annotation.allowed_emotions, result.annotation.emotion_presets)
    if result.audio.format.strip(".") == "":
        raise ConfigurationError("audio.format cannot be empty")
    if result.audio.format.lstrip(".").lower() not in {"wav", "opus"}:
        raise ConfigurationError("audio.format must be wav or opus")
    if result.audio.intermediate_format.lstrip(".").lower() != "wav":
        raise ConfigurationError("audio.intermediate_format must be wav")
    if result.audio.sample_rate < 1:
        raise ConfigurationError("audio.sample_rate must be positive")
    if result.audio.channels < 1:
        raise ConfigurationError("audio.channels must be positive")
    if result.audio.sample_width not in {1, 2, 3, 4}:
        raise ConfigurationError("audio.sample_width must be between 1 and 4")
    if result.audio.bitrate_kbps < 6:
        raise ConfigurationError("audio.bitrate_kbps must be at least 6")
    if not result.audio.ffmpeg_executable:
        raise ConfigurationError("audio.ffmpeg_executable cannot be empty")
    if not result.audio.ffprobe_executable:
        raise ConfigurationError("audio.ffprobe_executable cannot be empty")
    if result.audio.minimum_duration_seconds < 0:
        raise ConfigurationError("audio.minimum_duration_seconds cannot be negative")
    if (
        result.audio.maximum_duration_seconds
        < result.audio.minimum_duration_seconds
    ):
        raise ConfigurationError("audio duration range is invalid")
    return result
