"""Semantic preset registration and backend-owned, offline preparation contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re
import tomllib
from pathlib import Path
from typing import Mapping

from .emotions import EMOTION_LABELS

PresetCatalog = Mapping[str, tuple[str, str]]


def load_emotion_catalog(path: Path) -> dict[str, tuple[str, str]]:
    """Overlay workspace definitions without changing process-global defaults."""
    result = dict(EMOTION_LABELS)
    if path.exists():
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        if set(document) - {"emotions"}:
            raise ValueError("Emotion catalog only accepts [emotions.NAME] tables")
        table = document.get("emotions", {})
        if not isinstance(table, dict):
            raise ValueError("emotions must be a TOML table")
        for name, value in table.items():
            if not isinstance(value, dict) or set(value) != {"description", "example"}:
                raise ValueError(f"Preset {name!r} requires description and example only")
            validate_preset(name, value["description"], value["example"])
            result[name] = (value["description"], value["example"])
    return result


def validate_preset(name: str, description: str, example: str) -> None:
    if not isinstance(name, str) or name == "arbitrary_emotion" or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("Use one lowercase semantic label, not a combination")
    if not all(isinstance(value, str) and value.strip() for value in (description, example)):
        raise ValueError("Preset description and example must be nonempty strings")


def add_emotion_preset(path: Path, name: str, description: str, example: str) -> None:
    """Persist a new definition. Existing labels must be edited explicitly."""
    validate_preset(name, description, example)
    if name in load_emotion_catalog(path):
        raise ValueError(f"Emotion preset already exists: {name}")
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    # JSON basic strings are valid TOML strings for ordinary text. Reject control
    # characters whose JSON escape spelling is not supported by TOML.
    addition = (f"\n[emotions.{name}]\n"
                f"description = {json.dumps(description, ensure_ascii=False)}\n"
                f"example = {json.dumps(example, ensure_ascii=False)}\n")
    tomllib.loads(original + addition)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(original + addition, encoding="utf-8")


class EmotionPresetPreprocessor(ABC):
    """Compile portable definitions to a backend-owned cache, without TTS."""

    @property
    @abstractmethod
    def backend(self) -> str:
        """Stable backend identifier."""

    @abstractmethod
    def prepare(self, catalog: PresetCatalog, output: Path, *, force: bool = False) -> Path:
        """Prepare all definitions; validate/resume cache, or raise on failure."""


def cli_main(args, root: Path) -> int:
    path = root / "configs/emotions.toml"
    if args.emotion_command == "add":
        add_emotion_preset(path, args.name, args.description, args.example)
        print(f"Registered {args.name} in {path}. Prepare each backend before enabling it in annotation.allowed_emotions.")
    elif args.emotion_command == "list":
        from .emotions import emotion_definitions
        catalog = load_emotion_catalog(path)
        print(json.dumps(emotion_definitions(catalog, catalog), indent=2))
    else:
        from .synthesis.backends.index_tts.emotion_preparation import IndexTtsEmotionPreprocessor
        compiler = IndexTtsEmotionPreprocessor.from_profile(root, args.profile, args.input_kind)
        print(compiler.prepare(load_emotion_catalog(path), root / args.output, force=args.force))
    return 0


def register_cli(commands) -> None:
    parser = commands.add_parser("emotion-presets", help="Register semantic presets and prepare backend caches")
    actions = parser.add_subparsers(dest="emotion_command", required=True)
    actions.add_parser("list")
    add = actions.add_parser("add")
    add.add_argument("name")
    add.add_argument("--description", required=True)
    add.add_argument("--example", required=True)
    prepare = actions.add_parser("prepare")
    prepare.add_argument("--backend", choices=["index-tts"], required=True)
    prepare.add_argument("--profile", type=Path, required=True,
                         help="Backend configuration; no character or reference audio is loaded")
    prepare.add_argument("--input-kind", choices=["description", "example", "combined"], default="description")
    prepare.add_argument("--output", type=Path, default=Path("build/cache/index-tts/emotion-vectors.json"))
    prepare.add_argument("--force", action="store_true", help="Regenerate all predictions")
