"""Render effects or compare a reusable set of effects on one dry take."""

import argparse
import json
from pathlib import Path
import shutil
import tomllib

from ..hashing import file_hash
from .catalog import DESCRIPTIONS, EffectError, load_effect_library
from .processor import CoreEffectProcessor
from .pcm import read_pcm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Effects TOML; omitted uses core defaults")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="Print resolved presets and chains")
    render = commands.add_parser("render", help="Apply an ordered chain to a PCM WAV")
    render.add_argument("--input", type=Path, help="Omit for an effect-only beep")
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--effects", nargs="+", required=True)
    audition = commands.add_parser("audition", help="Compare fixture chains using an existing dry take; no TTS")
    audition.add_argument("--input", type=Path, required=True)
    audition.add_argument("--project", type=Path, default=Path("auditions/effects/project.toml"))
    audition.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    try:
        library = load_effect_library(args.config)
        processor = CoreEffectProcessor(library, ffmpeg_executable=args.ffmpeg)
        if args.command == "list":
            print(json.dumps({**library.to_dict(), "descriptions": DESCRIPTIONS}, indent=2))
            return 0
        if args.command == "render":
            if args.output.exists() or args.output.with_suffix(".effects.json").exists():
                raise EffectError("Output already exists; choose a new output path")
            print(processor.process(args.input, args.output, tuple(args.effects)))
            return 0
        from ..config import find_repository_root
        from ..audition.types import slug
        root = find_repository_root()
        slug(args.run)
        with args.project.open("rb") as stream:
            project = tomllib.load(stream)
        cases = project.get("cases", {})
        if not cases:
            raise EffectError("Audition fixture needs a non-empty cases table")
        for name, chain in cases.items():
            slug(name)
            if not isinstance(chain, list) or not chain:
                raise EffectError(f"Audition case {name} must be an array of effect names")
            library.resolve(tuple(chain))
        read_pcm(args.input)
        directory = root / "build/auditions" / args.run
        if not directory.resolve().is_relative_to((root / "build/auditions").resolve()):
            raise EffectError("Audition path escapes build/auditions")
        directory.mkdir(parents=True, exist_ok=False)
        shutil.copy2(args.input, directory / "dry.wav")
        manifest = {"project": project, "source": str(args.input.resolve()),
                    "source_sha256": file_hash(args.input), "configuration": processor.configuration,
                    "status": "running", "results": {}}
        manifest_path = directory / "manifest.json"

        def save() -> None:
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        save()
        try:
            for name, chain in cases.items():
                print(f"{name}: rendering", flush=True)
                output = processor.process(directory / "dry.wav", directory / f"{name}.wav", tuple(chain))
                manifest["results"][name] = {"audio": output.name, "sha256": file_hash(output),
                                             "audit": output.with_suffix(".effects.json").name}
                save()
        except Exception as exc:
            manifest.update(status="failed", error=str(exc))
            save()
            raise
        manifest["status"] = "complete"
        save()
        lines = ["# Core effects audition", "", "Compare against [dry input](dry.wav). Start at low playback volume.", "",
                 "| Case | Audio | Chain |", "| --- | --- | --- |"]
        lines.extend(f"| {name} | [Listen]({name}.wav) | {' → '.join(chain)} |" for name, chain in cases.items())
        lines.extend(["", "Inspect each .effects.json for resolved parameters, timing and limitations. Technical completion is not listening acceptance.", ""])
        (directory / "README.md").write_text("\n".join(lines), encoding="utf-8")
        print(directory)
        return 0
    except (EffectError, OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        parser.error(str(exc))
        return 2
