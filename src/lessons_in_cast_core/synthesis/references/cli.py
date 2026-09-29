"""Command-line entry point for reference-audio construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .builder import (
    ReferenceBuildError,
    ReferenceBuildSettings,
    build_reference_from_directory,
    prepare_reference_sources,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build model-ready voice reference audio."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-dir", type=Path)
    inputs.add_argument("--source", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-db", type=float, default=40.0)
    parser.add_argument("--padding-ms", type=int, default=150)
    parser.add_argument("--gate-hold-ms", type=int, default=500)
    parser.add_argument("--minimum-speech-seconds", type=float, default=3.0)
    parser.add_argument("--minimum-duration-seconds", type=float, default=15.0)
    parser.add_argument("--no-trim-silence", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    settings = ReferenceBuildSettings(
        top_db=args.top_db,
        padding_ms=args.padding_ms,
        gate_hold_ms=args.gate_hold_ms,
        minimum_speech_seconds=args.minimum_speech_seconds,
        minimum_duration_seconds=args.minimum_duration_seconds,
        trim_silence=not args.no_trim_silence,
    )
    if settings.top_db <= 0:
        raise ReferenceBuildError("top_db must be positive")
    if settings.padding_ms < 0 or settings.gate_hold_ms < 0:
        raise ReferenceBuildError("gate timing values cannot be negative")
    if settings.minimum_speech_seconds <= 0:
        raise ReferenceBuildError("minimum speech must be positive")
    if settings.minimum_duration_seconds < 15:
        raise ReferenceBuildError("minimum duration must be at least 15 seconds")
    if args.source:
        result = prepare_reference_sources(
            args.source,
            args.output,
            settings,
        )
    else:
        assert args.input_dir is not None
        result = build_reference_from_directory(
            args.input_dir,
            args.output,
            settings,
        )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
