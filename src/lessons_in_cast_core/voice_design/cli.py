"""Run one voice-design audition in an isolated backend environment."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Sequence

from ..config import find_repository_root
from ..model_registry import load_model_registry
from .types import VoiceDesignRequest
from .voxcpm2 import VoxCPM2Settings, VoxCPM2VoiceDesigner


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lessons-in-cast-voice-design")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--model-id", default="voxcpm2")
    parser.add_argument("--text", required=True, help="Spoken audition script.")
    parser.add_argument("--instruction", default="", help="Voice description or reference delivery guidance.")
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New WAV path, normally under build/auditions/.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--inference-timesteps", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--optimize", action="store_true", help="Enable upstream torch.compile and warm-up.")
    parser.add_argument("--retry-badcase", action="store_true", help="Enable upstream automatic generation retries.")
    args = parser.parse_args(argv)
    root = (args.root or find_repository_root()).expanduser().resolve()

    def rooted(path: Path) -> Path:
        path = path.expanduser()
        return path if path.is_absolute() else root / path

    designer = None
    try:
        request = VoiceDesignRequest(
            text=args.text,
            instruction=args.instruction,
            reference_audio=rooted(args.reference_audio) if args.reference_audio else None,
            seed=args.seed,
        )
        designer = VoxCPM2VoiceDesigner(
            load_model_registry(repository_root=root),
            args.model_id,
            settings=VoxCPM2Settings(
                device=args.device,
                cfg_value=args.cfg_value,
                inference_timesteps=args.inference_timesteps,
                max_length=args.max_length,
                optimize=args.optimize,
                retry_badcase=args.retry_badcase,
            ),
        )
        # Keep upstream progress visible without corrupting the JSON result.
        with contextlib.redirect_stdout(sys.stderr):
            result = designer.generate(request, rooted(args.output))
        print(json.dumps({
            "audio_path": str(result.audio_path),
            "metadata_path": str(result.metadata_path),
            "sample_rate": result.sample_rate,
            "duration_seconds": result.duration_seconds,
        }))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Voice design failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if designer is not None:
            designer.close()
    return 0
