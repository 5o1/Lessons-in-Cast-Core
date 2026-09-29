"""JSON-lines worker running inside the official IndexTTS environment."""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

from ...references.builder import (
    ReferenceBuildSettings,
    prepare_reference_sources,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--use-bf16", action="store_true")
    return parser


def _convert_audio(path: Path, sample_rate: int, channels: int) -> None:
    import librosa
    import numpy
    import soundfile

    audio, _ = librosa.load(path, sr=sample_rate, mono=channels == 1)
    if channels > 1 and audio.ndim == 1:
        audio = numpy.repeat(audio[numpy.newaxis, :], channels, axis=0)
    soundfile.write(
        path,
        audio.T if getattr(audio, "ndim", 1) > 1 else audio,
        sample_rate,
        subtype="PCM_16",
        format="WAV",
    )


def _prepare_reference(
    request: dict[str, Any],
    directory: Path,
    cache: dict[tuple[Any, ...], Path],
) -> Path:
    sources = tuple(Path(value).resolve() for value in request["references"])
    if not bool(request.get("prepare_references", True)):
        if len(sources) != 1:
            raise ValueError(
                "A prepared reference must contain exactly one audio file"
            )
        if not sources[0].is_file():
            raise FileNotFoundError(
                f"Prepared reference audio is missing: {sources[0]}"
            )
        return sources[0]
    settings = ReferenceBuildSettings(
        top_db=float(request["reference_trim_top_db"]),
        padding_ms=int(request["reference_trim_padding_ms"]),
        gate_hold_ms=int(request["reference_gate_hold_ms"]),
        minimum_speech_seconds=float(
            request["minimum_reference_speech_seconds"]
        ),
        minimum_duration_seconds=float(
            request["minimum_reference_duration_seconds"]
        ),
        trim_silence=bool(request["trim_reference_silence"]),
    )
    key = (sources, settings)
    cached = cache.get(key)
    if cached is not None:
        return cached
    destination = directory / f"reference-{len(cache):04d}.wav"
    prepare_reference_sources(sources, destination, settings)
    cache[key] = destination
    return destination


def _infer(
    tts: Any,
    request: dict[str, Any],
    language: str,
    reference_directory: Path,
    reference_cache: dict[tuple[Any, ...], Path],
) -> dict[str, Any]:
    import numpy
    import torch

    reference = _prepare_reference(request, reference_directory, reference_cache)

    seed = int(request["seed"])
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    output = Path(request["output"])
    temporary = output.with_suffix(".index-tts.tmp.wav")
    temporary.unlink(missing_ok=True)
    generation_options = {
        "do_sample": bool(request["do_sample"]),
        "num_beams": int(request["num_beams"]),
        "repetition_penalty": float(request["repetition_penalty"]),
        "length_penalty": float(request["length_penalty"]),
        "max_mel_tokens": int(request["max_mel_tokens"]),
    }
    if request["do_sample"]:
        generation_options |= {
            "top_p": float(request["top_p"]),
            "top_k": int(request["top_k"]),
            "temperature": float(request["temperature"]),
        }
    original_inference_speech = tts.gpt.inference_speech
    original_split = tts.split_text_by_tokens
    frontend = []

    def traced_split(text, max_tokens, lang_prefix=""):
        # Consume one-shot iterables once, then share the materialized data.
        segments = list(original_split(text, max_tokens, lang_prefix))
        frontend.append({"normalized_text": text, "language_prefix": lang_prefix,
                         "segments": list(segments),
                         "token_counts": [tts._token_len(lang_prefix + part) for part in segments]})
        return segments

    def configured_inference_speech(*args: Any, **kwargs: Any) -> Any:
        kwargs["do_sample"] = bool(request["do_sample"])
        if not kwargs["do_sample"]:
            for name in ("temperature", "top_p", "top_k"):
                kwargs.pop(name, None)
        return original_inference_speech(*args, **kwargs)

    tts.gpt.inference_speech = configured_inference_speech
    tts.split_text_by_tokens = traced_split
    try:
        tts.infer(
            spk_audio_prompt=str(reference),
            text=request["text"],
            output_path=str(temporary),
            lang=language,
            emo_vector=request["emotion_vector"],
            emo_alpha=float(request["emotion_alpha"]),
            use_random=bool(request["use_random_emotion"]),
            interval_silence=int(request["interval_silence_ms"]),
            max_text_tokens_per_segment=int(request["max_text_tokens_per_segment"]),
            duration_factor=float(request["duration_factor"]),
            text_normalization=bool(request["text_normalization"]),
            **generation_options,
            verbose=False,
        )
        if not temporary.is_file():
            raise RuntimeError("IndexTTS returned without producing audio")
        _convert_audio(
            temporary, int(request["sample_rate"]), int(request["channels"])
        )
        temporary.replace(output)
        return {"input_text": request["text"], "frontend_calls": frontend,
                "interval_silence_ms": int(request["interval_silence_ms"])}
    finally:
        temporary.unlink(missing_ok=True)
        tts.gpt.inference_speech = original_inference_speech
        tts.split_text_by_tokens = original_split


def main() -> int:
    args = _parser().parse_args()
    # The JSON-lines protocol is UTF-8 regardless of the platform's locale.
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(args.source_root.resolve()))
    with contextlib.redirect_stdout(sys.stderr):
        from indextts.infer_v2_5 import IndexTTS2

        tts = IndexTTS2(
            cfg_path=str(args.model_path / "config.yaml"),
            model_dir=str(args.model_path),
            use_bf16=args.use_bf16,
            use_qwen_emo=False,
        )
    reference_temporary_directory = tempfile.TemporaryDirectory(
        prefix="lessons-in-cast-index-tts-"
    )
    reference_directory = Path(reference_temporary_directory.name)
    reference_cache: dict[tuple[Any, ...], Path] = {}
    emotion_engine = None
    for line in sys.stdin:
        request: Any = None
        try:
            request = json.loads(line)
            emotion_resolution = None
            with contextlib.redirect_stdout(sys.stderr):
                if request.get("arbitrary_emotion") is not None:
                    if request.get("emotion_vector") is not None:
                        raise ValueError("arbitrary_emotion cannot overlap a preset vector")
                    from .qwen_emotion import load_qwen_emotion, resolve_arbitrary_emotion
                    if emotion_engine is None:
                        emotion_engine = load_qwen_emotion(args.model_path / tts.cfg.qwen_emo_path)
                    emotion_resolution = resolve_arbitrary_emotion(request["arbitrary_emotion"], emotion_engine,
                                                                  request.get("emotion_energy"))
                    request["emotion_vector"] = emotion_resolution["vector"]
                frontend = _infer(
                    tts,
                    request,
                    args.language,
                    reference_directory,
                    reference_cache,
                )
            response = {"id": request.get("id"), "ok": True, "text_frontend": frontend}
            if emotion_resolution is not None:
                response["emotion_resolution"] = emotion_resolution
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {
                "id": request.get("id") if isinstance(request, dict) else None,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    reference_temporary_directory.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
