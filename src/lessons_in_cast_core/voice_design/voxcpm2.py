"""Lazy, local-only VoxCPM2 voice design and reference-conditioned style control."""

from __future__ import annotations

import json
import math
import tempfile
import wave
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..hashing import file_hash
from ..model_registry import ModelRegistry
from .types import VoiceDesignRequest, VoiceDesignResult


@dataclass(frozen=True, slots=True)
class VoxCPM2Settings:
    """Official inference controls; compilation is opt-in to avoid warm-up costs."""

    device: str = "auto"
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    max_length: int = 4096
    optimize: bool = False
    retry_badcase: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must not be empty")
        if (
            isinstance(self.cfg_value, bool)
            or not isinstance(self.cfg_value, (int, float))
            or not math.isfinite(self.cfg_value)
            or self.cfg_value <= 0
        ):
            raise ValueError("cfg_value must be positive and finite")
        if type(self.inference_timesteps) is not int or self.inference_timesteps < 1:
            raise ValueError("inference_timesteps must be a positive integer")
        if type(self.max_length) is not int or self.max_length < 2:
            raise ValueError("max_length must be an integer of at least two")
        if type(self.optimize) is not bool or type(self.retry_badcase) is not bool:
            raise ValueError("optimize and retry_badcase must be booleans")


def compile_voxcpm2_text(request: VoiceDesignRequest) -> str:
    """Compile the separate instruction to the upstream parenthesized prefix."""

    instruction = " ".join(request.instruction.split())
    text = " ".join(request.text.split())
    return f"({instruction}){text}" if instruction else text


def _write_waveform(path: Path, waveform: Any, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    audio = np.asarray(waveform)
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeError("VoxCPM2 returned empty, non-mono, or non-finite audio")
    sf.write(str(path), audio, sample_rate, format="WAV", subtype="PCM_16")


class VoxCPM2VoiceDesigner:
    """Reuse one loaded model across auditions in the backend Python environment.

    Importing this module never imports PyTorch or downloads weights. Callers
    resolve models by registry ID, and generation only uses local artifacts.
    This object is intended for sequential use, not concurrent GPU requests.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        model_id: str = "voxcpm2",
        *,
        settings: VoxCPM2Settings | None = None,
    ) -> None:
        self._definition = registry.require(model_id)
        self._model_path = registry.resolve_path(model_id)
        self._settings = settings or VoxCPM2Settings()
        self._model: Any = None

    @property
    def name(self) -> str:
        return "voxcpm2"

    @property
    def configuration(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model": self._definition.to_dict(),
            "resolved_model_path": str(self._model_path),
            "settings": asdict(self._settings),
            "load_denoiser": False,
            "normalize": False,
        }

    def _load_model(self) -> Any:
        if self._model is None:
            for filename in ("config.json", "model.safetensors", "audiovae.pth"):
                path = self._model_path / filename
                if not path.is_file() or not path.stat().st_size:
                    raise FileNotFoundError(f"Download VoxCPM2 model artifact first: {path}")
            config = json.loads((self._model_path / "config.json").read_text("utf-8"))
            if config.get("architecture", "").lower() != "voxcpm2":
                raise ValueError("Voice design requires VoxCPM2, not a VoxCPM 1.x model")
            try:
                from voxcpm import VoxCPM
            except ImportError as exc:
                raise RuntimeError(
                    "VoxCPM2 dependencies are unavailable. Run voice design in its "
                    "own Conda environment after installing external/VoxCPM; "
                    "see docs/voice-design.md."
                ) from exc
            self._model = VoxCPM.from_pretrained(
                str(self._model_path),
                local_files_only=True,
                load_denoiser=False,
                optimize=self._settings.optimize,
                device=self._settings.device,
            )
        return self._model

    def generate(
        self, request: VoiceDesignRequest, output_path: Path
    ) -> VoiceDesignResult:
        output = output_path.expanduser().resolve()
        if output.suffix.lower() != ".wav":
            raise ValueError("Voice-design reference auditions must use a .wav path")
        metadata_path = output.with_suffix(".json")
        for path in (output, metadata_path):
            if path.exists():
                raise FileExistsError(f"Choose a new audition path; refusing to overwrite {path}")
        reference = (
            request.reference_audio.expanduser().resolve()
            if request.reference_audio is not None else None
        )
        if reference is not None and not reference.is_file():
            raise FileNotFoundError(f"Reference audio is missing: {reference}")
        reference_hash = file_hash(reference) if reference is not None else None
        compiled_text = compile_voxcpm2_text(request)
        model = self._load_model()
        settings = self._settings
        waveform = model.generate(
            text=compiled_text,
            reference_wav_path=str(reference) if reference else None,
            cfg_value=settings.cfg_value,
            inference_timesteps=settings.inference_timesteps,
            max_len=settings.max_length,
            seed=request.seed,
            normalize=False,
            denoise=False,
            retry_badcase=settings.retry_badcase,
        )
        sample_rate = int(model.tts_model.sample_rate)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".voice-design-", dir=output.parent) as directory:
            temporary_audio = Path(directory) / "audio.wav"
            _write_waveform(temporary_audio, waveform, sample_rate)
            with wave.open(str(temporary_audio), "rb") as audio:
                if audio.getnframes() == 0 or audio.getnchannels() != 1:
                    raise RuntimeError("VoxCPM2 did not produce a non-empty mono audition")
                duration = audio.getnframes() / audio.getframerate()
            try:
                package_version = version("voxcpm")
            except PackageNotFoundError:
                package_version = "unknown"
            metadata = {
                "schema_version": 1,
                "configuration": self.configuration,
                "voxcpm_version": package_version,
                "mode": "reference_style" if reference else "voice_design",
                "request": {
                    "text": request.text,
                    "instruction": request.instruction,
                    "reference_audio": str(reference) if reference else None,
                    "reference_sha256": reference_hash,
                    "seed": request.seed,
                },
                "compiled_text": compiled_text,
                "audio": {
                    "path": str(output),
                    "sha256": file_hash(temporary_audio),
                    "sample_rate": sample_rate,
                    "duration_seconds": duration,
                    "format": "wav",
                    "subtype": "PCM_16",
                },
            }
            temporary_metadata = Path(directory) / "metadata.json"
            temporary_metadata.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            temporary_audio.replace(output)
            temporary_metadata.replace(metadata_path)
        return VoiceDesignResult(output, metadata_path, sample_rate, duration)

    def close(self) -> None:
        """Drop this adapter's model reference; do not alter other backends' GPU state."""

        self._model = None
