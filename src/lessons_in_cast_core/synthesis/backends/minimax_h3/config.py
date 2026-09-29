"""Pinned native ComfyUI implementation of reference-conditioned H3 audio."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import tomllib
from urllib.parse import urlsplit


COMFY_REVISION = "efa6c8f804bff78b46a0fd458ebd2e47bba07a30"
MODEL_REPOSITORY = "Comfy-Org/MiniMax-H3"
MODEL_REVISION = "a98869194787969724c7425d95d0ed73ce9202af"
COMPONENTS = {
    "diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors": "f86f2f79ebd2d76eb8eeb46091e83982e6ff51d255747e7b16e92834b392b8e9",
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6",
    "vae/minimax_h3_audio_vae_fp32.safetensors": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
    "vae/minimax_h3_video_vae_fp16.safetensors": "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
}


def validate_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}):
        raise ValueError("H3 requires a local loopback ComfyUI HTTP endpoint")
    if parsed.port is None:
        raise ValueError("H3 endpoint must include an explicit port")
    return endpoint.rstrip("/")


@dataclass(frozen=True)
class MiniMaxH3Config:
    model_id: str
    reference_audio: str = "assets/references/default.wav"
    source_directory: str = "external/ComfyUI"
    endpoint: str = "http://127.0.0.1:8196"
    duration_seconds: float = 10.0
    steps: int = 20
    seed: int = 233333
    base_speed: float = 1.0
    language: str = "English"
    direction: str = "Natural conversational speech, with clear but unforced articulation."
    timeout_seconds: float = 1800.0

    def __post_init__(self):
        validate_endpoint(self.endpoint)
        for name in ("model_id", "reference_audio", "source_directory", "language", "direction"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"H3 {name} must be a nonempty string")
        for name, low, high in (("duration_seconds", 4, 15), ("base_speed", .5, 2),
                                ("timeout_seconds", 1, 86400)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"H3 {name} must be between {low} and {high}")
        for name, low, high in (("steps", 1, 100), ("seed", 0, 2**64 - 1)):
            if type(getattr(self, name)) is not int or not low <= getattr(self, name) <= high:
                raise ValueError(f"Invalid H3 {name}")

    @property
    def frame_count(self) -> int:
        # Native H3 accepts lengths 17*k+5. Never exceed its 15-second window.
        requested = math.ceil(self.duration_seconds * 24)
        return min(345, requested + (5 - requested) % 17)

    def to_dict(self):
        return asdict(self)


def load_config(path: Path) -> MiniMaxH3Config:
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    if set(document) != {"minimax_h3"}:
        raise ValueError("H3 profile requires only a [minimax_h3] table")
    return MiniMaxH3Config(**document["minimax_h3"])
