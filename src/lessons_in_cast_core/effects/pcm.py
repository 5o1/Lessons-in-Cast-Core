"""Small, dependency-free 16-bit PCM operations for line-level effects."""

from array import array
from dataclasses import dataclass
from pathlib import Path
import sys
import wave

from .catalog import EffectError


@dataclass
class Pcm:
    samples: array
    rate: int
    channels: int

    @property
    def frames(self) -> int:
        return len(self.samples) // self.channels

    @property
    def seconds(self) -> float:
        return self.frames / self.rate


def read_pcm(path: Path) -> Pcm:
    try:
        with wave.open(str(path), "rb") as stream:
            if stream.getsampwidth() != 2 or stream.getcomptype() != "NONE":
                raise EffectError("Effects require uncompressed 16-bit PCM WAV intermediates")
            rate, channels, frames = stream.getframerate(), stream.getnchannels(), stream.getnframes()
            if rate < 8000 or channels not in (1, 2) or not 0 < frames / rate <= 120:
                raise EffectError("Effects support 8 kHz or higher, mono/stereo, non-empty clips up to 120 seconds")
            samples = array("h")
            samples.frombytes(stream.readframes(frames))
            if sys.byteorder != "little":
                samples.byteswap()
            if len(samples) != frames * channels:
                raise EffectError("Truncated PCM input")
            return Pcm(samples, rate, channels)
    except (OSError, EOFError, wave.Error) as exc:
        raise EffectError(f"Cannot read PCM {path}: {exc}") from exc


def write_pcm(path: Path, pcm: Pcm) -> None:
    samples = array("h", pcm.samples)
    if sys.byteorder != "little":
        samples.byteswap()
    with wave.open(str(path), "wb") as stream:
        stream.setparams((pcm.channels, 2, pcm.rate, 0, "NONE", "not compressed"))
        stream.writeframes(samples.tobytes())


def ramp(samples: array, channels: int, frames: int, *, out: bool = False) -> array:
    result = array("h", samples)
    total = len(result) // channels
    frames = min(frames, total)
    for index in range(frames):
        gain = index / max(1, frames - 1)
        target = total - 1 - index if out else index
        for channel in range(channels):
            offset = target * channels + channel
            result[offset] = round(result[offset] * gain)
    return result


def edge_window(samples: array, channels: int, frames: int) -> array:
    frames = min(frames, len(samples) // channels // 2)
    return ramp(ramp(samples, channels, frames), channels, frames, out=True)
