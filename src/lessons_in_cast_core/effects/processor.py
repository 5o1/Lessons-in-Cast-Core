"""Ordered, backend-independent audio effect execution and provenance."""

from __future__ import annotations

from array import array
import json
import math
from pathlib import Path
import subprocess
import tempfile

from ..hashing import file_hash, content_hash
from .catalog import EffectError, EffectLibrary, EffectSpec
from .pcm import Pcm, read_pcm, write_pcm, ramp, edge_window


class CoreEffectProcessor:
    """Implements synthesis.api.AudioEffectProcessor; no TTS model dependencies."""

    VERSION = 1

    def __init__(self, library: EffectLibrary | None = None, *, sample_rate: int = 48000,
                 channels: int = 1, ffmpeg_executable: str = "ffmpeg") -> None:
        self.library = library or EffectLibrary()
        self.sample_rate = sample_rate
        self.channels = channels
        self.ffmpeg = ffmpeg_executable

    @property
    def configuration(self) -> dict:
        return {"processor": "core-effects", "version": self.VERSION,
                "library": self.library.to_dict(), "sample_rate": self.sample_rate,
                "channels": self.channels, "ffmpeg_executable": self.ffmpeg}

    def process(self, source: Path | None, destination: Path, effects: tuple[str, ...]) -> Path:
        specs = self.library.resolve(effects)
        if not specs:
            raise EffectError("An effect chain must not be empty")
        if destination.suffix.lower() != ".wav":
            raise EffectError("Effects output is PCM WAV; delivery encoding belongs to WaveRenderer")
        if source is not None and source.resolve() == destination.resolve():
            raise EffectError("Effects must not overwrite their input")
        if source is None and specs[0].kind != "censor_beep":
            raise EffectError("An effect-only chain must start with censor_beep")
        source_hash = file_hash(source) if source else None
        pcm = read_pcm(source) if source else None
        input_seconds = pcm.seconds if pcm else None
        steps = []
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Never publish a partially rendered chain. Keep source TTS cache intact.
        with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
            temporary = Path(directory)
            for spec in specs:
                before = pcm.seconds if pcm else None
                pcm, details = self._apply(pcm, spec, temporary)
                if not 0 < pcm.seconds <= 120:
                    raise EffectError("Effect chain output must be non-empty and at most 120 seconds")
                steps.append({**spec.to_dict(), "input_seconds": before, "output_seconds": pcm.seconds, **details})
            output = temporary / "output.wav"
            write_pcm(output, pcm)
            audit = {"processor": "core-effects", "version": self.VERSION,
                     "configuration_hash": content_hash(self.configuration),
                     "source_sha256": source_hash, "input_seconds": input_seconds,
                     "requested": list(effects), "steps": steps,
                     "sample_rate": pcm.rate, "channels": pcm.channels,
                     "output_seconds": pcm.seconds, "output_sha256": file_hash(output)}
            report = temporary / "report.json"
            report.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
            output.replace(destination)
            report.replace(destination.with_suffix(".effects.json"))
        return destination

    def _apply(self, pcm: Pcm | None, spec: EffectSpec, temporary: Path) -> tuple[Pcm, dict]:
        p = spec.resolved()
        details: dict = {"implementation": "core-pcm", "limitations": []}
        if spec.kind == "censor_beep":
            if pcm is None:
                if self.sample_rate < 8000 or self.channels not in (1, 2):
                    raise EffectError("Effect-only output requires >=8 kHz and mono/stereo")
                pcm = Pcm(array("h", [0]) * (round(p["duration_seconds"] * self.sample_rate) * self.channels),
                          self.sample_rate, self.channels)
            if p["frequency_hz"] >= pcm.rate / 2:
                raise EffectError("Beep frequency must be below Nyquist")
            start = round(p["start_seconds"] * pcm.rate)
            end = pcm.frames if p["end_seconds"] == -1 else round(p["end_seconds"] * pcm.rate)
            if not 0 <= start < end <= pcm.frames:
                raise EffectError("Censor interval is outside the current audio")
            amplitude = 32767 * 10 ** (p["level_db"] / 20)
            tone = array("h")
            for index in range(end - start):
                value = round(amplitude * math.sin(2 * math.pi * p["frequency_hz"] * index / pcm.rate))
                tone.extend([value] * pcm.channels)
            tone = edge_window(tone, pcm.channels, round(p["edge_seconds"] * pcm.rate))
            samples = array("h", pcm.samples)
            samples[start * pcm.channels:end * pcm.channels] = tone
            details["interval_seconds"] = [start / pcm.rate, end / pcm.rate]
            details["limitations"] = ["Time-based replacement; no automatic word alignment. Lossy delivery codecs may smear interval edges."]
            return Pcm(samples, pcm.rate, pcm.channels), details
        if pcm is None:
            raise EffectError(f"{spec.kind} needs source speech")
        if spec.kind in {"fade_in", "fade_out"}:
            frames = max(1, round(p["duration_seconds"] * pcm.rate))
            details["effective_seconds"] = min(frames, pcm.frames) / pcm.rate
            return Pcm(ramp(pcm.samples, pcm.channels, frames, out=spec.kind == "fade_out"), pcm.rate, pcm.channels), details
        if spec.kind == "glitch":
            return self._glitch(pcm, p, details)
        if spec.kind == "telephone":
            if p["high_hz"] >= min(4000, pcm.rate / 2):
                raise EffectError("Telephone high_hz must be below 4000 Hz and source Nyquist")
            mono = "pan=mono|c0=c0" if pcm.channels == 1 else "pan=mono|c0=0.5*c0+0.5*c1"
            filters = f"{mono},highpass=f={p['low_hz']},lowpass=f={p['high_hz']},aresample=8000"
            filtered, version = self._ffmpeg(pcm, filters, temporary)
            # Quantize the band-limited signal without adding random noise.
            step = 2 ** (16 - p["bits"])
            filtered.samples = array("h", (max(-32768, min(32767, round(sample / step) * step)) for sample in filtered.samples))
            details.update(implementation="ffmpeg+core-pcm", filtergraph=filters, ffmpeg_version=version)
            details["limitations"] = ["Stylized narrow-band/quantized telephone sound, not an exact telecom codec simulation; stereo becomes dual mono."]
            return filtered, details
        if spec.kind == "monster":
            ratio = 2 ** (p["semitones"] / 12)
            rate = round(pcm.rate * ratio)
            filters = (f"asetrate={rate},aresample={pcm.rate},atempo={pcm.rate / rate:.10f},"
                       f"bass=g={p['bass_db']}:f=150,volume={p['gain_db']}dB,"
                       "alimiter=limit=0.95:level=false:latency=true")
            filtered, version = self._ffmpeg(pcm, filters, temporary)
            details.update(implementation="ffmpeg", filtergraph=filters, ffmpeg_version=version)
            details["limitations"] = ["Resampling lowers pitch and formants together; tempo compensation is approximate and may create artifacts."]
            return filtered, details
        raise EffectError(f"No renderer for {spec.kind}")

    def _ffmpeg(self, pcm: Pcm, filters: str, temporary: Path) -> tuple[Pcm, str]:
        source, output = temporary / "filter-input.wav", temporary / "filter-output.wav"
        write_pcm(source, pcm)
        try:
            version = subprocess.run([self.ffmpeg, "-version"], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True, timeout=10).stdout.splitlines()[0]
            result = subprocess.run([self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                                     "-i", str(source), "-af", filters, "-ar", str(pcm.rate),
                                     "-ac", str(pcm.channels), "-c:a", "pcm_s16le", str(output)],
                                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            raise EffectError(f"Effect FFmpeg execution failed: {exc}") from exc
        if result.returncode:
            raise EffectError(f"Effect FFmpeg filter failed: {result.stderr.strip()}")
        return read_pcm(output), version

    @staticmethod
    def _glitch(pcm: Pcm, p: dict, details: dict) -> tuple[Pcm, dict]:
        rate, channels = pcm.rate, pcm.channels
        start = round(p["position"] * pcm.frames)
        size = max(1, round(p["chunk_seconds"] * rate))
        end = start + size
        if end >= pcm.frames:
            raise EffectError("Glitch needs a complete fragment and a normal recovery tail; move position earlier or shorten chunk_seconds")
        grain_size = max(1, round(p["grain_seconds"] * rate))
        edge = round(p["edge_seconds"] * rate)
        if grain_size > size or grain_size < max(2, 2 * edge):
            raise EffectError("Glitch grain must fit inside the fragment and be at least twice edge_seconds")
        clip = pcm.samples[start * channels:end * channels]
        grain = clip[-grain_size * channels:]
        chunk = edge_window(clip, channels, edge)
        grain = edge_window(grain, channels, edge)
        gap = array("h", [0]) * (round(p["gap_seconds"] * rate) * channels)
        hold_frames, dropout_frames = round(p["hold_seconds"] * rate), round(p["dropout_seconds"] * rate)
        predicted = pcm.frames + p["repeats"] * (size + len(gap) // channels) + hold_frames + dropout_frames
        if predicted / rate > 120:
            raise EffectError("Glitch would exceed the 120-second effect limit")
        prefix = pcm.samples[:end * channels]
        output = ramp(prefix, channels, edge, out=True)
        # The original fragment is followed by N extra repetitions.
        for _ in range(p["repeats"]):
            output.extend(gap)
            output.extend(chunk)
        for count, gated in ((hold_frames, False), (dropout_frames, True)):
            sustained = (grain * (count // grain_size + 1))[:count * channels]
            if gated:
                on, off = max(1, round(p["gate_on_seconds"] * rate)), max(1, round(p["gate_off_seconds"] * rate))
                cycle = on + off
                for frame in range(count):
                    phase = frame % cycle
                    gain = 0.0 if phase >= on else min(1.0, phase / max(1, edge), (on - 1 - phase) / max(1, edge))
                    for channel in range(channels):
                        index = frame * channels + channel
                        sustained[index] = round(sustained[index] * gain)
            output.extend(edge_window(sustained, channels, edge))
        recovery = ramp(pcm.samples[end * channels:], channels, edge)
        recovery_start = len(output) // channels
        output.extend(recovery)
        details.update(fragment_seconds=[start / rate, end / rate], recovery_output_seconds=recovery_start / rate,
                       inserted_seconds=(len(output) // channels - pcm.frames) / rate)
        details["limitations"] = ["Granular freeze, not phoneme-aware vowel stretching; choose a voiced fragment for a long-vowel effect. Recovery retains the original tail with a short anti-click ramp."]
        return Pcm(output, rate, channels), details
