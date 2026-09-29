"""PCM rendering, delivery encoding, and technical quality checks."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import wave
from array import array
from pathlib import Path

from ..config import AudioConfig
from ..effects import CoreEffectProcessor, EffectError
from .api import AudioEffectProcessor
from .types import AudioQualityResult, RenderTask, TtsJob
from ..keyframes import KeyframeProcessor


class AudioRenderError(RuntimeError):
    """Raised when a render task requires unavailable audio behavior."""


class FfmpegAudioEffectProcessor:
    """Apply the project's small, fixed effect vocabulary with FFmpeg."""

    _filters = {
        "chorus": "chorus=0.5:0.9:50|60:0.4|0.3:0.25|0.2:2|2.3",
        "distortion": "acompressor=threshold=0.1:ratio=9:attack=5:release=50:makeup=2",
        "echo": "aecho=0.8:0.88:160:0.3",
        "glitch": "acrusher=bits=6:mix=0.35",
        "reverb": "aecho=0.8:0.7:45|90:0.25|0.12",
        "telephone": "highpass=f=300,lowpass=f=3400",
    }

    def __init__(self, executable: str = "ffmpeg") -> None:
        self._executable = executable

    def process(
        self,
        source: Path | None,
        destination: Path,
        effects: tuple[str, ...],
    ) -> Path:
        try:
            filters = ",".join(self._filters[name] for name in effects)
        except KeyError as exc:
            raise AudioRenderError(f"Unsupported audio effect: {exc.args[0]}") from exc
        command = [self._executable, "-hide_banner", "-loglevel", "error", "-y"]
        if source is None:
            command += ["-f", "lavfi", "-i", "anoisesrc=color=pink:duration=0.8:amplitude=0.06"]
        else:
            command += ["-i", str(source)]
        command += ["-af", filters, str(destination)]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        except OSError as exc:
            raise AudioRenderError(f"Unable to start audio effect processor: {exc}") from exc
        if completed.returncode or not destination.is_file():
            detail = completed.stderr.strip() or completed.stdout.strip() or "no output file"
            raise AudioRenderError(f"Audio effect processing failed: {detail}")
        return destination


class WaveRenderer:
    """Render WAV intermediates and encode the configured delivery format."""

    def __init__(
        self,
        effect_processor: AudioEffectProcessor | None = None,
        *,
        audio_config: AudioConfig | None = None,
        keyframe_processor: KeyframeProcessor | None = None,
    ) -> None:
        self._config = audio_config or AudioConfig()
        self._effect_processor = effect_processor if effect_processor is not None else CoreEffectProcessor(
            sample_rate=self._config.sample_rate,
            channels=self._config.channels,
            ffmpeg_executable=self._config.ffmpeg_executable,
        )
        self._keyframe_processor = keyframe_processor

    def render(
        self,
        task: RenderTask,
        jobs_by_id: dict[str, TtsJob],
        artifact_root: Path,
    ) -> Path:
        if task.effects and self._effect_processor is None:
            raise AudioRenderError(
                f"{task.dialogue_id}: effects require a configured post-processor"
            )
        components = [
            artifact_root / jobs_by_id[job_id].output_path
            for job_id in task.component_job_ids
        ]
        if task.keyframe_program and len(components) != 1:
            raise AudioRenderError("Keyframe alignment currently requires one final dry voice, not a chorus mix")
        if not components and not task.effects:
            raise AudioRenderError(
                f"{task.dialogue_id}: render task has no audio components"
            )
        destination = artifact_root / task.output_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        delivery_format = destination.suffix.lstrip(".").lower()
        if delivery_format == "wav" and not task.effects and not task.keyframe_program:
            self._render_components(task, components, destination)
            destination.with_suffix(".effects.json").unlink(missing_ok=True)
            return destination
        if delivery_format not in {"wav", "opus"}:
            raise AudioRenderError(
                f"{task.dialogue_id}: unsupported delivery format "
                f"{delivery_format!r}"
            )

        with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
            temporary_root = Path(directory)
            source = temporary_root / "source.wav"
            rendered = self._render_components(task, components, source)
            if rendered is None and not task.effects:
                raise AudioRenderError(
                    f"{task.dialogue_id}: render task produced no audio"
                )
            if task.keyframe_program:
                if task.keyframe_program["text"] != jobs_by_id[task.component_job_ids[0]].text:
                    raise AudioRenderError("Keyframe program does not match the synthesized text")
                try:
                    processor = self._keyframe_processor or KeyframeProcessor.from_repository()
                    rendered = processor.process(
                        rendered, temporary_root / "keyframed.wav", task.keyframe_program,
                        cache_root=artifact_root / "alignment/cache",
                        trace_path=artifact_root / "alignment" / f"{task.dialogue_id}.json",
                    )
                except (ValueError, OSError, subprocess.SubprocessError) as exc:
                    raise AudioRenderError(f"{task.dialogue_id}: keyframe rendering failed: {exc}") from exc
            if task.effects:
                assert self._effect_processor is not None
                effected = temporary_root / "effected.wav"
                try:
                    rendered = self._effect_processor.process(rendered, effected, task.effects)
                except EffectError as exc:
                    raise AudioRenderError(f"{task.dialogue_id}: {exc}") from exc
            assert rendered is not None
            if delivery_format == "wav":
                shutil.copy2(rendered, destination)
            else:
                self._encode_opus(rendered, destination, task.dialogue_id)
            report = rendered.with_suffix(".effects.json")
            final_report = destination.with_suffix(".effects.json")
            if task.effects and report.is_file():
                from ..hashing import file_hash
                audit = json.loads(report.read_text(encoding="utf-8"))
                audit["dialogue_id"] = task.dialogue_id
                audit["delivery"] = {"format": delivery_format, "sha256": file_hash(destination)}
                final_report.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
            else:
                final_report.unlink(missing_ok=True)
        return destination

    def _encode_opus(
        self,
        source: Path,
        destination: Path,
        dialogue_id: str,
    ) -> None:
        command = [
            self._config.ffmpeg_executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            str(self._config.channels),
            "-ar",
            str(self._config.sample_rate),
            "-c:a",
            "libopus",
            "-b:a",
            f"{self._config.bitrate_kbps}k",
            "-application",
            "voip",
            "-vbr",
            "on",
            str(destination),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8", errors="replace",
                check=False,
            )
        except OSError as exc:
            raise AudioRenderError(
                f"{dialogue_id}: unable to start Opus encoder: {exc}"
            ) from exc
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise AudioRenderError(
                f"{dialogue_id}: Opus encoding failed: {detail}"
            )

    def _render_components(
        self,
        task: RenderTask,
        components: list[Path],
        destination: Path,
    ) -> Path | None:
        if not components:
            return None
        if len(components) == 1:
            shutil.copy2(components[0], destination)
            return destination
        if task.render_mode != "unison":
            raise AudioRenderError(
                f"{task.dialogue_id}: unsupported render mode {task.render_mode!r}"
            )
        self._mix_unison(components, destination)
        return destination

    @staticmethod
    def _mix_unison(components: list[Path], destination: Path) -> None:
        parameters: tuple[int, int, int] | None = None
        streams: list[array[int]] = []
        for path in components:
            with wave.open(str(path), "rb") as source:
                current = (
                    source.getnchannels(),
                    source.getsampwidth(),
                    source.getframerate(),
                )
                if current[1] != 2:
                    raise AudioRenderError(
                        "Unison mixing currently requires 16-bit PCM WAV"
                    )
                if parameters is None:
                    parameters = current
                elif parameters != current:
                    raise AudioRenderError(
                        "Unison components have incompatible WAV parameters"
                    )
                samples = array("h")
                samples.frombytes(source.readframes(source.getnframes()))
                streams.append(samples)
        assert parameters is not None
        sample_count = max(len(stream) for stream in streams)
        mixed = array("h")
        for index in range(sample_count):
            total = sum(
                stream[index] if index < len(stream) else 0
                for stream in streams
            )
            mixed.append(round(total / len(streams)))
        with wave.open(str(destination), "wb") as output:
            output.setnchannels(parameters[0])
            output.setsampwidth(parameters[1])
            output.setframerate(parameters[2])
            output.writeframes(mixed.tobytes())


class AudioQualityChecker:
    def __init__(self, config: AudioConfig) -> None:
        self._config = config

    def check(
        self,
        dialogue_id: str,
        path: Path,
    ) -> AudioQualityResult:
        issues: list[str] = []
        duration: float | None = None
        if not path.is_file():
            issues.append("missing_file")
        elif path.suffix.lower() == ".wav":
            duration = self._check_wav(path, issues)
        else:
            duration = self._check_encoded(path, issues)
        if duration is not None and duration < self._config.minimum_duration_seconds:
            issues.append("duration_too_short")
        if duration is not None and duration > self._config.maximum_duration_seconds:
            issues.append("duration_too_long")
        return AudioQualityResult(
            dialogue_id=dialogue_id,
            path=str(path),
            valid=not issues,
            duration_seconds=duration,
            issues=tuple(issues),
        )

    def _check_wav(self, path: Path, issues: list[str]) -> float | None:
        try:
            with wave.open(str(path), "rb") as source:
                if source.getframerate() != self._config.sample_rate:
                    issues.append("sample_rate")
                if source.getnchannels() != self._config.channels:
                    issues.append("channels")
                if source.getsampwidth() != self._config.sample_width:
                    issues.append("sample_width")
                frame_rate = source.getframerate()
                return source.getnframes() / frame_rate if frame_rate else 0.0
        except (OSError, EOFError, wave.Error):
            issues.append("invalid_wav")
            return None

    def _check_encoded(self, path: Path, issues: list[str]) -> float | None:
        command = [
            self._config.ffprobe_executable,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,sample_rate,channels:format=duration",
            "-of",
            "json",
            str(path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8", errors="replace",
                check=False,
            )
        except OSError:
            issues.append("probe_unavailable")
            return None
        if completed.returncode:
            issues.append("invalid_audio")
            return None
        try:
            payload = json.loads(completed.stdout)
            stream = payload["streams"][0]
            duration = float(payload["format"]["duration"])
            sample_rate = int(stream["sample_rate"])
            channels = int(stream["channels"])
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            issues.append("invalid_probe_result")
            return None
        if path.suffix.lower() == ".opus" and stream.get("codec_name") != "opus":
            issues.append("codec")
        if sample_rate != self._config.sample_rate:
            issues.append("sample_rate")
        if channels != self._config.channels:
            issues.append("channels")
        return duration
