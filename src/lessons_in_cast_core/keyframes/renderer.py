"""Resolve only requested anchors and apply envelopes to final dry PCM audio."""

from array import array
import json
import math
from pathlib import Path
import subprocess
import sys
import tomllib
import wave

from ..config import find_repository_root
from ..hashing import content_hash, file_hash
from .program import compile_keyframes


class KeyframeProcessor:
    """An optional external aligner supplies measured text-boundary timestamps.

    The executable receives JSON on stdin and returns JSON on stdout. It must
    resolve grapheme boundaries through pronunciation/phone alignment, never by
    distributing word duration evenly among letters. No shell is involved.
    """

    def __init__(self, *, command=(), revision="", minimum_confidence=.8, timeout_seconds=600,
                 pronunciation_path: Path | None = None, allow_unscored=False):
        if not isinstance(command, (list, tuple)) or not all(isinstance(x, str) and x for x in command):
            raise ValueError("Alignment command must be an argv array")
        if command and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("An alignment command needs a model/config revision for caching")
        if isinstance(minimum_confidence, bool) or not isinstance(minimum_confidence, (int, float)) or not 0 <= minimum_confidence <= 1:
            raise ValueError("Alignment confidence threshold must be in [0, 1]")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Alignment timeout must be positive and finite")
        self.command = tuple(command)
        self.revision = revision
        self.minimum_confidence = minimum_confidence
        self.timeout_seconds = timeout_seconds
        self.pronunciation_path = pronunciation_path
        if not isinstance(allow_unscored, bool):
            raise ValueError("allow_unscored must be boolean")
        self.allow_unscored = allow_unscored

    @classmethod
    def from_repository(cls):
        root = find_repository_root()
        path = root / "configs/keyframes.toml"
        config = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return cls(**config.get("alignment", {}), pronunciation_path=root / "configs/pronunciations.toml")

    def _times(self, source, program, duration, cache_root):
        offsets = sorted(set(program["anchors"].values()))
        times = {0: 0., len(program["text"]): duration}
        needed = [x for x in offsets if x not in times]
        if not needed:
            return times, None
        if not self.command:
            raise ValueError("Interior keyframe anchors require a configured phoneme/text aligner in configs/keyframes.toml; no character-proportional fallback is allowed")
        request = {
            "version": 1, "audio_path": str(source.resolve()), "audio_sha256": file_hash(source),
            "text": program["text"], "offsets": needed,
            "pronunciations_path": str(self.pronunciation_path.resolve()) if self.pronunciation_path else None,
        }
        binding = {"audio_sha256": request["audio_sha256"], "text": request["text"], "offsets": needed,
                   "command": self.command, "revision": self.revision,
                   "pronunciations_sha256": file_hash(self.pronunciation_path) if self.pronunciation_path else None}
        identity = content_hash(binding)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache = cache_root / f"{identity}.json"
        if cache.exists():
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if saved.get("binding_hash") != identity:
                raise ValueError("Alignment cache binding mismatch")
            response = saved["response"]
        else:
            result = subprocess.run(self.command, input=json.dumps(request), text=True, encoding="utf-8",
                                    capture_output=True, timeout=self.timeout_seconds, check=False)
            if result.returncode:
                raise ValueError(f"Alignment process failed (exit {result.returncode})")
            response = json.loads(result.stdout)
        if not isinstance(response, dict) or response.get("version") != 1 or response.get("audio_sha256") != request["audio_sha256"] or response.get("text") != program["text"]:
            raise ValueError("Alignment result does not match dry audio and text")
        boundaries = response.get("boundaries")
        if not isinstance(boundaries, list):
            raise ValueError("Alignment boundaries must be an array")
        seen = set()
        for point in boundaries:
            if not isinstance(point, dict) or set(point) != {"offset", "time", "confidence", "source"}:
                raise ValueError("Invalid alignment boundary fields")
            offset, time, confidence = point["offset"], point["time"], point["confidence"]
            if isinstance(offset, bool) or not isinstance(offset, int) or offset not in needed or offset in seen:
                raise ValueError("Unexpected or duplicate alignment offset")
            if isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time) or not 0 <= time <= duration:
                raise ValueError("Alignment time is outside the audio")
            if confidence is None and self.allow_unscored:
                pass
            elif isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not self.minimum_confidence <= confidence <= 1:
                raise ValueError("Alignment confidence is below the required threshold")
            if point["source"] not in ("phone_alignment", "word_alignment", "backend_alignment", "human_alignment"):
                raise ValueError("Alignment must identify a measured boundary source")
            seen.add(offset)
            times[offset] = float(time)
        if seen != set(needed):
            raise ValueError("Alignment is missing requested offsets")
        sequence = [times[x] for x in sorted(times)]
        if any(a > b for a, b in zip(sequence, sequence[1:])):
            raise ValueError("Alignment timestamps are not monotonic")
        # Cache only complete validated measurements, never invented estimates.
        if not cache.exists():
            cache.write_text(json.dumps({"binding_hash": identity, "binding": binding, "response": response}, indent=2) + "\n", encoding="utf-8")
        return times, identity

    def process(self, source: Path, destination: Path, program: dict, *, cache_root: Path, trace_path: Path) -> Path:
        if not isinstance(program, dict) or set(program) != {"version", "text", "anchors", "effects"} or program["version"] != 1:
            raise ValueError("Invalid compiled keyframe program")
        # Rebuild symbolic text and revalidate persisted programs before DSP.
        text = program["text"]
        if not isinstance(text, str) or not isinstance(program["anchors"], dict):
            raise ValueError("Invalid program text or anchors")
        markers = {}
        for identity, offset in program["anchors"].items():
            if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= len(text):
                raise ValueError("Invalid compiled anchor offset")
            markers.setdefault(offset, []).append("{" + identity + "}")
        marked = "".join("".join(markers.get(i, [])) + ch.replace("{", "{{").replace("}", "}}") for i, ch in enumerate(text)) + "".join(markers.get(len(text), []))
        if compile_keyframes(marked, program["effects"]) != program:
            raise ValueError("Inconsistent compiled keyframe program")
        with wave.open(str(source), "rb") as stream:
            params = stream.getparams()
            if params.sampwidth != 2 or params.comptype != "NONE" or params.nframes <= 0:
                raise ValueError("Keyframe effects require nonempty 16-bit PCM WAV")
            samples = array("h", stream.readframes(params.nframes))
        if sys.byteorder != "little":
            samples.byteswap()
        times, alignment_id = self._times(source, program, params.nframes / params.framerate, cache_root)
        resolved = []
        for curve in program["effects"]:
            points = [(times[program["anchors"][p["anchor"]]], p["gain"]) for p in curve["keyframes"]]
            if any(a[0] >= b[0] for a, b in zip(points, points[1:])):
                raise ValueError("Resolved curve has collapsed or reversed keyframes")
            resolved.append({"interpolation": curve["interpolation"], "points": points})
            interval = 0
            for frame in range(params.nframes):
                time = frame / params.framerate
                while interval + 1 < len(points) and time >= points[interval + 1][0]:
                    interval += 1
                if time <= points[0][0]:
                    gain = points[0][1]
                elif interval == len(points) - 1:
                    gain = points[-1][1]
                else:
                    start, end = points[interval], points[interval + 1]
                    fraction = (time - start[0]) / (end[0] - start[0])
                    if curve["interpolation"] == "smooth":
                        fraction = fraction * fraction * (3 - 2 * fraction)
                    gain = start[1] + (end[1] - start[1]) * fraction
                for channel in range(params.nchannels):
                    index = frame * params.nchannels + channel
                    samples[index] = round(samples[index] * gain)
        if sys.byteorder != "little":
            samples.byteswap()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(destination), "wb") as stream:
            stream.setparams(params)
            stream.writeframes(samples.tobytes())
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps({"program": program, "audio_sha256": file_hash(source),
                                         "alignment_id": alignment_id, "resolved_curves": resolved}, indent=2) + "\n", encoding="utf-8")
        return destination
