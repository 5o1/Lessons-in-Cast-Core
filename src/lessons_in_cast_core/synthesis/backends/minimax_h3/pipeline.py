"""Reusable local H3 profile; no cloud API key or hosted voice ID."""

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

from ....emotion_presets import load_emotion_catalog
from ....pronunciations import load_pronunciation_lexicon
from ...profiles.api import VoicePipeline
from ...references.voices import list_voice_references, reference_audio_files, voice_reference_fingerprint
from .client import ComfyClient
from .config import COMFY_REVISION, COMPONENTS, MODEL_REPOSITORY, MODEL_REVISION, load_config
from .prompt import compile_prompt
from .workflow import build_workflow


def verify_installation(source: Path, models: Path):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True, encoding="utf-8").strip()
    if actual != COMFY_REVISION:
        raise ValueError(f"H3 expects ComfyUI {COMFY_REVISION}, found {actual}")
    if subprocess.call(["git", "diff", "--quiet", "HEAD", "--"], cwd=source):
        raise ValueError("ComfyUI tracked sources changed; restore the pinned implementation explicitly")
    manifest = json.loads((models / "verified.json").read_text(encoding="utf-8"))
    if (manifest.get("repository"), manifest.get("revision")) != (MODEL_REPOSITORY, MODEL_REVISION):
        raise ValueError("H3 model verification manifest has a different revision")
    for name, expected in COMPONENTS.items():
        record = manifest["files"].get(name, {})
        stat = (models / name).stat()
        if (record.get("sha256"), record.get("size"), record.get("mtime_ns")) != (expected, stat.st_size, stat.st_mtime_ns):
            raise ValueError(f"Unverified or modified H3 component {name}; rerun scripts/download_minimax_h3.py")
    return manifest


class MiniMaxH3Pipeline(VoicePipeline):
    configuration_path = "config.toml"

    def __init__(self, context):
        if context.character.id != self.character_id:
            raise ValueError(f"{self.pipeline_id} handles {self.character_id}, not {context.character.id}")
        self.context = context
        self.config = load_config(context.resolve_resource(self.configuration_path))
        self.reference = context.resolve_resource(self.config.reference_audio)
        self.model = context.model_registry.require(self.config.model_id)
        if (self.model.repository, self.model.revision) != (MODEL_REPOSITORY, MODEL_REVISION):
            raise ValueError("This H3 implementation requires the pinned Comfy-Org component package")
        self.model_root = context.model_registry.resolve_path(self.config.model_id)
        self.source = context.resolve_repository_path(self.config.source_directory)
        self.catalog = load_emotion_catalog(context.repository_root / "configs/emotions.toml")
        self.lexicon = load_pronunciation_lexicon(context.repository_root / "configs/pronunciations.toml",
                                                 repository_root=context.repository_root).select(
                                                     ("ipa", "respelling", "romaji", "kana"))
        self.client = ComfyClient(self.config.endpoint)

    @property
    def configuration(self):
        return {"pipeline_id": self.pipeline_id, "character_id": self.character_id,
                "profile": self.config.to_dict(), "model": self.model.to_dict(),
                "source_revision": COMFY_REVISION, "compiler_version": 1,
                "reference_audio": str(self.reference),
                "references": voice_reference_fingerprint(self.reference),
                "emotion_catalog": self.catalog, "pronunciations": [rule.to_dict() for rule in self.lexicon],
                "audio": {"sample_rate": self.context.project_config.audio.sample_rate,
                          "channels": self.context.project_config.audio.channels,
                          "sample_width": self.context.project_config.audio.sample_width}}

    def override_reference_audio(self, path):
        self.reference = Path(path).resolve()

    def list_voice_tags(self):
        return tuple(list_voice_references(self.reference))

    def prepare(self):
        verify_installation(self.source, self.model_root)
        if not self.reference.is_file():
            raise FileNotFoundError(self.reference)
        audio = self.context.project_config.audio
        for program in (audio.ffmpeg_executable, audio.ffprobe_executable):
            if not shutil.which(program):
                raise FileNotFoundError(f"H3 requires {program} on PATH")
        self.client.request("/system_stats")
        # Fail before queueing if this endpoint lacks the expected native nodes.
        for name in ("MiniMaxH3ReferenceToVideo", "LoadAudio", "SaveAudio", "VAEDecodeAudio"):
            if name not in self.client.request(f"/object_info/{name}"):
                raise ValueError(f"The local worker lacks the required native node {name}")
        return (self.reference, *reference_audio_files(self.reference))

    def adapt(self, job):
        if job.character_id != self.character_id:
            raise ValueError("Wrong character for H3 profile")
        return compile_prompt(job, self.config, self.reference, self.catalog, self.lexicon)

    def render(self, job, artifact_root):
        adaptation = self.adapt(job)
        references = [Path(value) for value in adaptation.parameters["references"]]
        audio = self.context.project_config.audio
        durations = []
        for path in references:
            result = subprocess.check_output([audio.ffprobe_executable, "-v", "error", "-show_entries",
                                               "format=duration", "-of", "json", str(path)], text=True, encoding="utf-8")
            durations.append(float(json.loads(result)["format"]["duration"]))
        if any(not math.isfinite(value) or not 2 <= value <= 15 for value in durations) or sum(durations) > 15:
            raise ValueError("H3 reference clips must each be 2–15 seconds and total at most 15 seconds; "
                             f"found {durations}. Prepare a shorter reference explicitly; no silent truncation.")
        destination = Path(job.output_path)
        if not destination.is_absolute():
            destination = artifact_root / destination
        destination = destination.resolve()
        if not destination.is_relative_to(artifact_root.resolve()):
            raise ValueError("H3 output must remain inside the artifact directory")
        cache = destination.parent / (destination.stem + ".minimax-h3")
        names = [self.client.upload_audio(path) for path in references]
        key = hashlib.sha256(str(destination).encode()).hexdigest()[:24]
        graph = build_workflow(adaptation, names, output_prefix=f"lic-h3/{key}")
        native = self.client.execute(graph, cache, identity={"job_cache_key": job.cache_key,
                                     "configuration": self.configuration}, timeout_seconds=self.config.timeout_seconds)
        destination.parent.mkdir(parents=True, exist_ok=True)
        codec = {2: "pcm_s16le", 3: "pcm_s24le", 4: "pcm_s32le"}.get(audio.sample_width)
        if codec is None:
            raise ValueError("H3 WAV normalization requires 16-, 24- or 32-bit PCM")
        temporary = destination.with_suffix(".partial.wav")
        subprocess.run([audio.ffmpeg_executable, "-nostdin", "-y", "-v", "error", "-i", str(native),
                        "-map", "0:a:0", "-ar", str(audio.sample_rate), "-ac", str(audio.channels),
                        "-c:a", codec, str(temporary)], check=True)
        temporary.replace(destination)
        return destination
