"""Lower inline emotion spans to ordered single-emotion requests and one WAV."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import wave

from ..hashing import content_hash, file_hash
from ..performance import AdaptationFidelity, FeatureAdaptation, SpeechAdaptation
from .types import TtsJob


def split_emotion_job(job: TtsJob) -> tuple[TtsJob, ...]:
    if not job.segments or "".join(s.text for s in job.segments) != job.text:
        raise ValueError("Speech segments must exactly cover the decoded job text")
    # Multiple sentences with the same controls remain one request.
    groups = []
    for segment in job.segments:
        if groups and (groups[-1].emotion, groups[-1].voice, groups[-1].arbitrary_emotion) == (segment.emotion, segment.voice, segment.arbitrary_emotion):
            groups[-1] = replace(groups[-1], text=groups[-1].text + segment.text)
        else:
            groups.append(segment)
    jobs = []
    offset = 0
    for index, segment in enumerate(groups):
        end = offset + len(segment.text)
        cues = tuple(replace(cue, offset=cue.offset-offset) for cue in job.performance.cues
                     if offset <= cue.offset < end or (index == len(groups)-1 and cue.offset == end))
        performance = replace(job.performance, cues=cues)
        key = content_hash({"parent": job.cache_key, "segment": segment.to_dict(),
                            "performance": performance.to_dict(), "lowering_version": 2})
        output = Path(job.output_path)
        output = output.parent / (output.stem + ".segments") / f"{index:03d}-{key}.wav"
        jobs.append(replace(job, id=f"{job.id}-{index:03d}", text=segment.text,
                            emotion=segment.emotion,
                            arbitrary_emotion=segment.arbitrary_emotion,
                            voice=segment.voice,
                            performance=performance, segments=(), cache_key=key, output_path=str(output)))
        offset = end
    return tuple(jobs)


def adapt_segments(job: TtsJob, adapt) -> SpeechAdaptation:
    adaptations = [adapt(part) for part in split_emotion_job(job)]
    features = [FeatureAdaptation("speech_spans", AdaptationFidelity.APPROXIMATED,
                                 "generate ordered single-emotion/voice takes and concatenate without trimming; boundary continuity is not guaranteed")]
    for index, adaptation in enumerate(adaptations):
        features.extend(replace(feature, feature=f"segments[{index}].{feature.feature}") for feature in adaptation.features)
    return SpeechAdaptation(job.id, job.dialogue_id, adaptations[0].backend,
                            "".join(item.text for item in adaptations), None,
                            {"segments": [item.to_dict() for item in adaptations], "join": "pcm_concat"}, tuple(features))


def concatenate_wav(paths: tuple[Path, ...], destination: Path) -> None:
    """Stream PCM in order, preserving every sample; publish only a complete file."""
    if not paths:
        raise ValueError("No segments to concatenate")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        combined = Path(temporary) / "combined.wav"
        parameters = None
        with wave.open(str(combined), "wb") as output:
            for path in paths:
                with wave.open(str(path), "rb") as source:
                    current = (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype())
                    if source.getnframes() < 1 or current[3] != "NONE":
                        raise ValueError(f"Empty or non-PCM segment: {path}")
                    if parameters is None:
                        parameters = current
                        output.setnchannels(current[0])
                        output.setsampwidth(current[1])
                        output.setframerate(current[2])
                    elif parameters != current:
                        raise ValueError(f"Segment audio formats differ: {path}")
                    remaining = source.getnframes()
                    while remaining:
                        frames = source.readframes(min(remaining, 65536))
                        count = len(frames) // (current[0] * current[1])
                        if not count:
                            raise ValueError(f"Truncated segment audio: {path}")
                        output.writeframesraw(frames)
                        remaining -= count
        combined.replace(destination)


def synthesize_segments(job: TtsJob, artifact_root: Path, synthesize) -> Path:
    destination = (artifact_root / job.output_path).resolve()
    parts = split_emotion_job(job)
    paths = []
    for part in parts:
        path = (artifact_root / part.output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".job.json").write_text(json.dumps(part.to_dict(), indent=2) + "\n", encoding="utf-8")
        rendered = synthesize(part, artifact_root)
        if rendered.resolve() != path:
            raise ValueError("Backend returned a different segment path")
        paths.append(path)
    concatenate_wav(tuple(paths), destination)
    manifest = [{"job_id": part.id, "audio": str(path), "sha256": file_hash(path)}
                for part, path in zip(parts, paths)]
    (paths[0].parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return destination
