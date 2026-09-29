"""Run source-backed sides through the existing backend-neutral profile contract."""

from dataclasses import replace
import json
import os
from pathlib import Path
import wave

from ..characters import load_characters
from ..config import load_pipeline_config
from ..hashing import content_hash, file_hash
from ..model_registry import load_model_registry
from ..paths import PACKAGE_ROOT, SOURCE_ROOT
from ..synthesis.profiles import load_voice_profile
from ..synthesis.planner import SynthesisPlanner
from ..synthesis.audio import WaveRenderer
from ..effects import CoreEffectProcessor
from ..annotation import DialogueAction, ValidationStatus
from .cleaning import validate_cleaning
from .types import load_project, slug


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_listening_sheet(directory: Path, project, cases, manifest: dict) -> None:
    lines = [f"# {project.id}: {manifest['candidate']}", "", project.brief, "",
             f"Coverage: {project.coverage['status']}. {project.coverage['notes']}", "",
             "Background and intention are reader/director notes, not spoken dialogue. Backend adaptation records describe any lost direction.", "",
             "Compare character fit, identity continuity, diction, intention and usable dynamic range. Do not select by loudness alone.", ""]
    for case in cases:
        result = manifest["results"].get(case.id, {})
        take = (f"Take: omitted — {result.get('reason', '')}" if result.get("status") == "omitted"
                else f"Take: [{case.id}](takes/{case.id}.wav) — {result.get('status', 'pending')}")
        lines.extend([f"## {case.title}", "", f"State: {case.state}", "", f"Background: {case.background}", "",
                      f"Addressee: {case.addressee}. Intention: {case.intention}", "", f"Source text: {case.source.dialogue}", "",
                      f"Cleaned spoken text: {result.get('spoken_text', 'See the validated cleaning result.')}", "",
                      f"Direction: {case.direction}", "", "Listen for: " + "; ".join(case.listen_for), "",
                      "Avoid: " + "; ".join(case.avoid), "",
                      f"Source: {case.source.filename}:{case.source.line_number}, label {case.source.label}, scene {case.source.scene}", "",
                      take, ""])
    lines.extend(["## Listening decisions", "", "Record shortlist / callback / reject / undecided decisions in casting_notes.json via the review command. Technical completion is not casting acceptance.", ""])
    (directory / "README.md").write_text("\n".join(lines), encoding="utf-8")


def render_project(root: Path, project_path: Path, run: str, *, cleaning: Path | None = None, profile: Path | None = None,
                   reference_audio: Path | None = None, case_ids: tuple[str, ...] = (), candidate: str = "default") -> Path:
    root = root.resolve()
    slug(run)
    project = load_project(project_path)
    selected = set(case_ids)
    if selected - {case.id for case in project.cases}:
        raise ValueError(f"Unknown cases: {sorted(selected - {case.id for case in project.cases})}")
    cases = tuple(case for case in project.cases if not selected or case.id in selected)
    config = load_pipeline_config(repository_root=root)
    if cleaning is None:
        raise ValueError("A validated cleaning run is required; pass --cleaning")
    metadata, records, validated = validate_cleaning(root, project, cleaning)
    for case in cases:
        result = validated.get(metadata["case_targets"][case.id])
        if result is None or result.status is not ValidationStatus.ACCEPTED or result.annotation is None:
            raise ValueError(f"Polish not accepted for {case.id}; resolve missing/review/retry results first")
        config.effects.resolve(result.annotation.effects)
    character = load_characters(repository_root=root)[project.character]
    chosen = profile or Path(character.default_voice_profile)
    if profile is None and not character.default_voice_profile:
        raise ValueError("No default profile; pass --profile")
    entrypoint = (chosen if chosen.is_absolute() else root / chosen).resolve()
    directory = root / "build/auditions" / run
    if not directory.resolve().is_relative_to((root / "build/auditions").resolve()):
        raise ValueError("Run directory escapes build/auditions")
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".render.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    pipeline = None
    try:
        pipeline = load_voice_profile(root, entrypoint, character,
            replace(config, audio=replace(config.audio, format=config.audio.intermediate_format)),
            model_registry=load_model_registry(repository_root=root))
        if reference_audio is not None:
            pipeline.override_reference_audio(reference_audio.resolve())
        dependencies = pipeline.prepare()
        inputs = {"project": project.document, "cases": [case.id for case in cases], "candidate": candidate,
                  "effects": config.effects.to_dict(),
                  "cleaning": {"directory": str(cleaning.resolve()), "requests_sha256": metadata["requests_sha256"],
                               "responses_sha256": file_hash(cleaning / "annotation_responses.jsonl")},
                  "polish_responses_sha256": metadata.get("polish_responses_sha256"),
                  "profile": str(entrypoint.relative_to(root)), "configuration": pipeline.configuration,
                  "reference_override": str(reference_audio.resolve()) if reference_audio else None,
                  "dependencies": {str(path): file_hash(path) for path in dependencies},
                  "profile_files": {str(path.relative_to(root)): file_hash(path) for path in sorted(entrypoint.parent.rglob("*"))
                                    if path.is_file() and path.suffix in {".py", ".toml", ".json"} and "assets" not in path.relative_to(entrypoint.parent).parts},
                  "core_files": {str(path.relative_to(SOURCE_ROOT.parent)): file_hash(path) for path in sorted(PACKAGE_ROOT.rglob("*.py"))},
                  "pronunciations_sha256": file_hash(root / "configs/pronunciations.toml")}
        fingerprint = content_hash(inputs)
        planned_character = replace(character, default_voice_profile=str(entrypoint.relative_to(root)), variants=())
        planner = SynthesisPlanner({project.character: planned_character}, audio_config=replace(config.audio, format="wav"),
                                   synthesizer_configuration={"profile": pipeline.configuration, "audition_inputs": fingerprint})
        jobs = {}
        tasks = {}
        renderer = WaveRenderer(CoreEffectProcessor(config.effects, sample_rate=config.audio.sample_rate,
                                                    channels=config.audio.channels, ffmpeg_executable=config.audio.ffmpeg_executable),
                                audio_config=replace(config.audio, format="wav"))
        for case in cases:
            target_id = metadata["case_targets"][case.id]
            # Each side is an independent reading, even when two sides share
            # source text or a source identifier. Do not deduplicate takes.
            planned = planner.plan(records, [validated[target_id]])
            if planned.issues:
                raise ValueError(f"Synthesis planning rejected audition targets: {planned.issues}")
            if planned.render_tasks:
                tasks[target_id] = planned.render_tasks[0]
            if validated[target_id].annotation.action in {DialogueAction.SPEAK, DialogueAction.SPEAK_WITH_EFFECT}:
                if len(planned.jobs) != 1:
                    raise ValueError("An audition side requires a single-speaker profile")
                jobs[target_id] = planned.jobs[0]
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["fingerprint"] != fingerprint:
                raise ValueError("Run inputs changed; use a new --run name (existing takes are preserved)")
        else:
            manifest = {"fingerprint": fingerprint, "candidate": candidate, "status": "running", "results": {}}
        save(directory / "inputs.json", inputs)
        save(manifest_path, manifest)
        write_listening_sheet(directory, project, cases, manifest)
        (directory / "takes").mkdir(exist_ok=True)
        if not (directory / "casting_notes.json").exists():
            save(directory / "casting_notes.json", {case.id: {"decision": "undecided", "notes": ""} for case in cases})
        for number, case in enumerate(cases, 1):
            output = directory / "takes" / f"{case.id}.wav"
            target_id = metadata["case_targets"][case.id]
            annotation = validated[target_id].annotation
            if annotation.action is DialogueAction.OMIT:
                manifest["results"][case.id] = {"status": "omitted", "spoken_text": "", "reason": annotation.reason}
                save(manifest_path, manifest)
                continue
            previous = manifest["results"].get(case.id)
            if output.exists():
                if previous is None or previous.get("status") != "complete" or previous["sha256"] != file_hash(output):
                    raise ValueError(f"Unverified or modified take: {output}; use a new run")
                print(f"[{number}/{len(cases)}] {case.id}: cached", flush=True)
                continue
            # The normal planner supplies text, labels and performance from validated cleaning.
            # Audition changes only the take's artifact identity and output location.
            task = tasks[target_id]
            job = None
            if target_id in jobs:
                raw_output = directory / "raw" / f"{case.id}.wav" if task.effects else output
                raw_output.parent.mkdir(parents=True, exist_ok=True)
                job = replace(jobs[target_id], id=case.id, output_path=str(raw_output))
                save(directory / "takes" / f"{case.id}.job.json", job.to_dict())
                save(directory / "takes" / f"{case.id}.adaptation.json", pipeline.adapt(job).to_dict())
            print(f"[{number}/{len(cases)}] {case.id}: rendering", flush=True)
            try:
                if job is not None:
                    result = pipeline.render(job, directory)
                    if result.resolve() != Path(job.output_path).resolve():
                        raise ValueError("Profile returned a different path than the requested raw take")
                if task.effects:
                    effect_task = replace(task, output_path=str(output), component_job_ids=(job.id,) if job else ())
                    save(directory / "takes" / f"{case.id}.render.json", effect_task.to_dict())
                    result = renderer.render(effect_task, {job.id: job} if job else {}, directory)
                if result.resolve() != output.resolve():
                    raise ValueError("Profile returned a different path than the requested take")
                with wave.open(str(output)) as audio:
                    seconds = audio.getnframes() / audio.getframerate()
                    if seconds <= 0:
                        raise ValueError("Empty audition take")
                manifest["results"][case.id] = {"status": "complete", "audio": str(output.relative_to(directory)),
                                               "seconds": seconds, "sha256": file_hash(output), "spoken_text": job.text if job else "",
                                               "effects": list(task.effects)}
            except Exception as exc:
                manifest["status"] = "failed"
                manifest["results"][case.id] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                save(manifest_path, manifest)
                write_listening_sheet(directory, project, cases, manifest)
                raise
            save(manifest_path, manifest)
        manifest["status"] = "complete"
        save(manifest_path, manifest)
        write_listening_sheet(directory, project, cases, manifest)
        return directory
    finally:
        try:
            if pipeline is not None:
                pipeline.close()
        finally:
            lock.unlink()
