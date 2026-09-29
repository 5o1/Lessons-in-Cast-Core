"""Prepare audition targets for the normal Codex annotation/validation stages."""

from collections import defaultdict
from dataclasses import asdict, replace
import json
import shlex
from pathlib import Path

from ..annotation import build_annotation_request, ValidatedAnnotation
from ..annotation.codex import CodexAnnotationWorkflow, CodexWorkspace
from ..characters import load_characters
from ..config import load_pipeline_config, load_workspace_config
from ..dialogue import DialogueBatch, DialogueRecord
from ..galgame import load_galgame_backend
from ..hashing import content_hash, file_hash
from ..jsonl import read_jsonl, write_jsonl
from ..pipeline import DialoguePipeline
from ..workflow.artifacts import ArtifactLayout
from .types import load_project


def annotation_configuration(config) -> dict:
    values = asdict(config.annotation)
    values.pop("allowed_emotions")
    values.pop("emotion_presets", None)
    values["allowed_effects"] = sorted(values["allowed_effects"])
    return values


def workflow(root: Path, directory: Path, *, stage: str = "cleaning") -> CodexAnnotationWorkflow:
    config = load_pipeline_config(repository_root=root)
    # Audition targets span all chapters; do not inherit the production demo filter.
    prefix = "python -m lessons_in_cast_core.audition"
    return CodexAnnotationWorkflow(root / (config.codex.polish_prompt_path if stage == "polish" else config.codex.prompt_path),
        {key: character.name for key, character in load_characters(repository_root=root).items()},
        task_commands=(f"{prefix} {stage}-import --cleaning {shlex.quote(str(directory))}",
                       f"{prefix} {stage}-next --cleaning {shlex.quote(str(directory))}"))


def prepare_cleaning(root: Path, project_path: Path, dialogue: Path, directory: Path) -> dict:
    project = load_project(project_path)
    if directory.exists():
        raise FileExistsError("Cleaning directory exists; use a new run name")
    config = load_pipeline_config(repository_root=root)
    backend = load_galgame_backend(config.galgame.backend)
    release = load_workspace_config(repository_root=root).release_path
    relevant = {case.source.filename for case in project.cases}
    groups = defaultdict(list)
    for record in backend.read_dialogue(dialogue, allowed_sources=relevant, source_root=release):
        groups[(record.filename, record.label)].append(record)
    targets, requests, cases = [], [], {}
    for case in project.cases:
        rows = sorted(groups[(case.source.filename, case.source.label)], key=lambda r: (r.line_number, r.sequence))
        hits = [i for i, row in enumerate(rows) if row.id == case.source.id]
        if len(hits) != 1 or rows[hits[0]].dialogue != case.source.dialogue:
            raise ValueError(f"Case source is missing or changed: {case.id}")
        index = hits[0]
        target = replace(case.source, id=content_hash({"project": project.id, "case": case.id})[:24], character=project.character)
        targets.append(target)
        cases[case.id] = target.id
        batch = DialogueBatch(content_hash({"project": project.id, "case": case.id, "source": case.source.id})[:24],
            tuple(rows[max(0, index-config.batching.context_before):index]), (target,),
            tuple(rows[index+1:index+1+config.batching.context_after]))
        request = build_annotation_request(batch, prompt_version="cleaning-v4", annotation_config=config.annotation, stage="cleaning")
        request["independent_context"] = True
        request["director_notes"] = {target.id: {"role_brief": project.brief, "case": case.id,
            "state": case.state, "background": case.background, "addressee": case.addressee,
            "intention": case.intention, "direction": case.direction,
            "listen_for": list(case.listen_for), "avoid": list(case.avoid), "original_speaker": case.source.character}}
        requests.append(request)
    # Source-file/line order preserves the normal independent-thread reading workflow.
    requests.sort(key=lambda r: (r["batch"]["targets"][0]["filename"], r["batch"]["targets"][0]["line_number"]))
    from ..kantoku import Kantoku, bind_direction
    director = Kantoku(root, config, load_characters(repository_root=root), release if config.galgame.backend == "renpy" else None)
    requests = [bind_direction(request, director) for request in requests]
    directory.mkdir(parents=True)
    layout = ArtifactLayout(directory)
    write_jsonl((target.to_dict() for target in targets), layout.raw_dialogue)
    write_jsonl(requests, layout.annotation_requests)
    metadata = {"project_sha256": content_hash(project.document), "project": project.document, "case_targets": cases,
                "dialogue_sha256": file_hash(dialogue), "raw_sha256": file_hash(layout.raw_dialogue),
                "requests_sha256": file_hash(layout.annotation_requests), "annotation_config": annotation_configuration(config),
                "prompt_sha256": file_hash(root / config.codex.prompt_path)}
    (directory / "audition-cleaning.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    if config.cleaning.backend == "codex":
        workflow(root, directory).export_next(layout.annotation_requests, layout.annotation_responses, CodexWorkspace(directory / "codex"), batches_per_packet=1)
    return {"targets": len(targets), "directory": str(directory), "status": f"awaiting_{config.cleaning.backend}_annotation"}


def validate_cleaning(root: Path, project, directory: Path, *, require_polish: bool = True) -> tuple[dict, dict, dict]:
    metadata = json.loads((directory / "audition-cleaning.json").read_text(encoding="utf-8"))
    config = load_pipeline_config(repository_root=root)
    layout = ArtifactLayout(directory)
    expected = {"project_sha256": content_hash(project.document), "raw_sha256": file_hash(layout.raw_dialogue),
                "requests_sha256": file_hash(layout.annotation_requests),
                "prompt_sha256": file_hash(root / config.codex.prompt_path)}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Cleaning input/project/prompt changed; prepare a fresh cleaning run")
    if content_hash(metadata["annotation_config"]) != content_hash(annotation_configuration(config)):
        raise ValueError("Annotation configuration changed; prepare a fresh cleaning run")
    if not layout.annotation_responses.exists():
        raise ValueError("No cleaning annotations. Complete the independent Codex workflow before rendering")
    pipeline = DialoguePipeline(config=config, characters=load_characters(repository_root=root))
    pipeline.validate(layout)
    selected = layout
    if require_polish:
        pipeline.validate_polish(layout)
        selected = layout.polish
        metadata["polish_responses_sha256"] = file_hash(selected.annotation_responses)
    records = {row["id"]: DialogueRecord.from_dict(row) for row in read_jsonl(layout.raw_dialogue)}
    validated = {row["dialogue_id"]: ValidatedAnnotation.from_dict(row) for row in read_jsonl(selected.validated)}
    return metadata, records, validated
