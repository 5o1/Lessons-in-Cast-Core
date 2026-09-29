"""Prepare and validate independent Codex polish packets, bound to cleaning."""

from dataclasses import asdict, replace
import json
from pathlib import Path

from ..annotation import ValidatedAnnotation, ValidationStatus, build_annotation_request
from ..config import find_repository_root
from ..dialogue import DialogueBatch, DialogueRecord
from ..hashing import content_hash, file_hash
from ..jsonl import read_jsonl, write_jsonl
from ..workflow.artifacts import ArtifactLayout
from ..workflow.validation import AnnotationValidationStage


def _needs_gain_envelope(original: str, cleaned: str) -> bool:
    heard = original.lstrip(". …")
    return (len(heard) < len(original) and bool(heard)
            and cleaned == cleaned.lstrip(". …")
            and len(cleaned) > len(heard)
            and cleaned.casefold().endswith(heard.casefold()))


class PolishStage:
    def __init__(self, config):
        self.config = config

    def _inputs(self, layout):
        from ..emotions import emotion_definitions
        return {"raw": file_hash(layout.raw_dialogue), "requests": file_hash(layout.annotation_requests),
                "responses": file_hash(layout.annotation_responses), "validated": file_hash(layout.validated),
                "labels": emotion_definitions(self.config.annotation.allowed_emotions, self.config.annotation.emotion_presets or None),
                "effects": sorted(self.config.annotation.allowed_effects),
                "length_limits": [self.config.annotation.minimum_length_ratio, self.config.annotation.maximum_length_ratio],
                "codex": asdict(self.config.codex)}

    def prepare(self, layout: ArtifactLayout, *, prompt_path: Path | None = None) -> int:
        child = layout.polish
        if child.root.exists():
            raise FileExistsError("Polish already exists; preserve it and prepare a new run for changed cleaning inputs")
        accepted = {row["dialogue_id"]: ValidatedAnnotation.from_dict(row) for row in read_jsonl(layout.validated)}
        original_requests = list(read_jsonl(layout.annotation_requests))
        targets = {row["id"] for request in original_requests for row in request["batch"]["targets"]}
        if not targets or set(accepted) != targets or any(result.status is not ValidationStatus.ACCEPTED or result.annotation is None for result in accepted.values()):
            raise ValueError("Polish requires accepted cleaning results for every target; resolve missing, review and retry results first")
        if any(request.get("stage") != "cleaning" for request in original_requests):
            raise ValueError("Polish input must come from the separate cleaning stage")
        clean = {key: item.annotation.to_dict() for key, item in accepted.items()}
        records = [DialogueRecord.from_dict(row) for row in read_jsonl(layout.raw_dialogue)]
        cleaned_records = {record.id: replace(record, dialogue=clean[record.id]["spoken_text"]) if record.id in clean else record for record in records}
        original_texts = {record.id: record.dialogue for record in records}
        requests = []
        for original in original_requests:
            batch = DialogueBatch.from_dict(original["batch"])
            batch = replace(batch, targets=tuple(cleaned_records[row.id] for row in batch.targets))
            request = build_annotation_request(batch, prompt_version=self.config.codex.polish_prompt_version,
                                               annotation_config=self.config.annotation, stage="polish")
            request["cleaned_annotations"] = {row.id: clean[row.id] for row in batch.targets}
            request["original_texts"] = {row.id: original_texts[row.id] for row in batch.targets}
            request["keyframe_required"] = [row.id for row in batch.targets
                if _needs_gain_envelope(original_texts[row.id], clean[row.id]["spoken_text"])]
            request["director_notes"] = original.get("director_notes", {})
            for key in ("direction_context", "kantoku_hash", "independent_context"):
                if key in original:
                    request[key] = original[key]
            requests.append(request)
        prompt = (prompt_path or (self.config.repository_root or find_repository_root()) / self.config.codex.polish_prompt_path).resolve()
        prompt_hash = file_hash(prompt)
        inputs = self._inputs(layout)
        child.root.mkdir(parents=True)
        write_jsonl((row.to_dict() for row in cleaned_records.values()), child.raw_dialogue)
        write_jsonl(requests, child.annotation_requests)
        metadata = {"cleaning_inputs": inputs, "prompt": str(prompt), "prompt_sha256": prompt_hash,
                    "raw_sha256": file_hash(child.raw_dialogue), "requests_sha256": file_hash(child.annotation_requests)}
        (child.root / "polish.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return len(requests)

    def check_inputs(self, layout: ArtifactLayout) -> None:
        child = layout.polish
        if not (child.root / "polish.json").is_file():
            raise ValueError("No polish stage prepared; accepted cleaning alone cannot be synthesized")
        metadata = json.loads((child.root / "polish.json").read_text(encoding="utf-8"))
        from ..kantoku import Kantoku
        from ..characters import load_characters
        root = self.config.repository_root or find_repository_root()
        director = Kantoku(root, self.config, load_characters(repository_root=root))
        if any(row.get("kantoku_hash", director.fingerprint) != director.fingerprint
               for row in read_jsonl(layout.annotation_requests)):
            raise ValueError("Kantoku changed; prepare a fresh run")
        if (content_hash(metadata["cleaning_inputs"]) != content_hash(self._inputs(layout))
                or metadata["prompt_sha256"] != file_hash(Path(metadata["prompt"]))
                or metadata["raw_sha256"] != file_hash(child.raw_dialogue)
                or metadata["requests_sha256"] != file_hash(child.annotation_requests)):
            raise ValueError("Cleaning, polish inputs, labels or prompt changed; prepare a fresh run")

    def validate(self, layout: ArtifactLayout, *, retry: bool = False):
        self.check_inputs(layout)
        if not layout.polish.annotation_responses.is_file():
            raise ValueError("No polish responses; complete the independent polish workflow before synthesis")
        validator = AnnotationValidationStage(self.config)
        overrides = layout.root / "polish_overrides.toml"
        result = validator.run(layout.polish, retry=retry,
                               overrides_path=overrides if overrides.is_file() else None)
        if not retry and result.retryable_count and layout.polish.retry_responses.is_file():
            result = validator.run(layout.polish, retry=True,
                                   overrides_path=overrides if overrides.is_file() else None)
        return result
