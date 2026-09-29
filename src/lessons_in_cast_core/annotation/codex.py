"""File-based annotation workflow for a long-running Codex thread."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..hashing import content_hash, file_hash
from ..jsonl import AtomicJsonlWriter, JsonlIndex, read_jsonl, write_jsonl


CODEX_WORKFLOW_VERSION = 1


@dataclass(frozen=True, slots=True)
class CodexWorkspace:
    """Transient inbox and outbox used by one Codex annotation thread."""

    root: Path

    @property
    def inbox(self) -> Path:
        return self.root / "inbox.json"

    @property
    def outbox(self) -> Path:
        return self.root / "outbox.json"

    @property
    def task(self) -> Path:
        return self.root / "task.md"


@dataclass(frozen=True, slots=True)
class CodexWorkflowStatus:
    total_batches: int
    completed_batches: int

    @property
    def pending_batches(self) -> int:
        return self.total_batches - self.completed_batches

    def to_dict(self) -> dict[str, int]:
        return {
            "total_batches": self.total_batches,
            "completed_batches": self.completed_batches,
            "pending_batches": self.pending_batches,
        }


@dataclass(frozen=True, slots=True)
class CodexExportResult:
    packet_id: str | None
    batch_count: int
    target_count: int
    source_file: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "batch_count": self.batch_count,
            "target_count": self.target_count,
            "source_file": self.source_file,
        }


class CodexAnnotationWorkflow:
    """Exchange contiguous transcript packets with an independent Codex process."""

    def __init__(
        self,
        prompt_path: Path,
        character_names: Mapping[str, str] | None = None,
        source_files: tuple[str, ...] = (),
        task_commands: tuple[str, str] | None = None,
    ) -> None:
        self._prompt_path = prompt_path.resolve()
        if not self._prompt_path.is_file():
            raise FileNotFoundError(f"Codex prompt does not exist: {self._prompt_path}")
        self._character_names = dict(character_names or {})
        self._source_files = frozenset(source_files)
        self._task_commands = task_commands
        self._configuration = {
            "adapter": "codex-file-workflow",
            "workflow_version": CODEX_WORKFLOW_VERSION,
            "prompt_sha256": file_hash(self._prompt_path),
            "character_mapping_sha256": content_hash(self._character_names),
            "source_files": sorted(self._source_files),
        }

    @property
    def configuration(self) -> dict[str, Any]:
        return dict(self._configuration)

    def status(
        self,
        requests_path: Path,
        responses_path: Path,
    ) -> CodexWorkflowStatus:
        index_context = (
            JsonlIndex(responses_path, "batch_id")
            if responses_path.exists()
            else nullcontext(None)
        )
        total = 0
        completed = 0
        with index_context as responses:
            if responses is not None and responses.duplicates:
                raise ValueError(
                    "Annotation responses contain duplicate batch IDs: "
                    f"{sorted(responses.duplicates)!r}"
                )
            for request in read_jsonl(requests_path):
                if not self._in_scope(request):
                    continue
                total += 1
                if self._is_completed(request, responses):
                    completed += 1
        return CodexWorkflowStatus(total, completed)

    def export_next(
        self,
        requests_path: Path,
        responses_path: Path,
        workspace: CodexWorkspace,
        *,
        batches_per_packet: int = 2,
    ) -> CodexExportResult:
        """Export the next uninterrupted run of pending batches from one source."""

        if batches_per_packet < 1:
            raise ValueError("batches_per_packet must be positive")
        self._require_empty_workspace(workspace)
        index_context = (
            JsonlIndex(responses_path, "batch_id")
            if responses_path.exists()
            else nullcontext(None)
        )
        selected: list[dict[str, Any]] = []
        source_file: str | None = None
        with index_context as responses:
            if responses is not None and responses.duplicates:
                raise ValueError(
                    "Annotation responses contain duplicate batch IDs: "
                    f"{sorted(responses.duplicates)!r}"
                )
            for request in read_jsonl(requests_path):
                if not self._in_scope(request):
                    continue
                completed = self._is_completed(request, responses)
                if not selected:
                    if completed:
                        continue
                    source_file = self._source_file(request)
                    selected.append(request)
                    if len(selected) >= batches_per_packet:
                        break
                    continue
                if completed or self._source_file(request) != source_file:
                    break
                if request.get("independent_context") or selected[0].get("independent_context"):
                    break
                from .context import scope_for
                if scope_for(request, request["batch"]["targets"][0]) != scope_for(selected[0], selected[0]["batch"]["targets"][0]):
                    break
                selected.append(request)
                if len(selected) >= batches_per_packet:
                    break

        workspace.root.mkdir(parents=True, exist_ok=True)
        if not selected:
            workspace.inbox.unlink(missing_ok=True)
            workspace.outbox.unlink(missing_ok=True)
            self._write_task(workspace, None)
            return CodexExportResult(None, 0, 0, None)

        packet = self._build_packet(selected)
        self._write_json(workspace.inbox, packet)
        self._write_task(workspace, packet)
        return CodexExportResult(
            packet_id=packet["packet_id"],
            batch_count=len(packet["batches"]),
            target_count=sum(len(item["target_ids"]) for item in packet["batches"]),
            source_file=packet["source_file"],
        )

    def import_outbox(
        self,
        requests_path: Path,
        responses_path: Path,
        workspace: CodexWorkspace,
        *,
        replace: bool = False,
    ) -> int:
        """Split a packet response into batches and merge trusted provenance."""

        packet = self._read_json_object(workspace.inbox)
        output = self._read_json_object(workspace.outbox)
        supplied_packet_id = packet.get("packet_id")
        packet_content = {
            key: value for key, value in packet.items() if key != "packet_id"
        }
        if (
            not isinstance(supplied_packet_id, str)
            or supplied_packet_id != content_hash(packet_content)[:24]
        ):
            raise ValueError("Codex inbox packet content or identity was modified")
        if packet.get("prompt_sha256") != self._configuration["prompt_sha256"]:
            raise ValueError("Codex prompt changed after this packet was exported")
        if set(output) != {"packet_id", "annotations"}:
            raise ValueError("Codex outbox requires only packet_id and annotations")
        if output.get("packet_id") != packet.get("packet_id"):
            raise ValueError("Codex outbox packet_id does not match the inbox")
        annotations = output.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError("Codex outbox annotations must be an array")

        target_to_batch: dict[str, str] = {}
        batch_order: list[str] = []
        for batch in packet.get("batches", []):
            batch_id = batch.get("batch_id")
            target_ids = batch.get("target_ids")
            if not isinstance(batch_id, str) or not isinstance(target_ids, list):
                raise ValueError("Codex inbox contains an invalid batch index")
            batch_order.append(batch_id)
            for target_id in target_ids:
                if not isinstance(target_id, str) or target_id in target_to_batch:
                    raise ValueError(
                        "Codex inbox contains invalid or duplicate targets"
                    )
                target_to_batch[target_id] = batch_id

        annotations_by_batch = {batch_id: [] for batch_id in batch_order}
        for annotation in annotations:
            if not isinstance(annotation, dict) or not isinstance(
                annotation.get("id"), str
            ):
                raise ValueError("Each Codex annotation requires a string id")
            target_id = annotation["id"]
            if target_id not in target_to_batch:
                raise ValueError(f"Codex returned unknown target ID {target_id!r}")
            annotations_by_batch[target_to_batch[target_id]].append(annotation)

        request_by_batch: dict[str, dict[str, Any]] = {}
        wanted = set(batch_order)
        for request in read_jsonl(requests_path):
            batch_id = request.get("batch", {}).get("batch_id")
            if batch_id in wanted:
                request_by_batch[batch_id] = request
        missing_requests = wanted - set(request_by_batch)
        if missing_requests:
            raise ValueError(
                "Codex packet references missing requests: "
                f"{sorted(missing_requests)!r}"
            )

        generated_at = datetime.now(timezone.utc).isoformat()
        replacements: dict[str, dict[str, Any]] = {}
        for batch_id in batch_order:
            request = request_by_batch[batch_id]
            replacements[batch_id] = {
                "request_hash": content_hash(request),
                "batch_id": batch_id,
                "prompt_version": request["prompt_version"],
                "annotator_configuration": self.configuration,
                "codex_packet_id": packet["packet_id"],
                "generated_at": generated_at,
                "response": {
                    "batch_id": batch_id,
                    "annotations": annotations_by_batch[batch_id],
                },
            }

        if responses_path.exists():
            with JsonlIndex(responses_path, "batch_id") as existing:
                if existing.duplicates:
                    raise ValueError(
                        "Annotation responses contain duplicate batch IDs: "
                        f"{sorted(existing.duplicates)!r}"
                    )
                if not replace:
                    conflicts = [
                        batch_id
                        for batch_id, request in request_by_batch.items()
                        if self._is_completed(request, existing)
                    ]
                    if conflicts:
                        raise ValueError(
                            "Codex responses already exist for batches: "
                            f"{sorted(conflicts)!r}"
                        )

        seen: set[str] = set()
        with AtomicJsonlWriter(responses_path) as writer:
            if responses_path.exists():
                for value in read_jsonl(responses_path):
                    batch_id = value["batch_id"]
                    replacement = replacements.get(batch_id)
                    writer.write(replacement or value)
                    if replacement is not None:
                        seen.add(batch_id)
            for batch_id in batch_order:
                if batch_id not in seen:
                    writer.write(replacements[batch_id])

        workspace.inbox.unlink()
        workspace.outbox.unlink()
        self._write_task(workspace, None)
        return len(replacements)

    def _is_completed(
        self,
        request: dict[str, Any],
        responses: JsonlIndex | None,
    ) -> bool:
        if responses is None:
            return False
        batch_id = request.get("batch", {}).get("batch_id")
        if not isinstance(batch_id, str):
            raise ValueError("Annotation request is missing batch.batch_id")
        envelope = responses.get(batch_id)
        return bool(
            envelope is not None
            and envelope.get("request_hash") == content_hash(request)
            and envelope.get("prompt_version") == request.get("prompt_version")
            and envelope.get("annotator_configuration") == self._configuration
        )

    @staticmethod
    def _source_file(request: dict[str, Any]) -> str:
        batch = request.get("batch")
        if not isinstance(batch, dict):
            raise ValueError("Annotation request is missing batch")
        records = [
            *batch.get("context_before", []),
            *batch.get("targets", []),
            *batch.get("context_interleaved", []),
            *batch.get("context_after", []),
        ]
        filenames = {
            item.get("filename")
            for item in records
            if isinstance(item, dict) and isinstance(item.get("filename"), str)
        }
        if len(filenames) != 1:
            raise ValueError("Codex work packets cannot cross source files")
        return next(iter(filenames))

    def _in_scope(self, request: dict[str, Any]) -> bool:
        return (
            not self._source_files
            or self._source_file(request) in self._source_files
        )

    def _build_packet(self, requests: list[dict[str, Any]]) -> dict[str, Any]:
        source_file = self._source_file(requests[0])
        prompt_versions = {request.get("prompt_version") for request in requests}
        if len(prompt_versions) != 1:
            raise ValueError("Codex work packets require one prompt version")
        allowed_emotions = requests[0].get("allowed_emotions", [])
        allowed_effects = requests[0].get("allowed_effects", [])
        stage = requests[0].get("stage", "polish")
        batches: list[dict[str, Any]] = []
        target_ids: set[str] = set()
        ordered_records: dict[str, dict[str, Any]] = {}
        director_notes: dict[str, Any] = {}
        cleaned_annotations: dict[str, Any] = {}
        original_texts: dict[str, str] = {}
        direction_context: dict[str, Any] = {}
        for request in requests:
            if request.get("stage", "polish") != stage:
                raise ValueError("Codex packets cannot mix cleaning and polish")
            if self._source_file(request) != source_file:
                raise ValueError("Codex work packets cannot cross source files")
            batch = request["batch"]
            current_target_ids = [item["id"] for item in batch["targets"]]
            overlap = target_ids & set(current_target_ids)
            if overlap:
                raise ValueError(
                    f"Targets occur in multiple batches: {sorted(overlap)!r}"
                )
            target_ids.update(current_target_ids)
            supplied_notes = request.get("director_notes", {})
            if not isinstance(supplied_notes, dict) or set(supplied_notes) - set(current_target_ids):
                raise ValueError("Director notes must be keyed by this request's target IDs")
            director_notes.update(supplied_notes)
            cleaned_annotations.update(request.get("cleaned_annotations", {}))
            original_texts.update(request.get("original_texts", {}))
            direction_context.update(request.get("direction_context", {}))
            batches.append(
                {
                    "batch_id": batch["batch_id"],
                    "target_ids": current_target_ids,
                }
            )
            for section in (
                "context_before",
                "targets",
                "context_interleaved",
                "context_after",
            ):
                for record in batch[section]:
                    ordered_records.setdefault(record["id"], dict(record))

        records = []
        for record_id, record in sorted(
            ordered_records.items(),
            key=lambda item: (item[1]["line_number"], item[0]),
        ):
            character_id = record["character"] or "narrator"
            records.append(
                {
                    **record,
                    "character_name": self._character_names.get(
                        character_id,
                        character_id,
                    ),
                    "target": record_id in target_ids,
                }
            )
        line_numbers = [record["line_number"] for record in records]
        if line_numbers != sorted(line_numbers):
            raise ValueError("Codex work packet records are not chronological")
        response_schema = copy.deepcopy(requests[0]["response_schema"])
        response_schema["title"] = "Lessons in Cast Codex packet response"
        response_schema["required"] = ["packet_id", "annotations"]
        response_schema["properties"]["packet_id"] = response_schema[
            "properties"
        ].pop("batch_id")
        packet = {
            "schema_version": CODEX_WORKFLOW_VERSION,
            "stage": stage,
            "emotion_labels": requests[0].get("emotion_labels", {}),
            "cleaned_annotations": cleaned_annotations,
            "original_texts": original_texts,
            "source_file": source_file,
            "prompt_version": next(iter(prompt_versions)),
            "prompt_sha256": self._configuration["prompt_sha256"],
            "allowed_emotions": allowed_emotions,
            "allowed_effects": allowed_effects,
            "batches": batches,
            "records": records,
            "director_notes": director_notes if stage == "polish" else {},
            "direction_context": direction_context,
            "independent_context": any(request.get("independent_context", False) for request in requests),
            "response_schema": response_schema,
        }
        return {"packet_id": content_hash(packet)[:24], **packet}

    @staticmethod
    def _require_empty_workspace(workspace: CodexWorkspace) -> None:
        for path in (workspace.inbox, workspace.outbox):
            if path.exists():
                raise ValueError(
                    f"Codex workspace is active; import or clear {path} first"
                )

    @staticmethod
    def _read_json_object(path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as source:
                value = json.load(source)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Unable to read Codex workspace file {path}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"Codex workspace file must contain one object: {path}")
        return value

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
                json.dump(value, output, ensure_ascii=False, sort_keys=True, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def _write_task(
        self,
        workspace: CodexWorkspace,
        packet: dict[str, Any] | None,
    ) -> None:
        retry_option = " --retry" if workspace.root.name == "retry" else ""
        command = "python3 -m lessons_in_cast_core"
        import_command, next_command = self._task_commands or (
            f"{command} codex-import{retry_option}", f"{command} codex-next{retry_option}")
        if packet is None:
            content = (
                "# Codex dialogue annotation\n\n"
                f"No packet is currently active. Run `{next_command}` to export the next contiguous "
                "transcript segment.\n"
            )
        else:
            content = (
                "# Codex dialogue annotation\n\n"
                f"Read the instructions in `{self._prompt_path}` completely.\n\n"
                f"Input: `{workspace.inbox.resolve()}`\n\n"
                f"Output: `{workspace.outbox.resolve()}`\n\n"
                f"Packet: `{packet['packet_id']}` from `{packet['source_file']}`.\n\n"
                "Process the records in their given order as one continuous "
                "transcript. Write exactly one JSON object and no Markdown to "
                "the output file. Then run "
                f"`{import_command}`. Repeat "
                f"`{next_command}` and the import command in "
                "this same Codex thread to preserve conversational continuity.\n"
            )
        workspace.task.parent.mkdir(parents=True, exist_ok=True)
        if packet is not None and packet.get("independent_context"):
            content += "\nThis is an independent audition side. Reset prior story/acting assumptions and use only its supplied context.\n"
        if packet is not None and packet.get("direction_context"):
            content += (
                "\n## Scoped Kantoku guidance\n\n"
                "Read direction_context by dialogue ID. It contains human guidance snapshots for each source scope. "
                "Recompute guidance on file/label/scene entry; do not carry expired scene instructions forward. "
                "Across labels, reset transient story assumptions unless continuity_from explicitly names the preceding label. "
                "Cleaning uses background only and never assigns acting. Polish uses full character and acting guidance. "
                "Kantoku overrides matching upstream director_notes. Render settings are executed by the program, never spoken.\n"
            )
        if packet is not None and packet.get("director_notes"):
            content += (
                "\n## Director context\n\n"
                "Use director_notes keyed by target ID as non-spoken role/context/acting guidance. "
                "Preserve cleaned words and pauses; add paired semantic emotion labels, delivery and performance through "
                "the polish schema. Each emotion is a single preset label without numeric intensity. "
                "Do not produce model-specific emotion vectors. Do not copy background into spoken_text.\n"
                "Before choosing a voice wrapper, query the candidate profile's list_voice_tags() API "
                "or the project's voice-tags CLI; no wrapper means the default reference.\n"
            )
        workspace.task.write_text(content, encoding="utf-8", newline="\n")
