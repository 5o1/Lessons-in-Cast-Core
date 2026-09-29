"""Validation, retry generation, and status splitting for annotation results."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any

from ..annotation import (
    AnnotationValidator,
    ValidatedAnnotation,
    ValidationStatus,
    apply_override,
    build_annotation_request,
    load_overrides,
)
from ..config import PipelineConfig
from ..dialogue import DialogueBatch, DialogueRecord, JsonlDialogueReader
from ..hashing import content_hash
from ..jsonl import AtomicJsonlWriter, JsonlIndex, read_jsonl
from .artifacts import ArtifactLayout
from .manifest import update_run_manifest
from .types import ValidationSummary


class AnnotationValidationStage:
    """Validate untrusted model responses and prepare focused retries."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._validator = AnnotationValidator(config.annotation)

    def run(
        self,
        layout: ArtifactLayout,
        *,
        overrides_path: Path | None = None,
        retry: bool = False,
    ) -> ValidationSummary:
        requests_path = (
            layout.retry_requests if retry else layout.annotation_requests
        )
        responses_path = (
            layout.retry_responses if retry else layout.annotation_responses
        )
        overrides = (
            load_overrides(overrides_path)
            if overrides_path is not None
            else {}
        )
        override_records: dict[str, DialogueRecord] = {}
        missing_override_ids = set(overrides)
        if missing_override_ids:
            for record in JsonlDialogueReader().read(layout.raw_dialogue):
                if record.id in missing_override_ids:
                    override_records[record.id] = record
                    missing_override_ids.remove(record.id)
                    if not missing_override_ids:
                        break
        if missing_override_ids:
            raise ValueError(
                f"Overrides reference unknown IDs: {sorted(missing_override_ids)!r}"
            )

        requested_batch_ids: set[str] = set()
        requested_dialogue_ids: set[str] = set()
        current_output = layout.retry_validated if retry else layout.validated
        issue_path = (
            layout.retry_validation_issues
            if retry
            else layout.validation_issues
        )
        with JsonlIndex(responses_path, "batch_id") as response_index:
            with ExitStack() as stack:
                current_writer = stack.enter_context(
                    AtomicJsonlWriter(current_output)
                )
                issue_writer = stack.enter_context(
                    AtomicJsonlWriter(issue_path)
                )
                retry_writer = stack.enter_context(
                    AtomicJsonlWriter(layout.retry_requests)
                )
                for request in read_jsonl(requests_path):
                    batch = DialogueBatch.from_dict(request["batch"])
                    if batch.id in requested_batch_ids:
                        raise ValueError(
                            f"Duplicate request batch ID: {batch.id!r}"
                        )
                    requested_batch_ids.add(batch.id)
                    target_ids = {target.id for target in batch.targets}
                    duplicate_targets = target_ids & requested_dialogue_ids
                    if duplicate_targets:
                        raise ValueError(
                            "Dialogue targets occur in more than one request: "
                            f"{sorted(duplicate_targets)!r}"
                        )
                    if len(target_ids) != len(batch.targets):
                        raise ValueError(
                            f"Request batch {batch.id!r} contains duplicate targets"
                        )
                    requested_dialogue_ids.update(target_ids)
                    envelope = response_index.get(batch.id)
                    if batch.id in response_index.duplicates:
                        issue_writer.write(
                            self._batch_issue(
                                batch.id,
                                "duplicate_batch_response",
                                "More than one response exists for this batch.",
                            )
                        )
                        envelope = None
                    expected_request_hash = content_hash(request)
                    if (
                        envelope is not None
                        and envelope.get("request_hash")
                        != expected_request_hash
                    ):
                        issue_writer.write(
                            self._batch_issue(
                                batch.id,
                                "request_hash_mismatch",
                                "Response does not match the current annotation "
                                "request.",
                            )
                        )
                        envelope = None
                    if (
                        envelope is not None
                        and envelope.get("prompt_version")
                        != request["prompt_version"]
                    ):
                        issue_writer.write(
                            self._batch_issue(
                                batch.id,
                                "prompt_version_mismatch",
                                "Response prompt version does not match its request.",
                            )
                        )
                        envelope = None
                    response = envelope.get("response", {}) if envelope else {}
                    configuration = (
                        envelope.get("annotator_configuration", {})
                        if envelope
                        else {}
                    )
                    result = self._validator.validate_batch(
                        batch,
                        response,
                        prompt_version=request["prompt_version"],
                        annotator_configuration=configuration,
                        schema_version=request.get("schema_version", 1),
                        stage=request.get("stage", "polish"),
                        cleaned_annotations=request.get("cleaned_annotations"),
                        keyframe_required=request.get("keyframe_required", ()),
                        processed_at=(
                            envelope.get("generated_at")
                            if envelope
                            and isinstance(envelope.get("generated_at"), str)
                            else None
                        ),
                    )
                    for issue in result.issues:
                        issue_writer.write(
                            {**issue.to_dict(), "batch_id": batch.id}
                        )
                    for item in result.records:
                        configured_override = overrides.get(item.dialogue_id)
                        if configured_override is not None:
                            item = apply_override(
                                item,
                                override_records[item.dialogue_id],
                                configured_override,
                                self._validator,
                                stage=request.get("stage", "polish"),
                            )
                        current_writer.write(item.to_dict())
                        if item.status is ValidationStatus.RETRYABLE:
                            retry_writer.write(
                                self._build_retry_request(
                                    batch,
                                    item,
                                    request,
                                )
                            )
                for unknown_batch_id in (
                    set(response_index.offsets) - requested_batch_ids
                ):
                    issue_writer.write(
                        self._batch_issue(
                            unknown_batch_id,
                            "unknown_batch_response",
                            "Response does not correspond to a current request.",
                        )
                    )
            batch_issue_count = issue_writer.count
            retry_request_count = retry_writer.count

        if retry:
            self._merge_retry_results(layout)
        summary = self._split_validated_outputs(layout)
        manifest_key = "retry_validation" if retry else "validation"
        update_run_manifest(
            layout,
            {
                manifest_key: {
                    "counts": {
                        "accepted": summary.accepted_count,
                        "review_required": summary.review_required_count,
                        "retryable": summary.retryable_count,
                        "rejected": summary.rejected_count,
                    },
                    "retry_request_count": retry_request_count,
                    "batch_issue_count": batch_issue_count,
                },
            },
        )
        return ValidationSummary(
            accepted_count=summary.accepted_count,
            review_required_count=summary.review_required_count,
            retryable_count=summary.retryable_count,
            rejected_count=summary.rejected_count,
            retry_request_count=retry_request_count,
            batch_issue_count=batch_issue_count,
        )

    @staticmethod
    def _merge_retry_results(layout: ArtifactLayout) -> None:
        with JsonlIndex(layout.retry_validated, "dialogue_id") as replacements:
            with AtomicJsonlWriter(layout.validated) as output:
                seen: set[str] = set()
                for value in read_jsonl(layout.validated):
                    dialogue_id = value["dialogue_id"]
                    replacement = replacements.get(dialogue_id)
                    output.write(replacement or value)
                    if replacement is not None:
                        seen.add(dialogue_id)
                unknown = set(replacements.offsets) - seen
                if unknown:
                    raise ValueError(
                        "Retry results reference unknown dialogue IDs: "
                        f"{sorted(unknown)!r}"
                    )

    @staticmethod
    def _split_validated_outputs(
        layout: ArtifactLayout,
    ) -> ValidationSummary:
        counts = {status: 0 for status in ValidationStatus}
        with (
            AtomicJsonlWriter(layout.review_required) as review_writer,
            AtomicJsonlWriter(layout.retryable) as retryable_writer,
            AtomicJsonlWriter(layout.rejected) as rejected_writer,
        ):
            for value in read_jsonl(layout.validated):
                status = ValidationStatus(value["status"])
                counts[status] += 1
                if status is ValidationStatus.REVIEW_REQUIRED:
                    review_writer.write(value)
                elif status is ValidationStatus.RETRYABLE:
                    retryable_writer.write(value)
                elif status is ValidationStatus.REJECTED:
                    rejected_writer.write(value)
        return ValidationSummary(
            accepted_count=counts[ValidationStatus.ACCEPTED],
            review_required_count=counts[ValidationStatus.REVIEW_REQUIRED],
            retryable_count=counts[ValidationStatus.RETRYABLE],
            rejected_count=counts[ValidationStatus.REJECTED],
            retry_request_count=0,
            batch_issue_count=0,
        )

    def _build_retry_request(
        self,
        batch: DialogueBatch,
        item: ValidatedAnnotation,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        target = next(
            target for target in batch.targets if target.id == item.dialogue_id
        )
        ordered = sorted([*batch.context_before, *batch.targets, *batch.context_interleaved, *batch.context_after],
                         key=lambda row: (row.line_number, row.id))
        position = ordered.index(target)
        before = tuple(
            ordered[
                max(0, position - self._config.batching.context_before):
                position
            ]
        )
        after = tuple(
            ordered[
                position + 1:
                position + 1 + self._config.batching.context_after
            ]
        )
        attempt = request.get("attempt", 0) + 1
        retry_batch = DialogueBatch(
            id=content_hash(
                {
                    "retry_of": batch.id,
                    "dialogue_id": target.id,
                    "attempt": attempt,
                }
            )[:24],
            context_before=before,
            targets=(target,),
            context_after=after,
        )
        retry_request = build_annotation_request(
            retry_batch,
            prompt_version=request["prompt_version"],
            annotation_config=self._config.annotation,
            stage=request.get("stage", "polish"),
        )
        for key in ("director_notes", "cleaned_annotations", "original_texts"):
            if key in request:
                retry_request[key] = {target.id: request[key][target.id]} if target.id in request[key] else {}
        if target.id in request.get("keyframe_required", ()):
            retry_request["keyframe_required"] = [target.id]
        for key in ("direction_context", "kantoku_hash", "independent_context"):
            if key in request:
                retry_request[key] = request[key]
        retry_request["retry_of"] = request.get("retry_of", batch.id)
        retry_request["attempt"] = attempt
        return retry_request

    @staticmethod
    def _batch_issue(
        batch_id: str,
        code: str,
        message: str,
    ) -> dict[str, Any]:
        return {
            "code": code,
            "message": message,
            "severity": "error",
            "dialogue_id": None,
            "batch_id": batch_id,
        }
