"""Human-authored overrides applied after model validation."""

from __future__ import annotations

import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..config import ConfigurationError
from ..dialogue import DialogueBatch, DialogueRecord
from .types import ValidatedAnnotation, ValidationIssue, ValidationStatus
from .validation import AnnotationValidator


def load_overrides(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as source:
            data = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"Unable to load {path}: {exc}") from exc
    dialogue = data.get("dialogue", {})
    if not isinstance(dialogue, dict):
        raise ConfigurationError(f"{path}: [dialogue] must be a table")
    result: dict[str, dict[str, Any]] = {}
    for dialogue_id, value in dialogue.items():
        if not isinstance(value, dict):
            raise ConfigurationError(
                f"{path}: dialogue.{dialogue_id} must be a table"
            )
        result[dialogue_id] = dict(value)
    return result


def apply_overrides(
    validated: list[ValidatedAnnotation],
    records_by_id: dict[str, DialogueRecord],
    overrides: dict[str, dict[str, Any]],
    validator: AnnotationValidator,
) -> list[ValidatedAnnotation]:
    """Merge and revalidate manual fields; approved warnings become accepted."""

    known_ids = {item.dialogue_id for item in validated}
    unknown = set(overrides) - known_ids
    if unknown:
        raise ConfigurationError(f"Overrides reference unknown IDs: {sorted(unknown)!r}")

    result: list[ValidatedAnnotation] = []
    for item in validated:
        configured_override = overrides.get(item.dialogue_id)
        if configured_override is None:
            result.append(item)
            continue
        result.append(
            apply_override(
                item,
                records_by_id[item.dialogue_id],
                configured_override,
                validator,
            )
        )
    return result


def apply_override(
    item: ValidatedAnnotation,
    record: DialogueRecord,
    configured_override: dict[str, Any],
    validator: AnnotationValidator,
    *, stage: str = "polish",
) -> ValidatedAnnotation:
    """Apply and validate one human decision without mutating its source."""

    raw_override = dict(configured_override)
    approved = raw_override.pop("approved", False)
    requested_status = raw_override.pop("status", None)
    if not isinstance(approved, bool):
        raise ConfigurationError(
            f"Override {item.dialogue_id!r} approved must be a boolean"
        )
    if requested_status == ValidationStatus.REJECTED.value:
        return replace(
            item,
            status=ValidationStatus.REJECTED,
            issues=(
                *item.issues,
                ValidationIssue(
                    "manually_rejected",
                    "The annotation was rejected by a human override.",
                    "warning",
                    item.dialogue_id,
                ),
            ),
            source="manual",
        )
    if requested_status is not None:
        raise ConfigurationError(
            f"Override {item.dialogue_id!r} status must be 'rejected'"
        )
    base = item.annotation.to_dict() if item.annotation is not None else {}
    if stage == "cleaning":
        base.pop("emotion", None)
        base.pop("delivery", None)
    if (
        "spoken_text" in raw_override
        and raw_override["spoken_text"] != base.get("spoken_text")
        and "performance" not in raw_override
    ):
        performance = dict(base.get("performance", {}))
        performance["cues"] = []
        base["performance"] = performance
    merged = {**base, **raw_override, "id": item.dialogue_id}
    batch = DialogueBatch(
        id=f"manual:{item.dialogue_id}",
        context_before=(),
        targets=(record,),
        context_after=(),
    )
    checked = validator.validate_batch(
        batch,
        {"batch_id": batch.id, "annotations": [merged]},
        prompt_version=item.prompt_version,
        annotator_configuration={"adapter": "manual-override"},
        processed_at=item.processed_at,
        stage=stage,
    ).records[0]
    status = checked.status
    if (
        approved
        and checked.annotation is not None
        and status is ValidationStatus.REVIEW_REQUIRED
    ):
        status = ValidationStatus.ACCEPTED
    return replace(checked, status=status, source="manual")
