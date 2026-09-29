from __future__ import annotations

from pathlib import Path

from lessons_in_cast_core.annotation import (
    Annotation,
    DialogueAction,
    ValidatedAnnotation,
    ValidationStatus,
)
from lessons_in_cast_core.dialogue import DialogueRecord


def record(
    sequence: int,
    *,
    character: str = "a",
    dialogue: str = "Hello.",
    filename: str = "game/route_a.rpy",
) -> DialogueRecord:
    return DialogueRecord(
        id=f"id-{sequence}",
        sequence=sequence,
        identifier=f"line_{sequence}",
        character=character,
        dialogue=dialogue,
        filename=filename,
        line_number=sequence + 1,
        source_statement=f'{character} "[what]"',
    )


def accepted_annotation(
    item: DialogueRecord,
    *,
    action: DialogueAction = DialogueAction.SPEAK,
    effects: tuple[str, ...] = (),
) -> ValidatedAnnotation:
    annotation = Annotation(
        id=item.id,
        action=action,
        spoken_text=item.dialogue if "speak" in action.value else "",
        emotion="neutral" if "speak" in action.value else None,
        delivery={},
        effects=effects,
        confidence=1.0,
        review_required=False,
    )
    return ValidatedAnnotation(
        dialogue_id=item.id,
        batch_id="batch",
        status=ValidationStatus.ACCEPTED,
        annotation=annotation,
        issues=(),
        input_hash="input",
        prompt_version="1",
        annotator_config_hash="config",
        processed_at="2026-09-04T00:00:00+00:00",
    )


def write_minimal_workspace(root: Path) -> None:
    """Write the workspace files that the polish stage reads."""

    configs = root / "configs"
    configs.mkdir(exist_ok=True)
    (configs / "characters.toml").write_text(
        "[characters.a]\n"
        'name = "Alice"\n'
        'definition_path = "game/definitions.rpy"\n'
        "definition_line = 1\n"
        "built_in = false\n"
        'default_voice_profile = "profiles/a.py"\n',
        encoding="utf-8",
    )
    prompts = root / "prompts"
    prompts.mkdir(exist_ok=True)
    (prompts / "codex_dialogue_polish.md").write_text("Polish the dialogue.\n", encoding="utf-8")
