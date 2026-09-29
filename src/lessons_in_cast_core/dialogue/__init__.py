"""Dialogue records and streaming transformation contracts."""

from .api import DialogueProcessor, DialogueReader, DialogueWriter
from .audit import audit_dialogue
from .batching import DialogueBatch, DialogueBatchBuilder
from .jsonl import JsonlDialogueReader, JsonlDialogueWriter
from .scopes import DialogueScope, SpeakerOverride, load_dialogue_scopes
from .types import DialogueRecord

__all__ = [
    "DialogueBatch",
    "DialogueBatchBuilder",
    "DialogueProcessor",
    "DialogueReader",
    "DialogueRecord",
    "DialogueScope",
    "SpeakerOverride",
    "DialogueWriter",
    "JsonlDialogueReader",
    "JsonlDialogueWriter",
    "audit_dialogue",
    "load_dialogue_scopes",
]
