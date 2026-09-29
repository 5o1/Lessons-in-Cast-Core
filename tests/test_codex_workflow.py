from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lessons_in_cast_core.annotation import (
    CodexAnnotationWorkflow,
    CodexWorkspace,
    build_annotation_request,
)
from lessons_in_cast_core.config import AnnotationConfig, BatchingConfig
from lessons_in_cast_core.dialogue import DialogueBatchBuilder
from lessons_in_cast_core.jsonl import read_jsonl, write_jsonl

from .helpers import record


def annotation(target_id: str, text: str) -> dict[str, object]:
    return {
        "id": target_id,
        "action": "speak",
        "spoken_text": text,
        "emotion": "neutral",
        "delivery": {},
        "effects": [],
        "confidence": 1.0,
        "review_required": False,
        "reason": None,
    }


class CodexWorkflowTests(unittest.TestCase):
    def test_contiguous_packet_round_trip_and_resume(self) -> None:
        records = [record(index) for index in range(6)]
        annotation_config = AnnotationConfig(
            allowed_emotions=frozenset({"neutral"}),
            allowed_effects=frozenset(),
        )
        requests = [
            build_annotation_request(
                batch,
                prompt_version="codex-v1",
                annotation_config=annotation_config,
            )
            for batch in DialogueBatchBuilder(
                BatchingConfig(
                    target_size=1,
                    context_before=1,
                    context_after=1,
                )
            ).build(records)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = root / "prompt.md"
            prompt.write_text("test prompt\n", encoding="utf-8")
            requests_path = root / "requests.jsonl"
            responses_path = root / "responses.jsonl"
            workspace = CodexWorkspace(root / "codex" / "initial")
            write_jsonl(requests, requests_path)
            workflow = CodexAnnotationWorkflow(prompt)

            exported = workflow.export_next(
                requests_path,
                responses_path,
                workspace,
                batches_per_packet=3,
            )
            self.assertEqual(exported.batch_count, 3)
            self.assertEqual(exported.target_count, 3)
            packet = json.loads(workspace.inbox.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["line_number"] for item in packet["records"]],
                sorted(item["line_number"] for item in packet["records"]),
            )
            targets = [item for item in packet["records"] if item["target"]]
            self.assertEqual(
                [item["id"] for item in targets],
                [item.id for item in records[:3]],
            )

            workspace.outbox.write_text(
                json.dumps(
                    {
                        "packet_id": packet["packet_id"],
                        "annotations": [
                            annotation(item["id"], item["dialogue"])
                            for item in targets
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                workflow.import_outbox(
                    requests_path,
                    responses_path,
                    workspace,
                ),
                3,
            )
            status = workflow.status(requests_path, responses_path)
            self.assertEqual(status.completed_batches, 3)
            self.assertEqual(status.pending_batches, 3)
            envelopes = list(read_jsonl(responses_path))
            self.assertTrue(
                all(
                    item["annotator_configuration"]["adapter"]
                    == "codex-file-workflow"
                    for item in envelopes
                )
            )

            next_packet = workflow.export_next(
                requests_path,
                responses_path,
                workspace,
                batches_per_packet=3,
            )
            self.assertEqual(next_packet.target_count, 3)
            packet = json.loads(workspace.inbox.read_text(encoding="utf-8"))
            self.assertEqual(
                next(item["id"] for item in packet["records"] if item["target"]),
                records[3].id,
            )

    def test_packet_does_not_cross_source_file(self) -> None:
        records = [
            record(0, filename="game/one.rpy"),
            record(1, filename="game/one.rpy"),
            record(2, filename="game/two.rpy"),
        ]
        annotation_config = AnnotationConfig(
            allowed_emotions=frozenset({"neutral"}),
            allowed_effects=frozenset(),
        )
        requests = [
            build_annotation_request(batch, annotation_config=annotation_config)
            for batch in DialogueBatchBuilder(
                BatchingConfig(target_size=1, context_before=1, context_after=1)
            ).build(records)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = root / "prompt.md"
            prompt.write_text("test prompt\n", encoding="utf-8")
            requests_path = root / "requests.jsonl"
            write_jsonl(requests, requests_path)
            workflow = CodexAnnotationWorkflow(prompt)
            exported = workflow.export_next(
                requests_path,
                root / "responses.jsonl",
                CodexWorkspace(root / "codex" / "initial"),
                batches_per_packet=10,
            )
            self.assertEqual(exported.batch_count, 2)
            self.assertEqual(exported.source_file, "game/one.rpy")

    def test_source_scope_covers_every_batch_from_selected_file(self) -> None:
        records = [
            record(0, filename="game/one.rpy"),
            record(1, filename="game/two.rpy"),
            record(2, filename="game/two.rpy"),
        ]
        annotation_config = AnnotationConfig(
            allowed_emotions=frozenset({"neutral"}),
            allowed_effects=frozenset(),
        )
        requests = [
            build_annotation_request(batch, annotation_config=annotation_config)
            for batch in DialogueBatchBuilder(
                BatchingConfig(target_size=1, context_before=1, context_after=1)
            ).build(records)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = root / "prompt.md"
            prompt.write_text("test prompt\n", encoding="utf-8")
            requests_path = root / "requests.jsonl"
            responses_path = root / "responses.jsonl"
            workspace = CodexWorkspace(root / "codex" / "initial")
            write_jsonl(requests, requests_path)
            workflow = CodexAnnotationWorkflow(
                prompt,
                source_files=("game/two.rpy",),
            )

            status = workflow.status(requests_path, responses_path)
            self.assertEqual(status.total_batches, 2)
            exported = workflow.export_next(
                requests_path,
                responses_path,
                workspace,
                batches_per_packet=10,
            )
            self.assertEqual(exported.batch_count, 2)
            self.assertEqual(exported.target_count, 2)
            self.assertEqual(exported.source_file, "game/two.rpy")

    def test_import_rejects_a_modified_inbox(self) -> None:
        item = record(0)
        annotation_config = AnnotationConfig(
            allowed_emotions=frozenset({"neutral"}),
            allowed_effects=frozenset(),
        )
        request = build_annotation_request(
            next(DialogueBatchBuilder(BatchingConfig(target_size=1)).build([item])),
            annotation_config=annotation_config,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = root / "prompt.md"
            prompt.write_text("test prompt\n", encoding="utf-8")
            requests_path = root / "requests.jsonl"
            responses_path = root / "responses.jsonl"
            workspace = CodexWorkspace(root / "codex" / "initial")
            write_jsonl([request], requests_path)
            workflow = CodexAnnotationWorkflow(prompt)
            workflow.export_next(requests_path, responses_path, workspace)
            packet = json.loads(workspace.inbox.read_text(encoding="utf-8"))
            packet["records"][0]["dialogue"] = "tampered"
            workspace.inbox.write_text(json.dumps(packet), encoding="utf-8")
            workspace.outbox.write_text(
                json.dumps(
                    {
                        "packet_id": packet["packet_id"],
                        "annotations": [annotation(item.id, item.dialogue)],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "modified"):
                workflow.import_outbox(
                    requests_path,
                    responses_path,
                    workspace,
                )


if __name__ == "__main__":
    unittest.main()
