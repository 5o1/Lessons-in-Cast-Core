from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lessons_in_cast_core.config import BatchingConfig
from lessons_in_cast_core.dialogue import (
    DialogueBatchBuilder,
    JsonlDialogueReader,
    JsonlDialogueWriter,
    audit_dialogue,
)
from lessons_in_cast_core.galgame.renpy import DialogueTabError, TabDialogueReader

from .helpers import record


HEADER = (
    "Identifier\tCharacter\tDialogue\tFilename\tLine Number\tRen'Py Script\n"
)


class DialogueTabTests(unittest.TestCase):
    def test_tab_import_is_lossless_and_jsonl_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dialogue.tab"
            destination = Path(directory) / "raw.jsonl"
            source.write_text(
                HEADER
                + 'line_a\ta\tHello\\nworld.\tgame\\route_a.rpy\t12\ta "[what]"\n',
                encoding="utf-8",
            )
            records = list(TabDialogueReader().read(source))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].dialogue, r"Hello\nworld.")
            self.assertEqual(records[0].filename, "game/route_a.rpy")
            JsonlDialogueWriter().write(records, destination)
            self.assertEqual(list(JsonlDialogueReader().read(destination)), records)

    def test_enriches_rows_with_nearest_label_and_scene(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release"
            game = release / "game"
            game.mkdir(parents=True)
            (game / "chapter.rpy").write_text(
                "label opening:\n    scene classroom with dissolve\n    a \"Hello.\"\n",
                encoding="utf-8",
            )
            source = root / "dialogue.tab"
            source.write_text(
                HEADER
                + 'line_a\ta\tHello.\tgame/chapter.rpy\t3\ta "[what]"\n',
                encoding="utf-8",
            )

            record = next(TabDialogueReader(source_root=release).read(source))

            self.assertEqual(record.label, "opening")
            self.assertEqual(record.scene, "classroom")

    def test_missing_context_warns_and_truncates_at_first_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release"
            game = release / "game"
            game.mkdir(parents=True)
            (game / "no_label.rpy").write_text(
                "scene room\n    a \"Hello.\"\n",
                encoding="utf-8",
            )
            (game / "no_scene.rpy").write_text(
                "label opening:\n    a \"Hello.\"\n",
                encoding="utf-8",
            )
            source = root / "dialogue.tab"
            source.write_text(
                HEADER
                + 'one\ta\tOne.\tgame/no_label.rpy\t2\ta "[what]"\n'
                + 'two\ta\tTwo.\tgame/no_scene.rpy\t2\ta "[what]"\n'
                + 'three\ta\tThree.\tgame/missing.rpy\t1\ta "[what]"\n',
                encoding="utf-8",
            )

            with self.assertWarns(RuntimeWarning):
                records = list(
                    TabDialogueReader(source_root=release).read(source)
                )

            self.assertEqual((records[0].label, records[0].scene), ("", ""))
            self.assertEqual(
                (records[1].label, records[1].scene),
                ("opening", ""),
            )
            self.assertEqual((records[2].label, records[2].scene), ("", ""))

    def test_source_filter_resequences_selected_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dialogue.tab"
            source.write_text(
                HEADER
                + 'one\ta\tOne\tgame/route_a.rpy\t1\ta "[what]"\n'
                + 'two\tm\tTwo\tgame/route_b.rpy\t2\tm "[what]"\n',
                encoding="utf-8",
            )
            records = list(
                TabDialogueReader(["game/route_b.rpy"]).read(source)
            )
            self.assertEqual([item.identifier for item in records], ["two"])
            self.assertEqual(records[0].sequence, 0)

    def test_rejects_wrong_column_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dialogue.tab"
            source.write_text(HEADER + "too\tfew\n", encoding="utf-8")
            with self.assertRaises(DialogueTabError):
                list(TabDialogueReader().read(source))


class DialogueBatchTests(unittest.TestCase):
    def test_targets_are_emitted_once_and_never_cross_files(self) -> None:
        records = [
            record(index, filename="game/one.rpy" if index < 5 else "game/two.rpy")
            for index in range(8)
        ]
        builder = DialogueBatchBuilder(
            BatchingConfig(
                target_size=2,
                context_before=1,
                context_after=1,
                max_characters=1_000,
            )
        )
        batches = list(builder.build(records))
        target_ids = [item.id for batch in batches for item in batch.targets]
        self.assertEqual(target_ids, [item.id for item in records])
        for batch in batches:
            filenames = {
                item.filename
                for item in (
                    *batch.context_before,
                    *batch.targets,
                    *batch.context_after,
                )
            }
            self.assertEqual(len(filenames), 1)

    def test_character_budget_trims_far_context(self) -> None:
        records = [record(index, dialogue="x" * 10) for index in range(5)]
        builder = DialogueBatchBuilder(
            BatchingConfig(
                target_size=1,
                context_before=2,
                context_after=2,
                max_characters=12,
            )
        )
        batches = list(builder.build(records))
        self.assertTrue(all(len(batch.targets) == 1 for batch in batches))
        self.assertTrue(
            all(
                not batch.context_before and not batch.context_after
                for batch in batches
            )
        )

    def test_filters_targets_without_dropping_interleaved_context(self) -> None:
        records = [
            record(0, character="s"),
            record(1, character="a"),
            record(2, character="narrator"),
            record(3, character="m"),
        ]
        batches = list(
            DialogueBatchBuilder(
                BatchingConfig(
                    target_size=4,
                    context_before=0,
                    context_after=0,
                    max_characters=1_000,
                )
            ).build(records, target_characters={"a", "m"})
        )

        self.assertEqual(len(batches), 1)
        self.assertEqual(
            [item.character for item in batches[0].targets],
            ["a", "m"],
        )
        self.assertEqual(
            [item.character for item in batches[0].context_interleaved],
            ["s", "narrator"],
        )

    def test_structural_audit_reports_unknown_characters(self) -> None:
        report = audit_dialogue(
            [record(0, character="unknown")],
            known_characters={"a"},
        )
        self.assertFalse(report["valid"])
        self.assertEqual(report["unknown_characters"], {"unknown": 1})


if __name__ == "__main__":
    unittest.main()
