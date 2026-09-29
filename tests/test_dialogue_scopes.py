from __future__ import annotations

import unittest

from lessons_in_cast_core.dialogue import DialogueScope, SpeakerOverride

from .helpers import record


class DialogueScopeTests(unittest.TestCase):
    def test_scope_selects_boundaries_and_applies_speaker_override(self) -> None:
        records = [
            record(0, character="q", dialogue="before", filename="game/script.rpy"),
            record(1, character="q", dialogue="wake", filename="game/script.rpy"),
            record(2, character="a", dialogue="middle", filename="game/script.rpy"),
            record(3, character="a", dialogue="end", filename="game/script.rpy"),
            record(4, character="a", dialogue="after", filename="game/script.rpy"),
        ]
        scope = DialogueScope(
            name="demo",
            description="",
            source="game/script.rpy",
            start_identifier="line_1",
            end_identifier="line_3",
            inclusive=True,
            context_characters=("a",),
            voice_characters=("a",),
            context_only_characters=(),
            speaker_overrides=(
                SpeakerOverride("q", "a", "line_1", "line_2"),
            ),
        )

        selected = scope.apply(records)

        self.assertEqual([item.identifier for item in selected], ["line_1", "line_2", "line_3"])
        self.assertEqual([item.character for item in selected], ["a", "a", "a"])
        self.assertEqual([item.sequence for item in selected], [0, 1, 2])

    def test_scope_fails_when_a_boundary_is_missing(self) -> None:
        scope = DialogueScope(
            name="demo",
            description="",
            source="game/script.rpy",
            start_identifier="missing",
            end_identifier="line_1",
            inclusive=True,
            context_characters=(),
            voice_characters=(),
            context_only_characters=(),
            speaker_overrides=(),
        )
        with self.assertRaisesRegex(ValueError, "start identifier"):
            scope.apply([record(1, filename="game/script.rpy")])


if __name__ == "__main__":
    unittest.main()
