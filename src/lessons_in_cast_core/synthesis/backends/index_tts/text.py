"""Punctuation-preserving frontend repairs for the pinned IndexTTS worker."""

import re

from ....performance import (AdaptationFidelity, FeatureAdaptation, PerformanceCueKind,
                            approximate_cues_with_punctuation)

VERSION = 1
_PROTECTED = r"<\|SPECIAL_TOKEN_(\d+)\|>.*?<\|SPECIAL_TOKEN_\1\|>|<[^|>\n]+\|[^>\n]+>"
_MARKS = re.compile(r"[!?！？]+(?:[ \t]+[!?！？]+)*")
_END = re.compile(r"[.!?。！？…][\"'”’»）)]*\s*$")


def normalize_expressive_punctuation(text, marker="."):
    """Keep one of each mark, preserving mixed intent and pronunciation blocks."""
    def normalize(piece):
        def replace(match):
            marks = "".join(dict.fromkeys(ch for ch in match.group().translate(str.maketrans("！？", "!?")) if ch in "!?"))
            following = piece[match.end():]
            # Keep an existing comma, period or ellipsis: it already carries a pause.
            if following.lstrip().startswith((",", ".", "…", "，", "。")):
                return marks
            return marks + marker + (" " if following and not following[0].isspace() and following[0].isalnum() else "")
        return _MARKS.sub(replace, piece)
    pieces, offset = [], 0
    for match in re.finditer(_PROTECTED, text):
        pieces.extend((normalize(text[offset:match.start()]), match.group()))
        offset = match.end()
    return "".join(pieces) + normalize(text[offset:])


def lower_punctuation(text, cues, mode="period"):
    """Add an IndexTTS pause cue without replacing expressive punctuation.

    Cue offsets refer to the original decoded text. Perform all offset-sensitive
    work before collapsing repeated marks. Do not duplicate an explicit pause
    at an existing terminal boundary. Actual emotion comes from polish.
    """
    if mode not in {"native", "comma", "period"}:
        raise ValueError("IndexTTS expressive_pause must be native, comma or period")
    if mode == "native":
        return approximate_cues_with_punctuation(text, cues)
    if not any(ch.isalnum() for ch in text):
        raise ValueError("IndexTTS cannot speak punctuation-only input; resolve it in cleaning/polish")
    retained, notes = [], []
    for index, cue in enumerate(cues):
        if not 0 <= cue.offset <= len(text):
            raise ValueError("Performance cue offset is outside the dialogue")
        before, after = text[:cue.offset].rstrip(), text[cue.offset:].lstrip()
        # Quotes may stand between a terminal mark and the annotated boundary.
        already_terminal = bool(_END.search(before) or re.match(r"^[!?！？]", after))
        if cue.kind is PerformanceCueKind.PAUSE and already_terminal:
            notes.append(FeatureAdaptation(f"performance.cues[{index}]", AdaptationFidelity.APPROXIMATED,
                "reuse the existing terminal punctuation; do not append another pause marker; exact duration is not guaranteed"))
        else:
            _, cue_notes = approximate_cues_with_punctuation(text, (cue,))
            # Retain the cue itself; insert all markers together using original offsets.
            retained.append(cue)
            notes.extend(FeatureAdaptation(f"performance.cues[{index}]", note.fidelity, note.strategy)
                         for note in cue_notes)
    lowered, _ = approximate_cues_with_punctuation(text, tuple(retained))
    normalized = normalize_expressive_punctuation(lowered, "," if mode == "comma" else ".")
    if normalized != lowered:
        notes.append(FeatureAdaptation("punctuation", AdaptationFidelity.APPROXIMATED,
            f"retain !/? intent, collapse repeated marks, and append a {mode} pause cue; leave the supplied emotion unchanged; timing is approximate"))
    return normalized, tuple(notes)
