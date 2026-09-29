"""Strict, backend-neutral inline emotion markup for cleaned dialogue."""

from dataclasses import asdict, dataclass, replace
from html import escape
import re
from typing import Collection
from xml.etree import ElementTree


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    text: str
    emotion: str | None
    voice: str | None = None
    arbitrary_emotion: str | None = None

    def __post_init__(self):
        if (self.emotion is None) == (self.arbitrary_emotion is None):
            raise ValueError("Exactly one preset emotion or arbitrary_emotion is required per span")
        if self.emotion is not None and (not isinstance(self.emotion, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", self.emotion)):
            raise ValueError("Preset emotion must be one semantic label")
        if self.arbitrary_emotion is not None and (not isinstance(self.arbitrary_emotion, str) or not self.arbitrary_emotion.strip()):
            raise ValueError("arbitrary_emotion description cannot be empty")

    def to_dict(self) -> dict:
        result = asdict(self)
        if self.arbitrary_emotion is None:
            result.pop("arbitrary_emotion")
        return result


def parse_emotion_markup(text: str, allowed_emotions: Collection[str] | None = None) -> tuple[SpeechSegment, ...]:
    """Parse emotion spans, optionally grouped by a dynamic voice wrapper.

    Whitespace between elements is retained. All non-whitespace speech must be
    enclosed. XML declarations, DTDs, comments, nested voices/emotions and extra
    attributes are unsupported. Entity decoding happens exactly once.
    """
    if "<!" in text or "<?" in text:
        raise ValueError("Declarations, comments and DTDs are not speech markup")
    try:
        root = ElementTree.fromstring(f"<speech>{text}</speech>")
    except ElementTree.ParseError as exc:
        raise ValueError(f"Malformed emotion markup: {exc}") from exc
    if not len(root) or (root.text or "").strip():
        raise ValueError("All speech must be wrapped in emotion tags")
    segments = []
    prefix = root.text or ""

    def append_whitespace(value: str) -> None:
        nonlocal prefix
        if value.strip():
            raise ValueError("Unlabelled speech outside emotion tags")
        if value and segments:
            s = segments[-1]
            segments[-1] = replace(s, text=s.text + value)
        elif value:
            prefix += value

    def emotion_element(element, voice: str | None = None) -> None:
        if len(element):
            raise ValueError("Preset emotion and arbitrary_emotion must not overlap or nest")
        arbitrary = None
        if element.tag == "arbitrary_emotion" and set(element.attrib) == {"description"}:
            emotion = None
            arbitrary = element.attrib["description"]
        elif element.tag == "emotion" and set(element.attrib) == {"name"}:
            emotion = element.attrib["name"]
        else:
            raise ValueError("Expected emotion with name OR arbitrary_emotion with description; extra attributes are forbidden")
        if emotion is not None and (not re.fullmatch(r"[a-z][a-z0-9_]*", emotion) or (allowed_emotions is not None and emotion not in allowed_emotions)):
            raise ValueError(f"Unknown emotion: {emotion!r}")
        if not (element.text or "").strip():
            raise ValueError("Emotion spans cannot be empty")
        spoken = prefix if not segments else ""
        segments.append(SpeechSegment(spoken + element.text, emotion, voice, arbitrary))

    for element in root:
        if element.tag == "voice":
            if set(element.attrib) != {"name"} or not len(element):
                raise ValueError("A voice tag needs a name and one or more emotion spans")
            voice = validate_voice_name(element.attrib["name"])
            append_whitespace(element.text or "")
            for child in element:
                emotion_element(child, voice)
                append_whitespace(child.tail or "")
        else:
            emotion_element(element)
        append_whitespace(element.tail or "")
    return tuple(segments)


def validate_voice_name(name: str) -> str:
    """Names are dynamic file stems, never paths or a fixed emotion enum."""
    if not isinstance(name, str) or not name.strip() or name in {".", ".."} or any(c in name for c in "/\\\x00"):
        raise ValueError("Voice name must be a nonempty file stem, not a path")
    return name


def emotion_markup(segments: tuple[SpeechSegment, ...]) -> str:
    """Serialize spans without exposing raw text as markup."""
    parts = []
    for s in segments:
        if s.arbitrary_emotion is not None:
            tag, attribute, value = "arbitrary_emotion", "description", s.arbitrary_emotion
        else:
            tag, attribute, value = "emotion", "name", s.emotion
        text = (f'<{tag} {attribute}="{escape(value, quote=True)}">'
                f'{escape(s.text, quote=False)}</{tag}>')
        if s.voice is not None:
            text = f'<voice name="{escape(s.voice, quote=True)}">{text}</voice>'
        parts.append(text)
    return "".join(parts)
