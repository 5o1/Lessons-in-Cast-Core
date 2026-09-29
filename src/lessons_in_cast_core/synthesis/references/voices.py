"""Resolve dynamic voice names next to a profile's default reference audio."""

from pathlib import Path

from ...hashing import file_hash
from ...speech_markup import validate_voice_name


AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".ogg", ".opus", ".mp3", ".m4a", ".aac", ".aif", ".aiff"})


def reference_audio_files(default: Path) -> tuple[Path, ...]:
    directory = default.parent.resolve()
    if not directory.is_dir():
        return ()
    return tuple(path for path in sorted(directory.iterdir())
                 if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
                 and path.resolve().parent == directory)


def list_voice_references(default: Path) -> dict[str, Path]:
    result = {}
    for path in reference_audio_files(default):
        name = validate_voice_name(path.stem)
        if name in result:
            raise ValueError(f"Ambiguous voice {name!r}: multiple audio files in {default.parent}")
        result[name] = path.resolve()
    return result


def resolve_voice_reference(default: Path, name: str) -> Path:
    validate_voice_name(name)
    matches = list_voice_references(default)
    if name not in matches:
        raise ValueError(f"Voice {name!r} has no matching audio file in {default.parent}")
    return matches[name]


def voice_reference_fingerprint(default: Path) -> dict[str, str]:
    return {str(path): file_hash(path) for path in reference_audio_files(default)}
