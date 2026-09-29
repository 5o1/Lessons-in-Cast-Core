"""Optional MFA 3.x align_one bridge; needs an installed MFA and local models.

Run as a module with --dictionary and --acoustic-model. JSON stdin/stdout follows
the core alignment protocol. TextGrid does not expose calibrated confidence:
unscored boundaries require an explicit allow_unscored setting in the caller.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from ..hashing import file_hash
from ..pronunciations import load_pronunciation_lexicon


def read_intervals(text: str, suffix: str) -> list[tuple[float, float, str]]:
    """Read a single standard long TextGrid word/phone tier, including repeats."""
    result = []
    for tier in re.split(r"\bitem\s*\[\d+\]\s*:", text)[1:]:
        name = re.search(r'\bname\s*=\s*"((?:""|[^"])*)"', tier)
        if name is None or not name.group(1).endswith(suffix):
            continue
        if result:
            raise ValueError("Multiple speaker tiers are not supported for one dry voice")
        for interval in re.split(r"\bintervals\s*\[\d+\]\s*:", tier)[1:]:
            match = re.search(r'xmin\s*=\s*([\d.eE+-]+)\s+xmax\s*=\s*([\d.eE+-]+)\s+text\s*=\s*"((?:""|[^"])*)"', interval)
            if match is None:
                raise ValueError("Malformed MFA TextGrid interval")
            label = match[3].replace('""', '"')
            if label:
                start, end = float(match[1]), float(match[2])
                if not 0 <= start < end or (result and start < result[-1][1] - .000001):
                    raise ValueError("Invalid MFA interval ordering")
                result.append((start, end, label))
    if not result:
        raise ValueError(f"No MFA {suffix} intervals")
    return result


def boundary_times(text, offsets, words, phones, lexicon):
    """Map only exact word boundaries or explicit lexicon-unit boundaries.

    Unit labels must concatenate to the written word; their ARPABET phones must
    exactly match the aligned phone sequence (ignoring stress suffixes). This
    deliberately refuses unknown word interiors and positions inside a digraph.
    """
    tokens = list(re.finditer(r"[^\W_]+(?:['’-][^\W_]+)*", text))
    if len(tokens) != len(words):
        raise ValueError("MFA word sequence differs from the supplied transcript")
    entries = {entry.term.casefold(): entry for entry in lexicon.entries}
    normalize = lambda value: re.sub(r"[012]$", "", value).upper()
    mapping = {}
    for token, (start, end, label) in zip(tokens, words):
        if token.group().casefold() != label.casefold():
            raise ValueError("MFA word labels differ from the supplied transcript")
        mapping[token.start()] = (start, "word_alignment")
        mapping[token.end()] = (end, "word_alignment")
        if not any(token.start() < x < token.end() for x in offsets):
            continue
        entry = entries.get(token.group().casefold())
        units = entry.prosody.units if entry else ()
        if not units or "".join(unit.label for unit in units).casefold() != token.group().casefold():
            raise ValueError(f"Interior anchors need spelling-aligned pronunciation units for {token.group()!r}")
        expected = [normalize(phone) for unit in units for phone in (unit.for_system("arpabet") or "").split() if phone != "."]
        full = [normalize(phone) for phone in (entry.for_system("arpabet") or "").split() if phone != "."]
        actual = [(a, b, label) for a, b, label in phones if a >= start - .000001 and b <= end + .000001]
        if not expected or expected != full or expected != [normalize(x[2]) for x in actual]:
            raise ValueError("MFA phones differ from the explicit pronunciation units")
        char_offset, phone_offset = token.start(), 0
        for unit in units[:-1]:
            unit_phones = [p for p in (unit.for_system("arpabet") or "").split() if p != "."]
            if not unit_phones or not unit.label:
                raise ValueError("Pronunciation units need both spelling and phones")
            char_offset += len(unit.label)
            phone_offset += len(unit_phones)
            mapping[char_offset] = (actual[phone_offset][0], "phone_alignment")
    result = []
    for offset in offsets:
        if offset not in mapping:
            raise ValueError(f"No measured phoneme boundary for text offset {offset}; no proportional fallback")
        time, source = mapping[offset]
        result.append({"offset": offset, "time": time, "confidence": None, "source": source})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mfa", default="mfa")
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--acoustic-model", type=Path, required=True)
    parser.add_argument("--keep-stress", action="store_true")
    args = parser.parse_args()
    for path in (args.dictionary, args.acoustic_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    request = json.load(sys.stdin)
    audio = Path(request["audio_path"])
    if request["version"] != 1 or file_hash(audio) != request["audio_sha256"]:
        raise ValueError("Audio binding mismatch")
    lexicon = load_pronunciation_lexicon(Path(request["pronunciations_path"]))
    with tempfile.TemporaryDirectory(prefix="lic-mfa-", dir=audio.parent) as directory:
        work = Path(directory)
        transcript, dictionary, output = work / "speech.lab", work / "dictionary.dict", work / "aligned.TextGrid"
        transcript.write_text(request["text"], encoding="utf-8")
        # Override dictionary pronunciations only for words present in this line.
        tokens = {m.group().casefold() for m in re.finditer(r"[^\W_]+(?:['’-][^\W_]+)*", request["text"])}
        overrides = {e.term.casefold(): e.for_system("arpabet") for e in lexicon.entries
                     if e.term.casefold() in tokens and e.for_system("arpabet")}
        with args.dictionary.open(encoding="utf-8") as source, dictionary.open("w", encoding="utf-8") as target:
            for line in source:
                fields = line.split()
                if fields and fields[0].casefold() not in overrides:
                    target.write(line.rstrip("\n") + "\n")
            for word, pronunciation in overrides.items():
                phones = [p for p in pronunciation.split() if p != "."]
                if not args.keep_stress:
                    phones = [re.sub(r"[012]$", "", p) for p in phones]
                target.write(word + "\t" + " ".join(phones) + "\n")
        result = subprocess.run([args.mfa, "align_one", str(audio.resolve()), str(transcript),
                                 str(dictionary), str(args.acoustic_model.resolve()), str(output),
                                 "--output_format", "long_textgrid", "--temporary_directory", str(work / "mfa")],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=540)
        if result.returncode:
            raise RuntimeError(f"MFA alignment failed (exit {result.returncode}); check model/dictionary compatibility")
        grid = output.read_text(encoding="utf-8")
        boundaries = boundary_times(request["text"], request["offsets"], read_intervals(grid, "words"),
                                    read_intervals(grid, "phones"), lexicon)
        json.dump({"version": 1, "text": request["text"], "audio_sha256": request["audio_sha256"],
                   "boundaries": boundaries}, sys.stdout)


if __name__ == "__main__":
    main()
