"""Backend-neutral gain envelopes and local numeric placeholders."""

import math
import re


def extract_anchors(text: str) -> tuple[str, dict[str, int]]:
    """Offsets count decoded Unicode characters, not markup or UTF-8 bytes.

    Doubled braces quote literals. Numeric IDs are local to a single dialogue.
    Non-numeric braces are literal text, not an expression language.
    """
    if not isinstance(text, str):
        raise ValueError("Keyframe text must be a string")
    pieces, anchors, position = [], {}, 0
    for match in re.finditer(r"\{\{|\}\}|\{[0-9]+\}|[\s\S]", text):
        token = match.group()
        if token in ("{{", "}}"):
            token = token[0]
        elif re.fullmatch(r"\{[0-9]+\}", token):
            identity = token[1:-1]
            if not re.fullmatch(r"[1-9][0-9]*", identity):
                raise ValueError("Anchor IDs must be canonical positive integers")
            if identity in anchors:
                raise ValueError(f"Duplicate keyframe anchor: {identity}")
            anchors[identity] = position
            continue
        pieces.append(token)
        position += len(token)
    return "".join(pieces), anchors


def compile_keyframes(text: str, effects: list) -> dict:
    """Validate symbolic curves without knowing the generated audio duration."""
    plain, anchors = extract_anchors(text)
    if not isinstance(effects, list) or len(effects) > 16:
        raise ValueError("keyframe_effects must be an array of at most 16 curves")
    curves, referenced = [], set()
    for effect in effects:
        if not isinstance(effect, dict) or set(effect) != {"type", "interpolation", "keyframes"}:
            raise ValueError("A curve requires only type, interpolation and keyframes")
        if effect["type"] != "gain_envelope":
            raise ValueError("Unsupported keyframe effect type")
        if effect["interpolation"] not in ("linear", "smooth"):
            raise ValueError("Interpolation must be linear or smooth")
        points = effect["keyframes"]
        if not isinstance(points, list) or not 2 <= len(points) <= 128:
            raise ValueError("An envelope requires 2..128 keyframes")
        previous, compiled = -1, []
        for point in points:
            if not isinstance(point, dict) or set(point) != {"anchor", "gain"}:
                raise ValueError("A keyframe requires only anchor and gain")
            identity, gain = point["anchor"], point["gain"]
            if not isinstance(identity, str) or identity not in anchors:
                raise ValueError(f"Undefined keyframe anchor: {identity!r}")
            if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or not 0 <= gain <= 1:
                raise ValueError("Keyframe gain must be a finite linear amplitude in [0, 1]")
            offset = anchors[identity]
            if offset <= previous:
                raise ValueError("Curve keyframes must follow strictly increasing text positions")
            previous = offset
            referenced.add(identity)
            compiled.append({"anchor": identity, "gain": float(gain)})
        curves.append({"type": effect["type"], "interpolation": effect["interpolation"], "keyframes": compiled})
    if set(anchors) != referenced:
        raise ValueError(f"Unused keyframe anchors: {sorted(set(anchors) - referenced)}")
    return {"version": 1, "text": plain, "anchors": anchors, "effects": curves}


def keyframe_effects_schema() -> dict:
    """Only polish may author these optional, strictly structured effects."""
    return {"type": "array", "maxItems": 16, "items": {
        "type": "object", "additionalProperties": False,
        "required": ["type", "interpolation", "keyframes"], "properties": {
            "type": {"const": "gain_envelope"},
            "interpolation": {"enum": ["linear", "smooth"]},
            "keyframes": {"type": "array", "minItems": 2, "maxItems": 128, "items": {
                "type": "object", "additionalProperties": False, "required": ["anchor", "gain"],
                "properties": {"anchor": {"type": "string", "pattern": "^[1-9][0-9]*$"},
                               "gain": {"type": "number", "minimum": 0, "maximum": 1}},
            }},
        },
    }}
